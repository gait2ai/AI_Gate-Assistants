"""
AI_Gate Website Processor - website_scraper.py (Version 2.2 - Final Enhanced)

A sophisticated web crawler and content processor that generates structured JSON data
from target websites for integration with AI_Gate intelligent systems.

This version is fully compatible with website_researcher.py requirements.
Enhanced with JavaScript support, improved SSL verification, flexible www handling,
and a more robust page discovery mechanism with configurable depth and trigger.
"""

import json
import os
import time
import re
import hashlib
from urllib.parse import urljoin, urlparse # Removed: urllib.robotparser (as robots.txt is ignored)
from bs4 import BeautifulSoup
import requests
from typing import Dict, List, Tuple, Optional, Any 
from datetime import datetime
import logging
from http.client import HTTPConnection

# JavaScript support imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Debug configuration
DEBUG = False # Set to True for verbose debug logging
if DEBUG:
    HTTPConnection.debuglevel = 1
    logging.basicConfig(level=logging.DEBUG) 
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

# Directories
CONFIG_DIR = "./config"
OUTPUT_DIR = "./data"
LOG_DIR = "./logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging
log_file_path = os.path.join(LOG_DIR, f"website_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

root_logger_for_setup = logging.getLogger()
if root_logger_for_setup.hasHandlers():
    for handler in root_logger_for_setup.handlers[:]:
        root_logger_for_setup.removeHandler(handler)

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AI_Gate_Processor")

logger_init_selenium = logging.getLogger("AI_Gate_Processor_Init_Selenium")
if SELENIUM_AVAILABLE:
    logger_init_selenium.info("Selenium libraries loaded successfully. JavaScript rendering is available.")
else:
    logger_init_selenium.warning("Selenium libraries not available. JavaScript rendering will be disabled.")


class ContentAnalyzer:
    """Utility class for content analysis and categorization."""
    
    @staticmethod
    def generate_content_id(url: str, title: str) -> str:
        content_string = f"{url}_{title}_{datetime.now().date()}"
        return hashlib.md5(content_string.encode('utf-8')).hexdigest()[:12]
    
    @staticmethod
    def generate_summary(content: str, max_length: int = 200) -> str:
        if not content: return ""
        cleaned_content = re.sub(r'\s+', ' ', content.strip())
        if len(cleaned_content) <= max_length: return cleaned_content
        sentences = re.split(r'[.!?]+', cleaned_content)
        summary = ""
        for sentence_part in sentences:
            sentence = sentence_part.strip()
            if not sentence: continue
            potential_summary = f"{summary} {sentence}".strip()
            if len(potential_summary) <= max_length:
                summary = potential_summary
            else:
                if summary: break
                else:
                    words = sentence[:max_length - 3].split()
                    summary = ' '.join(words[:-1]) + "..." if len(words) > 1 else sentence[:max_length - 3] + "..."
                    break
        if not summary:
            words = cleaned_content[:max_length - 3].split()
            summary = ' '.join(words[:-1]) + "..." if len(words) > 1 else cleaned_content[:max_length - 3] + "..."
        return summary[:max_length]
    
    @staticmethod
    def categorize_content(title: str, content: str, keywords: List[str]) -> str:
        title_lower = title.lower() if title else ""
        content_lower = content.lower() if content else ""
        keywords_lower = [k.lower() for k in keywords]
        categories = {
            'education': ['course', 'tutorial', 'learn', 'guide', 'education', 'training', 'lesson', 'study', 'academic', 'university', 'school', 'class'],
            'technical': ['api', 'code', 'programming', 'developer', 'software', 'algorithm', 'technical', 'documentation', 'implementation', 'framework'],
            'research': ['research', 'study', 'analysis', 'paper', 'publication', 'findings', 'methodology', 'data', 'results', 'experiment'],
            'news': ['news', 'announcement', 'update', 'press', 'release', 'breaking', 'latest', 'report', 'event', 'happening'],
            'product': ['product', 'service', 'feature', 'solution', 'tool', 'application', 'platform', 'system'],
            'about': ['about', 'company', 'team', 'mission', 'vision', 'history', 'organization', 'who we are', 'our story'],
            'support': ['help', 'support', 'faq', 'troubleshoot', 'problem', 'issue', 'contact', 'assistance', 'customer service'],
            'blog': ['blog', 'post', 'article', 'opinion', 'editorial', 'commentary']
        }
        category_scores: Dict[str, int] = {}
        for category, patterns in categories.items():
            score = sum(title_lower.count(p) * 3 + content_lower.count(p) + sum(k.count(p) for k in keywords_lower) * 2 for p in patterns)
            if score > 0: category_scores[category] = score
        return max(category_scores, key=category_scores.get) if category_scores else 'general'
    
    @staticmethod
    def calculate_metrics(content: str, title: str) -> Dict[str, int]:
        return {'word_count': len(content.split()) if content else 0, 'char_count': len(content) if content else 0, 'title_word_count': len(title.split()) if title else 0}
    
    @staticmethod
    def create_structured_content(soup: BeautifulSoup) -> Dict[str, Any]:
        structured: Dict[str, Any] = {}
        headings = [{'level': level, 'text': h.text.strip()} for level in range(1, 7) for h in soup.find_all(f'h{level}') if h.text.strip()]
        if headings: structured['headings'] = headings
        
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True) and len(p.get_text(strip=True)) > 20][:5]
        if paragraphs: structured['paragraphs'] = paragraphs
        
        page_lists = [{'type': ul_ol.name, 'items': [li.get_text(strip=True) for li in ul_ol.find_all('li') if li.get_text(strip=True)][:10]} for ul_ol in soup.find_all(['ul', 'ol'])[:3] if any(li.get_text(strip=True) for li in ul_ol.find_all('li'))]
        if page_lists: structured['lists'] = page_lists
        return structured

class JavaScriptRenderer:
    def __init__(self):
        self.driver: Optional[webdriver.Remote] = None
        self.driver_type: Optional[str] = None
        self.wait_timeout = 15
        self.page_load_timeout = 45
        
    def _setup_chrome_driver(self) -> Optional[webdriver.Chrome]:
        chrome_options = ChromeOptions()
        chrome_options.add_argument('--headless=new'); chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage'); chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--blink-settings=imagesEnabled=false')
        chrome_options.add_argument(f'--user-agent=AI_Gate_WebCrawler/2.2 (+https://ai-gate.org/bot)') # Version bump
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(self.page_load_timeout)
            return driver
        except WebDriverException as e:
            logger.warning(f"Chrome driver setup failed (WebDriverException): {e}. Ensure chromedriver is in PATH.")
            return None
        except Exception as e:
            logger.warning(f"Chrome driver setup failed (General Exception): {e}")
            return None
            
    def _setup_firefox_driver(self) -> Optional[webdriver.Firefox]:
        firefox_options = FirefoxOptions(); firefox_options.add_argument('--headless')
        firefox_options.set_preference('permissions.default.image', 2)
        firefox_options.set_preference('general.useragent.override', f'AI_Gate_WebCrawler/2.2 (+https://ai-gate.org/bot)') # Version bump
        try:
            driver = webdriver.Firefox(options=firefox_options)
            driver.set_page_load_timeout(self.page_load_timeout)
            return driver
        except WebDriverException as e:
            logger.warning(f"Firefox driver setup failed (WebDriverException): {e}. Ensure geckodriver is in PATH.")
            return None
        except Exception as e:
            logger.warning(f"Firefox driver setup failed (General Exception): {e}")
            return None
            
    def initialize_driver(self) -> bool:
        if not SELENIUM_AVAILABLE: return False
        driver_preference = ['Chrome', 'Firefox']
        for driver_name in driver_preference:
            setup_func = getattr(self, f'_setup_{driver_name.lower()}_driver', None)
            if not setup_func: continue
            try:
                logger.info(f"Attempting to initialize {driver_name} WebDriver...")
                self.driver = setup_func()
                if self.driver:
                    self.driver_type = driver_name
                    logger.info(f"Successfully initialized {driver_name} WebDriver.")
                    return True
            except Exception as e: logger.warning(f"Failed to initialize {driver_name} during setup: {e}")
        logger.warning("No WebDriver (Chrome/Firefox) initialized. JS rendering disabled.")
        return False
        
    def render_page(self, url: str, wait_for_element_tag: str = "body") -> Tuple[str, bool]:
        if not self.driver: return "", False
        try:
            logger.debug(f"Rendering JS page: {url} with {self.driver_type}")
            self.driver.get(url)
            try: WebDriverWait(self.driver, self.wait_timeout).until(EC.presence_of_element_located((By.TAG_NAME, wait_for_element_tag)))
            except TimeoutException: logger.debug(f"Timeout waiting for '{wait_for_element_tag}' on {url}. Proceeding.")
            time.sleep(3)
            return self.driver.page_source, True
        except WebDriverException as e:
            logger.warning(f"WebDriver error rendering {url}: {e}")
            if "target window already closed" in str(e).lower() or "session deleted" in str(e).lower():
                logger.error(f"Critical WebDriver error for {url}. Re-initializing driver.")
                self.close(); self.initialize_driver()
            return "", False
        except Exception as e:
            logger.error(f"Unexpected error rendering {url}: {e}", exc_info=DEBUG)
            return "", False
            
    def close(self):
        if self.driver:
            name = self.driver_type or "WebDriver"
            try: self.driver.quit(); logger.info(f"{name} closed.")
            except Exception as e: logger.warning(f"Error closing {name}: {e}")
            finally: self.driver = None; self.driver_type = None

class WebsiteProcessor:
    def __init__(self, config_path: str = os.path.join(CONFIG_DIR, "sites.json")):
        self.config = self._load_config(config_path)
        self.base_url = self._normalize_url(self.config["website_url"])
        self.session = self._configure_session()
        self.start_time = time.time()
        self.pages_data: List[Dict] = []
        self.content_analyzer = ContentAnalyzer()
        self.js_renderer = JavaScriptRenderer()
        self.use_javascript = self.js_renderer.initialize_driver()
        logger.info(f"Initializing processor for {self.base_url}")
        logger.info(f"JavaScript rendering: {'Enabled' if self.use_javascript else 'Disabled'}")

    def _load_config(self, config_path: str) -> Dict:
        if not os.path.exists(config_path): raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f: return json.load(f)

    def _normalize_url(self, url: str) -> str:
        return ('https://' if not url.startswith(('http://', 'https://')) else '') + url.rstrip('/')

    def _configure_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'User-Agent': f'AI_Gate_WebCrawler/2.2 (+https://ai-gate.org/bot)', # Version bump
            'Accept-Language': 'en-US,en;q=0.9,ar;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
        })
        session.max_redirects = 5; session.timeout = 20; session.verify = True
        return session

    def _verify_site_accessibility(self) -> Tuple[bool, str]:
        try:
            logger.info("Verifying website accessibility...")
            resp = self.session.get(self.base_url, timeout=20)
            resp.raise_for_status()
            return True, "Website accessible with SSL verification"
        except requests.exceptions.SSLError as ssl_e:
            logger.warning(f"SSL verification failed for {self.base_url}: {ssl_e}. Retrying without verification...")
            self.session.verify = False # Disable for this session
            try:
                resp_no_ssl = self.session.get(self.base_url, timeout=20)
                resp_no_ssl.raise_for_status()
                logger.info(f"Accessed {self.base_url} without SSL verification. SSL verification disabled for session.")
                return True, "Website accessible (SSL verification disabled for this session)"
            except Exception as e_no_ssl:
                logger.error(f"Access failed for {self.base_url} even without SSL: {e_no_ssl}")
                return False, f"SSL error, and no-SSL access failed: {e_no_ssl}"
        except requests.exceptions.RequestException as req_e: # Catch other request exceptions
            logger.error(f"Request exception for {self.base_url}: {req_e}")
            return False, f"Request exception: {req_e}"
        except Exception as e:
            logger.error(f"Unexpected error during access check for {self.base_url}: {e}", exc_info=DEBUG)
            return False, f"Access check failed (General): {e}"

    def _check_url_permissions(self, url: str) -> bool:
        logger.debug(f"URL permission check for {url}: Bypassed (assumed True, robots.txt not checked).")
        return True

    def _is_valid_page_url(self, url: str) -> bool:
        if not url or not url.startswith("http"): return False
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc: return False
        excluded_exts = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.webp', '.bmp', '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.css', '.js', '.xml', '.rss', '.atom', '.zip', '.tar', '.gz', '.rar', '.7z', '.exe', '.dmg', '.apk')
        if parsed.path.lower().endswith(excluded_exts): return False
        excluded_paths = ['/wp-admin/', '/admin/', '/login', '/logout', '/register', '/cart', '/checkout', '/feed/', '/wp-json/', '/api/']
        if any(ex_path in parsed.path.lower() for ex_path in excluded_paths): return False
        if len(parsed.query) > 200: return False # Increased query length tolerance slightly
        return True

    def _requires_javascript(self, url: str, html_content: str) -> bool:
        if not self.use_javascript or not html_content: return False
        soup = BeautifulSoup(html_content, 'html.parser')
        for s_tag in soup(['script', 'style', 'noscript', 'meta', 'link']): s_tag.decompose()
        text = soup.get_text(strip=True, separator=' ')
        if len(text) < 350: # Adjusted threshold
            logger.debug(f"Page {url} has minimal static content ({len(text)} chars), may require JS.")
            js_indicators = ['<app-root>', 'id="root"', 'data-reactroot', 'ng-app', '__NEXT_DATA__', 'application/ld+json', '__INITIAL_STATE__']
            if any(indicator in html_content.lower() for indicator in js_indicators):
                logger.info(f"JS indicators found on {url} with minimal content. Flagging for JS rendering.")
                return True
            return True 
        return False

    def _crawl_sitemap(self) -> List[str]:
        sitemap_variants = ["sitemap.xml", "sitemap_index.xml", "sitemap.php", "sitemap.txt"]
        discovered_urls_set = set()
        for variant in sitemap_variants:
            sitemap_url = urljoin(self.base_url, variant)
            try:
                logger.info(f"Checking sitemap: {sitemap_url}")
                resp = self.session.get(sitemap_url, timeout=15); resp.raise_for_status()
                content_type = resp.headers.get('Content-Type', '').lower()
                urls_from_variant = set()
                if 'xml' in content_type or variant.endswith(('.xml', '.php')):
                    soup = BeautifulSoup(resp.content, 'xml')
                    sitemap_tags = soup.find_all('sitemap') # Check for sitemap index
                    if sitemap_tags:
                        logger.info(f"Sitemap index found in {variant}. Extracting sub-sitemap URLs.")
                        for s_tag in sitemap_tags:
                            sub_sitemap_loc = s_tag.find('loc')
                            if sub_sitemap_loc and sub_sitemap_loc.text:
                                # Simple fetch for sub-sitemaps (non-recursive for this version to keep it manageable)
                                try:
                                    sub_s_resp = self.session.get(sub_sitemap_loc.text.strip(), timeout=15)
                                    if sub_s_resp.status_code == 200:
                                        sub_s_soup = BeautifulSoup(sub_s_resp.content, 'xml')
                                        urls_from_variant.update(l.text.strip() for l in sub_s_soup.find_all('loc') if l.text and self._is_valid_page_url(l.text.strip()))
                                except Exception as sub_e: logger.warning(f"Failed to process sub-sitemap {sub_sitemap_loc.text.strip()}: {sub_e}")
                    else: # Regular sitemap
                        urls_from_variant.update(l.text.strip() for l in soup.find_all('loc') if l.text and self._is_valid_page_url(l.text.strip()))
                elif 'text/plain' in content_type or variant.endswith('.txt'):
                    urls_from_variant.update(line.strip() for line in resp.text.splitlines() if line.strip() and self._is_valid_page_url(line.strip()))
                if urls_from_variant: logger.info(f"Found {len(urls_from_variant)} valid page URLs in {variant}"); discovered_urls_set.update(urls_from_variant)
            except requests.exceptions.RequestException as e_req: logger.warning(f"Sitemap request error for {sitemap_url}: {e_req}")
            except Exception as e_gen: logger.warning(f"Sitemap processing error for {sitemap_url} ({variant}): {e_gen}")
        logger.info(f"Total valid page URLs from all sitemaps: {len(discovered_urls_set)}")
        return sorted(list(discovered_urls_set))

    def _discover_urls_via_crawl(self, start_urls: Optional[List[str]] = None, 
                                 max_pages_to_collect: int = 400, # TARGETED: Max pages to collect from this crawl
                                 max_crawl_depth: int = 5,    # TARGETED: Increased depth
                                 max_queue_multiplier: int = 5 # Safety: Queue size not to exceed max_pages_to_collect * multiplier
                                 ) -> List[str]:
        if start_urls is None: start_urls = [self.base_url]
        logger.info(f"Starting crawl. Max pages to collect: {max_pages_to_collect}, Max depth: {max_crawl_depth}, Start URLs: {len(start_urls)}")
        
        collected_valid_pages = set()
        queue: List[Tuple[str, int]] = []
        visited_for_queue = set() # URLs added to queue or processed to avoid re-queueing

        for s_url in start_urls:
            if s_url not in visited_for_queue:
                queue.append((s_url, 0)); visited_for_queue.add(s_url)

        base_netloc_norm = urlparse(self.base_url).netloc.replace("www.", "", 1)
        
        pages_fetched_count = 0 # How many pages we actually GET request for
        max_fetch_limit = max_pages_to_collect * max_queue_multiplier # Safety break for fetched pages

        while queue and len(collected_valid_pages) < max_pages_to_collect and pages_fetched_count < max_fetch_limit:
            current_url, depth = queue.pop(0)
            pages_fetched_count += 1

            if depth > max_crawl_depth: continue

            # Add current URL to collected_valid_pages if it's valid and we still need more pages
            if self._is_valid_page_url(current_url) and len(collected_valid_pages) < max_pages_to_collect:
                collected_valid_pages.add(current_url)
                if DEBUG: logger.debug(f"CRAWL: Valid page added to collection: {current_url} (Total collected: {len(collected_valid_pages)})")
                if len(collected_valid_pages) >= max_pages_to_collect: break 

            try:
                html_content, success_fetch = "", False
                try:
                    if DEBUG: logger.debug(f"CRAWL: Fetching (requests): {current_url} (Depth: {depth}, Fetched: {pages_fetched_count})")
                    response = self.session.get(current_url, timeout=self.session.timeout); response.raise_for_status()
                    html_content = response.text; success_fetch = True
                except requests.exceptions.RequestException as req_err: logger.warning(f"CRAWL: Request failed for {current_url}: {req_err}")
                
                if success_fetch and self._requires_javascript(current_url, html_content) and self.use_javascript:
                    if DEBUG: logger.debug(f"CRAWL: JS rendering for {current_url}")
                    rendered_html, js_s = self.js_renderer.render_page(current_url)
                    if js_s and rendered_html: html_content = rendered_html
                    else: logger.warning(f"CRAWL: JS rendering failed for {current_url}, using static HTML (if any).")
                
                if not html_content: continue

                soup = BeautifulSoup(html_content, 'html.parser')
                for a_tag in soup.find_all('a', href=True):
                    href, full_url_abs, parsed_f_url, full_url_norm = a_tag['href'].strip(), "", None, ""
                    if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')): continue
                    try:
                        full_url_abs = urljoin(current_url, href)
                        parsed_f_url = urlparse(full_url_abs)
                        full_url_norm = parsed_f_url._replace(fragment="").geturl()
                    except ValueError: continue # If URL is badly formed

                    link_netloc_norm = parsed_f_url.netloc.replace("www.", "", 1)
                    if link_netloc_norm != base_netloc_norm: continue 

                    if self._is_valid_page_url(full_url_norm):
                        if len(collected_valid_pages) < max_pages_to_collect:
                            is_new = full_url_norm not in collected_valid_pages
                            collected_valid_pages.add(full_url_norm)
                            if is_new and DEBUG: logger.debug(f"CRAWL: Valid page added to collection via link: {full_url_norm}")
                        
                        if full_url_norm not in visited_for_queue and depth < max_crawl_depth:
                            if len(visited_for_queue) < max_fetch_limit : # Control queue growth based on fetch limit
                                visited_for_queue.add(full_url_norm)
                                queue.append((full_url_norm, depth + 1))
                                if DEBUG: logger.debug(f"CRAWL: Page added to queue: {full_url_norm} (Depth: {depth+1})")
                
                if len(queue) > 0: time.sleep(0.35) # Increased crawl delay slightly

            except Exception as e: logger.warning(f"CRAWL: Error processing page {current_url} for links: {e}", exc_info=DEBUG)
        
        logger.info(f"Crawl finished. Fetched ~{pages_fetched_count} URLs. Collected {len(collected_valid_pages)} unique valid page URLs.")
        return list(collected_valid_pages)

    def _extract_keywords(self, soup: BeautifulSoup, text_content: str) -> List[str]:
        keywords_set = set()
        if soup.find('meta', attrs={'name': re.compile(r'keywords', re.I)}) and soup.find('meta', attrs={'name': re.compile(r'keywords', re.I)}).get('content'):
            keywords_set.update(k.strip().lower() for k in soup.find('meta', attrs={'name': re.compile(r'keywords', re.I)})['content'].split(',') if k.strip() and len(k.strip()) > 2)
        if soup.find('meta', attrs={'name': re.compile(r'description', re.I)}) and soup.find('meta', attrs={'name': re.compile(r'description', re.I)}).get('content'):
            keywords_set.update(re.findall(r'\b[a-zA-Z]{4,}\b', soup.find('meta', attrs={'name': re.compile(r'description', re.I)})['content'].lower())[:7])
        for h_name in ['h1', 'h2']:
            for h_el in soup.find_all(h_name):
                if h_el.text: keywords_set.update(re.findall(r'\b[a-zA-Z]{4,}\b', h_el.text.lower())[:5])
        if soup.title and soup.title.string: keywords_set.update(re.findall(r'\b[a-zA-Z]{4,}\b', soup.title.string.lower()))
        
        if len(keywords_set) < 7 and text_content: # Adjusted threshold
            logger.debug("Few keywords from meta/headings, trying main text.")
            text_words = re.findall(r'\b[a-zA-Z]{5,15}\b', text_content.lower())
            if text_words:
                from collections import Counter
                word_counts = Counter(text_words)
                stopwords = {"the", "and", "for", "with", "this", "that", "our", "are", "you", "your", "not", "all", "from", "more", "about", "also", "have", "was", "but", "can", "page", "site", "click", "here", "content"}
                for word, _ in word_counts.most_common(20): # Check more common words
                    if word not in stopwords and len(keywords_set) < 20: keywords_set.add(word)
        
        return [kw for kw in list(keywords_set) if len(kw) >=3][:20]

    def _process_page(self, url: str) -> bool:
        try:
            logger.info(f"Processing page: {url}")
            html_content, js_used = "", False
            try:
                resp = self.session.get(url, timeout=self.session.timeout); resp.raise_for_status()
                html_content = resp.text
            except requests.exceptions.RequestException as req_e:
                logger.warning(f"Initial HTTP request for {url} failed: {req_e}.")
                if not self.use_javascript: return False # Cannot proceed

            if (not html_content or self._requires_javascript(url, html_content)) and self.use_javascript:
                logger.info(f"Using JS rendering for {url}")
                rendered_html, success_js = self.js_renderer.render_page(url)
                if success_js and rendered_html: html_content, js_used = rendered_html, True
                elif not html_content: logger.error(f"Both HTTP and JS rendering failed for {url}"); return False
                else: logger.warning(f"JS rendering failed for {url}, using static HTML (if any).")
            
            if not html_content: logger.error(f"No HTML content for {url}"); return False

            soup = BeautifulSoup(html_content, 'html.parser')
            title = (soup.find('title').string.strip() if soup.find('title') and soup.find('title').string else urlparse(url).path) or "Untitled"
            for s_tag_remove in soup(["script", "style", "noscript", "link", "meta", "header", "nav", "footer", "aside"]): s_tag_remove.decompose() # More aggressive cleaning
            
            main_area = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find(class_=re.compile(r'content|main|article|post-body|entry', re.I)) or soup.body
            text_raw = main_area.get_text(separator=' ', strip=True) if main_area else soup.get_text(separator=' ', strip=True)
            cleaned_text = re.sub(r'\s+', ' ', text_raw).strip()
            if len(cleaned_text) > 75000: cleaned_text = cleaned_text[:75000] + "..."

            images = list(set(urljoin(url, i['src']) for i in soup.find_all('img', src=True) if i['src'] and not i['src'].startswith('data:image') and len(i['src']) > 4 and urljoin(url,i['src']).startswith('http')))[:20]
            page_links_raw = list(set(urljoin(url, a['href']) for a in soup.find_all('a', href=True) if a['href'] and not a['href'].startswith(('#', 'mailto:', 'tel:', 'javascript:')) and urljoin(url,a['href']).startswith('http')))
            
            base_netloc_n = urlparse(self.base_url).netloc.replace("www.", "", 1)
            internal_l, external_l = [l for l in page_links_raw if urlparse(l).netloc.replace("www.","",1) == base_netloc_n], [l for l in page_links_raw if urlparse(l).netloc.replace("www.","",1) != base_netloc_n]
            
            kws = self._extract_keywords(soup, cleaned_text)
            c_id, summ, cat, mets, struct_c = self.content_analyzer.generate_content_id(url, title), self.content_analyzer.generate_summary(cleaned_text), self.content_analyzer.categorize_content(title, cleaned_text, kws), self.content_analyzer.calculate_metrics(cleaned_text, title), self.content_analyzer.create_structured_content(soup)
            
            page_data = {"content_id": c_id, "url": url, "title": title, "summary": summ, "content": cleaned_text, "search_text": cleaned_text, "keywords": kws, "category": cat, "images": images, "internal_links": internal_l, "external_links": external_l, "structured_content": struct_c, "metrics": mets, "last_updated": datetime.now().isoformat(), "processing_info": {"processed_at": datetime.now().isoformat(), "content_length": len(cleaned_text), "links_found": len(page_links_raw), "images_found": len(images), "javascript_rendered": js_used}}

            ind_dir = os.path.join(OUTPUT_DIR, "individual_pages"); os.makedirs(ind_dir, exist_ok=True)
            with open(os.path.join(ind_dir, self._safe_filename(url) + ".json"), 'w', encoding='utf-8') as f: json.dump(page_data, f, ensure_ascii=False, indent=2)
            self.pages_data.append(page_data)
            logger.info(f"Successfully processed: {title} ({url})")
            return True
        except Exception as e: logger.error(f"Failed to process page {url}: {e}", exc_info=DEBUG); return False

    def _safe_filename(self, url: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]', '_', url.replace("https://", "").replace("http://", ""))[:100]

    def _save_consolidated_output(self):
        if not self.pages_data: logger.warning("No page data to save."); return
        metadata = {"processed_at": datetime.now().isoformat(), "last_crawl": datetime.now().isoformat(), "total_pages": len(self.pages_data), "total_urls_found": len(self.pages_data), "version": "2.2", "target_website": self.base_url, "javascript_support_used": self.use_javascript, "ssl_verification_status": "Enabled" if self.session.verify else "Disabled for session", "processing_summary": {"start_time": datetime.fromtimestamp(self.start_time).isoformat(), "end_time": datetime.now().isoformat(), "duration_seconds": round(time.time() - self.start_time, 2), "categories": self._get_category_stats(), "javascript_rendered_pages_count": sum(1 for p in self.pages_data if p.get('processing_info', {}).get('javascript_rendered', False))}}
        consolidated_data = {"pages": self.pages_data, "metadata": metadata}
        out_path = os.path.join(OUTPUT_DIR, "pages.json")
        with open(out_path, 'w', encoding='utf-8') as f: json.dump(consolidated_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Consolidated output saved to {out_path} ({len(self.pages_data)} pages)")
        logger.info(f"Category distribution: {self._get_category_stats()}")

    def _get_category_stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p_item in self.pages_data: counts[p_item.get('category', 'unknown')] = counts.get(p_item.get('category', 'unknown'), 0) + 1
        return counts

    def run(self):
        """Main processing routine."""
        logger.info("=" * 60 + f"\nStarting AI_Gate Website Processing (v2.2)\n" + "=" * 60)
        accessible, msg = self._verify_site_accessibility()
        if not accessible:
            logger.error(f"Access denied to {self.base_url}: {msg}"); self._generate_performance_report(); raise SystemExit(f"Fatal Error: {msg}")
        logger.info(f"Website access verified for {self.base_url}: {msg}")
        
        urls_to_process = []
        try:
            logger.info("Discovering URLs...")
            sitemap_urls = self._crawl_sitemap()
            discovered_set = set(sitemap_urls)
            logger.info(f"Found {len(sitemap_urls)} URLs from sitemap(s).")

            crawl_trigger_thresh = 350  # TARGETED: Sitemap URL count threshold
            max_pages_from_crawl_total = 400 # TARGETED: Max pages to try and get from crawl if triggered
            crawl_depth_config = 5       # TARGETED: Max depth for crawl

            if len(discovered_set) < crawl_trigger_thresh:
                logger.info(f"Sitemap URLs ({len(discovered_set)}) < threshold ({crawl_trigger_thresh}). Initiating broader crawl.")
                crawl_start_pts = [self.base_url] if not sitemap_urls else sitemap_urls # Start crawl from base or sitemap URLs
                
                # How many more pages do we want to find via crawling?
                needed_from_crawl = max_pages_from_crawl_total - len(discovered_set)
                if needed_from_crawl > 0:
                    additional_urls = self._discover_urls_via_crawl(
                        start_urls=crawl_start_pts, 
                        max_pages_to_collect=needed_from_crawl, # Try to collect up to this many *new* pages
                        max_crawl_depth=crawl_depth_config
                    )
                    discovered_set.update(additional_urls)
                    logger.info(f"Added {len(additional_urls)} URLs from broader crawl. Total unique now: {len(discovered_set)}")
            
            if self._is_valid_page_url(self.base_url) and self.base_url not in discovered_set:
                discovered_set.add(self.base_url); logger.info(f"Base URL {self.base_url} added. Total unique now: {len(discovered_set)}")

            urls_to_process = sorted(list(discovered_set))
            if not urls_to_process: logger.warning("No URLs discovered."); return # Exit run, finally will execute

            logger.info(f"Total unique URLs to process: {len(urls_to_process)}")
            success_c, failed_c, total_urls_count = 0, 0, len(urls_to_process)
            
            for i, url_p in enumerate(urls_to_process, 1):
                try:
                    if self._check_url_permissions(url_p): # Always true for now
                        if self._process_page(url_p): success_c += 1
                        else: failed_c += 1
                    else: failed_c += 1 # Should not happen with current permission check
                        
                    if i % 5 == 0 or i == total_urls_count:
                        rate = (success_c / i) * 100 if i > 0 else 0
                        logger.info(f"Progress: {i}/{total_urls_count} - Success: {success_c}, Failed: {failed_c} ({rate:.1f}%)")
                    time.sleep(0.95) # Slightly longer delay for page processing
                except KeyboardInterrupt: logger.warning("Processing interrupted by user in page loop."); break 
                except Exception as e_l: logger.error(f"Unexpected error in loop for {url_p}: {e_l}", exc_info=DEBUG); failed_c += 1; continue

            if self.pages_data: logger.info("Saving processed data..."); self._save_consolidated_output(); logger.info("✅ Website processing completed.")
            else: logger.warning("⚠️ No page data successfully processed.")

            logger.info("=" * 60 + "\nWEBSITE PROCESSING SUMMARY\n" + "=" * 60)
            logger.info(f"Total URLs for processing: {total_urls_count}\nSuccessfully processed: {success_c}\nFailed: {failed_c}")
            if total_urls_count > 0: logger.info(f"Success rate: {(success_c/total_urls_count*100):.1f}%")
            logger.info(f"Total time: {time.time() - self.start_time:.2f}s")

        except KeyboardInterrupt: logger.warning("Processing interrupted by user (main).")
        except SystemExit as se: logger.info(f"System exit: {se}")
        except Exception as e: logger.critical(f"Critical error: {e}", exc_info=True)
        finally:
            if self.pages_data and (getattr(self, 'success_c', 0) > 0 or getattr(self, 'failed_c', 0) > 0) : # Check if some processing occurred
                logger.info("Attempting to save partial results...")
                try: self._save_consolidated_output()
                except Exception as s_e: logger.error(f"Failed to save partial results: {s_e}")
            self._generate_performance_report()
            if hasattr(self, 'js_renderer') and self.js_renderer: self.js_renderer.close()

    def _generate_performance_report(self):
        global log_file_path
        log_fn = os.path.basename(log_file_path) if 'log_file_path' in globals() and log_file_path else "N/A"
        try:
            report = {"timestamp": datetime.now().isoformat(), "processing_time_seconds": round(time.time() - self.start_time, 2), "target_url": self.base_url, "pages_in_output": len(self.pages_data), "log_file": log_fn, "output_file": os.path.join(OUTPUT_DIR, "pages.json"), "compatibility_version": "2.2", "categories_found_in_output": self._get_category_stats(), "system_info": {"python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}", "platform": os.sys.platform, "selenium_available": SELENIUM_AVAILABLE, "javascript_rendering_used_in_session": getattr(self, 'use_javascript', "N/A")}, "processing_notes": ["Output compatible with website_researcher.py", "Individual pages in individual_pages/", "Main output: pages.json", f"JS rendering actively used: {'Yes' if getattr(self, 'use_javascript', False) and any(p.get('processing_info', {}).get('javascript_rendered') for p in self.pages_data) else 'No/Not needed'}", f"SSL verification status: {'Enabled' if getattr(self.session, 'verify', True) else 'Disabled'}"]}
            report_f_path = os.path.join(LOG_DIR, f"performance_report_website_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_f_path, 'w', encoding='utf-8') as f: json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Performance report saved to {report_f_path}")
        except Exception as e: logger.error(f"Failed to generate performance report: {e}")

if __name__ == "__main__":
    try:
        processor = WebsiteProcessor()
        processor.run()
    except KeyboardInterrupt: logger.info("Processing stopped by user (main exec).")
    except SystemExit: logger.info("Application exited (main exec).")
    except Exception as e_s: logger.critical(f"App startup/critical error: {e_s}", exc_info=True)