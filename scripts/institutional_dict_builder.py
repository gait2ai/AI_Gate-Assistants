"""
AI Gate for Artificial Intelligence Applications
Institutional Dictionary Builder Script - institutional_dictionary_builder.py

This script processes the website_scraper's output (pages.json) to generate
a categorized keyword dictionary (institutional_keywords.yaml) used for
enhancing topic analysis within the AI Gate system.

Enhanced with Arabic language support and external stopword loading.
"""

import json
import yaml
import logging
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional
import argparse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InstitutionalDictionaryBuilder:
    """
    Builds a categorized keyword dictionary from scraped website pages.
    Enhanced with Arabic language support and external stopword loading.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the dictionary builder with optional configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.stopwords_en, self.stopwords_ar = self._get_stopwords()
        self.combined_stopwords = self.stopwords_en.union(self.stopwords_ar)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            'min_keyword_frequency': 1,
            'exclude_categories': ['general'],
            'sort_keywords': True,
            'custom_stopwords': [],
            'extract_from_title_summary': True,
            'lemmatization': False,
            'stopwords_path': './nltk_data_local/corpora/stopwords/'
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info("Using default configuration")
            
        return default_config
    
    def _load_external_stopwords(self, language: str, stopwords_dir: str) -> Set[str]:
        """
        Load stopwords from external NLTK-style directory.
        
        Args:
            language: Language code ('english' or 'arabic')
            stopwords_dir: Path to stopwords directory
            
        Returns:
            Set of stopwords for the specified language
        """
        stopwords_file = os.path.join(stopwords_dir, language)
        stopwords = set()
        
        try:
            if os.path.exists(stopwords_file):
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    stopwords = {line.strip().lower() for line in f if line.strip()}
                logger.info(f"Loaded {len(stopwords)} {language} stopwords from {stopwords_file}")
            else:
                logger.warning(f"Stopwords file not found: {stopwords_file}")
        except Exception as e:
            logger.warning(f"Failed to load {language} stopwords from {stopwords_file}: {e}")
        
        return stopwords
    
    def _get_fallback_stopwords(self, language: str) -> Set[str]:
        """
        Get fallback stopwords if external files are not available.
        
        Args:
            language: Language code ('english' or 'arabic')
            
        Returns:
            Set of basic stopwords for the specified language
        """
        if language == 'english':
            return {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'were', 'will', 'with', 'would', 'you', 'your',
                'this', 'these', 'they', 'we', 'our', 'us', 'can', 'could',
                'should', 'may', 'might', 'must', 'shall', 'will', 'would'
            }
        elif language == 'arabic':
            # Basic Arabic stopwords
            return {
                'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
                'التي', 'الذي', 'التي', 'التي', 'كان', 'كانت', 'يكون', 'تكون',
                'هو', 'هي', 'هم', 'هن', 'أن', 'إن', 'كل', 'بعض', 'غير', 'سوف',
                'قد', 'لقد', 'كما', 'حيث', 'بين', 'عند', 'لدى', 'أمام', 'خلال'
            }
        else:
            return set()
    
    def _get_stopwords(self) -> tuple[Set[str], Set[str]]:
        """Get sets of stopwords for English and Arabic."""
        stopwords_dir = self.config.get('stopwords_path', './nltk_data_local/corpora/stopwords/')
        
        # Load English stopwords
        stopwords_en = self._load_external_stopwords('english', stopwords_dir)
        if not stopwords_en:
            logger.info("Using fallback English stopwords")
            stopwords_en = self._get_fallback_stopwords('english')
        
        # Load Arabic stopwords
        stopwords_ar = self._load_external_stopwords('arabic', stopwords_dir)
        if not stopwords_ar:
            logger.info("Using fallback Arabic stopwords")
            stopwords_ar = self._get_fallback_stopwords('arabic')
        
        # Add custom stopwords from config
        custom_stopwords = set(self.config.get('custom_stopwords', []))
        if custom_stopwords:
            logger.info(f"Adding {len(custom_stopwords)} custom stopwords")
            stopwords_en.update(custom_stopwords)
            stopwords_ar.update(custom_stopwords)
        
        return stopwords_en, stopwords_ar
    
    def _detect_arabic_text(self, text: str) -> bool:
        """
        Simple heuristic to detect if text contains Arabic characters.
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if text appears to contain Arabic content
        """
        if not text:
            return False
        
        # Count Arabic characters (Unicode range for Arabic script)
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        total_chars = len(re.findall(r'[^\s\d\W]', text))  # Non-whitespace, non-digit, non-punctuation
        
        # If more than 30% of characters are Arabic, consider it Arabic text
        return total_chars > 0 and (arabic_chars / total_chars) > 0.3
    
    def _clean_text(self, text: str) -> List[str]:
        """
        Clean and tokenize text into meaningful keywords.
        Enhanced to support Arabic text processing.
        
        Args:
            text: Input text to process
            
        Returns:
            List of cleaned keywords
        """
        if not text:
            return []
            
        # Convert to lowercase for English text, preserve case for Arabic
        is_arabic = self._detect_arabic_text(text)
        if not is_arabic:
            text = text.lower()
        
        # Enhanced regex to handle both English and Arabic words
        # English: [a-zA-Z]{2,}
        # Arabic: Unicode ranges for Arabic script
        if is_arabic:
            # Extract Arabic words (2+ characters) and preserve English words too
            words = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]{2,}|[a-zA-Z]{2,}', text)
        else:
            # Standard English word extraction with enhanced Unicode support
            words = re.findall(r'\b[a-zA-Z\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]{2,}\b', text)
        
        # Filter out stopwords and short words
        # Use appropriate stopword set based on detected language
        if is_arabic:
            relevant_stopwords = self.stopwords_ar
        else:
            relevant_stopwords = self.stopwords_en
        
        # For mixed content, use combined stopwords to be safe
        if is_arabic and any(re.match(r'[a-zA-Z]', word) for word in words):
            relevant_stopwords = self.combined_stopwords
        
        keywords = []
        for word in words:
            # Convert English words to lowercase for stopword checking
            check_word = word.lower() if re.match(r'^[a-zA-Z]+$', word) else word
            
            if check_word not in relevant_stopwords and len(word) >= 2:
                # Store original case for Arabic, lowercase for English
                if re.match(r'^[a-zA-Z]+$', word):
                    keywords.append(word.lower())
                else:
                    keywords.append(word)
        
        return keywords
    
    def _extract_keywords_from_page(self, page: Dict) -> List[str]:
        """
        Extract keywords from a single page object.
        
        Args:
            page: Page object from pages.json
            
        Returns:
            List of extracted keywords
        """
        keywords = []
        
        # Primary source: existing keywords field
        if 'keywords' in page and isinstance(page['keywords'], list):
            # Process existing keywords through cleaning pipeline
            for kw in page['keywords']:
                if kw and isinstance(kw, str):
                    # Clean existing keywords to ensure consistency
                    cleaned = self._clean_text(kw.strip())
                    keywords.extend(cleaned)
        
        # Optional enhancement: extract from title and summary
        if self.config.get('extract_from_title_summary', True):
            if 'title' in page:
                keywords.extend(self._clean_text(page['title']))
            if 'summary' in page:
                keywords.extend(self._clean_text(page['summary']))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw and kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _load_pages(self, pages_path: str) -> List[Dict]:
        """
        Load pages from JSON file.
        
        Args:
            pages_path: Path to pages.json file
            
        Returns:
            List of page objects
        """
        try:
            with open(pages_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list format and dict with 'pages' key
            if isinstance(data, list):
                pages = data
            elif isinstance(data, dict) and 'pages' in data:
                pages = data['pages']
            else:
                raise ValueError("Invalid pages.json format")
            
            logger.info(f"Loaded {len(pages)} pages from {pages_path}")
            return pages
            
        except FileNotFoundError:
            logger.error(f"Pages file not found: {pages_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {pages_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading pages from {pages_path}: {e}")
            raise
    
    def _process_pages(self, pages: List[Dict]) -> Dict[str, List[str]]:
        """
        Process all pages and build category-keyword dictionary.
        
        Args:
            pages: List of page objects
            
        Returns:
            Dictionary mapping categories to keyword lists
        """
        category_keywords = defaultdict(list)
        category_counters = defaultdict(Counter)
        
        processed_pages = 0
        arabic_pages = 0
        english_pages = 0
        
        for page in pages:
            if not isinstance(page, dict):
                logger.warning(f"Skipping invalid page object: {page}")
                continue
            
            # Get category (required field)
            category = page.get('category', '').strip().lower()
            if not category:
                logger.warning(f"Skipping page without category: {page.get('title', 'Unknown')}")
                continue
            
            # Skip excluded categories
            if category in self.config.get('exclude_categories', []):
                logger.debug(f"Skipping excluded category: {category}")
                continue
            
            # Extract keywords from this page
            keywords = self._extract_keywords_from_page(page)
            
            # Track language statistics
            page_text = f"{page.get('title', '')} {page.get('summary', '')}"
            if self._detect_arabic_text(page_text):
                arabic_pages += 1
            else:
                english_pages += 1
            
            # Count keyword occurrences within this category
            for keyword in keywords:
                category_counters[category][keyword] += 1
            
            processed_pages += 1
        
        logger.info(f"Processed {processed_pages} pages ({english_pages} English, {arabic_pages} Arabic)")
        
        # Apply frequency filtering and build final dictionary
        min_freq = self.config.get('min_keyword_frequency', 1)
        
        for category, counter in category_counters.items():
            # Filter by minimum frequency
            filtered_keywords = [
                keyword for keyword, count in counter.items()
                if count >= min_freq
            ]
            
            # Sort alphabetically if configured
            if self.config.get('sort_keywords', True):
                # Sort with proper Unicode support for Arabic
                filtered_keywords.sort(key=lambda x: x.lower() if re.match(r'^[a-zA-Z]+$', x) else x)
            
            category_keywords[category] = filtered_keywords
        
        return dict(category_keywords)
    
    def _save_dictionary(self, dictionary: Dict[str, List[str]], output_path: str):
        """
        Save the dictionary to YAML file.
        
        Args:
            dictionary: Category-keyword dictionary
            output_path: Path to output YAML file
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    dictionary, 
                    f, 
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=True
                )
            
            logger.info(f"Saved institutional dictionary to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save dictionary to {output_path}: {e}")
            raise
    
    def build_dictionary(self, pages_path: str, output_path: str):
        """
        Main method to build the institutional dictionary.
        
        Args:
            pages_path: Path to input pages.json file
            output_path: Path to output institutional_keywords.yaml file
        """
        logger.info("Starting institutional dictionary build process")
        logger.info(f"English stopwords: {len(self.stopwords_en)}, Arabic stopwords: {len(self.stopwords_ar)}")
        
        # Load pages
        pages = self._load_pages(pages_path)
        
        # Process pages to build dictionary
        dictionary = self._process_pages(pages)
        
        # Log statistics
        total_categories = len(dictionary)
        total_keywords = sum(len(keywords) for keywords in dictionary.values())
        
        logger.info(f"Generated dictionary with {total_categories} categories and {total_keywords} total keywords")
        
        # Log category summary
        for category, keywords in sorted(dictionary.items()):
            # Count Arabic vs English keywords
            arabic_kw = sum(1 for kw in keywords if self._detect_arabic_text(kw))
            english_kw = len(keywords) - arabic_kw
            logger.info(f"  {category}: {len(keywords)} keywords ({english_kw} English, {arabic_kw} Arabic)")
        
        # Save dictionary
        self._save_dictionary(dictionary, output_path)
        
        logger.info("Institutional dictionary build completed successfully")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Build institutional keyword dictionary from scraped pages'
    )
    parser.add_argument(
        '--pages', 
        default='data/pages.json',
        help='Path to input pages.json file (default: data/pages.json)'
    )
    parser.add_argument(
        '--output',
        default='data/institutional_keywords.yaml',
        help='Path to output YAML file (default: data/institutional_keywords.yaml)'
    )
    parser.add_argument(
        '--config',
        default='config/dictionary_builder_config.yaml',
        help='Path to configuration file (default: config/dictionary_builder_config.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize builder with configuration
        config_path = args.config if os.path.exists(args.config) else None
        builder = InstitutionalDictionaryBuilder(config_path)
        
        # Build dictionary
        builder.build_dictionary(args.pages, args.output)
        
        print(f"✅ Successfully built institutional dictionary: {args.output}")
        
    except Exception as e:
        logger.error(f"Failed to build institutional dictionary: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
