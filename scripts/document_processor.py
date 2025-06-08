"""
AI_Gate Advanced Multilingual Document Processor - document_processor.py (Version 2.1)

A sophisticated multilingual document processor that generates structured JSON data from local 
institution documents (DOCX, XLSX, PDF) with advanced NLP capabilities for Arabic and English.
Utilizes local NLTK resources exclusively and maintains output compatibility with v1.0.

Enhancements in v2.1:
- Improved NLTK initialization with correct path prepending
- Enhanced Arabic text normalization and stemming
- Optimized English lemmatization with batch POS tagging
- More robust stopword loading
- Optimized semantic similarity calculation
- Better language detection for categorization
"""

import json
import os
import time
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from collections import Counter
import sys

# Configure NLTK to use local data exclusively
LOCAL_NLTK_DATA_PATH = Path(__file__).parent.parent / "nltk_data_local"
if LOCAL_NLTK_DATA_PATH.exists():
    import nltk
    nltk.data.path.insert(0, str(LOCAL_NLTK_DATA_PATH))  # Prepend to maintain system NLTK data as fallback
    print(f"NLTK configured to use local data (prepended to path): {LOCAL_NLTK_DATA_PATH}")
else:
    print(f"WARNING: Local NLTK data path not found: {LOCAL_NLTK_DATA_PATH}")

# Language detection
try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# NLTK imports (using local resources)
try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer, SnowballStemmer
    from nltk.tag import pos_tag
    from nltk.corpus.reader import WordNetError
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Arabic processing libraries
try:
    from nltk.stem.isri import ISRIStemmer
    ARABIC_ISRI_AVAILABLE = True
except ImportError:
    ARABIC_ISRI_AVAILABLE = False

# Fallback Arabic stemmer options
try:
    # Try qutuf if available
    import qutuf
    QUTUF_AVAILABLE = True
except ImportError:
    QUTUF_AVAILABLE = False

try:
    # Try Arabycia if available
    import arabycia
    ARABYCIA_AVAILABLE = True
except ImportError:
    ARABYCIA_AVAILABLE = False

# Document processing imports (same as v1.0)
try:
    import docx
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import openpyxl
    from openpyxl import load_workbook
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False

try:
    import PyPDF2
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pypdf
        from pypdf import PdfReader
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Debug configuration
DEBUG = False
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)

# Directories (same as v1.0)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "institution_info"
LOG_DIR = BASE_DIR / "logs"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
log_file_path = LOG_DIR / f"document_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

root_logger = logging.getLogger()
if root_logger.hasHandlers():
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AI_Gate_AdvancedDocumentProcessor")

# Log library availability
logger_init = logging.getLogger("AI_Gate_DocumentProcessor_Init")
logger_init.info(f"Advanced Multilingual Document Processor v2.1 initializing...")
logger_init.info(f"Language Detection Available: {LANGDETECT_AVAILABLE}")
logger_init.info(f"NLTK Available: {NLTK_AVAILABLE}")
logger_init.info(f"Arabic ISRI Stemmer Available: {ARABIC_ISRI_AVAILABLE}")
logger_init.info(f"Qutuf Available: {QUTUF_AVAILABLE}")
logger_init.info(f"Arabycia Available: {ARABYCIA_AVAILABLE}")


class LanguageDetector:
    """Language detection and configuration manager."""
    
    def __init__(self):
        self.supported_languages = {'en': 'english', 'ar': 'arabic'}
        self.fallback_language = 'en'
    
    def detect_language(self, text: str) -> str:
        """Detect the primary language of text content with improved sampling."""
        if not text or not text.strip():
            return self.fallback_language
        
        # Take a representative sample (first 1000 chars + middle 1000 chars if available)
        sample_size = 1000
        sample = text[:sample_size]
        if len(text) > 2 * sample_size:
            middle_start = len(text) // 2 - sample_size // 2
            sample += text[middle_start:middle_start + sample_size]
        
        if not LANGDETECT_AVAILABLE:
            # Enhanced heuristic fallback
            arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', sample))
            english_chars = len(re.findall(r'[a-zA-Z]', sample))
            
            if arabic_chars > english_chars * 1.5:  # More strict threshold
                return 'ar'
            return 'en'
        
        try:
            detected = detect(sample)
            return detected if detected in self.supported_languages else self.fallback_language
        except (LangDetectError, Exception):
            return self.fallback_language
    
    def get_nltk_language_name(self, lang_code: str) -> str:
        """Get NLTK language name from language code."""
        return self.supported_languages.get(lang_code, 'english')


class AdvancedTextProcessor:
    """Advanced NLP text processor with multilingual support using local NLTK."""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self._initialize_nltk_resources()
        self._initialize_arabic_stemmers()
    
    def _initialize_nltk_resources(self):
        """Initialize NLTK resources from local data with improved robustness."""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available - text processing will be limited")
            return
        
        try:
            # Initialize stopwords with fallback to basic sets
            self.stopwords_en = set([
                'the', 'and', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
            ])
            self.stopwords_ar = set([
                'في', 'من', 'على', 'إلى', 'عن', 'أن', 'إن', 'أن', 'ما', 'هذا', 
                'هذه', 'ذلك', 'هؤلاء', 'كان', 'يكون', 'يكون', 'كانت'
            ])
            
            try:
                self.stopwords_en = set(stopwords.words('english'))
                logger.info("English stopwords loaded from local NLTK data")
            except Exception as e:
                logger.warning(f"Could not load English stopwords from NLTK: {e}. Using basic set.")
            
            try:
                self.stopwords_ar = set(stopwords.words('arabic'))
                logger.info("Arabic stopwords loaded from local NLTK data")
            except Exception as e:
                logger.warning(f"Could not load Arabic stopwords from NLTK: {e}. Using basic set.")
            
            # Initialize lemmatizer with error handling
            try:
                self.lemmatizer = WordNetLemmatizer()
                logger.info("WordNet lemmatizer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize lemmatizer: {e}")
                self.lemmatizer = None
            
            # Initialize stemmers
            try:
                self.english_stemmer = SnowballStemmer('english')
                logger.info("English stemmer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize English stemmer: {e}")
                self.english_stemmer = None
            
        except Exception as e:
            logger.error(f"Error initializing NLTK resources: {e}")
    
    def _initialize_arabic_stemmers(self):
        """Initialize Arabic stemming capabilities with improved error handling."""
        self.arabic_stemmer = None
        
        # Priority 1: NLTK ISRI stemmer
        if ARABIC_ISRI_AVAILABLE:
            try:
                self.arabic_stemmer = ISRIStemmer()
                logger.info("Arabic ISRI stemmer initialized from NLTK")
                return
            except Exception as e:
                logger.warning(f"Could not initialize ISRI stemmer: {e}")
        
        # Priority 2: External Arabic processing libraries
        if QUTUF_AVAILABLE:
            try:
                # Test qutuf functionality
                test_stem = qutuf.stem("يكتب")
                if test_stem:  # Simple functionality check
                    self.arabic_stemmer = "qutuf"
                    logger.info("Qutuf Arabic processor initialized and verified")
                    return
                else:
                    logger.warning("Qutuf returned empty stem - may not be functioning correctly")
            except Exception as e:
                logger.warning(f"Could not initialize Qutuf: {e}")
        
        if ARABYCIA_AVAILABLE:
            try:
                # Test arabycia functionality
                test_stem = arabycia.stem("يكتب")
                if test_stem:  # Simple functionality check
                    self.arabic_stemmer = "arabycia"
                    logger.info("Arabycia Arabic processor initialized and verified")
                    return
                else:
                    logger.warning("Arabycia returned empty stem - may not be functioning correctly")
            except Exception as e:
                logger.warning(f"Could not initialize Arabycia: {e}")
        
        logger.warning("No Arabic stemmer available - using basic tokenization")
    
    def preprocess_text(self, text: str, language: str) -> List[str]:
        """Advanced text preprocessing based on detected language."""
        if not text or not NLTK_AVAILABLE:
            return self._basic_tokenize(text)
        
        try:
            # Tokenization
            if language == 'ar':
                tokens = self._arabic_tokenize(text)
            else:
                tokens = word_tokenize(text.lower())
            
            # Remove stopwords
            stopwords_set = self.stopwords_ar if language == 'ar' else self.stopwords_en
            tokens = [token for token in tokens if token not in stopwords_set]
            
            # Remove punctuation and short tokens
            tokens = [token for token in tokens if token.isalnum() and len(token) > 2]
            
            # Stemming/Lemmatization
            if language == 'ar':
                tokens = self._arabic_stem_tokens(tokens)
            else:
                tokens = self._english_lemmatize_tokens(tokens)
            
            return tokens
            
        except Exception as e:
            logger.warning(f"Error in advanced preprocessing: {e}")
            return self._basic_tokenize(text)
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """Fallback basic tokenization."""
        if not text:
            return []
        words = re.findall(r'\b[a-zA-Z\u0600-\u06FF]{3,15}\b', text.lower())
        return words
    
    def _arabic_tokenize(self, text: str) -> List[str]:
        """Arabic-specific tokenization with improved normalization."""
        try:
            # Use NLTK tokenizer if available
            tokens = word_tokenize(text)
            # Clean Arabic tokens with enhanced normalization
            cleaned_tokens = []
            for token in tokens:
                normalized = self._normalize_arabic(token)
                if normalized and len(normalized) > 2:
                    cleaned_tokens.append(normalized)
            return cleaned_tokens
        except Exception:
            # Fallback to regex-based tokenization
            return re.findall(r'[\u0600-\u06FF]+', text)
    
    def _normalize_arabic(self, text: str) -> str:
        """Comprehensive normalization of Arabic text."""
        if not text:
            return ""
        
        # Remove diacritics and tatweel
        text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
        
        # Normalize Alef variants (including isolated forms)
        text = re.sub(r'[إأآاٱ]', 'ا', text)
        
        # Normalize Teh Marbuta to Heh
        text = re.sub(r'[ة]', 'ه', text)
        
        # Normalize Yeh variants (including dots and hamza)
        text = re.sub(r'[يىئ]', 'ى', text)
        
        # Normalize Kaf variants
        text = re.sub(r'[كڪ]', 'ك', text)
        
        # Normalize Heh variants
        text = re.sub(r'[هھ]', 'ه', text)
        
        # Remove non-Arabic characters
        text = re.sub(r'[^\u0600-\u06FF]', '', text)
        
        return text.strip()
    
    def _arabic_stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply Arabic stemming to tokens with improved external library handling."""
        if not tokens:
            return []
        
        stemmed_tokens = []
        
        if isinstance(self.arabic_stemmer, ISRIStemmer):
            # Use NLTK ISRI stemmer
            for token in tokens:
                try:
                    stemmed = self.arabic_stemmer.stem(token)
                    stemmed_tokens.append(stemmed if stemmed else token)
                except Exception:
                    stemmed_tokens.append(token)
        
        elif self.arabic_stemmer == "qutuf":
            # Use Qutuf with error handling
            try:
                for token in tokens:
                    try:
                        stemmed = qutuf.stem(token)
                        stemmed_tokens.append(stemmed if stemmed else token)
                    except Exception:
                        stemmed_tokens.append(token)
            except Exception:
                stemmed_tokens = tokens
        
        elif self.arabic_stemmer == "arabycia":
            # Use Arabycia with error handling
            try:
                for token in tokens:
                    try:
                        stemmed = arabycia.stem(token)
                        stemmed_tokens.append(stemmed if stemmed else token)
                    except Exception:
                        stemmed_tokens.append(token)
            except Exception:
                stemmed_tokens = tokens
        
        else:
            # Fallback: return original tokens
            stemmed_tokens = tokens
        
        return stemmed_tokens
    
    def _english_lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply optimized English lemmatization with batch POS tagging."""
        if not tokens or not self.lemmatizer:
            return tokens
        
        # Batch POS tagging for performance
        try:
            tagged_tokens = pos_tag(tokens)
        except Exception:
            tagged_tokens = [(token, 'NN') for token in tokens]
        
        lemmatized_tokens = []
        for token, tag in tagged_tokens:
            try:
                # Get WordNet POS tag
                pos = self._get_wordnet_pos(tag)
                lemmatized = self.lemmatizer.lemmatize(token, pos)
                lemmatized_tokens.append(lemmatized if lemmatized else token)
            except Exception:
                lemmatized_tokens.append(token)
        
        return lemmatized_tokens
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Get WordNet POS tag from treebank tag."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def extract_advanced_keywords(self, text: str, language: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords using advanced NLP techniques."""
        if not text:
            return []
        
        # Preprocess text
        processed_tokens = self.preprocess_text(text, language)
        
        if not processed_tokens:
            return self._fallback_keyword_extraction(text, max_keywords)
        
        # Count token frequency
        token_counts = Counter(processed_tokens)
        
        # Get base keywords from frequency
        base_keywords = [token for token, count in token_counts.most_common(max_keywords * 2)]
        
        # Enhance with WordNet synonyms if available
        enhanced_keywords = self._enhance_with_wordnet(base_keywords, language)
        
        # Return top keywords
        return enhanced_keywords[:max_keywords]
    
    def _enhance_with_wordnet(self, keywords: List[str], language: str) -> List[str]:
        """Enhance keywords using WordNet synonyms."""
        if not NLTK_AVAILABLE:
            return keywords
        
        enhanced = []
        seen = set()
        
        try:
            for keyword in keywords:
                if keyword in seen:
                    continue
                
                enhanced.append(keyword)
                seen.add(keyword)
                
                # Get WordNet synsets
                synsets = wordnet.synsets(keyword, lang='arb' if language == 'ar' else 'eng')
                
                # Add related terms
                for synset in synsets[:2]:  # Limit to first 2 synsets
                    for lemma in synset.lemmas():
                        related_word = lemma.name().replace('_', ' ').lower()
                        if (related_word not in seen and 
                            len(related_word) > 2 and 
                            related_word != keyword):
                            enhanced.append(related_word)
                            seen.add(related_word)
                            if len(enhanced) >= len(keywords) * 1.5:
                                break
                    if len(enhanced) >= len(keywords) * 1.5:
                        break
                
                if len(enhanced) >= len(keywords) * 1.5:
                    break
        
        except Exception as e:
            logger.warning(f"WordNet enhancement failed: {e}")
            return keywords
        
        return enhanced
    
    def _fallback_keyword_extraction(self, text: str, max_keywords: int) -> List[str]:
        """Fallback keyword extraction method."""
        words = re.findall(r'\b[a-zA-Z\u0600-\u06FF]{4,15}\b', text.lower())
        word_counts = Counter(words)
        
        # Filter stopwords manually
        basic_stopwords = {
            'the', 'and', 'for', 'with', 'this', 'that', 'من', 'في', 'على', 'إلى',
            'هذا', 'هذه', 'التي', 'الذي', 'كان', 'كانت', 'يكون', 'تكون'
        }
        
        keywords = []
        for word, count in word_counts.most_common(max_keywords * 2):
            if word not in basic_stopwords and len(word) >= 4:
                keywords.append(word)
                if len(keywords) >= max_keywords:
                    break
        
        return keywords


class AdvancedContentAnalyzer:
    """Enhanced content analyzer with multilingual NLP capabilities."""
    
    def __init__(self):
        self.text_processor = AdvancedTextProcessor()
        self.language_detector = LanguageDetector()
        self._initialize_category_concepts()
    
    def _initialize_category_concepts(self):
        """Initialize category concepts in multiple languages."""
        self.category_concepts = {
            'academic': {
                'en': ['syllabus', 'curriculum', 'course', 'program', 'degree', 'academic', 
                       'semester', 'credit', 'transcript', 'grade', 'study', 'education',
                       'learning', 'teaching', 'instruction', 'lecture', 'assignment'],
                'ar': ['منهج', 'دراسي', 'مقرر', 'برنامج', 'درجة', 'أكاديمي', 'فصل', 
                       'ساعة', 'كشف', 'درجات', 'دراسة', 'تعليم', 'تعلم', 'تدريس', 'محاضرة']
            },
            'administrative': {
                'en': ['policy', 'procedure', 'regulation', 'handbook', 'guide', 'manual', 
                       'administration', 'office', 'department', 'management', 'governance'],
                'ar': ['سياسة', 'إجراء', 'لائحة', 'دليل', 'إدارة', 'مكتب', 'قسم', 'إدارية']
            },
            'financial': {
                'en': ['tuition', 'fee', 'cost', 'payment', 'scholarship', 'grant', 'budget', 
                       'financial', 'aid', 'billing', 'expense', 'fund'],
                'ar': ['رسوم', 'تكلفة', 'دفع', 'منحة', 'مالي', 'مساعدة', 'فاتورة', 'مصروف']
            },
            'student_services': {
                'en': ['housing', 'dining', 'health', 'counseling', 'career', 'recreation', 
                       'activities', 'clubs', 'organizations', 'student', 'services'],
                'ar': ['سكن', 'طعام', 'صحة', 'إرشاد', 'مهني', 'ترفيه', 'أنشطة', 'نوادي', 'طلاب']
            },
            'admissions': {
                'en': ['admission', 'application', 'requirement', 'prerequisite', 'enrollment', 
                       'registration', 'apply', 'accept', 'candidate'],
                'ar': ['قبول', 'طلب', 'متطلب', 'تسجيل', 'التحاق', 'مرشح', 'شرط']
            },
            'faculty': {
                'en': ['faculty', 'professor', 'instructor', 'staff', 'directory', 'research', 
                       'publication', 'academic', 'teacher'],
                'ar': ['أعضاء', 'هيئة', 'تدريس', 'أستاذ', 'مدرس', 'بحث', 'نشر', 'أكاديمي']
            },
            'facilities': {
                'en': ['campus', 'building', 'library', 'laboratory', 'facility', 'equipment', 
                       'resources', 'infrastructure'],
                'ar': ['حرم', 'مبنى', 'مكتبة', 'مختبر', 'مرافق', 'معدات', 'موارد', 'بنية']
            },
            'events': {
                'en': ['event', 'calendar', 'schedule', 'meeting', 'conference', 'workshop', 
                       'seminar', 'activity', 'program'],
                'ar': ['فعالية', 'تقويم', 'جدول', 'اجتماع', 'مؤتمر', 'ورشة', 'ندوة', 'نشاط']
            },
            'research': {
                'en': ['research', 'study', 'analysis', 'paper', 'publication', 'findings', 
                       'methodology', 'data', 'results', 'experiment', 'investigation'],
                'ar': ['بحث', 'دراسة', 'تحليل', 'ورقة', 'نشر', 'نتائج', 'منهجية', 'بيانات', 'تجربة']
            },
            'general': {
                'en': ['information', 'overview', 'introduction', 'welcome', 'mission', 
                       'vision', 'history', 'about', 'general'],
                'ar': ['معلومات', 'نظرة', 'مقدمة', 'مرحبا', 'رسالة', 'رؤية', 'تاريخ', 'عام']
            }
        }
    
    def generate_content_id(self, filepath: str, title: str) -> str:
        """Generate unique content ID based on filepath and title."""
        content_string = f"{filepath}_{title}_{datetime.now().date()}"
        return hashlib.md5(content_string.encode('utf-8')).hexdigest()[:12]
    
    def generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate summary from content with language awareness."""
        if not content:
            return ""
        
        # Detect language for better sentence segmentation
        language = self.language_detector.detect_language(content)
        
        cleaned_content = re.sub(r'\s+', ' ', content.strip())
        if len(cleaned_content) <= max_length:
            return cleaned_content
        
        # Use NLTK sentence tokenizer if available
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(cleaned_content)
            except Exception:
                sentences = re.split(r'[.!?؟]+', cleaned_content)
        else:
            # Fallback for both Arabic and English punctuation
            sentences = re.split(r'[.!?؟]+', cleaned_content)
        
        summary = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_summary = f"{summary} {sentence}".strip()
            if len(potential_summary) <= max_length:
                summary = potential_summary
            else:
                if summary:
                    break
                else:
                    # Truncate single long sentence
                    words = sentence[:max_length - 3].split()
                    summary = ' '.join(words[:-1]) + "..." if len(words) > 1 else sentence[:max_length - 3] + "..."
                    break
        
        if not summary:
            words = cleaned_content[:max_length - 3].split()
            summary = ' '.join(words[:-1]) + "..." if len(words) > 1 else cleaned_content[:max_length - 3] + "..."
        
        return summary[:max_length]
    
    def categorize_content(self, title: str, content: str, keywords: List[str], filename: str = "") -> str:
        """Advanced categorization using semantic analysis."""
        # Detect language from representative sample
        full_text = f"{title} {content}"
        language = self.language_detector.detect_language(full_text)
        
        # Prepare text for analysis with enhanced keyword weighting
        text_to_analyze = f"{title} {' '.join(keywords)} {content} {filename}".lower()
        
        # Process text with NLP
        processed_tokens = self.text_processor.preprocess_text(text_to_analyze, language)
        
        # Calculate semantic scores for each category
        category_scores = {}
        
        for category, concepts in self.category_concepts.items():
            score = 0
            
            # Get concepts for detected language (fallback to English)
            lang_concepts = concepts.get(language, concepts.get('en', []))
            
            # Direct matching score (higher weight for title and keywords)
            for concept in lang_concepts:
                # Higher weight for matches in title
                score += title.lower().count(concept.lower()) * 2
                # Standard weight for matches in keywords and content
                score += ' '.join(keywords).lower().count(concept.lower()) * 1.5
                score += content.lower().count(concept.lower())
            
            # Enhanced matching with processed tokens
            if processed_tokens:
                for concept in lang_concepts:
                    processed_concept_tokens = self.text_processor.preprocess_text(concept, language)
                    for proc_concept in processed_concept_tokens:
                        if proc_concept in processed_tokens:
                            score += 2  # Higher weight for processed matches
            
            # Optimized semantic similarity calculation using top keywords
            if NLTK_AVAILABLE and processed_tokens and len(processed_tokens) > 5:
                try:
                    # Use top 10 keywords for semantic similarity
                    top_keywords = Counter(processed_tokens).most_common(10)
                    semantic_score = self._calculate_semantic_similarity(
                        [kw[0] for kw in top_keywords], lang_concepts, language
                    )
                    score += semantic_score
                except Exception as e:
                    logger.debug(f"Semantic similarity calculation failed: {e}")
            
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, default to 'general'
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            logger.debug(f"Categorization scores: {category_scores}, selected: {best_category}")
            return best_category
        
        return 'general'
    
    def _calculate_semantic_similarity(self, tokens: List[str], concepts: List[str], language: str) -> float:
        """Optimized semantic similarity calculation using WordNet."""
        if not NLTK_AVAILABLE or not tokens or not concepts:
            return 0.0
        
        similarity_score = 0.0
        lang_code = 'arb' if language == 'ar' else 'eng'
        
        try:
            # Limit tokens and concepts for efficiency
            tokens = tokens[:5]
            concepts = concepts[:5]
            
            for token in tokens:
                for concept in concepts:
                    try:
                        token_synsets = wordnet.synsets(token, lang=lang_code)
                        concept_synsets = wordnet.synsets(concept, lang=lang_code)
                        
                        if token_synsets and concept_synsets:
                            # Calculate path similarity between first synsets only
                            try:
                                sim = token_synsets[0].path_similarity(concept_synsets[0])
                                if sim and sim > 0.1:  # Threshold to ignore very weak similarities
                                    similarity_score += sim
                            except Exception:
                                continue
                    except Exception:
                        continue
        
        except Exception as e:
            logger.debug(f"WordNet similarity calculation error: {e}")
        
        return similarity_score * 2  # Scale score to match other scoring components
    
    def calculate_metrics(self, content: str, title: str) -> Dict[str, int]:
        """Calculate content metrics with language detection."""
        language = self.language_detector.detect_language(f"{title} {content}")
        
        return {
            'word_count': len(content.split()) if content else 0,
            'char_count': len(content) if content else 0,
            'title_word_count': len(title.split()) if title else 0,
            'detected_language': language  # This will be filtered out in final output
        }
    
    def extract_keywords_from_text(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords using advanced multilingual NLP."""
        if not text:
            return []
        
        # Detect language from representative sample
        language = self.language_detector.detect_language(text)
        
        # Use advanced extraction
        keywords = self.text_processor.extract_advanced_keywords(text, language, max_keywords)
        
        return keywords


class DocumentProcessor:
    """Main document processor class with advanced multilingual capabilities."""
    
    def __init__(self, input_directory: str = None):
        self.input_dir = Path(input_directory) if input_directory else DATA_DIR
        self.start_time = time.time()
        self.documents_data: List[Dict] = []
        self.content_analyzer = AdvancedContentAnalyzer()  # Use advanced analyzer
        
        logger.info(f"Initializing advanced multilingual document processor for directory: {self.input_dir}")
        
        # Verify input directory exists
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

    def _extract_docx_content(self, filepath: Path) -> tuple[str, str, Dict[str, Any]]:
        """Extract content from DOCX file."""
        if not DOCX_AVAILABLE:
            logger.error(f"DOCX processing not available for {filepath}")
            return "", "", {}
        
        try:
            doc = Document(str(filepath))
            
            # Extract title (from document properties or first heading)
            title = ""
            if doc.core_properties.title:
                title = doc.core_properties.title
            else:
                # Try to get title from first paragraph or heading
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        title = paragraph.text.strip()
                        break
            
            if not title:
                title = filepath.stem
            
            # Extract all text content
            content_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content_parts.append(paragraph.text.strip())
            
            # Extract table content if any
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        content_parts.append(" | ".join(row_text))
            
            content = "\n".join(content_parts)
            
            # Extract metadata
            metadata = {
                "author": doc.core_properties.author or "",
                "created": str(doc.core_properties.created) if doc.core_properties.created else "",
                "modified": str(doc.core_properties.modified) if doc.core_properties.modified else "",
                "subject": doc.core_properties.subject or "",
                "keywords": doc.core_properties.keywords or ""
            }
            
            return title, content, metadata
            
        except Exception as e:
            logger.error(f"Error processing DOCX file {filepath}: {e}")
            return "", "", {}

    def _extract_xlsx_content(self, filepath: Path) -> tuple[str, str, Dict[str, Any]]:
        """Extract content from XLSX file with improved sheet handling."""
        if not XLSX_AVAILABLE:
            logger.error(f"XLSX processing not available for {filepath}")
            return "", "", {}
        
        try:
            workbook = load_workbook(str(filepath), read_only=True, data_only=True)
            
            title = filepath.stem
            content_parts = []
            
            # Process each worksheet with priority to likely content sheets
            for sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]
                
                # Skip likely metadata sheets
                if any(name.lower() in sheet_name.lower() for name in ['metadata', 'summary', 'info']):
                    continue
                
                # Add sheet name as section header
                content_parts.append(f"\n[Sheet: {sheet_name}]")
                
                # Extract data from first 50 rows (institutional docs rarely need more)
                row_count = 0
                for row in worksheet.iter_rows(values_only=True):
                    row_data = []
                    for cell_value in row:
                        if cell_value is not None:
                            row_data.append(str(cell_value).strip())
                    
                    if row_data and any(cell for cell in row_data if cell):
                        content_parts.append(" | ".join(row_data))
                    
                    row_count += 1
                    if row_count >= 50:
                        break
            
            content = "\n".join(content_parts)
            
            # Basic metadata
            metadata = {
                "sheets": list(workbook.sheetnames),
                "sheet_count": len(workbook.sheetnames)
            }
            
            workbook.close()
            return title, content, metadata
            
        except Exception as e:
            logger.error(f"Error processing XLSX file {filepath}: {e}")
            return "", "", {}

    def _extract_pdf_content(self, filepath: Path) -> tuple[str, str, Dict[str, Any]]:
        """Extract content from PDF file."""
        title, content, metadata = "", "", {}
        
        # Try pdfplumber first (better text extraction)
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(str(filepath)) as pdf:
                    title = filepath.stem
                    content_parts = []
                    
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            content_parts.append(f"[Page {page_num}]\n{page_text.strip()}")
                    
                    content = "\n\n".join(content_parts)
                    
                    metadata = {
                        "pages": len(pdf.pages),
                        "extractor": "pdfplumber"
                    }
                    
                    # Try to get PDF metadata
                    if hasattr(pdf, 'metadata') and pdf.metadata:
                        if pdf.metadata.get('Title'):
                            title = pdf.metadata['Title']
                        metadata.update({
                            "author": pdf.metadata.get('Author', ''),
                            "subject": pdf.metadata.get('Subject', ''),
                            "creator": pdf.metadata.get('Creator', '')
                        })
                
                if content.strip():
                    return title, content, metadata
                    
            except Exception as e:
                logger.warning(f"pdfplumber failed for {filepath}: {e}")
        
        # Fallback to PyPDF2/pypdf
        if PDF_AVAILABLE:
            try:
                with open(filepath, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    
                    title = filepath.stem
                    content_parts = []
                    
                    # Extract text from all pages
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                content_parts.append(f"[Page {page_num}]\n{page_text.strip()}")
                        except Exception as e:
                            logger.warning(f"Error extracting page {page_num} from {filepath}: {e}")
                    
                    content = "\n\n".join(content_parts)
                    
                    metadata = {
                        "pages": len(pdf_reader.pages),
                        "extractor": "PyPDF2/pypdf"
                    }
                    
                    # Try to get PDF metadata
                    if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
                        if pdf_reader.metadata.get('/Title'):
                            title = pdf_reader.metadata['/Title']
                        metadata.update({
                            "author": pdf_reader.metadata.get('/Author', ''),
                            "subject": pdf_reader.metadata.get('/Subject', ''),
                            "creator": pdf_reader.metadata.get('/Creator', '')
                        })
                
                return title, content, metadata
                
            except Exception as e:
                logger.error(f"Error processing PDF file {filepath}: {e}")
        
        return title, content, metadata

    def _process_document(self, filepath: Path) -> bool:
        """Process a single document file with advanced multilingual capabilities."""
        try:
            logger.info(f"Processing document: {filepath}")
            
            # Determine file type and extract content
            suffix = filepath.suffix.lower()
            title, content, metadata = "", "", {}
            
            if suffix == '.docx':
                title, content, metadata = self._extract_docx_content(filepath)
            elif suffix == '.xlsx':
                title, content, metadata = self._extract_xlsx_content(filepath)
            elif suffix == '.pdf':
                title, content, metadata = self._extract_pdf_content(filepath)
            else:
                logger.warning(f"Unsupported file type: {suffix} for {filepath}")
                return False
            
            if not content or not content.strip():
                logger.warning(f"No content extracted from {filepath}")
                return False
            
            # Clean and limit content
            cleaned_content = re.sub(r'\s+', ' ', content).strip()
            if len(cleaned_content) > 75000:
                cleaned_content = cleaned_content[:75000] + "..."
            
            # Generate structured data matching website_scraper.py output format
            keywords = self.content_analyzer.extract_keywords_from_text(cleaned_content)
            content_id = self.content_analyzer.generate_content_id(str(filepath), title)
            summary = self.content_analyzer.generate_summary(cleaned_content)
            category = self.content_analyzer.categorize_content(title, cleaned_content, keywords, filepath.name)
            metrics = self.content_analyzer.calculate_metrics(cleaned_content, title)
            
            # Remove language-specific metric if present to maintain output structure
            if 'detected_language' in metrics:
                del metrics['detected_language']
            
            # Create document data structure matching pages.json format
            document_data = {
                "content_id": content_id,
                "url": str(filepath),  # Using filepath as URL equivalent
                "title": title,
                "summary": summary,
                "content": cleaned_content,
                "search_text": cleaned_content,  # Same as content for documents
                "keywords": keywords,
                "category": category,
                "images": [],  # Documents don't have image URLs
                "internal_links": [],  # Not applicable for documents
                "external_links": [],  # Not applicable for documents
                "structured_content": {},  # Could be enhanced later for document structure
                "metrics": metrics,
                "last_updated": datetime.now().isoformat(),
                "processing_info": {
                    "processed_at": datetime.now().isoformat(),
                    "content_length": len(cleaned_content),
                    "links_found": 0,  # N/A for documents
                    "images_found": 0,  # N/A for documents
                    "javascript_rendered": False,  # N/A for documents
                    "file_type": suffix,
                    "file_size_bytes": filepath.stat().st_size,
                    "metadata": metadata
                }
            }
            
            self.documents_data.append(document_data)
            logger.info(f"Successfully processed: {title} ({filepath.name})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process document {filepath}: {e}")
            return False

    def _find_supported_documents(self) -> List[Path]:
        """Find all supported document files in the input directory."""
        supported_extensions = ['.docx', '.xlsx', '.pdf']
        documents = []
        
        for ext in supported_extensions:
            pattern = f"*{ext}"
            found_files = list(self.input_dir.glob(pattern))
            documents.extend(found_files)
            logger.info(f"Found {len(found_files)} {ext.upper()} files")
        
        # Also check subdirectories
        for ext in supported_extensions:
            pattern = f"**/*{ext}"
            found_files = list(self.input_dir.glob(pattern))
            # Remove duplicates from direct directory scan
            new_files = [f for f in found_files if f not in documents]
            documents.extend(new_files)
            if new_files:
                logger.info(f"Found {len(new_files)} additional {ext.upper()} files in subdirectories")
        
        return sorted(documents)

    def _save_consolidated_output(self):
        """Save consolidated output to JSON file."""
        if not self.documents_data:
            logger.warning("No document data to save.")
            return
        
        # Create metadata matching website_scraper.py format
        metadata = {
            "processed_at": datetime.now().isoformat(),
            "last_crawl": datetime.now().isoformat(),  # Using same field name for compatibility
            "total_pages": len(self.documents_data),  # Using same field name for compatibility
            "total_urls_found": len(self.documents_data),  # Using same field name for compatibility
            "version": "2.1",  # Updated version number
            "target_website": str(self.input_dir),  # Using input directory as "target"
            "javascript_support_used": False,  # N/A for documents
            "ssl_verification_status": "N/A",  # N/A for documents
            "processing_summary": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": round(time.time() - self.start_time, 2),
                "categories": self._get_category_stats(),
                "javascript_rendered_pages_count": 0,  # N/A for documents
                "document_types_processed": self._get_file_type_stats()
            }
        }
        
        # Create consolidated data structure matching website_scraper.py
        consolidated_data = {
            "pages": self.documents_data,  # Using same field name for compatibility
            "metadata": metadata
        }
        
        # Save to output file
        output_path = OUTPUT_DIR / "ins_info.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Consolidated output saved to {output_path} ({len(self.documents_data)} documents)")
        logger.info(f"Category distribution: {self._get_category_stats()}")
        logger.info(f"File type distribution: {self._get_file_type_stats()}")

    def _get_category_stats(self) -> Dict[str, int]:
        """Get category distribution statistics."""
        counts: Dict[str, int] = {}
        for doc in self.documents_data:
            category = doc.get('category', 'unknown')
            counts[category] = counts.get(category, 0) + 1
        return counts

    def _get_file_type_stats(self) -> Dict[str, int]:
        """Get file type distribution statistics."""
        counts: Dict[str, int] = {}
        for doc in self.documents_data:
            file_type = doc.get('processing_info', {}).get('file_type', 'unknown')
            counts[file_type] = counts.get(file_type, 0) + 1
        return counts

    def run(self):
        """Main processing routine."""
        logger.info("=" * 60)
        logger.info("Starting AI_Gate Advanced Multilingual Document Processing (v2.1)")
        logger.info("=" * 60)
        
        try:
            # Find all supported documents
            documents = self._find_supported_documents()
            
            if not documents:
                logger.warning(f"No supported documents found in {self.input_dir}")
                logger.info("Supported formats: DOCX, XLSX, PDF")
                return
            
            logger.info(f"Found {len(documents)} documents to process")
            
            # Process each document
            success_count = 0
            failed_count = 0
            
            for i, doc_path in enumerate(documents, 1):
                try:
                    if self._process_document(doc_path):
                        success_count += 1
                    else:
                        failed_count += 1
                    
                    # Progress update
                    if i % 5 == 0 or i == len(documents):
                        rate = (success_count / i) * 100 if i > 0 else 0
                        logger.info(f"Progress: {i}/{len(documents)} - Success: {success_count}, Failed: {failed_count} ({rate:.1f}%)")
                
                except KeyboardInterrupt:
                    logger.warning("Processing interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error processing {doc_path}: {e}")
                    failed_count += 1
                    continue
            
            # Save results
            if self.documents_data:
                logger.info("Saving processed data...")
                self._save_consolidated_output()
                logger.info("✅ Document processing completed successfully.")
            else:
                logger.warning("⚠️ No documents were successfully processed.")
            
            # Summary
            logger.info("=" * 60)
            logger.info("DOCUMENT PROCESSING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total documents found: {len(documents)}")
            logger.info(f"Successfully processed: {success_count}")
            logger.info(f"Failed: {failed_count}")
            if len(documents) > 0:
                logger.info(f"Success rate: {(success_count/len(documents)*100):.1f}%")
            logger.info(f"Total processing time: {time.time() - self.start_time:.2f}s")
            
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
        except Exception as e:
            logger.critical(f"Critical error: {e}", exc_info=True)
        finally:
            # Save partial results if any processing occurred
            if self.documents_data and (success_count > 0 or failed_count > 0):
                logger.info("Ensuring results are saved...")
                try:
                    self._save_consolidated_output()
                except Exception as save_error:
                    logger.error(f"Failed to save results: {save_error}")
            
            self._generate_performance_report()

    def _generate_performance_report(self):
        """Generate performance report."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(time.time() - self.start_time, 2),
                "input_directory": str(self.input_dir),
                "documents_processed": len(self.documents_data),
                "output_file": str(OUTPUT_DIR / "ins_info.json"),
                "version": "2.1",  # Updated version number
                "categories_found": self._get_category_stats(),
                "file_types_processed": self._get_file_type_stats(),
                "system_info": {
                    "docx_available": DOCX_AVAILABLE,
                    "xlsx_available": XLSX_AVAILABLE,
                    "pdf_available": PDF_AVAILABLE,
                    "pdfplumber_available": PDFPLUMBER_AVAILABLE,
                    "nltk_available": NLTK_AVAILABLE,
                    "arabic_stemming_available": ARABIC_ISRI_AVAILABLE or QUTUF_AVAILABLE or ARABYCIA_AVAILABLE
                },
                "processing_notes": [
                    "Output structure compatible with website_scraper.py",
                    "Documents processed from local files",
                    "Main output: ins_info.json",
                    "Maintains same JSON structure as pages.json",
                    "Advanced multilingual processing with local NLTK resources"
                ]
            }
            
            report_path = LOG_DIR / f"performance_report_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Performance report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")


if __name__ == "__main__":
    try:
        # Check if required libraries are available
        missing_libs = []
        if not DOCX_AVAILABLE:
            missing_libs.append("python-docx (for DOCX files)")
        if not XLSX_AVAILABLE:
            missing_libs.append("openpyxl (for XLSX files)")
        if not PDF_AVAILABLE:
            missing_libs.append("PyPDF2 or pypdf (for PDF files)")
        if not NLTK_AVAILABLE:
            missing_libs.append("NLTK (for advanced text processing)")
        
        if missing_libs:
            logger.warning("Missing optional libraries:")
            for lib in missing_libs:
                logger.warning(f"  - {lib}")
            logger.warning("Some file types or advanced features may not be processed.")
        
        # Verify local NLTK data path
        if not LOCAL_NLTK_DATA_PATH.exists():
            logger.warning(f"Local NLTK data path not found: {LOCAL_NLTK_DATA_PATH}")
            logger.warning("Advanced NLP features may be limited")
        
        # Create and run processor
        processor = DocumentProcessor()
        processor.run()
        
    except KeyboardInterrupt:
        logger.info("Processing stopped by user")
    except Exception as e:
        logger.critical(f"Application startup error: {e}", exc_info=True)