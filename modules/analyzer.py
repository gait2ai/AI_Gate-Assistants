"""
AI Gate for Artificial Intelligence Applications
NLP Core - analyzer.py

This module serves as the centralized Natural Language Processing (NLP) core for the
AI Gate data ingestion pipeline. It provides a suite of tools for advanced,
multilingual (English and Arabic) analysis of text content.

Its primary purpose is to receive raw text from any data source (web pages,
documents) and transform it into a structured, enriched JSON object. This ensures
data consistency and quality across the entire knowledge base.

Core Components:
- LanguageDetector: Detects the primary language of a text.
- AdvancedTextProcessor: Performs low-level NLP tasks like normalization,
  tokenization, stopword removal, and stemming/lemmatization.
- AdvancedContentAnalyzer: The main orchestrator class that uses the other
  components to perform high-level analysis, including semantic categorization,
  summarization, and keyword extraction.

This module is designed to be called by various data scraper and processor
scripts, decoupling the complex NLP logic from the data extraction logic.
"""

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set, Union
import logging
from functools import lru_cache

# Third-party imports with fallbacks
try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect library not available. Using fallback language detection.")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, wordpunct_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.stem.isri import ISRIStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK library not available. Using fallback tokenization and processing.")

# Set up logging
logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detects the primary language of text content.
    
    Supports English and Arabic with both library-based detection
    (langdetect) and fallback heuristic analysis.
    """
    
    def __init__(self):
        """Initialize the language detector with optimized patterns."""
        # Comprehensive Arabic character ranges
        self.arabic_pattern = re.compile(
            r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
        )
        self.english_pattern = re.compile(r'[a-zA-Z]')
        
        # Cached detection threshold
        self._detection_cache = {}
        self.cache_size_limit = 1000
        
    @lru_cache(maxsize=1000)
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the given text with caching.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            str: Language code ('en' for English, 'ar' for Arabic)
        """
        if not text or not text.strip():
            return 'en'  # Default to English for empty text
            
        # Clean text for more accurate detection
        clean_text = self._clean_text_for_detection(text)
        
        # Try langdetect first if available
        if LANGDETECT_AVAILABLE and len(clean_text) > 10:
            try:
                detected = detect(clean_text)
                if detected in ['en', 'ar']:
                    return detected
                # If detected language is neither English nor Arabic, fall back to heuristic
            except LangDetectError:
                logger.debug("Language detection failed, using heuristic method")
        
        # Fallback heuristic method
        return self._heuristic_detection(clean_text)
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text by removing noise for better language detection."""
        # Remove URLs, emails, and special characters that might confuse detection
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', ' ', text)
        return text.strip()
    
    def _heuristic_detection(self, text: str) -> str:
        """
        Improved heuristic language detection based on character analysis.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            str: Language code ('en' for English, 'ar' for Arabic)
        """
        # Count Arabic and English characters
        arabic_chars = len(self.arabic_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        
        total_chars = arabic_chars + english_chars
        
        if total_chars == 0:
            return 'en'  # Default to English if no recognizable characters
        
        # Use a slightly higher threshold for Arabic to account for mixed content
        arabic_ratio = arabic_chars / total_chars
        threshold = 0.3  # Lower threshold to catch Arabic text with some English numbers/terms
        
        return 'ar' if arabic_ratio > threshold else 'en'
    
    def get_language_confidence(self, text: str) -> Dict[str, float]:
        """
        Get confidence scores for language detection.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict[str, float]: Confidence scores for each language
        """
        if not text or not text.strip():
            return {'en': 1.0, 'ar': 0.0}
        
        clean_text = self._clean_text_for_detection(text)
        arabic_chars = len(self.arabic_pattern.findall(clean_text))
        english_chars = len(self.english_pattern.findall(clean_text))
        
        total_chars = arabic_chars + english_chars
        if total_chars == 0:
            return {'en': 1.0, 'ar': 0.0}
        
        arabic_confidence = arabic_chars / total_chars
        english_confidence = english_chars / total_chars
        
        return {
            'ar': round(arabic_confidence, 3),
            'en': round(english_confidence, 3)
        }


class AdvancedTextProcessor:
    """
    Low-level NLP processing for multilingual text.
    
    Handles tokenization, normalization, stopword removal,
    and stemming/lemmatization for English and Arabic.
    """
    
    def __init__(self):
        """Initialize the text processor with necessary components."""
        self.lemmatizer = None
        self.arabic_stemmer = None
        self.stopwords_en = set()
        self.stopwords_ar = set()
        
        self._initialize_nlp_tools()
        self._load_stopwords()
        
        # Arabic normalization patterns
        self._arabic_normalization_patterns = self._compile_arabic_patterns()
        
    def _initialize_nlp_tools(self):
        """Initialize NLTK tools if available."""
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.arabic_stemmer = ISRIStemmer()
                logger.info("NLTK tools initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK tools: {e}")
        else:
            logger.info("NLTK not available, using fallback methods")
                
    def _load_stopwords(self):
        """Load stopwords for English and Arabic with comprehensive fallbacks."""
        if NLTK_AVAILABLE:
            try:
                self.stopwords_en = set(stopwords.words('english'))
                self.stopwords_ar = set(stopwords.words('arabic'))
                logger.info("NLTK stopwords loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load NLTK stopwords: {e}")
        
        # Enhanced fallback stopwords
        if not self.stopwords_en:
            self.stopwords_en = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                'who', 'whom', 'whose', 'why', 'how', 'all', 'any', 'both', 'each',
                'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                'only', 'own', 'same', 'so', 'than', 'too', 'very'
            }
            
        if not self.stopwords_ar:
            self.stopwords_ar = {
                'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
                'التي', 'الذي', 'اللذان', 'اللتان', 'اللذين', 'اللتين', 'اللاتي',
                'اللواتي', 'هو', 'هي', 'هم', 'هن', 'أن', 'إن', 'كان', 'كانت',
                'يكون', 'تكون', 'أنا', 'أنت', 'نحن', 'أنتم', 'أنتن', 'إياه', 'إياها',
                'إياهم', 'إياهن', 'إياي', 'إياك', 'إيانا', 'إياكم', 'إياكن',
                'لا', 'لم', 'لن', 'ما', 'لما', 'إذا', 'إذ', 'حيث', 'بينما', 'لكن',
                'غير', 'سوى', 'خلال', 'عبر', 'حول', 'دون', 'بعد', 'قبل', 'أمام',
                'وراء', 'تحت', 'فوق', 'بين', 'وسط', 'ضد', 'نحو', 'صوب', 'حتى',
                'منذ', 'مذ', 'لدى', 'عند', 'كل', 'جميع', 'بعض', 'معظم', 'أكثر',
                'أقل', 'كثير', 'قليل', 'جدا', 'فقط', 'أيضا', 'كذلك', 'هكذا',
                'هنا', 'هناك', 'حيث', 'أين', 'متى', 'كيف', 'ماذا', 'لماذا', 'أي'
            }
    
    def _compile_arabic_patterns(self) -> Dict[str, re.Pattern]:
        """Compile Arabic text normalization patterns for better performance."""
        return {
            'diacritics': re.compile(r'[\u064B-\u0652\u0670\u0640]'),
            'alef_variations': re.compile(r'[أإآ]'),
            'yeh_variations': re.compile(r'[يى]'),
            'teh_marbuta': re.compile(r'ة'),
            'whitespace': re.compile(r'\s+'),
            'punctuation': re.compile(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        }
    
    def normalize_arabic(self, text: str) -> str:
        """
        Enhanced Arabic text normalization with better performance.
        
        Args:
            text (str): Arabic text to normalize
            
        Returns:
            str: Normalized Arabic text
        """
        if not text:
            return text
        
        # Remove diacritics (tashkeel)
        text = self._arabic_normalization_patterns['diacritics'].sub('', text)
        
        # Normalize alef variations
        text = self._arabic_normalization_patterns['alef_variations'].sub('ا', text)
        
        # Normalize yeh variations
        text = self._arabic_normalization_patterns['yeh_variations'].sub('ي', text)
        
        # Normalize teh marbuta
        text = self._arabic_normalization_patterns['teh_marbuta'].sub('ه', text)
        
        # Normalize whitespace
        text = self._arabic_normalization_patterns['whitespace'].sub(' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str, language: str) -> List[str]:
        """
        Enhanced tokenization with proper Arabic support using wordpunct_tokenize.
        
        Args:
            text (str): Text to tokenize
            language (str): Language code ('en' or 'ar')
            
        Returns:
            List[str]: List of tokens
        """
        if not text or not text.strip():
            return []
        
        # Normalize Arabic text if needed
        if language == 'ar':
            text = self.normalize_arabic(text)
        
        # Use NLTK tokenizer if available
        if NLTK_AVAILABLE:
            try:
                if language == 'ar':
                    # Use wordpunct_tokenize for Arabic to avoid arabic.pickle dependency
                    tokens = wordpunct_tokenize(text)
                else:
                    # Use word_tokenize for English (more sophisticated)
                    tokens = word_tokenize(text)
                
                # Filter and clean tokens
                cleaned_tokens = []
                for token in tokens:
                    token = token.lower().strip()
                    # Keep alphanumeric tokens and Arabic characters
                    if token and (token.isalnum() or 
                                re.match(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', token)):
                        cleaned_tokens.append(token)
                
                return cleaned_tokens
                
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}")
        
        # Enhanced fallback tokenization
        if language == 'ar':
            # For Arabic, use word boundaries and Arabic character patterns
            tokens = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+|\w+', text.lower())
        else:
            # For English, use standard word boundaries
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out very short tokens
        return [token for token in tokens if len(token) > 1]
    
    def remove_stopwords(self, tokens: List[str], language: str) -> List[str]:
        """
        Remove stopwords from tokens with improved filtering.
        
        Args:
            tokens (List[str]): List of tokens
            language (str): Language code ('en' or 'ar')
            
        Returns:
            List[str]: Filtered tokens
        """
        if not tokens:
            return []
        
        stopwords_set = self.stopwords_ar if language == 'ar' else self.stopwords_en
        
        # Additional filtering: remove very short tokens and numbers-only tokens
        filtered_tokens = []
        for token in tokens:
            if (token not in stopwords_set and 
                len(token) > 1 and 
                not token.isdigit() and
                not re.match(r'^[^\w]+$', token)):  # Remove punctuation-only tokens
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def stem_and_lemmatize(self, tokens: List[str], language: str) -> List[str]:
        """
        Apply stemming and lemmatization to tokens with error handling.
        
        Args:
            tokens (List[str]): List of tokens
            language (str): Language code ('en' or 'ar')
            
        Returns:
            List[str]: Processed tokens
        """
        if not tokens:
            return []
        
        processed_tokens = []
        
        for token in tokens:
            try:
                if language == 'en' and self.lemmatizer:
                    processed_token = self.lemmatizer.lemmatize(token)
                elif language == 'ar' and self.arabic_stemmer:
                    processed_token = self.arabic_stemmer.stem(token)
                else:
                    processed_token = token
                
                # Ensure processed token is valid
                if processed_token and len(processed_token) > 1:
                    processed_tokens.append(processed_token)
                else:
                    processed_tokens.append(token)
                    
            except Exception as e:
                logger.debug(f"Processing failed for token '{token}': {e}")
                processed_tokens.append(token)
        
        return processed_tokens
    
    def extract_keywords(self, tokens: List[str], top_k: int = 10, min_freq: int = 1) -> List[Tuple[str, int]]:
        """
        Enhanced keyword extraction with frequency filtering.
        
        Args:
            tokens (List[str]): Processed tokens
            top_k (int): Number of top keywords to return
            min_freq (int): Minimum frequency for a token to be considered a keyword
            
        Returns:
            List[Tuple[str, int]]: List of (keyword, frequency) tuples
        """
        if not tokens:
            return []
        
        # Filter out very short tokens and apply minimum length
        filtered_tokens = [
            token for token in tokens 
            if len(token) >= 3 and not token.isdigit()
        ]
        
        if not filtered_tokens:
            return []
        
        # Count frequency
        counter = Counter(filtered_tokens)
        
        # Filter by minimum frequency
        frequent_tokens = {token: freq for token, freq in counter.items() 
                          if freq >= min_freq}
        
        # Return top keywords
        return Counter(frequent_tokens).most_common(top_k)
    
    def process_text(self, text: str, language: str) -> Dict[str, Union[List[str], List[Tuple[str, int]], int]]:
        """
        Complete text processing pipeline with enhanced metrics.
        
        Args:
            text (str): Raw text to process
            language (str): Language code ('en' or 'ar')
            
        Returns:
            Dict: Processed text data including tokens, keywords, etc.
        """
        if not text:
            return self._empty_processing_result()
        
        # Tokenize
        tokens = self.tokenize(text, language)
        
        # Remove stopwords
        filtered_tokens = self.remove_stopwords(tokens, language)
        
        # Stem and lemmatize
        processed_tokens = self.stem_and_lemmatize(filtered_tokens, language)
        
        # Extract keywords
        keywords = self.extract_keywords(processed_tokens)
        
        return {
            'original_tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'processed_tokens': processed_tokens,
            'keywords': keywords,
            'token_count': len(tokens),
            'filtered_token_count': len(filtered_tokens),
            'unique_tokens': len(set(processed_tokens)),
            'vocabulary_richness': len(set(processed_tokens)) / len(processed_tokens) if processed_tokens else 0
        }
    
    def _empty_processing_result(self) -> Dict[str, Union[List, int, float]]:
        """Return empty processing result for invalid input."""
        return {
            'original_tokens': [],
            'filtered_tokens': [],
            'processed_tokens': [],
            'keywords': [],
            'token_count': 0,
            'filtered_token_count': 0,
            'unique_tokens': 0,
            'vocabulary_richness': 0.0
        }


class AdvancedContentAnalyzer:
    """
    High-level content analyzer that orchestrates the entire NLP pipeline.
    
    This is the main class that scraper scripts will interact with.
    Enhanced with better categorization and improved analysis methods.
    """
    
    def __init__(self):
        """Initialize the content analyzer with enhanced components."""
        self.language_detector = LanguageDetector()
        self.text_processor = AdvancedTextProcessor()
        
        # Enhanced semantic categorization concepts with more comprehensive terms
        self.category_concepts = {
            'academic': {
                'en': [
                    'research', 'study', 'university', 'education', 'academic', 'scholar',
                    'journal', 'publication', 'thesis', 'dissertation', 'course', 'lecture',
                    'professor', 'student', 'curriculum', 'degree', 'bachelor', 'master',
                    'phd', 'doctorate', 'conference', 'symposium', 'peer-review', 'citation'
                ],
                'ar': [
                    'بحث', 'دراسة', 'جامعة', 'تعليم', 'أكاديمي', 'عالم', 'مجلة',
                    'نشر', 'رسالة', 'محاضرة', 'دورة', 'أستاذ', 'طالب', 'منهج',
                    'درجة', 'بكالوريوس', 'ماجستير', 'دكتوراه', 'مؤتمر', 'ندوة',
                    'مراجعة', 'استشهاد', 'كلية', 'معهد', 'تخصص'
                ]
            },
            'financial': {
                'en': [
                    'money', 'finance', 'investment', 'bank', 'economy', 'market',
                    'stock', 'profit', 'revenue', 'budget', 'cost', 'price',
                    'trading', 'forex', 'cryptocurrency', 'bitcoin', 'portfolio',
                    'dividend', 'interest', 'loan', 'mortgage', 'insurance'
                ],
                'ar': [
                    'مال', 'تمويل', 'استثمار', 'بنك', 'اقتصاد', 'سوق',
                    'سهم', 'ربح', 'إيراد', 'ميزانية', 'تكلفة', 'سعر',
                    'تداول', 'عملة', 'محفظة', 'فائدة', 'قرض', 'رهن',
                    'تأمين', 'أرباح', 'خسارة', 'صرف'
                ]
            },
            'technology': {
                'en': [
                    'technology', 'computer', 'software', 'digital', 'internet',
                    'data', 'artificial', 'intelligence', 'machine', 'learning',
                    'programming', 'coding', 'algorithm', 'database', 'network',
                    'cybersecurity', 'blockchain', 'cloud', 'mobile', 'app'
                ],
                'ar': [
                    'تكنولوجيا', 'حاسوب', 'برمجيات', 'رقمي', 'إنترنت',
                    'بيانات', 'ذكي', 'اصطناعي', 'آلة', 'تعلم',
                    'برمجة', 'خوارزمية', 'قاعدة', 'شبكة', 'أمان',
                    'سحابة', 'محمول', 'تطبيق', 'نظام'
                ]
            },
            'health': {
                'en': [
                    'health', 'medical', 'medicine', 'doctor', 'patient',
                    'treatment', 'disease', 'hospital', 'clinic', 'therapy',
                    'diagnosis', 'symptoms', 'medication', 'surgery', 'nursing',
                    'pharmacy', 'epidemic', 'vaccine', 'wellness', 'fitness'
                ],
                'ar': [
                    'صحة', 'طبي', 'طب', 'طبيب', 'مريض',
                    'علاج', 'مرض', 'مستشفى', 'عيادة', 'علاج',
                    'تشخيص', 'أعراض', 'دواء', 'جراحة', 'تمريض',
                    'صيدلة', 'وباء', 'لقاح', 'عافية', 'لياقة'
                ]
            },
            'business': {
                'en': [
                    'business', 'company', 'corporate', 'management', 'strategy',
                    'marketing', 'sales', 'customer', 'service', 'product',
                    'entrepreneur', 'startup', 'brand', 'competition', 'merger',
                    'acquisition', 'leadership', 'teamwork', 'innovation', 'growth'
                ],
                'ar': [
                    'أعمال', 'شركة', 'إدارة', 'استراتيجية', 'تسويق',
                    'مبيعات', 'عميل', 'خدمة', 'منتج', 'ريادة',
                    'ناشئة', 'علامة', 'منافسة', 'اندماج', 'استحواذ',
                    'قيادة', 'فريق', 'ابتكار', 'نمو', 'تطوير'
                ]
            },
            'sports': {
                'en': [
                    'sport', 'football', 'basketball', 'tennis', 'soccer',
                    'athlete', 'team', 'competition', 'championship', 'tournament',
                    'coach', 'training', 'fitness', 'exercise', 'match'
                ],
                'ar': [
                    'رياضة', 'كرة', 'قدم', 'سلة', 'تنس',
                    'رياضي', 'فريق', 'منافسة', 'بطولة', 'دوري',
                    'مدرب', 'تدريب', 'لياقة', 'تمرين', 'مباراة'
                ]
            },
            'politics': {
                'en': [
                    'politics', 'government', 'policy', 'election', 'democracy',
                    'parliament', 'minister', 'president', 'vote', 'campaign',
                    'law', 'legislation', 'public', 'citizen', 'rights'
                ],
                'ar': [
                    'سياسة', 'حكومة', 'سياسة', 'انتخابات', 'ديمقراطية',
                    'برلمان', 'وزير', 'رئيس', 'تصويت', 'حملة',
                    'قانون', 'تشريع', 'عام', 'مواطن', 'حقوق'
                ]
            }
        }
    
    def categorize_content(self, processed_tokens: List[str], language: str, 
                          use_weighted_scoring: bool = True) -> Dict[str, float]:
        """
        Enhanced content categorization with weighted scoring.
        
        Args:
            processed_tokens (List[str]): Processed text tokens
            language (str): Language code
            use_weighted_scoring (bool): Whether to use weighted scoring based on token frequency
            
        Returns:
            Dict[str, float]: Category scores
        """
        if not processed_tokens:
            return {}
        
        category_scores = {}
        token_counter = Counter(processed_tokens) if use_weighted_scoring else None
        total_tokens = len(processed_tokens)
        
        for category, concepts in self.category_concepts.items():
            lang_concepts = concepts.get(language, [])
            if not lang_concepts:
                continue
            
            if use_weighted_scoring and token_counter:
                # Weighted scoring based on token frequency
                weighted_matches = sum(
                    token_counter.get(concept, 0) for concept in lang_concepts
                )
                score = (weighted_matches / total_tokens) * 100 if total_tokens > 0 else 0
            else:
                # Simple presence-based scoring
                token_set = set(processed_tokens)
                matches = sum(1 for concept in lang_concepts if concept in token_set)
                score = (matches / len(lang_concepts)) * 100 if lang_concepts else 0
            
            category_scores[category] = round(score, 2)
        
        return dict(sorted(category_scores.items(), key=lambda x: x[1], reverse=True))
    
    def generate_summary(self, text: str, max_sentences: int = 3, max_length: int = 500) -> str:
        """
        Enhanced summary generation with better sentence selection.
        
        Args:
            text (str): Original text
            max_sentences (int): Maximum number of sentences in summary
            max_length (int): Maximum character length of summary
            
        Returns:
            str: Generated summary
        """
        if not text or not text.strip():
            return ""
        
        # Split into sentences with better regex
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        if len(sentences) <= max_sentences:
            summary = '. '.join(sentences)
            if not summary.endswith(('.', '!', '?')):
                summary += '.'
        else:
            # Select sentences with better distribution
            if max_sentences == 1:
                selected = [sentences[0]]
            elif max_sentences == 2:
                selected = [sentences[0], sentences[-1]] if len(sentences) > 1 else sentences
            else:
                # Take first, middle, and last sentences for better coverage
                indices = [0]
                if len(sentences) > 2:
                    indices.append(len(sentences) // 2)
                if len(sentences) > 1:
                    indices.append(-1)
                indices = indices[:max_sentences]
                selected = [sentences[i] for i in indices]
            
            summary = '. '.join(selected)
            if not summary.endswith(('.', '!', '?')):
                summary += '.'
        
        # Ensure summary doesn't exceed maximum length
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."
        
        return summary
    
    def calculate_readability_metrics(self, text: str, language: str = 'en') -> Dict[str, Union[int, float]]:
        """
        Enhanced readability metrics calculation with language-specific considerations.
        
        Args:
            text (str): Text to analyze
            language (str): Language code for language-specific calculations
            
        Returns:
            Dict[str, Union[int, float]]: Enhanced readability metrics
        """
        if not text or not text.strip():
            return self._empty_readability_metrics()
        
        # Count words with language-specific patterns
        if language == 'ar':
            # Arabic word pattern
            words = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+|\w+', text)
        else:
            # English word pattern
            words = re.findall(r'\b\w+\b', text)
        
        word_count = len(words)
        
        # Count sentences with better patterns
        sentence_pattern = r'[.!?؟]+(?:\s|$)'
        sentences = re.split(sentence_pattern, text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Count paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Calculate averages
        avg_words_per_sentence = round(word_count / sentence_count, 1) if sentence_count > 0 else 0
        avg_sentences_per_paragraph = round(sentence_count / paragraph_count, 1) if paragraph_count > 0 else 0
        
        # Character count (excluding whitespace)
        char_count = len(re.sub(r'\s+', '', text))
        avg_chars_per_word = round(char_count / word_count, 1) if word_count > 0 else 0
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'character_count': char_count,
            'avg_words_per_sentence': avg_words_per_sentence,
            'avg_sentences_per_paragraph': avg_sentences_per_paragraph,
            'avg_chars_per_word': avg_chars_per_word
        }
    
    def _empty_readability_metrics(self) -> Dict[str, int]:
        """Return empty readability metrics for invalid input."""
        return {
            'word_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'character_count': 0,
            'avg_words_per_sentence': 0,
            'avg_sentences_per_paragraph': 0,
            'avg_chars_per_word': 0
        }
    
    def extract_entities(self, text: str, language: str) -> Dict[str, List[str]]:
        """
        Basic named entity extraction using pattern matching.
        
        Args:
            text (str): Text to analyze
            language (str): Language code
            
        Returns:
            Dict[str, List[str]]: Extracted entities by type
        """
        entities = {
            'emails': [],
            'urls': [],
            'numbers': [],
            'dates': []
        }
        
        if not text:
            return entities
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = list(set(re.findall(email_pattern, text)))
        
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        entities['urls'] = list(set(re.findall(url_pattern, text)))
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        entities['numbers'] = list(set(re.findall(number_pattern, text)))
        
        # Extract date patterns (basic)
        if language == 'en':
            date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'
        else:  # Arabic
            date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        
        entities['dates'] = list(set(re.findall(date_pattern, text)))
        
        return entities
    
    def analyze_content(self, raw_text: str, title: str = "", 
                       metadata: Optional[Dict] = None,
                       include_entities: bool = True,
                       detailed_analysis: bool = True) -> Dict:
        """
        Enhanced main analysis method with comprehensive content analysis.
        
        Args:
            raw_text (str): Raw text content to analyze
            title (str): Title of the content (optional)
            metadata (Dict): Additional metadata (optional)
            include_entities (bool): Whether to extract entities
            detailed_analysis (bool): Whether to perform detailed analysis
            
        Returns:
            Dict: Complete analysis results in structured format
        """
        if not raw_text or not raw_text.strip():
            return self._empty_analysis_result()
        
        # Combine title and text for analysis
        full_text = f"{title} {raw_text}" if title else raw_text
        
        # Step 1: Language detection with confidence
        language = self.language_detector.detect_language(full_text)
        language_confidence = self.language_detector.get_language_confidence(full_text)
        
        # Step 2: Text processing
        processing_result = self.text_processor.process_text(full_text, language)
        
        # Step 3: Semantic categorization
        categories = self.categorize_content(
            processing_result['processed_tokens'], 
            language,
            use_weighted_scoring=detailed_analysis
        )
        
        # Step 4: Summary generation
        summary = self.generate_summary(raw_text)
        
        # Step 5: Readability metrics
        readability = self.calculate_readability_metrics(raw_text, language)
        
        # Step 6: Entity extraction (if requested)
        entities = self.extract_entities(raw_text, language) if include_entities else {}
        
        # Step 7: Determine primary category
        primary_category = max(categories.items(), key=lambda x: x[1])[0] if categories else 'general'
        
        # Step 8: Calculate content quality score
        quality_score = self._calculate_content_quality(processing_result, readability, categories)
        
        # Assemble final result
        analysis_result = {
            'title': title,
            'summary': summary,
            'language': language,
            'language_confidence': language_confidence,
            'primary_category': primary_category,
            'categories': categories,
            'keywords': [keyword for keyword, freq in processing_result['keywords']],
            'keyword_frequencies': dict(processing_result['keywords']),
            'readability_metrics': readability,
            'processing_stats': {
                'total_tokens': processing_result['token_count'],
                'filtered_tokens': processing_result['filtered_token_count'],
                'unique_tokens': processing_result['unique_tokens'],
                'vocabulary_richness': processing_result['vocabulary_richness']
            },
            'quality_score': quality_score,
            'metadata': metadata or {}
        }
        
        # Add entities if requested
        if include_entities:
            analysis_result['entities'] = entities
        
        return analysis_result
    
    def _calculate_content_quality(self, processing_result: Dict, 
                                  readability: Dict, categories: Dict) -> float:
        """
        Calculate a content quality score based on various metrics.
        
        Args:
            processing_result (Dict): Text processing results
            readability (Dict): Readability metrics
            categories (Dict): Category scores
            
        Returns:
            float: Quality score between 0 and 100
        """
        try:
            # Vocabulary richness (0-30 points)
            vocab_score = min(processing_result.get('vocabulary_richness', 0) * 60, 30)
            
            # Content length score (0-25 points)
            word_count = readability.get('word_count', 0)
            if word_count < 50:
                length_score = word_count * 0.3
            elif word_count < 200:
                length_score = 15 + (word_count - 50) * 0.1
            else:
                length_score = 25
            
            # Sentence structure score (0-20 points)
            avg_words = readability.get('avg_words_per_sentence', 0)
            if avg_words == 0:
                structure_score = 0
            elif 10 <= avg_words <= 25:
                structure_score = 20
            elif 5 <= avg_words < 10 or 25 < avg_words <= 35:
                structure_score = 15
            else:
                structure_score = 10
            
            # Category relevance (0-25 points)
            max_category_score = max(categories.values()) if categories else 0
            category_score = min(max_category_score * 0.5, 25)
            
            total_score = vocab_score + length_score + structure_score + category_score
            return round(min(total_score, 100), 1)
            
        except Exception as e:
            logger.warning(f"Error calculating quality score: {e}")
            return 50.0  # Default middle score
    
    def _empty_analysis_result(self) -> Dict:
        """Return empty analysis result for invalid input."""
        return {
            'title': '',
            'summary': '',
            'language': 'en',
            'language_confidence': {'en': 1.0, 'ar': 0.0},
            'primary_category': 'general',
            'categories': {},
            'keywords': [],
            'keyword_frequencies': {},
            'readability_metrics': self._empty_readability_metrics(),
            'processing_stats': {
                'total_tokens': 0,
                'filtered_tokens': 0,
                'unique_tokens': 0,
                'vocabulary_richness': 0.0
            },
            'quality_score': 0.0,
            'entities': {},
            'metadata': {}
        }
    
    def batch_analyze(self, texts: List[Dict[str, str]], 
                     detailed_analysis: bool = True) -> List[Dict]:
        """
        Analyze multiple texts efficiently.
        
        Args:
            texts (List[Dict[str, str]]): List of texts with 'text' and optional 'title' keys
            detailed_analysis (bool): Whether to perform detailed analysis
            
        Returns:
            List[Dict]: List of analysis results
        """
        results = []
        
        for i, text_data in enumerate(texts):
            try:
                text = text_data.get('text', '')
                title = text_data.get('title', '')
                metadata = text_data.get('metadata', {'batch_index': i})
                
                result = self.analyze_content(
                    text, title, metadata, 
                    detailed_analysis=detailed_analysis
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing text at index {i}: {e}")
                results.append(self._empty_analysis_result())
        
        return results


# Utility functions for external usage
def quick_analyze(text: str, title: str = "") -> Dict:
    """
    Quick analysis function for simple use cases.
    
    Args:
        text (str): Text to analyze
        title (str): Optional title
        
    Returns:
        Dict: Analysis results
    """
    analyzer = AdvancedContentAnalyzer()
    return analyzer.analyze_content(text, title, detailed_analysis=False)


def detect_language_simple(text: str) -> str:
    """
    Simple language detection function.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Language code ('en' or 'ar')
    """
    detector = LanguageDetector()
    return detector.detect_language(text)


# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AdvancedContentAnalyzer()
    
    # Test with English text
    english_text = """
    Artificial intelligence is revolutionizing the way we approach complex problems
    in technology and business. Machine learning algorithms are being used to
    analyze vast amounts of data and provide insights that were previously impossible
    to obtain. This technology has applications in healthcare, finance, and education.
    Companies are investing billions of dollars in AI research and development.
    The future of work will be significantly impacted by these technological advances.
    """
    
    # Test with Arabic text
    arabic_text = """
    الذكاء الاصطناعي يثور في طريقة تعاملنا مع المشاكل المعقدة في التكنولوجيا والأعمال.
    خوارزميات التعلم الآلي تستخدم لتحليل كميات كبيرة من البيانات وتوفير رؤى
    كانت مستحيلة الحصول عليها سابقاً. هذه التكنولوجيا لها تطبيقات في الصحة والمالية والتعليم.
    الشركات تستثمر مليارات الدولارات في بحث وتطوير الذكاء الاصطناعي.
    مستقبل العمل سيتأثر بشكل كبير بهذه التطورات التكنولوجية.
    """
    
    # Analyze both texts
    print("🔍 Analyzing English content...")
    english_result = analyzer.analyze_content(english_text, "AI Revolution in Technology")
    
    print("🔍 Analyzing Arabic content...")
    arabic_result = analyzer.analyze_content(arabic_text, "ثورة الذكاء الاصطناعي في التكنولوجيا")
    
    # Display results
    print("\n" + "="*60)
    print("📊 ENGLISH ANALYSIS RESULTS")
    print("="*60)
    print(f"Language: {english_result['language']} (Confidence: {english_result['language_confidence']})")
    print(f"Primary Category: {english_result['primary_category']}")
    print(f"Quality Score: {english_result['quality_score']}/100")
    print(f"Top Keywords: {english_result['keywords'][:5]}")
    print(f"Word Count: {english_result['readability_metrics']['word_count']}")
    print(f"Vocabulary Richness: {english_result['processing_stats']['vocabulary_richness']:.3f}")
    print(f"Summary: {english_result['summary'][:100]}...")
    
    print("\n" + "="*60)
    print("📊 ARABIC ANALYSIS RESULTS")
    print("="*60)
    print(f"Language: {arabic_result['language']} (Confidence: {arabic_result['language_confidence']})")
    print(f"Primary Category: {arabic_result['primary_category']}")
    print(f"Quality Score: {arabic_result['quality_score']}/100")
    print(f"Top Keywords: {arabic_result['keywords'][:5]}")
    print(f"Word Count: {arabic_result['readability_metrics']['word_count']}")
    print(f"Vocabulary Richness: {arabic_result['processing_stats']['vocabulary_richness']:.3f}")
    print(f"Summary: {arabic_result['summary'][:100]}...")
    
    # Test batch analysis
    print("\n" + "="*60)
    print("📦 BATCH ANALYSIS TEST")
    print("="*60)
    
    test_texts = [
        {"text": "This is a short health article about medicine.", "title": "Health Topic"},
        {"text": "هذا مقال قصير عن التكنولوجيا والذكاء الاصطناعي.", "title": "موضوع تقني"}
    ]
    
    batch_results = analyzer.batch_analyze(test_texts)
    for i, result in enumerate(batch_results):
        print(f"Text {i+1}: {result['language']} - {result['primary_category']} - Quality: {result['quality_score']}")
    
    print("\n✅ Analysis completed successfully!")