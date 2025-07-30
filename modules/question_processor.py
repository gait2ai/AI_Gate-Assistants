"""
Question Processor Module for AI Gate Application
Handles question validation, analysis, and preprocessing with enhanced Arabic support

This module is responsible for:
- Validating user input for completeness and coherence
- Extracting main topics and keywords from questions with language-specific tokenization
- Performing language detection and preprocessing optimized for Arabic and other languages
- Loading organization-specific keywords from YAML configuration
- Returning structured question analysis with confidence scores
- Using appropriate tokenizers for different languages (wordpunct_tokenize for Arabic)
"""

import re
import logging
import hashlib
import yaml
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
from functools import lru_cache

# Third-party imports for advanced processing
try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
    # Set seed for consistent language detection
    DetectorFactory.seed = 0
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class LanguageCode(Enum):
    """Enumeration of supported language codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ARABIC = "ar"


class TokenizationStrategy(Enum):
    """Enumeration of tokenization strategies."""
    WORD_TOKENIZE = "word_tokenize"
    WORDPUNCT_TOKENIZE = "wordpunct_tokenize"
    BASIC_SPLIT = "basic_split"


@dataclass
class LanguageConfig:
    """Configuration for language-specific processing."""
    code: str
    nltk_stopwords_name: Optional[str]
    tokenization_strategy: TokenizationStrategy
    requires_special_handling: bool = False
    character_pattern: Optional[str] = None


@dataclass
class QuestionAnalysis:
    """Data class to hold question analysis results with enhanced type hints."""
    is_valid: bool = False
    original_question: str = ""
    cleaned_question: str = ""
    language: str = "ar"  # Default to Arabic
    language_confidence: float = 0.0
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    question_type: str = "general"
    complexity_score: float = 0.0
    confidence_score: float = 0.0
    error_message: Optional[str] = None
    processing_time: float = 0.0
    word_count: int = 0
    sentence_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LanguageDetector:
    """Handles language detection with fallback mechanisms."""
    
    def __init__(self, fallback_language: str = "ar"):
        self.fallback_language = fallback_language
        
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of the input text with enhanced Arabic support.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language, confidence, and debug information
        """
        debug_info = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'langdetect_available': LANGDETECT_AVAILABLE,
            'detection_method': 'unknown',
            'character_analysis': {},
            'fallback_used': False
        }
        
        logger.debug(f"[LANG_DETECT] Starting detection for text: '{text[:50]}...' (length: {len(text)})")
        
        # Enhanced character analysis
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        debug_info['character_analysis'] = {
            'arabic_chars': arabic_chars,
            'latin_chars': latin_chars,
            'total_chars': total_chars,
            'arabic_percentage': (arabic_chars / total_chars * 100) if total_chars > 0 else 0,
            'latin_percentage': (latin_chars / total_chars * 100) if total_chars > 0 else 0
        }
        
        logger.debug(f"[LANG_DETECT] Arabic: {arabic_chars} ({debug_info['character_analysis']['arabic_percentage']:.1f}%), "
                    f"Latin: {latin_chars} ({debug_info['character_analysis']['latin_percentage']:.1f}%)")
        
        # Use character-based detection for very short text or when langdetect unavailable
        if not LANGDETECT_AVAILABLE or len(text.split()) < 3:
            debug_info['detection_method'] = 'character_based'
            return self._character_based_detection(text, debug_info)
        
        # Use langdetect for longer text with fallback
        try:
            debug_info['detection_method'] = 'langdetect'
            detected_lang = detect(text)
            confidence = self._calculate_detection_confidence(text, detected_lang, debug_info['character_analysis'])
            
            logger.debug(f"[LANG_DETECT] Result: {detected_lang} (confidence: {confidence})")
            
            return {
                'language': detected_lang,
                'confidence': confidence,
                'debug_info': debug_info
            }
            
        except (LangDetectException, Exception) as e:
            logger.warning(f"[LANG_DETECT] Detection failed: {e}, using fallback")
            debug_info['detection_method'] = 'fallback_after_exception'
            debug_info['fallback_used'] = True
            debug_info['exception'] = str(e)
            
            return self._character_based_detection(text, debug_info)
    
    def _character_based_detection(self, text: str, debug_info: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback character-based language detection."""
        char_analysis = debug_info['character_analysis']
        
        if char_analysis['arabic_chars'] > char_analysis['latin_chars']:
            logger.debug("[LANG_DETECT] Character-based: Arabic")
            return {
                'language': 'ar',
                'confidence': 0.7 if char_analysis['arabic_percentage'] > 50 else 0.6,
                'debug_info': debug_info
            }
        elif char_analysis['latin_chars'] > 0:
            logger.debug("[LANG_DETECT] Character-based: English")
            return {
                'language': 'en',
                'confidence': 0.6 if char_analysis['latin_percentage'] > 50 else 0.5,
                'debug_info': debug_info
            }
        else:
            logger.debug(f"[LANG_DETECT] Character-based: Fallback to {self.fallback_language}")
            debug_info['fallback_used'] = True
            return {
                'language': self.fallback_language,
                'confidence': 0.5,
                'debug_info': debug_info
            }
    
    def _calculate_detection_confidence(self, text: str, detected_lang: str, char_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for language detection."""
        # Base confidence from text length
        confidence = min(0.9, 0.5 + (len(text.split()) * 0.05))
        
        # Boost confidence based on character analysis alignment
        if detected_lang == 'ar' and char_analysis['arabic_percentage'] > 50:
            confidence = min(0.95, confidence + 0.2)
        elif detected_lang == 'en' and char_analysis['latin_percentage'] > 50:
            confidence = min(0.95, confidence + 0.2)
        
        # Language-specific indicators
        confidence += self._check_language_indicators(text, detected_lang)
        
        return max(0.0, min(1.0, confidence))
    
    def _check_language_indicators(self, text: str, language: str) -> float:
        """Check for language-specific indicators to boost confidence."""
        text_lower = text.lower()
        boost = 0.0
        
        if language == 'en':
            english_indicators = ['the', 'and', 'or', 'but', 'what', 'how', 'why', 'when', 'where', 'who']
            matches = sum(1 for word in english_indicators if word in text_lower)
            boost = min(0.15, matches * 0.03)
        
        elif language == 'ar':
            arabic_indicators = ['ما', 'هذا', 'هذه', 'كيف', 'لماذا', 'متى', 'أين', 'من', 'في', 'على']
            matches = sum(1 for word in arabic_indicators if word in text)
            boost = min(0.15, matches * 0.03)
        
        return boost


class LanguageSpecificTokenizer:
    """Handles language-specific tokenization strategies."""
    
    def __init__(self):
        self.language_configs = {
            LanguageCode.ENGLISH.value: LanguageConfig(
                code="en",
                nltk_stopwords_name="english",
                tokenization_strategy=TokenizationStrategy.WORD_TOKENIZE,
                character_pattern=r'[a-zA-Z]'
            ),
            LanguageCode.SPANISH.value: LanguageConfig(
                code="es",
                nltk_stopwords_name="spanish",
                tokenization_strategy=TokenizationStrategy.WORD_TOKENIZE,
                character_pattern=r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]'
            ),
            LanguageCode.FRENCH.value: LanguageConfig(
                code="fr",
                nltk_stopwords_name="french",
                tokenization_strategy=TokenizationStrategy.WORD_TOKENIZE,
                character_pattern=r'[a-zA-ZàâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ]'
            ),
            LanguageCode.GERMAN.value: LanguageConfig(
                code="de",
                nltk_stopwords_name="german", 
                tokenization_strategy=TokenizationStrategy.WORD_TOKENIZE,
                character_pattern=r'[a-zA-ZäöüßÄÖÜ]'
            ),
            LanguageCode.ARABIC.value: LanguageConfig(
                code="ar",
                nltk_stopwords_name="arabic",
                tokenization_strategy=TokenizationStrategy.WORDPUNCT_TOKENIZE,
                requires_special_handling=True,
                character_pattern=r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
            )
        }
    
    def tokenize_text(self, text: str, language: str) -> List[str]:
        """
        Tokenize text using the appropriate strategy for the language.
        
        Args:
            text: Text to tokenize
            language: Language code
            
        Returns:
            List of tokens
        """
        config = self.language_configs.get(language, self.language_configs[LanguageCode.ARABIC.value])
        
        try:
            if not NLTK_AVAILABLE:
                return self._basic_tokenize(text)
            
            if config.tokenization_strategy == TokenizationStrategy.WORDPUNCT_TOKENIZE:
                logger.debug(f"Using wordpunct_tokenize for language: {language}")
                tokens = wordpunct_tokenize(text.lower())
            elif config.tokenization_strategy == TokenizationStrategy.WORD_TOKENIZE:
                logger.debug(f"Using word_tokenize for language: {language}")
                tokens = word_tokenize(text.lower())
            else:
                tokens = self._basic_tokenize(text)
            
            # Additional filtering for Arabic to remove punctuation-only tokens
            if language == LanguageCode.ARABIC.value:
                tokens = [token for token in tokens if not re.match(r'^[^\w\u0600-\u06FF]+$', token)]
            
            return tokens
            
        except Exception as e:
            logger.warning(f"Tokenization failed for language {language}: {e}, using basic tokenization")
            return self._basic_tokenize(text)
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization fallback."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def get_language_config(self, language: str) -> LanguageConfig:
        """Get configuration for a specific language."""
        return self.language_configs.get(language, self.language_configs[LanguageCode.ARABIC.value])


class InstitutionalKeywordManager:
    """Manages institutional keywords from YAML configuration."""
    
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self._keywords: Dict[str, List[str]] = {}
        self._load_keywords()
    
    def _load_keywords(self) -> None:
        """Load institutional keywords from YAML configuration file."""
        try:
            possible_paths = [
                self.yaml_path,
                os.path.join(os.path.dirname(__file__), '..', self.yaml_path),
                os.path.join(os.getcwd(), self.yaml_path),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'institutional_keywords.yaml')
            ]
            
            for path in possible_paths:
                try:
                    full_path = Path(path).resolve()
                    if full_path.exists():
                        with open(full_path, 'r', encoding='utf-8') as file:
                            yaml_content = yaml.safe_load(file)
                            self._process_yaml_content(yaml_content, str(full_path))
                            return
                except Exception as e:
                    logger.debug(f"Failed to load from {path}: {e}")
                    continue
            
            logger.warning("Could not load institutional keywords, using fallback")
            self._keywords = self._get_fallback_keywords()
            
        except Exception as e:
            logger.error(f"Error loading institutional keywords: {e}")
            self._keywords = self._get_fallback_keywords()
    
    def _process_yaml_content(self, yaml_content: Any, path: str) -> None:
        """Process and validate YAML content."""
        if not isinstance(yaml_content, dict):
            raise ValueError("Invalid YAML structure: root should be a dictionary")
        
        institutional_keywords = yaml_content.get('institutional_keywords', {})
        if not isinstance(institutional_keywords, dict):
            raise ValueError("Invalid YAML structure: 'institutional_keywords' should be a dictionary")
        
        validated_keywords = {}
        for category, keywords in institutional_keywords.items():
            if isinstance(keywords, list):
                validated_keywords[category] = [
                    str(kw).strip() for kw in keywords 
                    if kw and str(kw).strip()
                ]
                logger.debug(f"Loaded {len(validated_keywords[category])} keywords for category '{category}'")
        
        if not validated_keywords:
            raise ValueError("No valid institutional keywords found in YAML")
        
        self._keywords = validated_keywords
        total_keywords = sum(len(keywords) for keywords in validated_keywords.values())
        logger.info(f"Loaded {total_keywords} institutional keywords from: {path}")
    
    def _get_fallback_keywords(self) -> Dict[str, List[str]]:
        """Get fallback institutional keywords."""
        return {
            'academic': [
                'course', 'class', 'degree', 'program', 'curriculum', 'syllabus', 
                'credit', 'semester', 'professor', 'instructor', 'lecture', 'lab',
                'دورة', 'صف', 'درجة', 'برنامج', 'منهج', 'ائتمان', 'فصل', 
                'أستاذ', 'مدرس', 'محاضرة', 'مختبر'
            ],
            'administrative': [
                'admission', 'enrollment', 'registration', 'fee', 'tuition', 
                'scholarship', 'financial', 'aid', 'deadline', 'application',
                'قبول', 'تسجيل', 'رسوم', 'منحة', 'مالي', 'مساعدة', 'موعد', 'طلب'
            ],
            'campus': [
                'facility', 'library', 'dormitory', 'housing', 'dining', 
                'parking', 'recreation', 'gym', 'health', 'campus',
                'مرفق', 'مكتبة', 'سكن', 'طعام', 'وقوف', 'ترفيه', 'رياضة', 'صحة', 'حرم'
            ],
            'technical': [
                'requirement', 'prerequisite', 'policy', 'procedure', 
                'application', 'system', 'portal', 'website', 'online',
                'متطلب', 'شرط', 'سياسة', 'إجراء', 'تطبيق', 'نظام', 'بوابة', 'موقع', 'إنترنت'
            ],
            'support': [
                'help', 'assistance', 'support', 'service', 'contact', 
                'office', 'hours', 'appointment', 'counseling',
                'مساعدة', 'دعم', 'خدمة', 'اتصال', 'مكتب', 'ساعات', 'موعد', 'استشارة'
            ]
        }
    
    @property
    def keywords(self) -> Dict[str, List[str]]:
        """Get the loaded keywords."""
        return self._keywords
    
    def reload(self) -> bool:
        """Reload keywords from YAML file."""
        try:
            old_keywords = self._keywords.copy()
            self._load_keywords()
            
            old_total = sum(len(keywords) for keywords in old_keywords.values())
            new_total = sum(len(keywords) for keywords in self._keywords.values())
            logger.info(f"Keywords reloaded. Count changed from {old_total} to {new_total}")
            return True
        except Exception as e:
            logger.error(f"Error reloading keywords: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get keyword statistics."""
        return {
            'categories': list(self._keywords.keys()),
            'total_categories': len(self._keywords),
            'keywords_per_category': {cat: len(kws) for cat, kws in self._keywords.items()},
            'total_keywords': sum(len(kws) for kws in self._keywords.values()),
            'yaml_path': self.yaml_path
        }


class QuestionProcessor:
    """
    Enhanced question processor with improved Arabic support and modular design.
    
    Key improvements:
    - Language-specific tokenization (wordpunct_tokenize for Arabic)
    - Modular architecture with separate components
    - Enhanced error handling and logging
    - Better caching mechanisms
    - Improved type hints and documentation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_manager=None):
        """
        Initialize the Question Processor with enhanced configuration.
        
        Args:
            config: Configuration dictionary with processing parameters
            cache_manager: Cache manager instance for storing processed results
        """
        self.config = config or {}
        self.cache_manager = cache_manager
        
        # Configuration parameters
        self.min_question_length = self.config.get('min_length', 3)
        self.max_question_length = self.config.get('max_length', 2000)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        self.enable_caching = self.config.get('enable_caching', True)
        self.supported_languages = self.config.get('supported_languages', ['en', 'es', 'fr', 'de', 'ar'])
        self.fallback_language = self.config.get('fallback_language', 'ar')
        
        # Initialize components
        self.language_detector = LanguageDetector(self.fallback_language)
        self.tokenizer = LanguageSpecificTokenizer()
        self.keyword_manager = InstitutionalKeywordManager(
            self.config.get('keywords_yaml_path', 'data/institutional_keywords.yaml')
        )
        
        # Initialize NLTK components
        self.nltk_is_operational = False
        self.lemmatizer = None
        self.stop_words_dict = {}
        self._initialize_nltk_components()
        
        # Question type patterns (multilingual)
        self.question_patterns = self._initialize_question_patterns()
        
        logger.info(f"Question Processor initialized with fallback language: {self.fallback_language}")
        logger.info(f"Tokenization strategies: Arabic=wordpunct_tokenize, Others=word_tokenize")
    
    def _initialize_nltk_components(self) -> None:
        """Initialize NLTK components with better error handling."""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, using basic text processing")
            self._initialize_fallback_components()
            return
        
        try:
            # Verify required datasets
            required_datasets = [
                'tokenizers/punkt',
                'corpora/stopwords',
                'corpora/wordnet',
                'taggers/averaged_perceptron_tagger'
            ]
            
            missing_datasets = []
            for dataset_path in required_datasets:
                try:
                    nltk.data.find(dataset_path)
                except LookupError:
                    missing_datasets.append(dataset_path)
            
            if missing_datasets:
                logger.error(f"Missing NLTK datasets: {missing_datasets}")
                self._initialize_fallback_components()
                return
            
            # Initialize components
            self.lemmatizer = WordNetLemmatizer()
            self._initialize_stopwords()
            self.nltk_is_operational = True
            
            logger.info("NLTK components initialized successfully")
            
        except Exception as e:
            logger.error(f"NLTK initialization failed: {e}")
            self._initialize_fallback_components()
    
    def _initialize_stopwords(self) -> None:
        """Initialize stopwords for supported languages."""
        lang_mapping = {
            'en': 'english', 'es': 'spanish', 'fr': 'french',
            'de': 'german', 'ar': 'arabic'
        }
        
        for lang in self.supported_languages:
            try:
                nltk_lang = lang_mapping.get(lang, 'arabic')
                if nltk_lang in stopwords.fileids():
                    self.stop_words_dict[lang] = set(stopwords.words(nltk_lang))
                else:
                    self.stop_words_dict[lang] = set(stopwords.words('arabic'))
            except Exception as e:
                logger.warning(f"Failed to load stopwords for {lang}: {e}")
                self.stop_words_dict[lang] = self._get_fallback_stopwords(lang)
    
    def _initialize_fallback_components(self) -> None:
        """Initialize fallback components."""
        self.lemmatizer = None
        self.nltk_is_operational = False
        
        for lang in self.supported_languages:
            self.stop_words_dict[lang] = self._get_fallback_stopwords(lang)
        
        logger.info("Fallback components initialized")
    
    def _get_fallback_stopwords(self, language: str) -> set:
        """Get fallback stopwords for a language."""
        fallback_stopwords = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'},
            'es': {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo'},
            'fr': {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour'},
            'de': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des'},
            'ar': {'في', 'من', 'إلى', 'على', 'هذا', 'هذه', 'التي', 'الذي', 'كان', 'كانت', 'يكون'}
        }
        return fallback_stopwords.get(language, fallback_stopwords['ar'])
    
    def _initialize_question_patterns(self) -> Dict[str, str]:
        """Initialize multilingual question patterns."""
        return {
            'what': r'\b(what|which|ما|ماذا|أي)\b',
            'how': r'\b(how|كيف)\b',
            'why': r'\b(why|لماذا|لم)\b',
            'when': r'\b(when|متى)\b',
            'where': r'\b(where|أين|حيث)\b',
            'who': r'\b(who|whom|من)\b',
            'definition': r'\b(define|definition|meaning|means|تعريف|معنى|يعني)\b',
            'comparison': r'\b(compare|comparison|difference|versus|vs|مقارنة|الفرق|مقابل)\b',
            'procedure': r'\b(steps|process|procedure|guide|tutorial|خطوات|عملية|إجراء|دليل)\b',
            'factual': r'\b(is|are|does|do|can|will|would|should|هل|أم|لعل)\b'
        }
    
    @lru_cache(maxsize=128)
    def _get_stopwords_for_language(self, language: str) -> frozenset:
        """Get stopwords for language (cached)."""
        return frozenset(self.stop_words_dict.get(language, self.stop_words_dict.get('ar', set())))
    
    def _validate_input(self, question: str) -> Dict[str, Any]:
        """Validate basic input requirements with enhanced checks."""
        if not question or not isinstance(question, str):
            return {'is_valid': False, 'error_message': 'Question must be a non-empty string'}
        
        question = question.strip()
        if len(question) < self.min_question_length:
            return {'is_valid': False, 'error_message': f'Question too short (minimum {self.min_question_length} characters)'}
        
        if len(question) > self.max_question_length:
            return {'is_valid': False, 'error_message': f'Question too long (maximum {self.max_question_length} characters)'}
        
        if self._contains_suspicious_patterns(question):
            return {'is_valid': False, 'error_message': 'Question contains invalid content'}
        
        # Check for meaningful content
        cleaned = re.sub(r'[^\w\s\u0600-\u06FF]', '', question.lower())
        words = cleaned.split()
        meaningful_words = [w for w in words if len(w) > 2]
        
        if len(meaningful_words) < 1:
            return {'is_valid': False, 'error_message': 'Question lacks meaningful content'}
        
        return {'is_valid': True}
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Enhanced suspicious pattern detection."""
        suspicious_patterns = [
            r'<script', r'javascript:', r'eval\(', r'document\.', r'window\.',
            r'\.exe\b', r'hack|crack|exploit', r'sql.*injection',
            r'<iframe', r'onload=', r'onerror=', r'style=.*expression'
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in suspicious_patterns)
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better Arabic support."""
        cleaned = text.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Preserve Arabic text and punctuation
        cleaned = re.sub(r'[^\w\s\.,;:?!؟\u0600-\u06FF-]', '', cleaned)
        
        # Normalize abbreviations
        abbreviations = {
            r'\bu\b': 'you', r'\bur\b': 'your', r'\br\b': 'are',
            r'\btho\b': 'though', r'\bthru\b': 'through',
            r'\bw/\b': 'with', r'\bw/o\b': 'without'
        }
        
        for abbrev, full in abbreviations.items():
            cleaned = re.sub(abbrev, full, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    async def _extract_topics_and_keywords(self, text: str, detected_language: str = 'ar') -> Dict[str, List[str]]:
        """
        Enhanced topic and keyword extraction with language-specific tokenization.
        
        Args:
            text: Cleaned text to analyze
            detected_language: Detected language code
            
        Returns:
            Dictionary with topics, keywords, and entities
        """
        # Get language-specific stopwords
        language_stopwords = self._get_stopwords_for_language(detected_language)
        
        # Use language-specific tokenization
        tokens = self.tokenizer.tokenize_text(text, detected_language)
        
        # Extract keywords using NLTK if available
        if self.nltk_is_operational and self.lemmatizer:
            try:
                # POS tagging for better keyword extraction
                pos_tags = pos_tag(tokens)
                important_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}
                
                keywords = []
                for word, pos in pos_tags:
                    if (pos in important_pos and 
                        word not in language_stopwords and 
                        len(word) > 2 and
                        not re.match(r'^[^\w\u0600-\u06FF]+, word)):  # Filter punctuation-only tokens
                        
                        # Lemmatize only for non-Arabic languages (lemmatizer doesn't handle Arabic well)
                        if detected_language != 'ar':
                            keywords.append(self.lemmatizer.lemmatize(word))
                        else:
                            keywords.append(word)
                            
            except Exception as e:
                logger.warning(f"POS tagging failed for {detected_language}: {e}, using basic extraction")
                keywords = self._extract_keywords_basic(tokens, language_stopwords)
        else:
            keywords = self._extract_keywords_basic(tokens, language_stopwords)
        
        # Remove duplicates while preserving order, limit to top 20
        keywords = list(dict.fromkeys(keywords))[:20]
        
        # Extract topics using institutional keywords
        topics = self._extract_topics(text.lower())
        
        # Extract entities
        entities = self._extract_entities(text)
        
        logger.debug(f"Extracted for {detected_language}: topics={len(topics)}, keywords={len(keywords)}, entities={len(entities)}")
        
        return {
            'topics': topics,
            'keywords': keywords,
            'entities': entities
        }
    
    def _extract_keywords_basic(self, tokens: List[str], stopwords: frozenset) -> List[str]:
        """Basic keyword extraction fallback."""
        return [
            word for word in tokens
            if (word not in stopwords and 
                len(word) > 2 and
                not re.match(r'^[^\w\u0600-\u06FF]+, word))
        ]
    
    def _extract_topics(self, text_lower: str) -> List[str]:
        """Extract topics based on institutional keywords."""
        topics = []
        
        logger.debug(f"Analyzing text for institutional topics: {text_lower[:100]}...")
        
        # Check institutional keyword categories
        for category, category_keywords in self.keyword_manager.keywords.items():
            matches = []
            for kw in category_keywords:
                kw_lower = kw.lower()
                if kw_lower in text_lower:
                    matches.append(kw)
                    logger.debug(f"Found keyword '{kw}' in category '{category}'")
            
            if matches:
                topics.append(category)
                logger.debug(f"Added topic '{category}' based on matches: {matches}")
        
        # Fallback to general topics if no institutional matches
        if not topics:
            topics = self._extract_general_topics(text_lower)
        
        return topics
    
    def _extract_general_topics(self, text_lower: str) -> List[str]:
        """Extract general topics when no institutional keywords match."""
        topics = []
        
        topic_terms = {
            'academic': ['course', 'class', 'study', 'learn', 'education', 'school', 'university', 
                        'college', 'degree', 'program', 'دورة', 'تعلم', 'دراسة', 'تعليم', 'جامعة'],
            'administrative': ['apply', 'admission', 'enroll', 'register', 'fee', 'tuition', 
                             'تسجيل', 'قبول', 'رسوم', 'التحاق'],
            'campus': ['campus', 'location', 'facility', 'building', 'library', 'lab',
                      'حرم', 'مكان', 'مبنى', 'مكتبة', 'مختبر'],
            'technical': ['system', 'website', 'portal', 'online', 'application', 'login',
                         'نظام', 'موقع', 'بوابة', 'إنترنت', 'تطبيق'],
            'support': ['help', 'support', 'assistance', 'service', 'contact',
                       'مساعدة', 'دعم', 'خدمة', 'اتصال']
        }
        
        for topic, terms in topic_terms.items():
            if any(term in text_lower for term in terms):
                topics.append(topic)
                logger.debug(f"Added general topic: {topic}")
        
        return topics if topics else ['general']
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text."""
        entities = []
        
        # Capitalized words (proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(capitalized_words[:10])
        
        # Numbers and dates
        numbers = re.findall(r'\b\d+\b', text)
        entities.extend(numbers[:5])
        
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        entities.extend(emails)
        
        # URLs
        urls = re.findall(r'https?://[^\s]+', text)
        entities.extend(urls)
        
        return list(set(entities))
    
    def _classify_question_type(self, text: str) -> str:
        """Classify question type using multilingual patterns."""
        text_lower = text.lower()
        
        for question_type, pattern in self.question_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return question_type
        
        # Fallback classification
        if text.strip().endswith('?') or text.strip().endswith('؟'):
            return 'interrogative'
        elif any(word in text_lower for word in ['help', 'assist', 'support', 'مساعدة', 'دعم']):
            return 'help_request'
        elif any(word in text_lower for word in ['explain', 'describe', 'tell', 'شرح', 'وصف']):
            return 'explanation'
        else:
            return 'general'
    
    def _calculate_complexity_score(self, analysis: QuestionAnalysis) -> float:
        """Calculate complexity score with improved metrics."""
        score = 0.0
        
        # Word count factor (normalized to 50 words max)
        word_factor = min(1.0, analysis.word_count / 50)
        score += word_factor * 0.25
        
        # Sentence count factor (normalized to 5 sentences max)
        sentence_factor = min(1.0, analysis.sentence_count / 5)
        score += sentence_factor * 0.15
        
        # Keywords complexity (normalized to 15 keywords max)
        keyword_factor = min(1.0, len(analysis.keywords) / 15)
        score += keyword_factor * 0.25
        
        # Topic diversity (normalized to 3 topics max)
        topic_factor = min(1.0, len(analysis.topics) / 3)
        score += topic_factor * 0.15
        
        # Question type complexity
        complex_types = ['comparison', 'procedure', 'definition', 'explanation']
        if analysis.question_type in complex_types:
            score += 0.15
        
        # Entity presence
        if analysis.entities:
            score += min(0.05, len(analysis.entities) * 0.01)
        
        return min(1.0, score)
    
    def _calculate_confidence_score(self, analysis: QuestionAnalysis) -> float:
        """Calculate confidence score with enhanced metrics."""
        score = 0.0
        
        # Language confidence (30% weight)
        score += analysis.language_confidence * 0.3
        
        # Content quality indicators
        if analysis.word_count >= 3:
            score += 0.15
        if analysis.word_count >= 5:
            score += 0.05  # Bonus for longer questions
        
        if len(analysis.keywords) > 0:
            score += 0.2
        if len(analysis.keywords) >= 3:
            score += 0.05  # Bonus for more keywords
        
        if len(analysis.topics) > 0:
            score += 0.15
        
        # Question structure quality
        if analysis.question_type != 'general':
            score += 0.1
        
        # Penalize extremes
        if analysis.word_count < 2:
            score -= 0.15
        elif analysis.word_count > 100:
            score -= 0.1
        
        # Language-specific adjustments
        if analysis.language == 'ar' and analysis.language_confidence > 0.8:
            score += 0.05  # Bonus for confident Arabic detection
        
        return max(0.0, min(1.0, score))
    
    def _generate_validation_message(self, analysis: QuestionAnalysis) -> str:
        """Generate helpful validation messages."""
        if analysis.language not in self.supported_languages:
            return f"Language '{analysis.language}' is not supported. Please ask in a supported language."
        
        if len(analysis.keywords) == 0:
            return "Please make your question more specific by including relevant keywords."
        
        if analysis.confidence_score < self.min_confidence_threshold:
            if analysis.word_count < 3:
                return "Your question is too short. Please provide more details."
            else:
                return "Please rephrase your question more clearly."
        
        return "Please provide a clearer, more specific question."
    
    def _generate_cache_key(self, question: str) -> str:
        """Generate stable cache key."""
        normalized_question = question.strip().lower()
        hash_object = hashlib.sha256(normalized_question.encode('utf-8'))
        return f"question_analysis:{hash_object.hexdigest()}"
    
    async def _get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis."""
        try:
            if self.cache_manager and hasattr(self.cache_manager, 'get'):
                return await self.cache_manager.get(cache_key)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        return None
    
    async def _cache_analysis(self, cache_key: str, analysis: Dict[str, Any]) -> None:
        """Cache analysis result."""
        try:
            if self.cache_manager and hasattr(self.cache_manager, 'set'):
                await self.cache_manager.set(
                    key=cache_key,
                    value=analysis,
                    category='question_analysis',
                    ttl=3600
                )
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    async def process_question(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process and analyze a user question with enhanced Arabic support.
        
        Args:
            question: The user's question string
            session_id: Optional session identifier for caching
            
        Returns:
            Dictionary containing comprehensive question analysis results
        """
        start_time = datetime.now()
        
        try:
            logger.debug(f"Processing question: {question[:100]}...")
            
            # Check cache first
            cache_key = None
            if self.enable_caching and self.cache_manager:
                cache_key = self._generate_cache_key(question)
                cached_result = await self._get_cached_analysis(cache_key)
                if cached_result:
                    logger.debug("Returning cached analysis")
                    return cached_result
            
            # Initialize analysis
            analysis = QuestionAnalysis(original_question=question)
            
            # Step 1: Input validation
            validation_result = self._validate_input(question)
            if not validation_result['is_valid']:
                analysis.is_valid = False
                analysis.error_message = validation_result['error_message']
                return self._finalize_analysis(analysis, start_time, session_id, cache_key)
            
            # Step 2: Text cleaning and preprocessing
            analysis.cleaned_question = self._clean_text(question)
            analysis.word_count = len(analysis.cleaned_question.split())
            analysis.sentence_count = len([s for s in analysis.cleaned_question.split('.') if s.strip()])
            
            # Step 3: Enhanced language detection
            lang_result = self.language_detector.detect_language(analysis.cleaned_question)
            analysis.language = lang_result['language']
            analysis.language_confidence = lang_result['confidence']
            
            # Step 4: Language-specific topic and keyword extraction
            extraction_result = await self._extract_topics_and_keywords(
                analysis.cleaned_question, 
                analysis.language
            )
            analysis.topics = extraction_result['topics']
            analysis.keywords = extraction_result['keywords']
            analysis.entities = extraction_result['entities']
            
            # Step 5: Question type classification
            analysis.question_type = self._classify_question_type(analysis.cleaned_question)
            
            # Step 6: Score calculations
            analysis.complexity_score = self._calculate_complexity_score(analysis)
            analysis.confidence_score = self._calculate_confidence_score(analysis)
            
            # Step 7: Final validation
            analysis.is_valid = (
                analysis.confidence_score >= self.min_confidence_threshold and
                analysis.language in self.supported_languages and
                len(analysis.keywords) > 0
            )
            
            if not analysis.is_valid and not analysis.error_message:
                analysis.error_message = self._generate_validation_message(analysis)
            
            return await self._finalize_analysis(analysis, start_time, session_id, cache_key, lang_result)
            
        except Exception as e:
            logger.error(f"Question processing error: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return QuestionAnalysis(
                is_valid=False,
                original_question=question,
                error_message=f"Processing error: {str(e)}",
                processing_time=processing_time
            ).__dict__
    
    async def _finalize_analysis(self, analysis: QuestionAnalysis, start_time: datetime, 
                               session_id: Optional[str], cache_key: Optional[str],
                               lang_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Finalize analysis with metadata and caching."""
        # Calculate processing time
        analysis.processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add comprehensive metadata
        analysis.metadata = {
            'session_id': session_id,
            'processing_timestamp': datetime.now().isoformat(),
            'nltk_available': self.nltk_is_operational,
            'langdetect_available': LANGDETECT_AVAILABLE,
            'cache_key': cache_key,
            'fallback_language': self.fallback_language,
            'tokenization_strategy': self.tokenizer.get_language_config(analysis.language).tokenization_strategy.value,
            'institutional_keywords_loaded': bool(self.keyword_manager.keywords),
            'institutional_keywords_categories': list(self.keyword_manager.keywords.keys()),
            'supported_languages': self.supported_languages
        }
        
        if lang_result:
            analysis.metadata['language_detection_details'] = lang_result.get('debug_info', {})
        
        # Cache successful analysis
        if (self.enable_caching and self.cache_manager and 
            analysis.is_valid and cache_key):
            await self._cache_analysis(cache_key, analysis.__dict__)
        
        logger.debug(f"Analysis completed in {analysis.processing_time:.3f}s "
                    f"(valid: {analysis.is_valid}, lang: {analysis.language}, "
                    f"confidence: {analysis.confidence_score:.2f})")
        
        return analysis.__dict__
    
    # Public utility methods
    
    def reload_institutional_keywords(self) -> bool:
        """Reload institutional keywords from YAML file."""
        return self.keyword_manager.reload()
    
    def get_institutional_keyword_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded institutional keywords."""
        return self.keyword_manager.get_stats()
    
    def get_language_config(self, language: str) -> Dict[str, Any]:
        """Get configuration for a specific language."""
        config = self.tokenizer.get_language_config(language)
        return {
            'code': config.code,
            'tokenization_strategy': config.tokenization_strategy.value,
            'nltk_stopwords_name': config.nltk_stopwords_name,
            'requires_special_handling': config.requires_special_handling,
            'character_pattern': config.character_pattern
        }
    
    def is_healthy(self) -> bool:
        """Check if the processor is healthy and ready."""
        try:
            # Test basic functionality
            test_result = self._validate_input("What is the meaning of life?")
            keywords_loaded = bool(self.keyword_manager.keywords)
            
            return test_result['is_valid'] and keywords_loaded
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processor statistics."""
        return {
            'processor_info': {
                'nltk_available': self.nltk_is_operational,
                'langdetect_available': LANGDETECT_AVAILABLE,
                'supported_languages': self.supported_languages,
                'fallback_language': self.fallback_language,
                'min_confidence_threshold': self.min_confidence_threshold,
                'caching_enabled': self.enable_caching
            },
            'tokenization_strategies': {
                lang: self.tokenizer.get_language_config(lang).tokenization_strategy.value
                for lang in self.supported_languages
            },
            'stopwords_available': list(self.stop_words_dict.keys()),
            'institutional_keywords': self.keyword_manager.get_stats(),
            'question_patterns': list(self.question_patterns.keys())
        }
    
    def cleanup(self) -> None:
        """Cleanup resources used by the processor."""
        # Clear caches
        self._get_stopwords_for_language.cache_clear()
        logger.info("Question Processor cleanup completed")


# Convenience functions for backward compatibility
async def process_question_simple(question: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Simple function to process a question without maintaining processor state.
    
    Args:
        question: Question to process
        config: Optional configuration
        
    Returns:
        Question analysis result
    """
    processor = QuestionProcessor(config)
    try:
        return await processor.process_question(question)
    finally:
        processor.cleanup()


def create_default_processor(**kwargs) -> QuestionProcessor:
    """
    Create a processor with default configuration.
    
    Args:
        **kwargs: Additional configuration options
        
    Returns:
        Configured QuestionProcessor instance
    """
    default_config = {
        'min_length': 3,
        'max_length': 2000,
        'min_confidence_threshold': 0.6,
        'enable_caching': True,
        'supported_languages': ['en', 'es', 'fr', 'de', 'ar'],
        'fallback_language': 'ar',
        'keywords_yaml_path': 'data/institutional_keywords.yaml'
    }
    
    default_config.update(kwargs)
    return QuestionProcessor(default_config)