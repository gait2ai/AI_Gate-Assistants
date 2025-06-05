"""
Question Processor Module for AI Gate Application
Handles question validation, analysis, and preprocessing

This module is responsible for:
- Validating user input for completeness and coherence
- Extracting main topics and keywords from questions
- Performing language detection and basic preprocessing
- Returning structured question analysis with confidence scores
"""

import re
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

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
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class QuestionAnalysis:
    """Data class to hold question analysis results."""
    is_valid: bool = False
    original_question: str = ""
    cleaned_question: str = ""
    language: str = "en"
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

class QuestionProcessor:
    """
    Advanced question processor for analyzing and validating user queries.
    
    This class provides comprehensive question analysis including:
    - Input validation and sanitization
    - Language detection
    - Topic and keyword extraction  
    - Question type classification
    - Confidence scoring
    """
    
    def __init__(self, config: Dict[str, Any] = None, cache_manager=None):
        """
        Initialize the Question Processor.
        
        Args:
            config: Configuration dictionary with processing parameters
            cache_manager: Cache manager instance for storing processed results
        """
        self.config = config or {}
        self.cache_manager = cache_manager
        
        # Configuration parameters with defaults
        self.min_question_length = self.config.get('min_length', 3)
        self.max_question_length = self.config.get('max_length', 2000)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        self.enable_caching = self.config.get('enable_caching', True)
        self.supported_languages = self.config.get('supported_languages', ['en', 'es', 'fr', 'de', 'ar'])
        
        # Instance attribute for NLTK operational status
        self.nltk_is_operational = False
        
        # Initialize processing components
        self._initialize_components()
        
        # Question type patterns
        self.question_patterns = {
            'what': r'\b(what|which)\b',
            'how': r'\b(how)\b',
            'why': r'\b(why)\b',
            'when': r'\b(when)\b',
            'where': r'\b(where)\b',
            'who': r'\b(who|whom)\b',
            'definition': r'\b(define|definition|meaning|means)\b',
            'comparison': r'\b(compare|comparison|difference|versus|vs)\b',
            'procedure': r'\b(steps|process|procedure|guide|tutorial)\b',
            'factual': r'\b(is|are|does|do|can|will|would|should)\b'
        }
        
        # Common academic/institutional keywords
        self.institutional_keywords = {
            'academic': ['course', 'class', 'degree', 'program', 'curriculum', 'syllabus', 'credit', 'semester', 'professor', 'instructor'],
            'administrative': ['admission', 'enrollment', 'registration', 'fee', 'tuition', 'scholarship', 'financial', 'aid', 'deadline'],
            'campus': ['facility', 'library', 'dormitory', 'housing', 'dining', 'parking', 'recreation', 'gym', 'health'],
            'technical': ['requirement', 'prerequisite', 'policy', 'procedure', 'application', 'system', 'portal', 'website']
        }
        
        logger.info("Question Processor initialized successfully")
    
    def _initialize_components(self):
        """Initialize NLTK components using bundled data."""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, using basic text processing")
            self._initialize_fallback_components()
            return
        
        try:
            # Verify presence of required NLTK data without downloading
            required_datasets = [
                'tokenizers/punkt',
                'corpora/stopwords', 
                'corpora/wordnet',
                'taggers/averaged_perceptron_tagger',
                'corpora/omw-1.4'
            ]
            
            missing_datasets = []
            for dataset_path in required_datasets:
                try:
                    nltk.data.find(dataset_path)
                    logger.debug(f"Found bundled NLTK dataset: {dataset_path}")
                except LookupError:
                    missing_datasets.append(dataset_path)
                    logger.error(f"Missing bundled NLTK dataset: {dataset_path}")
            
            if missing_datasets:
                logger.critical(f"Missing required NLTK datasets: {missing_datasets}. "
                              f"Application should include these bundled datasets.")
                self._initialize_fallback_components()
                return
            
            # Initialize NLTK components directly with bundled data
            self.lemmatizer = WordNetLemmatizer()
            
            # Initialize multi-language stopwords
            self.stop_words_dict = {}
            for lang in self.supported_languages:
                try:
                    # Map language codes to NLTK stopwords language names
                    lang_mapping = {
                        'en': 'english',
                        'es': 'spanish', 
                        'fr': 'french',
                        'de': 'german',
                        'ar': 'arabic'
                    }
                    
                    nltk_lang = lang_mapping.get(lang, 'english')
                    if nltk_lang in stopwords.fileids():
                        self.stop_words_dict[lang] = set(stopwords.words(nltk_lang))
                        logger.debug(f"Loaded stopwords for language: {lang} ({nltk_lang})")
                    else:
                        logger.warning(f"Stopwords not available for language: {lang}, using English fallback")
                        self.stop_words_dict[lang] = set(stopwords.words('english'))
                        
                except Exception as e:
                    logger.warning(f"Failed to load stopwords for {lang}: {e}, using English fallback")
                    self.stop_words_dict[lang] = set(stopwords.words('english'))
            
            # Set default English stopwords for general validation purposes
            self.stop_words = self.stop_words_dict.get('en', set(stopwords.words('english')))
            
            # Mark NLTK as operational
            self.nltk_is_operational = True
            logger.info("NLTK components initialized successfully with bundled data")
            
        except Exception as e:
            logger.critical(f"NLTK initialization failed despite bundled data: {e}")
            self._initialize_fallback_components()
    
    def _initialize_fallback_components(self):
        """Initialize fallback components when NLTK is not available or fails."""
        logger.info("Initializing fallback text processing components")
        
        self.lemmatizer = None
        self.nltk_is_operational = False
        
        # Fallback stopwords for multiple languages
        fallback_stopwords = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                   'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                   'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those'},
            'es': {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su',
                   'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'le'},
            'fr': {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce',
                   'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand', 'en', 'une'},
            'de': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für',
                   'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus', 'er'},
            'ar': {'في', 'من', 'إلى', 'على', 'هذا', 'هذه', 'التي', 'الذي', 'كان', 'كانت', 'يكون', 'تكون', 'له', 'لها',
                   'عن', 'مع', 'بعد', 'قبل', 'حتى', 'لكن', 'أو', 'كل', 'بعض', 'غير', 'أن', 'أنه', 'أنها', 'لم', 'لن'}
        }
        
        self.stop_words_dict = {}
        for lang in self.supported_languages:
            self.stop_words_dict[lang] = fallback_stopwords.get(lang, fallback_stopwords['en'])
        
        # Set default English stopwords for general validation purposes
        self.stop_words = self.stop_words_dict.get('en', fallback_stopwords['en'])
        
        logger.info("Fallback components initialized successfully")
    
    def _get_stopwords_for_language(self, language: str) -> set:
        """Get stopwords for the specified language."""
        return self.stop_words_dict.get(language, self.stop_words_dict.get('en', self.stop_words))
    
    def _generate_stable_cache_key(self, question: str) -> str:
        """
        Generate a stable cache key using SHA-256 hash.
        
        Args:
            question: The question string to hash
            
        Returns:
            Stable cache key string
        """
        # Normalize the question for consistent hashing
        normalized_question = question.strip().lower()
        
        # Create SHA-256 hash for stable cache key
        hash_object = hashlib.sha256(normalized_question.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        
        return f"question_analysis:{hash_hex}"
    
    async def process_question(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process and analyze a user question.
        
        Args:
            question: The user's question string
            session_id: Optional session identifier for caching
            
        Returns:
            Dictionary containing question analysis results
        """
        start_time = datetime.now()
        
        try:
            logger.debug(f"Processing question: {question[:100]}...")
            
            # Check cache first if enabled
            cache_key = None
            if self.enable_caching and self.cache_manager:
                cache_key = self._generate_stable_cache_key(question)
                cached_result = await self._get_cached_analysis(cache_key)
                if cached_result:
                    logger.debug("Returning cached question analysis")
                    return cached_result
            
            # Create analysis object
            analysis = QuestionAnalysis(original_question=question)
            
            # Step 1: Basic validation
            validation_result = self._validate_input(question)
            if not validation_result['is_valid']:
                analysis.is_valid = False
                analysis.error_message = validation_result['error_message']
                return analysis.__dict__
            
            # Step 2: Clean and preprocess
            analysis.cleaned_question = self._clean_text(question)
            analysis.word_count = len(analysis.cleaned_question.split())
            analysis.sentence_count = len([s for s in analysis.cleaned_question.split('.') if s.strip()])
            
            # Step 3: Language detection
            lang_result = self._detect_language(analysis.cleaned_question)
            analysis.language = lang_result['language']
            analysis.language_confidence = lang_result['confidence']
            
            # Step 4: Extract topics and keywords
            extraction_result = await self._extract_topics_and_keywords(analysis.cleaned_question, analysis.language)
            analysis.topics = extraction_result['topics']
            analysis.keywords = extraction_result['keywords']
            analysis.entities = extraction_result['entities']
            
            # Step 5: Classify question type
            analysis.question_type = self._classify_question_type(analysis.cleaned_question)
            
            # Step 6: Calculate complexity and confidence scores
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
            
            # Add metadata
            analysis.processing_time = (datetime.now() - start_time).total_seconds()
            analysis.metadata = {
                'session_id': session_id,
                'processing_timestamp': datetime.now().isoformat(),
                'nltk_available': self.nltk_is_operational,
                'langdetect_available': LANGDETECT_AVAILABLE,
                'cache_key': cache_key
            }
            
            # Cache result if enabled
            if self.enable_caching and self.cache_manager and analysis.is_valid and cache_key:
                await self._cache_analysis(cache_key, analysis.__dict__)
            
            logger.debug(f"Question processing completed in {analysis.processing_time:.3f}s")
            return analysis.__dict__
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return QuestionAnalysis(
                is_valid=False,
                original_question=question,
                error_message=f"Processing error: {str(e)}",
                processing_time=processing_time
            ).__dict__
    
    def _validate_input(self, question: str) -> Dict[str, Any]:
        """
        Validate basic input requirements.
        
        Args:
            question: Input question string
            
        Returns:
            Dictionary with validation results
        """
        # Check if question exists and is string
        if not question or not isinstance(question, str):
            return {
                'is_valid': False,
                'error_message': 'Question must be a non-empty string'
            }
        
        # Check length constraints
        question = question.strip()
        if len(question) < self.min_question_length:
            return {
                'is_valid': False,
                'error_message': f'Question too short (minimum {self.min_question_length} characters)'
            }
        
        if len(question) > self.max_question_length:
            return {
                'is_valid': False,
                'error_message': f'Question too long (maximum {self.max_question_length} characters)'
            }
        
        # Check for suspicious patterns
        if self._contains_suspicious_patterns(question):
            return {
                'is_valid': False,
                'error_message': 'Question contains invalid content'
            }
        
        # Check if question has some meaningful content
        # Using default English stopwords for generic validation
        cleaned = re.sub(r'[^\w\s]', '', question.lower())
        meaningful_words = [w for w in cleaned.split() if w not in self.stop_words and len(w) > 2]
        
        if len(meaningful_words) < 1:
            return {
                'is_valid': False,
                'error_message': 'Question lacks meaningful content'
            }
        
        return {'is_valid': True}
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check for suspicious or malicious patterns in text."""
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'eval\(',
            r'document\.',
            r'window\.',
            r'\.exe\b',
            r'hack|crack|exploit',
            r'sql.*injection',
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in suspicious_patterns)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text input.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text string
        """
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove or replace special characters but preserve punctuation
        cleaned = re.sub(r'[^\w\s\.,;:?!-]', '', cleaned)
        
        # Normalize common abbreviations
        abbreviations = {
            r'\bu\b': 'you',
            r'\bur\b': 'your',
            r'\br\b': 'are',
            r'\btho\b': 'though',
            r'\bthru\b': 'through',
            r'\bw/\b': 'with',
            r'\bw/o\b': 'without'
        }
        
        for abbrev, full in abbreviations.items():
            cleaned = re.sub(abbrev, full, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of the input text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language and confidence
        """
        if not LANGDETECT_AVAILABLE:
            return {'language': 'en', 'confidence': 0.5}
        
        try:
            # langdetect works better with longer text
            if len(text.split()) < 3:
                return {'language': 'en', 'confidence': 0.7}
            
            detected_lang = detect(text)
            
            # Simple confidence estimation based on text length and character patterns
            confidence = min(0.9, 0.5 + (len(text.split()) * 0.05))
            
            # Boost confidence for English if it contains common English words
            english_indicators = ['the', 'and', 'or', 'but', 'what', 'how', 'why', 'when', 'where', 'who']
            if detected_lang == 'en':
                text_lower = text.lower()
                english_word_count = sum(1 for word in english_indicators if word in text_lower)
                confidence = min(0.95, confidence + (english_word_count * 0.05))
            
            return {
                'language': detected_lang,
                'confidence': confidence
            }
            
        except LangDetectException:
            logger.warning("Language detection failed, defaulting to English")
            return {'language': 'en', 'confidence': 0.6}
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return {'language': 'en', 'confidence': 0.5}
    
    async def _extract_topics_and_keywords(self, text: str, detected_language: str = 'en') -> Dict[str, List[str]]:
        """
        Extract topics, keywords, and entities from text.
        
        Args:
            text: Cleaned text to analyze
            detected_language: Detected language code for appropriate stopwords
            
        Returns:
            Dictionary with topics, keywords, and entities
        """
        # Get appropriate stopwords for the detected language
        language_stopwords = self._get_stopwords_for_language(detected_language)
        
        # Tokenize text
        if self.nltk_is_operational and self.lemmatizer:
            words = word_tokenize(text.lower())
            # Part-of-speech tagging to identify important words
            pos_tags = pos_tag(words)
            
            # Extract nouns, verbs, and adjectives as potential keywords
            important_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}
            keywords = [
                self.lemmatizer.lemmatize(word) 
                for word, pos in pos_tags 
                if pos in important_pos and word not in language_stopwords and len(word) > 2
            ]
        else:
            # Basic keyword extraction without NLTK
            words = re.findall(r'\b\w+\b', text.lower())
            keywords = [
                word for word in words 
                if word not in language_stopwords and len(word) > 2
            ]
        
        # Remove duplicates while preserving order
        keywords = list(dict.fromkeys(keywords))[:20]  # Limit to top 20
        
        # Extract topics based on institutional keyword categories
        topics = []
        text_lower = text.lower()
        
        for category, category_keywords in self.institutional_keywords.items():
            matches = [kw for kw in category_keywords if kw in text_lower]
            if matches:
                topics.append(category)
        
        # Add general topics based on question content
        if not topics:
            if any(word in text_lower for word in ['course', 'class', 'study', 'learn']):
                topics.append('academic')
            elif any(word in text_lower for word in ['apply', 'admission', 'enroll']):
                topics.append('administrative')
            elif any(word in text_lower for word in ['campus', 'location', 'facility']):
                topics.append('campus')
            else:
                topics.append('general')
        
        # Extract potential entities (capitalized words, numbers, etc.)
        entities = []
        
        # Find capitalized words that might be proper nouns
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(capitalized_words[:10])  # Limit to 10
        
        # Find numbers and dates
        numbers = re.findall(r'\b\d+\b', text)
        entities.extend(numbers[:5])  # Limit to 5
        
        # Find email patterns or URLs
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        entities.extend(emails)
        
        return {
            'topics': topics,
            'keywords': keywords,
            'entities': list(set(entities))  # Remove duplicates
        }
    
    def _classify_question_type(self, text: str) -> str:
        """
        Classify the type of question based on patterns.
        
        Args:
            text: Question text to classify
            
        Returns:
            Question type string
        """
        text_lower = text.lower()
        
        # Check each pattern type
        for question_type, pattern in self.question_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return question_type
        
        # Default classification based on sentence structure
        if text.strip().endswith('?'):
            return 'interrogative'
        elif any(word in text_lower for word in ['help', 'assist', 'support']):
            return 'help_request'
        elif any(word in text_lower for word in ['explain', 'describe', 'tell']):
            return 'explanation'
        else:
            return 'general'
    
    def _calculate_complexity_score(self, analysis: QuestionAnalysis) -> float:
        """
        Calculate complexity score based on various factors.
        
        Args:
            analysis: Question analysis object
            
        Returns:
            Complexity score between 0.0 and 1.0
        """
        score = 0.0
        
        # Word count factor (normalized)
        word_factor = min(1.0, analysis.word_count / 50)
        score += word_factor * 0.3
        
        # Sentence count factor
        sentence_factor = min(1.0, analysis.sentence_count / 5)
        score += sentence_factor * 0.2
        
        # Keywords complexity
        keyword_factor = min(1.0, len(analysis.keywords) / 10)
        score += keyword_factor * 0.2
        
        # Topic diversity
        topic_factor = min(1.0, len(analysis.topics) / 3)
        score += topic_factor * 0.15
        
        # Question type complexity
        complex_types = ['comparison', 'procedure', 'definition']
        if analysis.question_type in complex_types:
            score += 0.15
        
        return min(1.0, score)
    
    def _calculate_confidence_score(self, analysis: QuestionAnalysis) -> float:
        """
        Calculate overall confidence score for the analysis.
        
        Args:
            analysis: Question analysis object
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        
        # Language confidence
        score += analysis.language_confidence * 0.3
        
        # Content quality
        if analysis.word_count >= 3:
            score += 0.2
        if len(analysis.keywords) > 0:
            score += 0.2
        if len(analysis.topics) > 0:
            score += 0.15
        
        # Question structure
        if analysis.question_type != 'general':
            score += 0.1
        
        # Penalize for very short or very long questions
        if analysis.word_count < 2 or analysis.word_count > 100:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_validation_message(self, analysis: QuestionAnalysis) -> str:
        """
        Generate helpful validation message for invalid questions.
        
        Args:
            analysis: Question analysis object
            
        Returns:
            Validation message string
        """
        if analysis.language not in self.supported_languages:
            return f"Language '{analysis.language}' is not supported. Please ask your question in English."
        
        if len(analysis.keywords) == 0:
            return "Your question needs to be more specific. Please include relevant keywords or topics."
        
        if analysis.confidence_score < self.min_confidence_threshold:
            return "I'm having difficulty understanding your question. Could you please rephrase it more clearly?"
        
        return "Please provide a clearer, more specific question."
    
    async def _get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis result using CacheManager's generic get method.
        
        Args:
            cache_key: The cache key to retrieve
            
        Returns:
            Cached analysis result or None if not found
        """
        try:
            if hasattr(self.cache_manager, 'get'):
                return await self.cache_manager.get(cache_key)
        except Exception as e:
            logger.error(f"Error retrieving cached analysis: {e}")
        return None
    
    async def _cache_analysis(self, cache_key: str, analysis: Dict[str, Any]) -> None:
        """
        Cache analysis result using CacheManager's generic set method.
        
        Args:
            cache_key: The cache key to store under
            analysis: The analysis result to cache
        """
        try:
            if hasattr(self.cache_manager, 'set'):
                # Use CacheManager's set method with appropriate parameters
                await self.cache_manager.set(
                    key=cache_key,
                    value=analysis,
                    category='question_analysis',
                    ttl=3600  # 1 hour TTL for question analysis
                )
        except Exception as e:
            logger.error(f"Error caching analysis: {e}")
    
    def is_healthy(self) -> bool:
        """
        Check if the question processor is healthy and ready to process questions.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Test basic functionality
            test_question = "What is the meaning of life?"
            test_result = self._validate_input(test_question)
            return test_result['is_valid']
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup resources used by the question processor."""
        logger.info("Question Processor cleanup completed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns:
            Dictionary with processor statistics
        """
        return {
            'nltk_available': self.nltk_is_operational,
            'langdetect_available': LANGDETECT_AVAILABLE,
            'supported_languages': self.supported_languages,
            'min_confidence_threshold': self.min_confidence_threshold,
            'caching_enabled': self.enable_caching,
            'multi_language_stopwords': list(self.stop_words_dict.keys()) if hasattr(self, 'stop_words_dict') else []
        }