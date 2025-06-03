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
    from nltk.tokenize import word_tokenize, sent_tokenize # sent_tokenize might not be used directly but good to have
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
                'corpora/omw-1.4' # For WordNetLemmatizer multi-language support
            ]
            
            missing_datasets = []
            for dataset_path in required_datasets:
                try:
                    nltk.data.find(dataset_path)
                    logger.debug(f"Found bundled NLTK dataset: {dataset_path}")
                except LookupError:
                    missing_datasets.append(dataset_path)
                    logger.error(f"Missing bundled NLTK dataset: {dataset_path}") # Keep as error to highlight missing bundled data
            
            if missing_datasets:
                logger.critical(f"Missing required NLTK datasets: {missing_datasets}. "
                              f"Application should include these bundled datasets in 'nltk_data_local'. "
                              f"Falling back to basic processing.")
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
                        # Add other mappings if NLTK supports them and you have the data
                    }
                    
                    nltk_lang = lang_mapping.get(lang, 'english') # Default to English if lang not in map
                    if nltk_lang in stopwords.fileids():
                        self.stop_words_dict[lang] = set(stopwords.words(nltk_lang))
                        logger.debug(f"Loaded stopwords for language: {lang} ({nltk_lang})")
                    else:
                        logger.warning(f"NLTK stopwords not available for language code: {lang} (mapped to '{nltk_lang}'). Using English fallback.")
                        self.stop_words_dict[lang] = set(stopwords.words('english')) # Fallback for this specific language
                        
                except Exception as e: # Catch any exception during stopword loading for a language
                    logger.warning(f"Failed to load NLTK stopwords for language '{lang}': {e}. Using English fallback.")
                    self.stop_words_dict[lang] = set(stopwords.words('english'))
            
            # Set default English stopwords for general validation purposes or if a lang is entirely missing
            self.stop_words = self.stop_words_dict.get('en', set(stopwords.words('english')))
            
            # Mark NLTK as operational
            self.nltk_is_operational = True
            logger.info("NLTK components initialized successfully with bundled data")
            
        except Exception as e: # Catch any broad exception during NLTK initialization
            logger.critical(f"NLTK initialization failed despite bundled data check: {e}", exc_info=True)
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
        # Ensure stop_words_dict is initialized
        if not hasattr(self, 'stop_words_dict') or not self.stop_words_dict:
             # This case should ideally not happen if _initialize_components or _initialize_fallback_components ran
            logger.warning("Stopwords dictionary not initialized, returning empty set for safety.")
            return set()
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
        analysis = QuestionAnalysis(original_question=question) # Initialize analysis dataclass early

        try:
            logger.debug(f"Processing question: {question[:100]}...")
            
            # Check cache first if enabled
            cache_key = None
            if self.enable_caching and self.cache_manager:
                cache_key = self._generate_stable_cache_key(question)
                # Use standardized CacheManager calls
                cached_result = await self._get_cached_analysis(cache_key)
                if cached_result:
                    logger.debug("Returning cached question analysis")
                    # Ensure the cached result is a dict before returning
                    if isinstance(cached_result, dict):
                        return cached_result
                    else:
                        logger.warning(f"Cached result for {cache_key} is not a dict. Re-processing.")
            
            # Step 1: Basic validation
            validation_result = self._validate_input(question)
            if not validation_result['is_valid']:
                analysis.is_valid = False
                analysis.error_message = validation_result['error_message']
                analysis.processing_time = (datetime.now() - start_time).total_seconds()
                return analysis.__dict__
            
            # Step 2: Clean and preprocess
            analysis.cleaned_question = self._clean_text(question)
            analysis.word_count = len(analysis.cleaned_question.split())
            # A more robust sentence count:
            analysis.sentence_count = len(nltk.sent_tokenize(analysis.cleaned_question) if self.nltk_is_operational and NLTK_AVAILABLE else analysis.cleaned_question.split('.'))


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
            
            # Step 7: Final validation based on scores and content
            analysis.is_valid = (
                analysis.confidence_score >= self.min_confidence_threshold and
                analysis.language in self.supported_languages and
                len(analysis.keywords) > 0 # Ensure some keywords were extracted
            )
            
            if not analysis.is_valid and not analysis.error_message: # If still not valid, generate a message
                analysis.error_message = self._generate_validation_message(analysis)
            
            # Add metadata
            analysis.processing_time = (datetime.now() - start_time).total_seconds()
            analysis.metadata = {
                'session_id': session_id,
                'processing_timestamp': datetime.now().isoformat(),
                'nltk_operational_at_init': self.nltk_is_operational, # To know NLTK status at init
                'langdetect_available': LANGDETECT_AVAILABLE,
                'cache_key': cache_key
            }
            
            # Cache result if enabled and valid
            if self.enable_caching and self.cache_manager and analysis.is_valid and cache_key:
                # Use standardized CacheManager calls
                await self._cache_analysis(cache_key, analysis.__dict__)
            
            logger.debug(f"Question processing completed in {analysis.processing_time:.3f}s. Valid: {analysis.is_valid}, Error: {analysis.error_message}")
            return analysis.__dict__
            
        except Exception as e:
            logger.error(f"Unhandled error during question processing: {e}", exc_info=True)
            analysis.is_valid = False
            analysis.error_message = f"Unexpected processing error: {str(e)}"
            analysis.processing_time = (datetime.now() - start_time).total_seconds()
            return analysis.__dict__
    
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
        question_stripped = question.strip() # Use stripped version for length checks
        if len(question_stripped) < self.min_question_length:
            return {
                'is_valid': False,
                'error_message': f'Question too short (minimum {self.min_question_length} characters)'
            }
        
        if len(question_stripped) > self.max_question_length:
            return {
                'is_valid': False,
                'error_message': f'Question too long (maximum {self.max_question_length} characters)'
            }
        
        # Check for suspicious patterns
        if self._contains_suspicious_patterns(question_stripped): # Use stripped version
            return {
                'is_valid': False,
                'error_message': 'Question contains invalid content'
            }
        
        # Check if question has some meaningful content
        # Using default English stopwords for generic validation
        cleaned_q_for_meaning = re.sub(r'[^\w\s]', '', question_stripped.lower())
        # Ensure self.stop_words is initialized (should be by __init__)
        current_stopwords = self.stop_words if hasattr(self, 'stop_words') else set()
        meaningful_words = [w for w in cleaned_q_for_meaning.split() if w not in current_stopwords and len(w) > 2]
        
        if len(meaningful_words) < 1:
            return {
                'is_valid': False,
                'error_message': 'Question lacks meaningful content after removing common words.'
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
        
        text_lower = text.lower() # Already stripped before calling this
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
        
        # Remove or replace special characters but preserve punctuation essential for context
        # This regex keeps letters, numbers, whitespace, and basic punctuation
        cleaned = re.sub(r'[^\w\s\.,;:?!\'"-]', '', cleaned) # Added apostrophe and hyphen
        
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
            logger.warning("Langdetect not available, defaulting to English for language detection.")
            return {'language': 'en', 'confidence': 0.5} # Low confidence if library is missing
        
        try:
            # langdetect works better with longer text; if too short, results can be unreliable
            if len(text.split()) < 3: # Heuristic: need at least 3 words for somewhat reliable detection
                logger.debug("Text too short for reliable language detection by langdetect, defaulting to English.")
                return {'language': 'en', 'confidence': 0.7} # Default for very short text
            
            detected_lang = detect(text)
            
            # Simple confidence estimation based on text length and common word patterns
            # This is a heuristic and not a true confidence score from langdetect
            confidence = min(0.9, 0.5 + (len(text.split()) * 0.05)) # Base confidence increases with length
            
            # Boost confidence for English if it contains common English words, as langdetect can sometimes misclassify
            english_indicators = ['the', 'and', 'or', 'but', 'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are']
            if detected_lang == 'en':
                text_lower = text.lower()
                english_word_count = sum(1 for word in english_indicators if word in text_lower)
                if english_word_count > 1: # If at least two English indicators are present
                    confidence = min(0.95, confidence + (english_word_count * 0.05))
            
            logger.debug(f"Language detected: {detected_lang} with estimated confidence {confidence:.2f}")
            return {
                'language': detected_lang,
                'confidence': confidence
            }
            
        except LangDetectException as lde: # Specific exception from langdetect
            logger.warning(f"Langdetect failed to detect language: {lde}. Defaulting to English.")
            return {'language': 'en', 'confidence': 0.6} # Slightly higher confidence than generic error
        except Exception as e: # Catch any other unexpected error during detection
            logger.error(f"Unexpected error during language detection: {e}", exc_info=True)
            return {'language': 'en', 'confidence': 0.5}
    
    async def _extract_topics_and_keywords(self, text: str, detected_language: str = 'en') -> Dict[str, List[str]]:
        """
        Extract topics, keywords, and entities from text.
        Implements runtime fallback for NLTK operations.
        
        Args:
            text: Cleaned text to analyze
            detected_language: Detected language code for appropriate stopwords
            
        Returns:
            Dictionary with topics, keywords, and entities
        """
        # Get appropriate stopwords for the detected language
        language_stopwords = self._get_stopwords_for_language(detected_language)
        
        keywords = []
        # Entities are extracted later, common to both paths
        
        use_nltk_for_this_run = self.nltk_is_operational and self.lemmatizer and NLTK_AVAILABLE

        if use_nltk_for_this_run:
            try:
                # Attempt NLTK-based keyword extraction
                logger.debug(f"Attempting NLTK keyword extraction for lang '{detected_language}'. Text: {text[:60]}...")
                words = word_tokenize(text.lower())
                pos_tags = pos_tag(words)
                
                important_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}
                
                # Ensure lemmatizer is available (checked by use_nltk_for_this_run)
                keywords = [
                    self.lemmatizer.lemmatize(word) 
                    for word, pos in pos_tags 
                    if pos in important_pos and word not in language_stopwords and len(word) > 2
                ]
                logger.debug(f"Successfully extracted {len(keywords)} keywords using NLTK.")
                
            except Exception as e:
                # Log the NLTK runtime error and set flag to fall back
                logger.error(
                    "NLTK runtime error during keyword/POS tagging: %s. Falling back to basic method for this request.", 
                    e, 
                    exc_info=True
                )
                use_nltk_for_this_run = False # Force fallback for this run
        
        # If NLTK was not operational initially OR failed during runtime
        if not use_nltk_for_this_run:
            logger.debug(f"Using basic keyword extraction for lang '{detected_language}'. Text: {text[:60]}...")
            words = re.findall(r'\b\w+\b', text.lower())
            keywords = [
                word for word in words 
                if word not in language_stopwords and len(word) > 2
            ]
            logger.debug(f"Fallback keyword extraction completed, found {len(keywords)} keywords.")
        
        # Remove duplicates while preserving order, and limit
        keywords = list(dict.fromkeys(keywords))[:20] 
        
        # --- Common logic for topics and entities (after keyword extraction) ---
        topics = []
        text_lower = text.lower() # Use the same lowercased text for consistency
        
        for category, category_keywords in self.institutional_keywords.items():
            # Check if any keyword from the category list is present in the text
            if any(kw in text_lower for kw in category_keywords):
                topics.append(category)
        
        # Add general topics based on question content if no specific institutional topics found
        if not topics:
            if any(word in text_lower for word in ['course', 'class', 'study', 'learn', 'academic', 'degree']):
                topics.append('academic')
            elif any(word in text_lower for word in ['apply', 'admission', 'enroll', 'register', 'application']):
                topics.append('administrative')
            elif any(word in text_lower for word in ['campus', 'location', 'facility', 'building', 'map']):
                topics.append('campus')
            else:
                topics.append('general') # Default topic
        
        # Extract potential entities (simplified)
        entities = []
        # Find capitalized words (potential proper nouns) - improved regex
        capitalized_words = re.findall(r'\b[A-Z][A-Za-z\'.-]*\b', text) # Allow apostrophes, hyphens, dots in names
        entities.extend(capitalized_words[:10]) 
        
        # Find numbers (could be IDs, years, quantities)
        numbers = re.findall(r'\b\d+\b', text)
        entities.extend(numbers[:5])
        
        # Find email patterns
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        entities.extend(emails)
        
        # Remove duplicates from entities
        entities = list(set(entities))
        
        logger.debug(f"Final keywords: {keywords}, Topics: {topics}, Entities: {entities}")
        return {
            'topics': topics,
            'keywords': keywords,
            'entities': entities
        }
    
    def _classify_question_type(self, text: str) -> str:
        """
        Classify the type of question based on patterns.
        
        Args:
            text: Question text to classify
            
        Returns:
            Question type string
        """
        text_lower = text.lower().strip() # Ensure it's stripped and lowercased
        
        # Check each pattern type
        for question_type_key, pattern in self.question_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return question_type_key
        
        # Default classification based on sentence structure or common keywords
        if text_lower.endswith('?'):
            return 'interrogative'
        # More specific keywords for help requests
        elif any(word in text_lower for word in ['help', 'assist', 'support', 'aid', 'guidance']):
            return 'help_request'
        # Keywords for explanation requests
        elif any(word in text_lower for word in ['explain', 'describe', 'tell me about', 'what is', 'what are']):
            if 'what is' in text_lower or 'what are' in text_lower: # Prioritize definition-like "what" questions
                 return 'definition' 
            return 'explanation'
        else:
            return 'general' # Default if no other type matches
    
    def _calculate_complexity_score(self, analysis: QuestionAnalysis) -> float:
        """
        Calculate complexity score based on various factors.
        
        Args:
            analysis: Question analysis object
            
        Returns:
            Complexity score between 0.0 and 1.0
        """
        score = 0.0
        
        # Word count factor (normalized to avoid extreme impact)
        # Max contribution from word count around 50-75 words
        word_factor = min(1.0, analysis.word_count / 75.0 if analysis.word_count > 0 else 0)
        score += word_factor * 0.3  # Weight: 30%
        
        # Sentence count factor - more sentences can imply more complex structure
        sentence_factor = min(1.0, analysis.sentence_count / 5.0 if analysis.sentence_count > 0 else 0)
        score += sentence_factor * 0.2 # Weight: 20%
        
        # Keywords complexity - more distinct, relevant keywords suggest higher complexity
        keyword_factor = min(1.0, len(analysis.keywords) / 10.0 if analysis.keywords else 0)
        score += keyword_factor * 0.2 # Weight: 20%
        
        # Topic diversity - multiple topics can indicate a multifaceted question
        topic_factor = min(1.0, len(analysis.topics) / 3.0 if analysis.topics else 0)
        score += topic_factor * 0.15 # Weight: 15%
        
        # Question type complexity - certain types are inherently more complex
        complex_types = ['comparison', 'procedure', 'definition', 'why'] # Added 'why'
        if analysis.question_type in complex_types:
            score += 0.15 # Weight: 15%
        
        return round(min(1.0, max(0.0, score)), 2) # Ensure score is between 0 and 1, rounded
    
    def _calculate_confidence_score(self, analysis: QuestionAnalysis) -> float:
        """
        Calculate overall confidence score for the analysis.
        
        Args:
            analysis: Question analysis object
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        
        # Language confidence - very important
        score += analysis.language_confidence * 0.4 # Weight: 40%
        
        # Content quality and richness
        if analysis.word_count >= 3: # Basic check for minimal content
            score += 0.15 # Weight: 15%
        if len(analysis.keywords) >= 2: # At least two keywords suggest some specificity
            score += 0.20 # Weight: 20%
        if len(analysis.topics) > 0 and analysis.topics != ['general']: # Specific topics found
            score += 0.15 # Weight: 15%
        
        # Question structure and clarity (indirectly via question_type)
        if analysis.question_type != 'general' and analysis.question_type != 'interrogative': # More specific types
            score += 0.10 # Weight: 10%
        
        # Penalize for very short or very long questions if they weren't caught by min/max length
        # This is a soft penalty if it passes initial validation but is still at extremes
        if analysis.word_count < 2 or analysis.word_count > 150: # Adjusted upper limit
            score -= 0.05 # Small penalty
        
        return round(min(1.0, max(0.0, score)), 2) # Ensure score is between 0 and 1, rounded
    
    def _generate_validation_message(self, analysis: QuestionAnalysis) -> str:
        """
        Generate helpful validation message for invalid questions.
        
        Args:
            analysis: Question analysis object
            
        Returns:
            Validation message string
        """
        if analysis.language not in self.supported_languages:
            # Provide a more generic message about supported languages if possible
            supported_lang_str = ", ".join(self.supported_languages)
            return (f"The detected language '{analysis.language}' is not currently supported. "
                    f"Please try asking your question in one of the following languages: {supported_lang_str}.")
        
        if len(analysis.keywords) == 0: # If no keywords were extracted at all
            return "Your question appears to lack specific keywords. Please try rephrasing or adding more details."
        
        # If confidence is low despite passing initial checks
        if analysis.confidence_score < self.min_confidence_threshold:
            if analysis.language_confidence < 0.6: # If language detection itself was very uncertain
                return ("I'm having trouble understanding the language of your question or its clarity. "
                        "Could you please rephrase it or ensure it's in a supported language?")
            return "I'm having difficulty fully understanding your question. Could you please try rephrasing it more clearly or with more specific terms?"
        
        # Generic fallback if other conditions didn't provide a specific message
        return "Please provide a clearer, more specific question for a better response."

    async def _get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result using the generic CacheManager.get method."""
        try:
            if self.cache_manager and hasattr(self.cache_manager, 'get'):
                # logger.debug(f"Attempting to retrieve from cache with key: {cache_key}")
                return await self.cache_manager.get(cache_key)
            # logger.debug(f"Cache manager or 'get' method not available for key: {cache_key}")
        except Exception as e:
            logger.error(f"Error retrieving cached analysis for key {cache_key}: {e}", exc_info=True)
        return None
    
    async def _cache_analysis(self, cache_key: str, analysis: Dict[str, Any]) -> None:
        """Cache analysis result using the generic CacheManager.set method."""
        try:
            if self.cache_manager and hasattr(self.cache_manager, 'set'):
                category = 'question_analysis'
                # Define a specific TTL for question_analysis if needed, otherwise CacheManager default applies
                # ttl_question_analysis = self.config.get('cache_ttl_question_analysis', 1800) # e.g., 30 minutes
                # logger.debug(f"Attempting to cache analysis for key: {cache_key} with category: {category}")
                await self.cache_manager.set(
                    key=cache_key, 
                    value=analysis, 
                    category=category
                    # ttl=ttl_question_analysis # Uncomment to use a specific TTL for this category
                )
            # else:
                # logger.debug(f"Cache manager or 'set' method not available for key: {cache_key}")
        except Exception as e:
            logger.error(f"Error caching analysis for key {cache_key}: {e}", exc_info=True)
    
    def is_healthy(self) -> bool:
        """
        Check if the question processor is healthy and ready to process questions.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Test basic functionality: input validation should always work
            test_question = "This is a valid test question for health check."
            validation_result = self._validate_input(test_question)
            if not validation_result['is_valid']:
                logger.error(f"Health check: Basic validation failed. Message: {validation_result.get('error_message')}")
                return False
            
            # Optionally, check if NLTK (if supposed to be operational) can perform a very basic task
            # This is tricky because NLTK itself might be the source of issues in some environments.
            # For now, basic validation is a good indicator.
            # if self.nltk_is_operational:
            #     nltk.word_tokenize("test") # Simple check, might still fail if punkt is problematic

            return True
        except Exception as e:
            logger.error(f"QuestionProcessor health check failed with exception: {e}", exc_info=True)
            return False
    
    def cleanup(self) -> None:
        """Cleanup resources used by the question processor."""
        # Currently, no specific resources to clean up in QuestionProcessor itself
        # NLTK data is loaded globally, CacheManager has its own cleanup
        logger.info("Question Processor cleanup called (currently no specific actions).")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns:
            Dictionary with processor statistics
        """
        return {
            'nltk_operational_at_init': self.nltk_is_operational, # Status after initialization
            'nltk_package_available': NLTK_AVAILABLE, # If NLTK module could be imported
            'langdetect_package_available': LANGDETECT_AVAILABLE, # If langdetect could be imported
            'supported_languages': self.supported_languages,
            'min_confidence_threshold': self.min_confidence_threshold,
            'enable_caching': self.enable_caching,
            'configured_stopwords_languages': list(self.stop_words_dict.keys()) if hasattr(self, 'stop_words_dict') else []
        }