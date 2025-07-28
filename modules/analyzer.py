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
from typing import Dict, List, Tuple, Optional, Set
import logging

# Third-party imports with fallbacks
try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.stem.isri import ISRIStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detects the primary language of text content.
    
    Supports English and Arabic with both library-based detection
    (langdetect) and fallback heuristic analysis.
    """
    
    def __init__(self):
        """Initialize the language detector."""
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        self.english_pattern = re.compile(r'[a-zA-Z]')
        
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the given text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            str: Language code ('en' for English, 'ar' for Arabic)
        """
        if not text or not text.strip():
            return 'en'  # Default to English for empty text
            
        # Try langdetect first if available
        if LANGDETECT_AVAILABLE:
            try:
                detected = detect(text)
                if detected in ['en', 'ar']:
                    return detected
                # If detected language is neither English nor Arabic, fall back to heuristic
            except LangDetectError:
                pass
        
        # Fallback heuristic method
        return self._heuristic_detection(text)
    
    def _heuristic_detection(self, text: str) -> str:
        """
        Heuristic language detection based on character analysis.
        
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
        
        # Return the language with higher character ratio
        arabic_ratio = arabic_chars / total_chars
        return 'ar' if arabic_ratio > 0.5 else 'en'


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
        
    def _initialize_nlp_tools(self):
        """Initialize NLTK tools if available."""
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.arabic_stemmer = ISRIStemmer()
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK tools: {e}")
                
    def _load_stopwords(self):
        """Load stopwords for English and Arabic."""
        if NLTK_AVAILABLE:
            try:
                self.stopwords_en = set(stopwords.words('english'))
                self.stopwords_ar = set(stopwords.words('arabic'))
            except Exception as e:
                logger.warning(f"Failed to load NLTK stopwords: {e}")
        
        # Fallback stopwords if NLTK is not available
        if not self.stopwords_en:
            self.stopwords_en = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
            }
            
        if not self.stopwords_ar:
            self.stopwords_ar = {
                'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
                'التي', 'الذي', 'التي', 'اللذان', 'اللتان', 'اللذين', 'اللتين',
                'هو', 'هي', 'هم', 'هن', 'أن', 'إن', 'كان', 'كانت', 'يكون', 'تكون'
            }
    
    def normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text by removing diacritics and standardizing characters.
        
        Args:
            text (str): Arabic text to normalize
            
        Returns:
            str: Normalized Arabic text
        """
        # Remove diacritics (tashkeel)
        text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
        
        # Normalize alef variations
        text = re.sub(r'[أإآ]', 'ا', text)
        
        # Normalize yeh variations
        text = re.sub(r'[يى]', 'ي', text)
        
        # Normalize teh marbuta
        text = re.sub(r'ة', 'ه', text)
        
        return text
    
    def tokenize(self, text: str, language: str) -> List[str]:
        """
        Tokenize text based on language.
        
        Args:
            text (str): Text to tokenize
            language (str): Language code ('en' or 'ar')
            
        Returns:
            List[str]: List of tokens
        """
        if not text:
            return []
            
        # Normalize Arabic text if needed
        if language == 'ar':
            text = self.normalize_arabic(text)
        
        # Use NLTK tokenizer if available
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text, language='arabic' if language == 'ar' else 'english')
                return [token.lower() for token in tokens if token.isalnum()]
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}")
        
        # Fallback tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def remove_stopwords(self, tokens: List[str], language: str) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens (List[str]): List of tokens
            language (str): Language code ('en' or 'ar')
            
        Returns:
            List[str]: Filtered tokens
        """
        stopwords_set = self.stopwords_ar if language == 'ar' else self.stopwords_en
        return [token for token in tokens if token not in stopwords_set]
    
    def stem_and_lemmatize(self, tokens: List[str], language: str) -> List[str]:
        """
        Apply stemming and lemmatization to tokens.
        
        Args:
            tokens (List[str]): List of tokens
            language (str): Language code ('en' or 'ar')
            
        Returns:
            List[str]: Processed tokens
        """
        processed_tokens = []
        
        for token in tokens:
            if language == 'en' and self.lemmatizer:
                try:
                    processed_token = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(processed_token)
                except Exception:
                    processed_tokens.append(token)
            elif language == 'ar' and self.arabic_stemmer:
                try:
                    processed_token = self.arabic_stemmer.stem(token)
                    processed_tokens.append(processed_token)
                except Exception:
                    processed_tokens.append(token)
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def extract_keywords(self, tokens: List[str], top_k: int = 10) -> List[Tuple[str, int]]:
        """
        Extract top keywords from processed tokens.
        
        Args:
            tokens (List[str]): Processed tokens
            top_k (int): Number of top keywords to return
            
        Returns:
            List[Tuple[str, int]]: List of (keyword, frequency) tuples
        """
        if not tokens:
            return []
        
        # Filter out very short tokens
        filtered_tokens = [token for token in tokens if len(token) > 2]
        
        # Count frequency
        counter = Counter(filtered_tokens)
        
        return counter.most_common(top_k)
    
    def process_text(self, text: str, language: str) -> Dict:
        """
        Complete text processing pipeline.
        
        Args:
            text (str): Raw text to process
            language (str): Language code ('en' or 'ar')
            
        Returns:
            Dict: Processed text data including tokens, keywords, etc.
        """
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
            'unique_tokens': len(set(processed_tokens))
        }


class AdvancedContentAnalyzer:
    """
    High-level content analyzer that orchestrates the entire NLP pipeline.
    
    This is the main class that scraper scripts will interact with.
    """
    
    def __init__(self):
        """Initialize the content analyzer."""
        self.language_detector = LanguageDetector()
        self.text_processor = AdvancedTextProcessor()
        
        # Semantic categorization concepts
        self.category_concepts = {
            'academic': {
                'en': ['research', 'study', 'university', 'education', 'academic', 'scholar',
                       'journal', 'publication', 'thesis', 'dissertation', 'course', 'lecture'],
                'ar': ['بحث', 'دراسة', 'جامعة', 'تعليم', 'أكاديمي', 'عالم', 'مجلة',
                       'نشر', 'رسالة', 'محاضرة', 'دورة']
            },
            'financial': {
                'en': ['money', 'finance', 'investment', 'bank', 'economy', 'market',
                       'stock', 'profit', 'revenue', 'budget', 'cost', 'price'],
                'ar': ['مال', 'تمويل', 'استثمار', 'بنك', 'اقتصاد', 'سوق',
                       'سهم', 'ربح', 'إيراد', 'ميزانية', 'تكلفة', 'سعر']
            },
            'technology': {
                'en': ['technology', 'computer', 'software', 'digital', 'internet',
                       'data', 'artificial', 'intelligence', 'machine', 'learning'],
                'ar': ['تكنولوجيا', 'حاسوب', 'برمجيات', 'رقمي', 'إنترنت',
                       'بيانات', 'ذكي', 'اصطناعي', 'آلة', 'تعلم']
            },
            'health': {
                'en': ['health', 'medical', 'medicine', 'doctor', 'patient',
                       'treatment', 'disease', 'hospital', 'clinic', 'therapy'],
                'ar': ['صحة', 'طبي', 'طب', 'طبيب', 'مريض',
                       'علاج', 'مرض', 'مستشفى', 'عيادة', 'علاج']
            },
            'business': {
                'en': ['business', 'company', 'corporate', 'management', 'strategy',
                       'marketing', 'sales', 'customer', 'service', 'product'],
                'ar': ['أعمال', 'شركة', 'إدارة', 'استراتيجية', 'تسويق',
                       'مبيعات', 'عميل', 'خدمة', 'منتج']
            }
        }
    
    def categorize_content(self, processed_tokens: List[str], language: str) -> Dict[str, float]:
        """
        Categorize content based on semantic analysis.
        
        Args:
            processed_tokens (List[str]): Processed text tokens
            language (str): Language code
            
        Returns:
            Dict[str, float]: Category scores
        """
        if not processed_tokens:
            return {}
        
        category_scores = {}
        token_set = set(processed_tokens)
        total_tokens = len(processed_tokens)
        
        for category, concepts in self.category_concepts.items():
            lang_concepts = concepts.get(language, [])
            if not lang_concepts:
                continue
                
            # Count matches
            matches = sum(1 for concept in lang_concepts if concept in token_set)
            
            # Calculate score as percentage
            score = (matches / len(lang_concepts)) * 100 if lang_concepts else 0
            category_scores[category] = round(score, 2)
        
        return category_scores
    
    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """
        Generate a concise summary of the content.
        
        Args:
            text (str): Original text
            max_sentences (int): Maximum number of sentences in summary
            
        Returns:
            str: Generated summary
        """
        if not text:
            return ""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text[:500] + "..." if len(text) > 500 else text
        
        # Simple extractive summarization - take first few sentences
        summary_sentences = sentences[:max_sentences]
        summary = '. '.join(summary_sentences)
        
        # Ensure summary doesn't exceed reasonable length
        if len(summary) > 500:
            summary = summary[:497] + "..."
        
        return summary
    
    def calculate_readability_metrics(self, text: str) -> Dict[str, int]:
        """
        Calculate basic readability metrics.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, int]: Readability metrics
        """
        if not text:
            return {'word_count': 0, 'sentence_count': 0, 'avg_words_per_sentence': 0}
        
        # Count words
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Average words per sentence
        avg_words = round(word_count / sentence_count) if sentence_count > 0 else 0
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': avg_words
        }
    
    def analyze_content(self, raw_text: str, title: str = "", metadata: Optional[Dict] = None) -> Dict:
        """
        Main analysis method - performs complete content analysis.
        
        Args:
            raw_text (str): Raw text content to analyze
            title (str): Title of the content (optional)
            metadata (Dict): Additional metadata (optional)
            
        Returns:
            Dict: Complete analysis results in structured format
        """
        if not raw_text:
            return self._empty_analysis_result()
        
        # Combine title and text for analysis
        full_text = f"{title} {raw_text}" if title else raw_text
        
        # Step 1: Language detection
        language = self.language_detector.detect_language(full_text)
        
        # Step 2: Text processing
        processing_result = self.text_processor.process_text(full_text, language)
        
        # Step 3: Semantic categorization
        categories = self.categorize_content(processing_result['processed_tokens'], language)
        
        # Step 4: Summary generation
        summary = self.generate_summary(raw_text)
        
        # Step 5: Readability metrics
        readability = self.calculate_readability_metrics(raw_text)
        
        # Step 6: Determine primary category
        primary_category = max(categories.items(), key=lambda x: x[1])[0] if categories else 'general'
        
        # Assemble final result
        analysis_result = {
            'title': title,
            'summary': summary,
            'language': language,
            'primary_category': primary_category,
            'categories': categories,
            'keywords': [keyword for keyword, freq in processing_result['keywords']],
            'keyword_frequencies': dict(processing_result['keywords']),
            'readability_metrics': readability,
            'processing_stats': {
                'total_tokens': processing_result['token_count'],
                'unique_tokens': processing_result['unique_tokens'],
                'processed_tokens_count': len(processing_result['processed_tokens'])
            },
            'metadata': metadata or {}
        }
        
        return analysis_result
    
    def _empty_analysis_result(self) -> Dict:
        """Return empty analysis result for invalid input."""
        return {
            'title': '',
            'summary': '',
            'language': 'en',
            'primary_category': 'general',
            'categories': {},
            'keywords': [],
            'keyword_frequencies': {},
            'readability_metrics': {'word_count': 0, 'sentence_count': 0, 'avg_words_per_sentence': 0},
            'processing_stats': {'total_tokens': 0, 'unique_tokens': 0, 'processed_tokens_count': 0},
            'metadata': {}
        }


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
    """
    
    # Test with Arabic text
    arabic_text = """
    الذكاء الاصطناعي يثور في طريقة تعاملنا مع المشاكل المعقدة في التكنولوجيا والأعمال.
    خوارزميات التعلم الآلي تستخدم لتحليل كميات كبيرة من البيانات وتوفير رؤى
    كانت مستحيلة الحصول عليها سابقاً. هذه التكنولوجيا لها تطبيقات في الصحة والمالية والتعليم.
    """
    
    # Analyze both texts
    english_result = analyzer.analyze_content(english_text, "AI Revolution")
    arabic_result = analyzer.analyze_content(arabic_text, "ثورة الذكاء الاصطناعي")
    
    print("English Analysis:")
    print(f"Language: {english_result['language']}")
    print(f"Primary Category: {english_result['primary_category']}")
    print(f"Keywords: {english_result['keywords'][:5]}")
    print(f"Summary: {english_result['summary'][:100]}...")
    print()
    
    print("Arabic Analysis:")
    print(f"Language: {arabic_result['language']}")
    print(f"Primary Category: {arabic_result['primary_category']}")
    print(f"Keywords: {arabic_result['keywords'][:5]}")
    print(f"Summary: {arabic_result['summary'][:100]}...")
