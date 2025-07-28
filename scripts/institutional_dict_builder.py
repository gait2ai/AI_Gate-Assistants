"""
AI Gate for Artificial Intelligence Applications
Enhanced Institutional Dictionary Builder - institutional_dict_builder.py

This script processes the knowledge base (knowledge_base.json) to generate
a categorized keyword dictionary (institutional_keywords.yaml) used for
enhancing topic analysis within the AI Gate system.

Features:
- Multi-language support (English and Arabic)
- Configurable stopword filtering
- Frequency-based keyword filtering
- Robust text processing and cleaning
- Comprehensive error handling and logging
- Performance optimizations
"""

import json
import yaml
import logging
import os
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional, Union, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import argparse
from functools import lru_cache
import time


@dataclass
class DictionaryConfig:
    """Configuration class for the dictionary builder."""
    min_keyword_frequency: int = 1
    exclude_categories: List[str] = field(default_factory=lambda: ['general'])
    sort_keywords: bool = True
    custom_stopwords: List[str] = field(default_factory=list)
    extract_from_title_summary: bool = True
    lemmatization: bool = False
    stopwords_path: str = './nltk_data_local/corpora/stopwords/'
    min_keyword_length: int = 2
    max_keyword_length: int = 50
    arabic_detection_threshold: float = 0.3
    case_sensitive_arabic: bool = True


class ProcessingStats:
    """Statistics tracking for processing operations."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.total_pages = 0
        self.processed_pages = 0
        self.skipped_pages = 0
        self.arabic_pages = 0
        self.english_pages = 0
        self.total_keywords = 0
        self.filtered_keywords = 0
        self.categories = 0
        self.start_time = None
        self.end_time = None
        self.errors = []
    
    def start_timing(self):
        """Start timing the processing operation."""
        self.start_time = time.time()
    
    def end_timing(self):
        """End timing the processing operation."""
        self.end_time = time.time()
    
    @property
    def processing_time(self) -> Optional[float]:
        """Get processing time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def add_error(self, error: str):
        """Add an error to the statistics."""
        self.errors.append(error)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of processing statistics."""
        return {
            'total_pages': self.total_pages,
            'processed_pages': self.processed_pages,
            'skipped_pages': self.skipped_pages,
            'arabic_pages': self.arabic_pages,
            'english_pages': self.english_pages,
            'total_keywords': self.total_keywords,
            'filtered_keywords': self.filtered_keywords,
            'categories': self.categories,
            'processing_time': self.processing_time,
            'errors_count': len(self.errors)
        }


class LanguageDetector:
    """Optimized language detection for Arabic and English text."""
    
    # Arabic Unicode ranges
    ARABIC_RANGES = [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ]
    
    def __init__(self, threshold: float = 0.3):
        """
        Initialize language detector.
        
        Args:
            threshold: Minimum ratio of Arabic characters to consider text as Arabic
        """
        self.threshold = threshold
        self._arabic_pattern = self._build_arabic_pattern()
        self._word_pattern = re.compile(r'[^\s\d\W]', re.UNICODE)
    
    def _build_arabic_pattern(self) -> re.Pattern:
        """Build optimized regex pattern for Arabic character detection."""
        ranges = ''.join([f'\\u{start:04X}-\\u{end:04X}' for start, end in self.ARABIC_RANGES])
        return re.compile(f'[{ranges}]')
    
    @lru_cache(maxsize=1000)
    def is_arabic(self, text: str) -> bool:
        """
        Detect if text contains Arabic characters above threshold.
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if text appears to contain Arabic content
        """
        if not text or len(text.strip()) < 2:
            return False
        
        arabic_chars = len(self._arabic_pattern.findall(text))
        total_chars = len(self._word_pattern.findall(text))
        
        return total_chars > 0 and (arabic_chars / total_chars) >= self.threshold


class StopwordManager:
    """Manages stopwords for multiple languages with caching and fallback support."""
    
    FALLBACK_STOPWORDS = {
        'english': {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'would', 'you', 'your',
            'this', 'these', 'they', 'we', 'our', 'us', 'can', 'could',
            'should', 'may', 'might', 'must', 'shall', 'would', 'about',
            'after', 'all', 'also', 'but', 'do', 'have', 'how', 'if', 'no',
            'not', 'or', 'so', 'some', 'such', 'than', 'then', 'very', 'what',
            'when', 'where', 'who', 'why', 'before', 'during', 'each', 'few',
            'more', 'most', 'other', 'since', 'too', 'under', 'until', 'up'
        },
        'arabic': {
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
            'التي', 'الذي', 'كان', 'كانت', 'يكون', 'تكون', 'هو', 'هي', 'هم', 'هن',
            'أن', 'إن', 'كل', 'بعض', 'غير', 'سوف', 'قد', 'لقد', 'كما', 'حيث',
            'بين', 'عند', 'لدى', 'أمام', 'خلال', 'بعد', 'قبل', 'عبر', 'ضد',
            'نحو', 'لكن', 'لكن', 'أو', 'أم', 'بل', 'لا', 'ما', 'لم', 'لن',
            'كيف', 'أين', 'متى', 'ماذا', 'لماذا', 'أي', 'كم', 'أكثر', 'أقل'
        }
    }
    
    def __init__(self, stopwords_path: str, custom_stopwords: List[str] = None):
        """
        Initialize stopword manager.
        
        Args:
            stopwords_path: Path to stopwords directory  
            custom_stopwords: Additional custom stopwords
        """
        self.stopwords_path = Path(stopwords_path)
        self.custom_stopwords = set(custom_stopwords or [])
        self._cache = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load all stopwords at initialization
        self.english_stopwords = self._load_stopwords('english')
        self.arabic_stopwords = self._load_stopwords('arabic')
        self.combined_stopwords = self.english_stopwords | self.arabic_stopwords
    
    def _load_stopwords(self, language: str) -> Set[str]:
        """
        Load stopwords for a specific language with fallback support.
        
        Args:
            language: Language code ('english' or 'arabic')
            
        Returns:
            Set of stopwords for the specified language
        """
        if language in self._cache:
            return self._cache[language]
        
        stopwords = set()
        stopwords_file = self.stopwords_path / language
        
        # Try to load from file
        if stopwords_file.exists():
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    stopwords = {line.strip().lower() for line in f if line.strip()}
                self.logger.info(f"Loaded {len(stopwords)} {language} stopwords from file")
            except Exception as e:
                self.logger.warning(f"Failed to load {language} stopwords from file: {e}")
        
        # Use fallback if file loading failed or no stopwords found
        if not stopwords and language in self.FALLBACK_STOPWORDS:
            stopwords = self.FALLBACK_STOPWORDS[language].copy()
            self.logger.info(f"Using {len(stopwords)} fallback {language} stopwords")
        
        # Add custom stopwords
        if self.custom_stopwords:
            stopwords.update(self.custom_stopwords)
        
        # Cache and return
        self._cache[language] = stopwords
        return stopwords
    
    def get_stopwords(self, language: str = None) -> Set[str]:
        """
        Get stopwords for a specific language or combined set.
        
        Args:
            language: Specific language ('english', 'arabic') or None for combined
            
        Returns:
            Set of stopwords
        """
        if language == 'english':
            return self.english_stopwords
        elif language == 'arabic':
            return self.arabic_stopwords
        else:
            return self.combined_stopwords


class TextProcessor:
    """Advanced text processing with multi-language support and optimization."""
    
    def __init__(self, config: DictionaryConfig, language_detector: LanguageDetector, 
                 stopword_manager: StopwordManager):
        """
        Initialize text processor.
        
        Args:
            config: Configuration object
            language_detector: Language detection utility
            stopword_manager: Stopword management utility
        """
        self.config = config
        self.language_detector = language_detector
        self.stopword_manager = stopword_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Precompile regex patterns for performance
        self._english_word_pattern = re.compile(r'\b[a-zA-Z]{2,}\b')
        self._arabic_word_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]{2,}')
        self._mixed_word_pattern = re.compile(r'\b[a-zA-Z\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]{2,}\b')
        self._english_only_pattern = re.compile(r'^[a-zA-Z]+$')
    
    @lru_cache(maxsize=5000)
    def clean_and_tokenize(self, text: str) -> tuple[List[str], bool]:
        """
        Clean and tokenize text into meaningful keywords with language detection.
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple of (keywords list, is_arabic boolean)
        """
        if not text or not text.strip():
            return [], False
        
        # Detect language
        is_arabic = self.language_detector.is_arabic(text)
        
        # Extract words based on language
        if is_arabic:
            # For Arabic text, extract both Arabic and English words
            words = self._mixed_word_pattern.findall(text)
        else:
            # For English text, normalize case and extract words
            text = text.lower()
            words = self._english_word_pattern.findall(text)
        
        # Filter and clean words
        keywords = self._filter_words(words, is_arabic)
        
        return keywords, is_arabic
    
    def _filter_words(self, words: List[str], is_arabic: bool) -> List[str]:
        """
        Filter words based on stopwords and length constraints.
        
        Args:
            words: List of extracted words
            is_arabic: Whether the text is primarily Arabic
            
        Returns:
            List of filtered keywords
        """
        # Choose appropriate stopwords
        if is_arabic:
            # Check if text has mixed languages
            has_english = any(self._english_only_pattern.match(word) for word in words)
            stopwords = (self.stopword_manager.combined_stopwords if has_english 
                        else self.stopword_manager.arabic_stopwords)
        else:
            stopwords = self.stopword_manager.english_stopwords
        
        keywords = []
        for word in words:
            # Length constraints
            if not (self.config.min_keyword_length <= len(word) <= self.config.max_keyword_length):
                continue
            
            # Stopword filtering
            check_word = word.lower() if self._english_only_pattern.match(word) else word
            if check_word in stopwords:
                continue
            
            # Case normalization
            if self._english_only_pattern.match(word):
                keywords.append(word.lower())
            elif self.config.case_sensitive_arabic:
                keywords.append(word)
            else:
                keywords.append(word.lower())
        
        return keywords
    
    def extract_page_keywords(self, page: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from a page object with comprehensive text processing.
        
        Args:
            page: Page object from knowledge base
            
        Returns:
            List of unique keywords
        """
        all_keywords = []
        
        # Process existing keywords field
        if 'keywords' in page and isinstance(page['keywords'], list):
            for keyword in page['keywords']:
                if keyword and isinstance(keyword, str):
                    cleaned_keywords, _ = self.clean_and_tokenize(keyword.strip())
                    all_keywords.extend(cleaned_keywords)
        
        # Optionally extract from title and summary
        if self.config.extract_from_title_summary:
            for field in ['title', 'summary', 'description']:
                if field in page and page[field]:
                    cleaned_keywords, _ = self.clean_and_tokenize(str(page[field]))
                    all_keywords.extend(cleaned_keywords)
        
        # Remove duplicates while preserving order
        return self._deduplicate_keywords(all_keywords)
    
    @staticmethod
    def _deduplicate_keywords(keywords: List[str]) -> List[str]:
        """Remove duplicate keywords while preserving order."""
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword and keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        return unique_keywords


class InstitutionalDictionaryBuilder:
    """
    Enhanced institutional dictionary builder with professional architecture.
    
    Processes knowledge base entries to generate categorized keyword dictionaries
    with multi-language support, robust error handling, and performance optimization.
    """
    
    def __init__(self, config: Optional[Union[str, DictionaryConfig]] = None):
        """
        Initialize the dictionary builder.
        
        Args:
            config: Configuration file path or DictionaryConfig object
        """
        self.config = self._load_configuration(config)
        self.stats = ProcessingStats()
        
        # Initialize components
        self.language_detector = LanguageDetector(self.config.arabic_detection_threshold)
        self.stopword_manager = StopwordManager(
            self.config.stopwords_path, 
            self.config.custom_stopwords
        )
        self.text_processor = TextProcessor(
            self.config, 
            self.language_detector, 
            self.stopword_manager
        )
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._log_initialization()
    
    def _load_configuration(self, config: Optional[Union[str, DictionaryConfig]]) -> DictionaryConfig:
        """
        Load and validate configuration.
        
        Args:
            config: Configuration file path, DictionaryConfig object, or None
            
        Returns:
            Validated DictionaryConfig object
        """
        if isinstance(config, DictionaryConfig):
            return config
        
        # Default configuration
        default_config = DictionaryConfig()
        
        if isinstance(config, str) and Path(config).exists():
            try:
                with open(config, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                
                # Update default config with user values
                for key, value in user_config.items():
                    if hasattr(default_config, key):
                        setattr(default_config, key, value)
                
                logging.info(f"Loaded configuration from {config}")
                return default_config
                
            except Exception as e:
                logging.warning(f"Failed to load config from {config}: {e}")
        
        logging.info("Using default configuration")
        return default_config
    
    def _log_initialization(self):
        """Log initialization information."""
        self.logger.info("Institutional Dictionary Builder initialized")
        self.logger.info(f"English stopwords: {len(self.stopword_manager.english_stopwords)}")
        self.logger.info(f"Arabic stopwords: {len(self.stopword_manager.arabic_stopwords)}")
        self.logger.info(f"Configuration: min_frequency={self.config.min_keyword_frequency}, "
                        f"excluded_categories={self.config.exclude_categories}")
    
    @contextmanager
    def _error_handling(self, operation: str):
        """Context manager for consistent error handling."""
        try:
            yield
        except FileNotFoundError as e:
            error_msg = f"{operation} failed - File not found: {e}"
            self.logger.error(error_msg)
            self.stats.add_error(error_msg)
            raise
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            error_msg = f"{operation} failed - Invalid file format: {e}"
            self.logger.error(error_msg)
            self.stats.add_error(error_msg)
            raise
        except Exception as e:
            error_msg = f"{operation} failed - Unexpected error: {e}"
            self.logger.error(error_msg)
            self.stats.add_error(error_msg)
            raise
    
    def load_knowledge_base(self, knowledge_base_path: str) -> List[Dict[str, Any]]:
        """
        Load knowledge base from JSON file with robust error handling.
        
        Args:
            knowledge_base_path: Path to knowledge_base.json file
            
        Returns:
            List of knowledge base entries
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            ValueError: If file format is invalid
        """
        with self._error_handling("Knowledge base loading"):
            knowledge_base_path = Path(knowledge_base_path)
            
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                # Try different possible keys
                for key in ['entries', 'pages', 'documents', 'items', 'data']:
                    if key in data:
                        entries = data[key]
                        break
                else:
                    raise ValueError(f"Unknown knowledge base format. Expected list or dict with "
                                   f"'entries'/'pages'/'documents' key")
            else:
                raise ValueError("Invalid knowledge base format - must be list or dict")
            
            if not isinstance(entries, list):
                raise ValueError("Knowledge base entries must be a list")
            
            self.stats.total_pages = len(entries)
            self.logger.info(f"Loaded {len(entries)} entries from {knowledge_base_path}")
            
            return entries
    
    def process_knowledge_base(self, entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Process knowledge base entries to build category-keyword dictionary.
        
        Args:
            entries: List of knowledge base entries
            
        Returns:
            Dictionary mapping categories to keyword lists
        """
        self.logger.info("Starting knowledge base processing")
        self.stats.start_timing()
        
        category_counters = defaultdict(Counter)
        
        for entry in entries:
            try:
                self._process_single_entry(entry, category_counters)
            except Exception as e:
                self.logger.warning(f"Error processing entry {entry.get('title', 'Unknown')}: {e}")
                self.stats.skipped_pages += 1
                continue
        
        # Build final filtered dictionary
        dictionary = self._build_filtered_dictionary(category_counters)
        
        self.stats.end_timing()
        self._log_processing_summary(dictionary)
        
        return dictionary
    
    def _process_single_entry(self, entry: Dict[str, Any], 
                            category_counters: Dict[str, Counter]):
        """
        Process a single knowledge base entry.
        
        Args:
            entry: Single knowledge base entry
            category_counters: Dictionary of category keyword counters
        """
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid entry format: {type(entry)}")
        
        # Extract and validate category
        category = str(entry.get('category', '')).strip().lower()
        if not category:
            raise ValueError("Entry missing category field")
        
        # Skip excluded categories
        if category in self.config.exclude_categories:
            self.stats.skipped_pages += 1
            return
        
        # Extract keywords
        keywords = self.text_processor.extract_page_keywords(entry)
        if not keywords:
            self.stats.skipped_pages += 1
            return
        
        # Update counters
        for keyword in keywords:
            category_counters[category][keyword] += 1
        
        # Update statistics
        self.stats.processed_pages += 1
        
        # Track language statistics
        entry_text = f"{entry.get('title', '')} {entry.get('summary', '')}"
        if self.language_detector.is_arabic(entry_text):
            self.stats.arabic_pages += 1
        else:
            self.stats.english_pages += 1
    
    def _build_filtered_dictionary(self, category_counters: Dict[str, Counter]) -> Dict[str, List[str]]:
        """
        Build final dictionary with frequency filtering.
        
        Args:
            category_counters: Dictionary of category keyword counters
            
        Returns:
            Filtered category-keyword dictionary
        """
        dictionary = {}
        min_freq = self.config.min_keyword_frequency
        
        for category, counter in category_counters.items():
            # Filter by frequency
            filtered_keywords = [
                keyword for keyword, count in counter.items()
                if count >= min_freq
            ]
            
            # Sort if configured
            if self.config.sort_keywords:
                filtered_keywords.sort(key=self._sort_key)
            
            if filtered_keywords:  # Only include non-empty categories
                dictionary[category] = filtered_keywords
                self.stats.total_keywords += len(counter)
                self.stats.filtered_keywords += len(filtered_keywords)
        
        self.stats.categories = len(dictionary)
        return dictionary
    
    def _sort_key(self, keyword: str) -> str:
        """Generate sort key for keyword with Unicode support."""
        return keyword.lower() if re.match(r'^[a-zA-Z]+$', keyword) else keyword
    
    def _log_processing_summary(self, dictionary: Dict[str, List[str]]):
        """Log comprehensive processing summary."""
        stats = self.stats.get_summary()
        
        self.logger.info("=" * 60)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total entries: {stats['total_pages']}")
        self.logger.info(f"Processed entries: {stats['processed_pages']}")
        self.logger.info(f"Skipped entries: {stats['skipped_pages']}")
        self.logger.info(f"Arabic entries: {stats['arabic_pages']}")
        self.logger.info(f"English entries: {stats['english_pages']}")
        self.logger.info(f"Total keywords found: {stats['total_keywords']}")
        self.logger.info(f"Keywords after filtering: {stats['filtered_keywords']}")
        self.logger.info(f"Categories generated: {stats['categories']}")
        
        if stats['processing_time']:
            self.logger.info(f"Processing time: {stats['processing_time']:.2f} seconds")
        
        if stats['errors_count'] > 0:
            self.logger.warning(f"Errors encountered: {stats['errors_count']}")
        
        # Log category details
        self.logger.info("\nCATEGORY BREAKDOWN:")
        for category, keywords in sorted(dictionary.items()):
            arabic_count = sum(1 for kw in keywords if self.language_detector.is_arabic(kw))
            english_count = len(keywords) - arabic_count
            self.logger.info(f"  {category}: {len(keywords)} keywords "
                           f"({english_count} English, {arabic_count} Arabic)")
    
    def save_dictionary(self, dictionary: Dict[str, List[str]], output_path: str):
        """
        Save dictionary to YAML file with proper formatting.
        
        Args:
            dictionary: Category-keyword dictionary
            output_path: Path to output YAML file
        """
        with self._error_handling("Dictionary saving"):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare metadata
            metadata = {
                'metadata': {
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                    'total_categories': len(dictionary),
                    'total_keywords': sum(len(keywords) for keywords in dictionary.values()),
                    'min_frequency': self.config.min_keyword_frequency,
                    'excluded_categories': self.config.exclude_categories
                },
                'categories': dictionary
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    metadata,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=True,
                    indent=2,
                    width=100
                )
            
            self.logger.info(f"Dictionary saved successfully to {output_path}")
    
    def build_dictionary(self, knowledge_base_path: str, output_path: str) -> Dict[str, Any]:
        """
        Main method to build the institutional dictionary.
        
        Args:
            knowledge_base_path: Path to knowledge_base.json file
            output_path: Path to output institutional_keywords.yaml file
            
        Returns:
            Processing statistics summary
        """
        self.logger.info("Starting institutional dictionary build process")
        
        try:
            # Load knowledge base
            entries = self.load_knowledge_base(knowledge_base_path)
            
            # Process entries
            dictionary = self.process_knowledge_base(entries)
            
            # Save results
            self.save_dictionary(dictionary, output_path)
            
            self.logger.info("Institutional dictionary build completed successfully")
            return self.stats.get_summary()
            
        except Exception as e:
            self.logger.error(f"Dictionary build failed: {e}")
            raise


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('institutional_dict_builder.log', encoding='utf-8')
        ]
    )


def main():
    """Main entry point with enhanced CLI interface."""
    parser = argparse.ArgumentParser(
        description='Build institutional keyword dictionary from knowledge base',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --knowledge-base data/knowledge_base.json --output data/keywords.yaml
  %(prog)s --config config/builder.yaml --verbose
  %(prog)s --knowledge-base kb.json --output keywords.yaml --min-frequency 3
        """
    )
    
    parser.add_argument(
        '--knowledge-base', '--kb',
        default='data/knowledge_base.json',
        help='Path to knowledge_base.json file (default: data/knowledge_base.json)'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/institutional_keywords.yaml',
        help='Path to output YAML file (default: data/institutional_keywords.yaml)'
    )
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        help='Minimum keyword frequency (overrides config)'
    )
    parser.add_argument(
        '--exclude-categories',
        nargs='*',
        help='Categories to exclude (overrides config)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Process knowledge base but do not save output file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load or create configuration
        if args.config and Path(args.config).exists():
            config = args.config
        else:
            config = DictionaryConfig()
            
            # Apply command-line overrides
            if args.min_frequency is not None:
                config.min_keyword_frequency = args.min_frequency
            if args.exclude_categories is not None:
                config.exclude_categories = args.exclude_categories
        
        # Initialize builder
        builder = InstitutionalDictionaryBuilder(config)
        
        # Validate input file
        if not Path(args.knowledge_base).exists():
            logger.error(f"Knowledge base file not found: {args.knowledge_base}")
            return 1
        
        # Build dictionary
        if args.dry_run:
            logger.info("DRY RUN: Processing knowledge base without saving output")
            entries = builder.load_knowledge_base(args.knowledge_base)
            dictionary = builder.process_knowledge_base(entries)
            logger.info(f"Would generate {len(dictionary)} categories with "
                       f"{sum(len(kw) for kw in dictionary.values())} keywords")
        else:
            stats = builder.build_dictionary(args.knowledge_base, args.output)
            
            # Print success message with statistics
            print("\n" + "="*60)
            print("✅ INSTITUTIONAL DICTIONARY BUILD SUCCESSFUL")
            print("="*60)
            print(f"📊 Processed: {stats['processed_pages']}/{stats['total_pages']} entries")
            print(f"📚 Generated: {stats['categories']} categories")
            print(f"🔑 Keywords: {stats['filtered_keywords']} (from {stats['total_keywords']} total)")
            print(f"🌐 Languages: {stats['english_pages']} English, {stats['arabic_pages']} Arabic")
            if stats['processing_time']:
                print(f"⏱️  Time: {stats['processing_time']:.2f} seconds")
            print(f"💾 Output: {args.output}")
            
            if stats['errors_count'] > 0:
                print(f"⚠️  Warnings: {stats['errors_count']} (see log for details)")
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\n❌ Process interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Build process failed: {e}")
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())