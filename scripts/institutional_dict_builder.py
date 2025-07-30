"""
AI Gate for Artificial Intelligence Applications
Enhanced Institutional Dictionary Builder - institutional_dict_builder.py

This script processes the knowledge base (knowledge_base.json) to generate
a categorized keyword dictionary (institutional_keywords.yaml) used for
enhancing topic analysis within the AI Gate system.

Refactored Features:
- Centralized NLP processing via AdvancedContentAnalyzer
- Eliminated code duplication and redundant processing
- Unified text analysis pipeline across AI-Gate ecosystem
- Enhanced performance and maintainability
- Comprehensive error handling and logging
- Multi-language support through centralized analyzer
"""

import json
import yaml
import logging
import os
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import argparse

# --- Path Correction ---
# Add the project's root directory to the Python path.
# This allows this script to be run from anywhere and still import modules 
# from the 'modules' directory correctly, just as if it were run from the root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import centralized analyzer
try:
    from modules.analyzer import AdvancedContentAnalyzer
except ImportError as e:
    print(f"Error: Could not import AdvancedContentAnalyzer from modules.analyzer: {e}")
    print("Please ensure the modules/analyzer.py file exists and is properly configured.")
    sys.exit(1)


@dataclass
class DictionaryConfig:
    """Configuration class for the dictionary builder."""
    min_keyword_frequency: int = 1
    exclude_categories: List[str] = field(default_factory=lambda: ['general'])
    sort_keywords: bool = True
    extract_from_title_summary: bool = True
    min_keyword_length: int = 2
    max_keyword_length: int = 50
    use_processed_tokens: bool = True
    use_high_quality_keywords: bool = True
    analyzer_config: Dict[str, Any] = field(default_factory=dict)


class ProcessingStats:
    """Enhanced statistics tracking for processing operations."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.total_pages = 0
        self.processed_pages = 0
        self.skipped_pages = 0
        self.arabic_pages = 0
        self.english_pages = 0
        self.mixed_pages = 0
        self.total_keywords = 0
        self.filtered_keywords = 0
        self.categories = 0
        self.analyzer_calls = 0
        self.failed_analyses = 0
        self.start_time = None
        self.end_time = None
        self.errors = []
        self.category_stats = defaultdict(int)
    
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
    
    def increment_analyzer_calls(self):
        """Increment analyzer call counter."""
        self.analyzer_calls += 1
    
    def increment_failed_analyses(self):
        """Increment failed analysis counter."""
        self.failed_analyses += 1
    
    def update_language_stats(self, language_info: Dict[str, Any]):
        """Update language statistics from analyzer results."""
        if language_info.get('is_arabic', False):
            self.arabic_pages += 1
        elif language_info.get('is_english', False):
            self.english_pages += 1
        else:
            self.mixed_pages += 1
    
    def update_category_stats(self, category: str, keyword_count: int):
        """Update category-specific statistics."""
        self.category_stats[category] += keyword_count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of processing statistics."""
        return {
            'total_pages': self.total_pages,
            'processed_pages': self.processed_pages,
            'skipped_pages': self.skipped_pages,
            'arabic_pages': self.arabic_pages,
            'english_pages': self.english_pages,
            'mixed_pages': self.mixed_pages,
            'total_keywords': self.total_keywords,
            'filtered_keywords': self.filtered_keywords,
            'categories': self.categories,
            'analyzer_calls': self.analyzer_calls,
            'failed_analyses': self.failed_analyses,
            'processing_time': self.processing_time,
            'errors_count': len(self.errors),
            'category_stats': dict(self.category_stats)
        }


class InstitutionalDictionaryBuilder:
    """
    Enhanced institutional dictionary builder with centralized NLP processing.
    
    This refactored version delegates all text analysis tasks to the centralized
    AdvancedContentAnalyzer, ensuring consistency across the AI-Gate ecosystem.
    """
    
    def __init__(self, config: Optional[Union[str, DictionaryConfig]] = None):
        """
        Initialize the dictionary builder.
        
        Args:
            config: Configuration file path or DictionaryConfig object
        """
        self.config = self._load_configuration(config)
        self.stats = ProcessingStats()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize centralized analyzer
        self._initialize_analyzer()
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
    
    def _initialize_analyzer(self):
        """Initialize the centralized AdvancedContentAnalyzer."""
        try:
            # Merge analyzer configuration if provided
            analyzer_config = self.config.analyzer_config or {}
            
            # Initialize the centralized analyzer
            self.analyzer = AdvancedContentAnalyzer(**analyzer_config)
            
            self.logger.info("Successfully initialized AdvancedContentAnalyzer")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AdvancedContentAnalyzer: {e}")
            raise RuntimeError(f"Could not initialize content analyzer: {e}")
    
    def _log_initialization(self):
        """Log initialization information."""
        self.logger.info("Institutional Dictionary Builder initialized")
        self.logger.info(f"Using centralized AdvancedContentAnalyzer")
        self.logger.info(f"Configuration: min_frequency={self.config.min_keyword_frequency}, "
                        f"excluded_categories={self.config.exclude_categories}")
        self.logger.info(f"Extraction settings: title_summary={self.config.extract_from_title_summary}, "
                        f"processed_tokens={self.config.use_processed_tokens}, "
                        f"high_quality_keywords={self.config.use_high_quality_keywords}")
    
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
            
            self.logger.info(f"Loading knowledge base from: {knowledge_base_path}")
            
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
            self.logger.info(f"Successfully loaded {len(entries)} entries from knowledge base")
            
            return entries
    
    def _extract_text_content(self, entry: Dict[str, Any]) -> str:
        """
        Extract all relevant text content from a knowledge base entry.
        
        Args:
            entry: Single knowledge base entry
            
        Returns:
            Combined text content for analysis
        """
        text_parts = []
        
        # Standard content fields
        content_fields = ['content', 'description', 'summary']
        if self.config.extract_from_title_summary:
            content_fields.extend(['title', 'name'])
        
        # Existing keywords field (if present)
        if 'keywords' in entry and isinstance(entry['keywords'], list):
            keywords_text = ' '.join([str(kw).strip() for kw in entry['keywords'] if kw])
            if keywords_text:
                text_parts.append(keywords_text)
        
        # Extract from specified fields
        for field in content_fields:
            if field in entry and entry[field]:
                text_content = str(entry[field]).strip()
                if text_content:
                    text_parts.append(text_content)
        
        return ' '.join(text_parts)
    
    def _extract_keywords_from_analysis(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from AdvancedContentAnalyzer results.
        
        Args:
            analysis_result: Result dictionary from analyzer.analyze_content()
            
        Returns:
            List of extracted keywords
        """
        keywords = []
        
        # Use high-quality keywords if available and configured
        if (self.config.use_high_quality_keywords and 
            'keywords' in analysis_result and 
            analysis_result['keywords']):
            keywords.extend(analysis_result['keywords'])
        
        # Use processed tokens if available and configured
        if (self.config.use_processed_tokens and 
            'processed_tokens' in analysis_result and 
            analysis_result['processed_tokens']):
            # Filter processed tokens by length constraints
            filtered_tokens = [
                token for token in analysis_result['processed_tokens']
                if self.config.min_keyword_length <= len(token) <= self.config.max_keyword_length
            ]
            keywords.extend(filtered_tokens)
        
        # Fallback to basic tokens if nothing else available
        if (not keywords and 
            'tokens' in analysis_result and 
            analysis_result['tokens']):
            # Apply basic filtering for fallback tokens
            filtered_tokens = [
                token for token in analysis_result['tokens']
                if (self.config.min_keyword_length <= len(token) <= self.config.max_keyword_length
                    and token.strip())
            ]
            keywords.extend(filtered_tokens)
        
        # Remove duplicates while preserving order
        return self._deduplicate_keywords(keywords)
    
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
    
    def process_knowledge_base(self, entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Process knowledge base entries to build category-keyword dictionary.
        
        Args:
            entries: List of knowledge base entries
            
        Returns:
            Dictionary mapping categories to keyword lists
        """
        self.logger.info("Starting knowledge base processing with centralized analyzer")
        self.stats.start_timing()
        
        category_counters = defaultdict(Counter)
        processed_count = 0
        
        for i, entry in enumerate(entries):
            try:
                self._process_single_entry(entry, category_counters)
                processed_count += 1
                
                # Log progress periodically
                if processed_count % 100 == 0:
                    self.logger.info(f"Processed {processed_count}/{len(entries)} entries...")
                    
            except Exception as e:
                entry_id = entry.get('title', entry.get('id', f'index_{i}'))
                self.logger.warning(f"Error processing entry '{entry_id}': {e}")
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
        Process a single knowledge base entry using centralized analyzer.
        
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
        
        # Extract text content for analysis
        text_content = self._extract_text_content(entry)
        if not text_content.strip():
            self.stats.skipped_pages += 1
            return
        
        # Analyze content using centralized analyzer
        try:
            self.stats.increment_analyzer_calls()
            analysis_result = self.analyzer.analyze_content(text_content)
            
            # Extract keywords from analysis results
            keywords = self._extract_keywords_from_analysis(analysis_result)
            
            if not keywords:
                self.stats.skipped_pages += 1
                return
            
            # Update counters with extracted keywords
            for keyword in keywords:
                category_counters[category][keyword] += 1
            
            # Update statistics
            self.stats.processed_pages += 1
            self.stats.update_category_stats(category, len(keywords))
            
            # Update language statistics if available
            if 'language_info' in analysis_result:
                self.stats.update_language_stats(analysis_result['language_info'])
            
        except Exception as e:
            self.stats.increment_failed_analyses()
            raise ValueError(f"Content analysis failed: {e}")
    
    def _build_filtered_dictionary(self, category_counters: Dict[str, Counter]) -> Dict[str, List[str]]:
        """
        Build final dictionary with frequency filtering and sorting.
        
        Args:
            category_counters: Dictionary of category keyword counters
            
        Returns:
            Filtered and sorted category-keyword dictionary
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
                filtered_keywords.sort(key=lambda x: x.lower())
            
            if filtered_keywords:  # Only include non-empty categories
                dictionary[category] = filtered_keywords
                self.stats.total_keywords += len(counter)
                self.stats.filtered_keywords += len(filtered_keywords)
        
        self.stats.categories = len(dictionary)
        return dictionary
    
    def _log_processing_summary(self, dictionary: Dict[str, List[str]]):
        """Log comprehensive processing summary."""
        stats = self.stats.get_summary()
        
        self.logger.info("=" * 70)
        self.logger.info("INSTITUTIONAL DICTIONARY PROCESSING SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Total entries in knowledge base: {stats['total_pages']}")
        self.logger.info(f"Successfully processed entries: {stats['processed_pages']}")
        self.logger.info(f"Skipped entries: {stats['skipped_pages']}")
        self.logger.info(f"Failed analyses: {stats['failed_analyses']}")
        self.logger.info(f"Total analyzer calls: {stats['analyzer_calls']}")
        
        # Language distribution
        self.logger.info("\nLANGUAGE DISTRIBUTION:")
        self.logger.info(f"  Arabic entries: {stats['arabic_pages']}")
        self.logger.info(f"  English entries: {stats['english_pages']}")
        self.logger.info(f"  Mixed/Other entries: {stats['mixed_pages']}")
        
        # Keyword statistics
        self.logger.info("\nKEYWORD STATISTICS:")
        self.logger.info(f"  Total keywords extracted: {stats['total_keywords']}")
        self.logger.info(f"  Keywords after filtering: {stats['filtered_keywords']}")
        self.logger.info(f"  Categories generated: {stats['categories']}")
        
        if stats['processing_time']:
            rate = stats['processed_pages'] / stats['processing_time']
            self.logger.info(f"\nPERFORMANCE:")
            self.logger.info(f"  Processing time: {stats['processing_time']:.2f} seconds")
            self.logger.info(f"  Processing rate: {rate:.1f} entries/second")
        
        if stats['errors_count'] > 0:
            self.logger.warning(f"\nERRORS: {stats['errors_count']} errors encountered")
        
        # Category breakdown
        self.logger.info("\nCATEGORY BREAKDOWN:")
        for category, keywords in sorted(dictionary.items()):
            avg_keywords = stats['category_stats'].get(category, 0) / stats['processed_pages'] if stats['processed_pages'] > 0 else 0
            self.logger.info(f"  {category}: {len(keywords)} unique keywords "
                           f"(avg {avg_keywords:.1f} per entry)")
    
    def save_dictionary(self, dictionary: Dict[str, List[str]], output_path: str):
        """
        Save dictionary to YAML file with comprehensive metadata.
        
        Args:
            dictionary: Category-keyword dictionary
            output_path: Path to output YAML file
        """
        with self._error_handling("Dictionary saving"):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare comprehensive metadata
            stats = self.stats.get_summary()
            metadata = {
                'metadata': {
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                    'generator': 'InstitutionalDictionaryBuilder (Refactored)',
                    'version': '2.0.0',
                    'analyzer': 'AdvancedContentAnalyzer (Centralized)',
                    'total_categories': len(dictionary),
                    'total_keywords': sum(len(keywords) for keywords in dictionary.values()),
                    'processing_stats': {
                        'total_entries': stats['total_pages'],
                        'processed_entries': stats['processed_pages'],
                        'skipped_entries': stats['skipped_pages'],
                        'failed_analyses': stats['failed_analyses'],
                        'analyzer_calls': stats['analyzer_calls'],
                        'processing_time_seconds': stats['processing_time']
                    },
                    'language_distribution': {
                        'arabic_entries': stats['arabic_pages'],
                        'english_entries': stats['english_pages'],
                        'mixed_entries': stats['mixed_pages']
                    },
                    'configuration': {
                        'min_frequency_threshold': self.config.min_keyword_frequency,
                        'excluded_categories': self.config.exclude_categories,
                        'extract_from_title_summary': self.config.extract_from_title_summary,
                        'use_processed_tokens': self.config.use_processed_tokens,
                        'use_high_quality_keywords': self.config.use_high_quality_keywords,
                        'min_keyword_length': self.config.min_keyword_length,
                        'max_keyword_length': self.config.max_keyword_length
                    }
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
            
            file_size = output_path.stat().st_size
            self.logger.info(f"Dictionary saved successfully to {output_path}")
            self.logger.info(f"Output file size: {file_size:,} bytes")
    
    def build_dictionary(self, knowledge_base_path: str, output_path: str) -> Dict[str, Any]:
        """
        Main method to build the institutional dictionary using centralized analysis.
        
        Args:
            knowledge_base_path: Path to knowledge_base.json file
            output_path: Path to output institutional_keywords.yaml file
            
        Returns:
            Processing statistics summary
        """
        self.logger.info("Starting institutional dictionary build process")
        self.logger.info("Using centralized AdvancedContentAnalyzer for consistent NLP processing")
        
        try:
            # Load knowledge base
            entries = self.load_knowledge_base(knowledge_base_path)
            
            # Process entries using centralized analyzer
            dictionary = self.process_knowledge_base(entries)
            
            # Save results with comprehensive metadata
            self.save_dictionary(dictionary, output_path)
            
            self.logger.info("Institutional dictionary build completed successfully")
            self.logger.info("All text processing performed via centralized AdvancedContentAnalyzer")
            
            return self.stats.get_summary()
            
        except Exception as e:
            self.logger.error(f"Dictionary build failed: {e}")
            raise


def setup_logging(verbose: bool = False):
    """Setup comprehensive logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging with both console and file output
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                logs_dir / 'institutional_dict_builder.log', 
                encoding='utf-8'
            )
        ]
    )


def main():
    """Enhanced main entry point with comprehensive CLI interface."""
    parser = argparse.ArgumentParser(
        description='Build institutional keyword dictionary using centralized NLP analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --knowledge-base data/knowledge_base.json --output data/keywords.yaml
  %(prog)s --config config/builder.yaml --verbose
  %(prog)s --knowledge-base kb.json --output keywords.yaml --min-frequency 3
  %(prog)s --dry-run --verbose  # Process without saving, show detailed logs

Features:
  • Centralized NLP processing via AdvancedContentAnalyzer
  • Multi-language support (Arabic, English, Mixed)
  • Frequency-based keyword filtering
  • Comprehensive metadata generation
  • Performance monitoring and statistics
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
    parser.add_argument(
        '--analyzer-config',
        help='Path to AdvancedContentAnalyzer configuration file'
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
            
            # Load analyzer configuration if provided
            if args.analyzer_config and Path(args.analyzer_config).exists():
                try:
                    with open(args.analyzer_config, 'r', encoding='utf-8') as f:
                        config.analyzer_config = yaml.safe_load(f) or {}
                    logger.info(f"Loaded analyzer configuration from {args.analyzer_config}")
                except Exception as e:
                    logger.warning(f"Failed to load analyzer config: {e}")
        
        # Validate input file
        if not Path(args.knowledge_base).exists():
            logger.error(f"Knowledge base file not found: {args.knowledge_base}")
            return 1
        
        # Initialize builder with centralized analyzer
        logger.info("Initializing Institutional Dictionary Builder with centralized NLP processing")
        builder = InstitutionalDictionaryBuilder(config)
        
        # Build dictionary
        if args.dry_run:
            logger.info("DRY RUN: Processing knowledge base without saving output")
            entries = builder.load_knowledge_base(args.knowledge_base)
            dictionary = builder.process_knowledge_base(entries)
            
            total_keywords = sum(len(kw) for kw in dictionary.values())
            logger.info(f"Would generate {len(dictionary)} categories with {total_keywords} keywords")
            
            # Show top categories
            top_categories = sorted(
                [(cat, len(kws)) for cat, kws in dictionary.items()], 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            print("\n" + "="*50)
            print("🔍 DRY RUN RESULTS")
            print("="*50)
            print(f"📊 Total categories: {len(dictionary)}")
            print(f"🔑 Total keywords: {total_keywords}")
            print(f"\n📈 Top 10 categories by keyword count:")
            for cat, count in top_categories:
                print(f"  • {cat}: {count} keywords")
            
        else:
            stats = builder.build_dictionary(args.knowledge_base, args.output)
            
            # Print comprehensive success message
            print("\n" + "="*70)
            print("✅ INSTITUTIONAL DICTIONARY BUILD SUCCESSFUL")
            print("="*70)
            print(f"🔧 Analyzer: AdvancedContentAnalyzer (Centralized)")
            print(f"📊 Processed: {stats['processed_pages']}/{stats['total_pages']} entries")
            print(f"📚 Generated: {stats['categories']} categories")
            print(f"🔑 Keywords: {stats['filtered_keywords']} (from {stats['total_keywords']} total)")
            print(f"🤖 Analyzer calls: {stats['analyzer_calls']}")
            
            # Language breakdown
            total_lang_entries = stats['arabic_pages'] + stats['english_pages'] + stats['mixed_pages']
            if total_lang_entries > 0:
                print(f"🌐 Languages: {stats['english_pages']} English, {stats['arabic_pages']} Arabic, {stats['mixed_pages']} Mixed")
            
            if stats['processing_time']:
                rate = stats['processed_pages'] / stats['processing_time']
                print(f"⏱️  Time: {stats['processing_time']:.2f}s ({rate:.1f} entries/sec)")
            
            print(f"💾 Output: {args.output}")
            
            if stats['failed_analyses'] > 0:
                print(f"⚠️  Failed analyses: {stats['failed_analyses']}")
            
            if stats['errors_count'] > 0:
                print(f"⚠️  Warnings: {stats['errors_count']} (see log for details)")
            
            print("\n🎯 Key improvements in this refactored version:")
            print("  • Centralized NLP processing via AdvancedContentAnalyzer")
            print("  • Eliminated code duplication and redundant text processing")
            print("  • Unified analysis pipeline across AI-Gate ecosystem")
            print("  • Enhanced metadata and performance tracking")
    
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