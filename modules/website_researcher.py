"""
AI Gate for Artificial Intelligence Applications
Website Researcher Module

This module handles intelligent content retrieval from institutional websites
by searching through processed website data and returning relevant passages
with source information for AI response generation.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import hashlib

# Third-party imports for text processing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, falling back to basic text matching")

@dataclass
class SearchResult:
    """Data class for search results."""
    content_id: str
    url: str
    title: str
    summary: str
    relevant_content: str
    relevance_score: float
    source_url: str
    keywords: List[str]
    category: str
    matched_terms: List[str]
    word_count: int

class WebsiteResearcher:
    """
    Intelligent website content researcher that searches through processed
    institutional website data to find relevant content for user queries.
    """
    
    def __init__(self, pages_file: Path, config: Dict[str, Any] = None, cache_manager=None):
        """
        Initialize the Website Researcher.
        
        Args:
            pages_file: Path to the pages.json file containing processed website data
            config: Configuration dictionary for research parameters
            cache_manager: Optional cache manager instance for performance optimization
        """
        self.pages_file = Path(pages_file)
        self.config = config or {}
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Configuration with defaults
        self.max_results = self.config.get('max_results', 10)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.1)
        self.content_snippet_length = self.config.get('content_snippet_length', 500)
        self.keyword_boost_factor = self.config.get('keyword_boost_factor', 1.5)
        self.title_boost_factor = self.config.get('title_boost_factor', 2.0)
        self.summary_boost_factor = self.config.get('summary_boost_factor', 1.3)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        # Internal state
        self._pages_data: List[Dict[str, Any]] = []
        self._last_loaded: Optional[datetime] = None
        self._search_cache: Dict[str, Tuple[List[SearchResult], datetime]] = {}
        
        # Text processing components
        self._tfidf_vectorizer = None
        self._document_vectors = None
        self._setup_text_processing()
        
        # Load initial data
        asyncio.create_task(self._load_pages_data())
        
        self.logger.info(f"WebsiteResearcher initialized with {len(self._pages_data)} pages")
    
    def _setup_text_processing(self):
        """Setup text processing components."""
        if SKLEARN_AVAILABLE:
            try:
                self._tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    lowercase=True,
                    token_pattern=r'\b\w{2,}\b'
                )
                self.logger.info("Advanced text processing initialized with TF-IDF")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TF-IDF: {e}, using basic matching")
                self._tfidf_vectorizer = None
        else:
            self.logger.info("Using basic text matching (sklearn not available)")
    
    async def _load_pages_data(self) -> bool:
        """
        Load pages data from JSON file.
        
        Returns:
            bool: True if data loaded successfully
        """
        try:
            if not self.pages_file.exists():
                self.logger.error(f"Pages file not found: {self.pages_file}")
                return False
            
            # Check if file has been modified since last load
            file_mtime = datetime.fromtimestamp(self.pages_file.stat().st_mtime)
            if self._last_loaded and file_mtime <= self._last_loaded:
                return True  # No need to reload
            
            self.logger.info(f"Loading pages data from {self.pages_file}")
            
            with open(self.pages_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            if not isinstance(data, dict) or 'pages' not in data:
                self.logger.error("Invalid pages.json structure - missing 'pages' key")
                return False
            
            self._pages_data = data['pages']
            self._last_loaded = datetime.now()
            
            # Update text processing vectors if using advanced processing
            if self._tfidf_vectorizer and self._pages_data:
                await self._update_document_vectors()
            
            # Clear search cache when data is reloaded
            self._search_cache.clear()
            
            self.logger.info(f"Loaded {len(self._pages_data)} pages successfully")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in pages file: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading pages data: {e}")
            return False
    
    async def _update_document_vectors(self):
        """Update TF-IDF document vectors for advanced similarity search."""
        if not self._tfidf_vectorizer or not self._pages_data:
            return
        
        try:
            # Prepare documents for vectorization
            documents = []
            for page in self._pages_data:
                # Combine title, summary, keywords, and search_text for comprehensive matching
                doc_text = f"{page.get('title', '')} {page.get('summary', '')} "
                doc_text += f"{' '.join(page.get('keywords', []))} "
                doc_text += page.get('search_text', '')
                documents.append(doc_text)
            
            # Fit and transform documents
            self._document_vectors = self._tfidf_vectorizer.fit_transform(documents)
            self.logger.info("Document vectors updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating document vectors: {e}")
    
    def _generate_cache_key(self, topics: List[str], keywords: List[str]) -> str:
        """Generate cache key for search parameters."""
        combined = sorted(topics + keywords)
        return hashlib.md5(''.join(combined).encode()).hexdigest()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def _extract_relevant_snippet(self, content: str, terms: List[str], max_length: int = None) -> str:
        """
        Extract the most relevant snippet from content based on search terms.
        
        Args:
            content: Full content text
            terms: Search terms to find
            max_length: Maximum snippet length
            
        Returns:
            str: Most relevant content snippet
        """
        if not content or not terms:
            return content[:max_length] if max_length else content
        
        max_length = max_length or self.content_snippet_length
        content = self._clean_text(content)
        
        if len(content) <= max_length:
            return content
        
        # Find sentences containing search terms
        sentences = re.split(r'[.!?]+', content)
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            score = 0
            matched_terms = []
            
            for term in terms:
                term_lower = term.lower()
                sentence_lower = sentence.lower()
                
                if term_lower in sentence_lower:
                    # Count occurrences and boost score
                    count = sentence_lower.count(term_lower)
                    score += count * len(term)
                    matched_terms.append(term)
            
            if score > 0:
                scored_sentences.append((sentence, score, i, matched_terms))
        
        if not scored_sentences:
            # No matches found, return beginning of content
            return content[:max_length] + "..." if len(content) > max_length else content
        
        # Sort by score and position (prefer higher scores and earlier positions)
        scored_sentences.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        
        # Build snippet around best matching sentences
        snippet = ""
        used_sentences = set()
        
        for sentence, score, pos, matched in scored_sentences:
            if len(snippet) + len(sentence) > max_length:
                break
            
            if pos not in used_sentences:
                if snippet:
                    snippet += " "
                snippet += sentence
                used_sentences.add(pos)
        
        if len(snippet) < max_length // 2 and scored_sentences:
            # Add context around best match
            best_pos = scored_sentences[0][2]
            start_pos = max(0, best_pos - 1)
            end_pos = min(len(sentences), best_pos + 2)
            
            context_snippet = " ".join(sentences[start_pos:end_pos]).strip()
            if len(context_snippet) <= max_length:
                snippet = context_snippet
        
        return snippet + "..." if len(snippet) == max_length else snippet
    
    def _calculate_basic_relevance(self, page: Dict[str, Any], topics: List[str], keywords: List[str]) -> Tuple[float, List[str]]:
        """
        Calculate relevance score using basic text matching.
        
        Args:
            page: Page data dictionary
            topics: List of topic terms
            keywords: List of keyword terms
            
        Returns:
            Tuple of (relevance_score, matched_terms)
        """
        search_terms = list(set(topics + keywords))  # Remove duplicates
        matched_terms = []
        total_score = 0.0
        
        # Prepare searchable text fields
        searchable_fields = {
            'title': (page.get('title', ''), self.title_boost_factor),
            'summary': (page.get('summary', ''), self.summary_boost_factor),
            'keywords': (' '.join(page.get('keywords', [])), self.keyword_boost_factor),
            'search_text': (page.get('search_text', ''), 1.0),
            'content': (str(page.get('structured_content', {})), 0.8)
        }
        
        for term in search_terms:
            term_lower = term.lower()
            term_score = 0.0
            
            for field_name, (field_text, boost) in searchable_fields.items():
                if not field_text:
                    continue
                
                field_text_lower = field_text.lower()
                
                # Exact phrase matching
                if term_lower in field_text_lower:
                    # Count occurrences
                    count = field_text_lower.count(term_lower)
                    
                    # Calculate score based on term length and frequency
                    base_score = count * (len(term) / 10.0)  # Longer terms are more valuable
                    field_score = base_score * boost
                    term_score += field_score
                    
                    if term not in matched_terms:
                        matched_terms.append(term)
                
                # Fuzzy matching for partial matches
                words = field_text_lower.split()
                for word in words:
                    similarity = SequenceMatcher(None, term_lower, word).ratio()
                    if similarity > 0.8:  # High similarity threshold
                        fuzzy_score = similarity * (len(term) / 15.0) * boost * 0.5
                        term_score += fuzzy_score
                        
                        if term not in matched_terms:
                            matched_terms.append(term)
            
            total_score += term_score
        
        # Normalize score by number of search terms
        if search_terms:
            total_score = total_score / len(search_terms)
        
        # Boost score for pages with multiple matches
        if len(matched_terms) > 1:
            total_score *= 1.2
        
        return total_score, matched_terms
    
    async def _calculate_advanced_relevance(self, query_text: str, page_index: int) -> float:
        """
        Calculate relevance using TF-IDF cosine similarity.
        
        Args:
            query_text: Combined query text
            page_index: Index of the page to score
            
        Returns:
            float: Relevance score (0.0 to 1.0)
        """
        if not self._tfidf_vectorizer or self._document_vectors is None:
            return 0.0
        
        try:
            # Transform query text
            query_vector = self._tfidf_vectorizer.transform([query_text])
            
            # Calculate cosine similarity
            page_vector = self._document_vectors[page_index]
            similarity = cosine_similarity(query_vector, page_vector)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Error in advanced relevance calculation: {e}")
            return 0.0
    
    async def search_content(self, topics: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search through website content for relevant passages.
        
        Args:
            topics: List of main topic terms extracted from the question
            keywords: List of keyword terms for detailed matching
            
        Returns:
            List of dictionaries containing relevant content with metadata
        """
        # Validate inputs
        if not topics and not keywords:
            self.logger.warning("No topics or keywords provided for search")
            return []
        
        # Ensure data is loaded
        await self._load_pages_data()
        
        if not self._pages_data:
            self.logger.error("No pages data available for search")
            return []
        
        # Check cache first
        cache_key = self._generate_cache_key(topics, keywords)
        if cache_key in self._search_cache:
            cached_results, cache_time = self._search_cache[cache_key]
            if datetime.now() - cache_time < timedelta(seconds=self.cache_ttl):
                self.logger.info(f"Returning cached search results for key: {cache_key}")
                return [asdict(result) for result in cached_results]
        
        self.logger.info(f"Searching content for topics: {topics}, keywords: {keywords}")
        
        search_results = []
        all_terms = list(set(topics + keywords))  # Remove duplicates
        query_text = ' '.join(all_terms)
        
        # Process each page
        for page_index, page in enumerate(self._pages_data):
            try:
                # Skip pages without required fields
                if not page.get('title') and not page.get('search_text'):
                    continue
                
                # Calculate relevance scores
                basic_score, matched_terms = self._calculate_basic_relevance(page, topics, keywords)
                
                # Use advanced scoring if available
                advanced_score = 0.0
                if self._tfidf_vectorizer:
                    advanced_score = await self._calculate_advanced_relevance(query_text, page_index)
                
                # Combine scores (weighted average)
                if advanced_score > 0:
                    final_score = (basic_score * 0.6) + (advanced_score * 0.4)
                else:
                    final_score = basic_score
                
                # Skip if score is too low
                if final_score < self.similarity_threshold:
                    continue
                
                # Extract relevant content snippet
                content_text = page.get('search_text', '') or page.get('summary', '')
                relevant_snippet = self._extract_relevant_snippet(
                    content_text, 
                    matched_terms, 
                    self.content_snippet_length
                )
                
                # Create search result
                result = SearchResult(
                    content_id=page.get('content_id', f"page_{page_index}"),
                    url=page.get('url', ''),
                    title=page.get('title', 'Untitled'),
                    summary=page.get('summary', ''),
                    relevant_content=relevant_snippet,
                    relevance_score=final_score,
                    source_url=page.get('url', ''),
                    keywords=page.get('keywords', []),
                    category=page.get('category', 'general'),
                    matched_terms=matched_terms,
                    word_count=page.get('metrics', {}).get('word_count', 0)
                )
                
                search_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing page {page_index}: {e}")
                continue
        
        # Sort by relevance score (descending)
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit results
        search_results = search_results[:self.max_results]
        
        # Cache results
        self._search_cache[cache_key] = (search_results, datetime.now())
        
        # Cleanup old cache entries (keep cache size manageable)
        if len(self._search_cache) > 100:
            # Remove oldest entries
            oldest_entries = sorted(
                self._search_cache.items(), 
                key=lambda x: x[1][1]
            )[:20]  # Remove 20 oldest
            
            for key, _ in oldest_entries:
                del self._search_cache[key]
        
        self.logger.info(f"Found {len(search_results)} relevant results")
        
        # Convert to dictionaries for API compatibility
        return [asdict(result) for result in search_results]
    
    async def get_page_by_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific page by its content ID.
        
        Args:
            content_id: Unique identifier for the page
            
        Returns:
            Dict containing page data or None if not found
        """
        await self._load_pages_data()
        
        for page in self._pages_data:
            if page.get('content_id') == content_id:
                return page
        
        return None
    
    async def get_pages_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all pages in a specific category.
        
        Args:
            category: Category name to filter by
            
        Returns:
            List of page dictionaries in the specified category
        """
        await self._load_pages_data()
        
        return [
            page for page in self._pages_data 
            if page.get('category', '').lower() == category.lower()
        ]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the website researcher.
        
        Returns:
            Dict containing various statistics
        """
        await self._load_pages_data()
        
        if not self._pages_data:
            return {
                'total_pages': 0,
                'cache_entries': len(self._search_cache),
                'last_loaded': None,
                'categories': {},
                'total_words': 0
            }
        
        # Calculate statistics
        categories = {}
        total_words = 0
        
        for page in self._pages_data:
            category = page.get('category', 'uncategorized')
            categories[category] = categories.get(category, 0) + 1
            
            word_count = page.get('metrics', {}).get('word_count', 0)
            total_words += word_count
        
        return {
            'total_pages': len(self._pages_data),
            'cache_entries': len(self._search_cache),
            'last_loaded': self._last_loaded.isoformat() if self._last_loaded else None,
            'categories': categories,
            'total_words': total_words,
            'advanced_processing': SKLEARN_AVAILABLE and self._tfidf_vectorizer is not None,
            'config': {
                'max_results': self.max_results,
                'similarity_threshold': self.similarity_threshold,
                'content_snippet_length': self.content_snippet_length
            }
        }
    
    def is_healthy(self) -> bool:
        """
        Check if the website researcher is healthy and operational.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Check if pages file exists
            if not self.pages_file.exists():
                return False
            
            # Check if data is loaded
            if not self._pages_data:
                return False
            
            # Check if data is not too stale (reload if needed)
            if self._last_loaded:
                file_mtime = datetime.fromtimestamp(self.pages_file.stat().st_mtime)
                if file_mtime > self._last_loaded:
                    # Data is stale, trigger reload
                    asyncio.create_task(self._load_pages_data())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources and cache."""
        self.logger.info("Cleaning up WebsiteResearcher...")
        
        # Clear caches
        self._search_cache.clear()
        
        # Clear TF-IDF components
        self._tfidf_vectorizer = None
        self._document_vectors = None
        
        self.logger.info("WebsiteResearcher cleanup completed")

# Example usage and testing functions
async def test_website_researcher():
    """Test function for the WebsiteResearcher module."""
    import tempfile
    import os
    
    # Create test data
    test_data = {
        "metadata": {
            "processed_at": "2024-01-15T10:30:00",
            "total_urls_found": 3
        },
        "pages": [
            {
                "content_id": "test1",
                "url": "https://example.com/ai-basics",
                "title": "Introduction to Artificial Intelligence",
                "summary": "Learn the fundamentals of AI and machine learning.",
                "keywords": ["artificial intelligence", "machine learning", "AI"],
                "category": "education",
                "search_text": "Artificial intelligence is revolutionizing technology...",
                "metrics": {"word_count": 500}
            },
            {
                "content_id": "test2",
                "url": "https://example.com/neural-networks",
                "title": "Neural Networks Explained",
                "summary": "Deep dive into neural network architectures.",
                "keywords": ["neural networks", "deep learning", "AI"],
                "category": "technical",
                "search_text": "Neural networks are the backbone of modern AI systems...",
                "metrics": {"word_count": 750}
            }
        ]
    }
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        test_file = f.name
    
    try:
        # Test the researcher
        researcher = WebsiteResearcher(test_file)
        
        # Test search functionality
        results = await researcher.search_content(
            topics=['artificial intelligence'],
            keywords=['machine learning']
        )
        
        print(f"Found {len(results)} results")
        for result in results:
            print(f"- {result['title']} (score: {result['relevance_score']:.3f})")
        
        # Test statistics
        stats = await researcher.get_statistics()
        print(f"Statistics: {stats}")
        
        # Test health check
        print(f"Health status: {researcher.is_healthy()}")
        
    finally:
        # Cleanup
        os.unlink(test_file)

if __name__ == "__main__":
    # Run tests if executed directly
    asyncio.run(test_website_researcher())
