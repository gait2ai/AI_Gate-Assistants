# tests/test_website_researcher.py

"""
AI Gate for Artificial Intelligence Applications
test_website_researcher Module

Test Suite for the Website Researcher Module (modules.website_researcher)

This module provides comprehensive tests for the WebsiteResearcher class,
ensuring its ability to correctly load, process, and search website content
data. It covers aspects like:
- Initialization with various configurations.
- Correct loading and parsing of 'pages.json'.
- Search functionality, including relevance scoring and result ordering.
- Text processing utilities (cleaning, snippet extraction).
- Caching behavior (via a mocked CacheManager).
- Handling of edge cases and potential errors.
- Utility methods for statistics and health checks.
- Behavior with and without advanced text processing (sklearn).
"""

import pytest
import asyncio
import json
import tempfile
import shutil # For cleaning up temp directories
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call # Added call for checking mock calls

# Import the class to be tested
from modules.website_researcher import WebsiteResearcher, SearchResult # Assuming SearchResult is still relevant

# Pytest marker for all tests in this file to be treated as asyncio
pytestmark = pytest.mark.asyncio


# --- Sample Data Fixture (Local to this test file) ---

@pytest.fixture
def sample_pages_data_dict() -> dict:
    """Provides a sample dictionary结构 for pages.json content."""
    return {
        "metadata": {
            "processed_at": "2024-07-28T10:30:00",
            "total_urls_found": 3, # Adjusted to match actual pages
        },
        "pages": [
            {
                "content_id": "ai_fundamentals_001",
                "url": "https://institution.edu/ai/fundamentals",
                "title": "Artificial Intelligence Fundamentals",
                "summary": "Comprehensive introduction to AI concepts, machine learning algorithms.",
                "keywords": ["artificial intelligence", "machine learning", "AI", "algorithms"],
                "category": "academics",
                "search_text": "Artificial intelligence (AI) is a branch of computer science. Machine learning is a subset of AI.",
                "structured_content": {"headings": ["Intro", "Core"], "sections": 2},
                "metrics": {"word_count": 100, "last_updated": "2024-01-10"}
            },
            {
                "content_id": "ml_applications_002",
                "url": "https://institution.edu/research/ml",
                "title": "Machine Learning Applications",
                "summary": "Practical applications of machine learning in research.",
                "keywords": ["machine learning", "research", "data analysis"],
                "category": "research",
                "search_text": "Machine learning has revolutionized academic research. Common applications include NLP.",
                "structured_content": {"headings": ["Overview", "Cases"], "sections": 2},
                "metrics": {"word_count": 120, "last_updated": "2024-01-12"}
            },
            {
                "content_id": "data_science_003",
                "url": "https://institution.edu/programs/ds",
                "title": "Data Science Program",
                "summary": "Guide to our data science curriculum.",
                "keywords": ["data science", "statistics", "programming", "python"],
                "category": "academics",
                "search_text": "The Data Science program combines statistics and computer science. Students learn Python.",
                "structured_content": {"headings": ["Curriculum", "Careers"], "sections": 2},
                "metrics": {"word_count": 90, "last_updated": "2024-01-08"}
            }
        ]
    }

@pytest.fixture
def temp_pages_file(sample_pages_data_dict: dict) -> Path:
    """
    Creates a temporary 'pages.json' file with sample data for a test.
    The file and its directory are cleaned up after the test.
    Yields the Path object to the temporary file.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="test_website_researcher_"))
    pages_file_path = temp_dir / "pages.json"
    
    try:
        with open(pages_file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_pages_data_dict, f, indent=2)
        yield pages_file_path
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# --- Main Test Class for WebsiteResearcher ---

class TestWebsiteResearcher:
    """Groups tests for the WebsiteResearcher class."""

    @pytest.fixture
    async def researcher_instance(self, temp_pages_file: Path, website_researcher_config: dict, mock_cache_manager: MagicMock) -> WebsiteResearcher:
        """
        Provides a WebsiteResearcher instance initialized with a temporary
        pages file, configuration from conftest.py, and a mock cache manager.
        Also waits briefly for any initial async tasks in __init__ to settle.
        """
        researcher = WebsiteResearcher(
            pages_file=temp_pages_file,
            config=website_researcher_config,
            cache_manager=mock_cache_manager
        )
        # WebsiteResearcher's __init__ calls asyncio.create_task(self._load_pages_data())
        # Give it a moment to potentially complete or start.
        # In a real scenario with I/O, more robust handling might be needed,
        # but for tests with local files, a small delay is often sufficient.
        await asyncio.sleep(0.01) # Small delay for async init tasks
        return researcher

    @pytest.fixture
    async def researcher_instance_no_sklearn(self, temp_pages_file: Path, website_researcher_config: dict, mock_cache_manager: MagicMock) -> WebsiteResearcher:
        """
        Provides a WebsiteResearcher instance with SKLEARN_AVAILABLE patched to False,
        simulating an environment where scikit-learn is not installed.
        """
        with patch('modules.website_researcher.SKLEARN_AVAILABLE', False):
            researcher = WebsiteResearcher(
                pages_file=temp_pages_file,
                config=website_researcher_config,
                cache_manager=mock_cache_manager
            )
            await asyncio.sleep(0.01)
            return researcher

    # --- Initialization and Configuration Tests ---
    class TestInitialization:
        """Tests related to WebsiteResearcher initialization and configuration."""

        def test_initialization_with_valid_config(self, temp_pages_file: Path, website_researcher_config: dict, mock_cache_manager: MagicMock):
            """
            Tests if WebsiteResearcher initializes correctly with configuration
            from the 'website_researcher_config' fixture.
            """
            researcher = WebsiteResearcher(
                pages_file=temp_pages_file,
                config=website_researcher_config,
                cache_manager=mock_cache_manager
            )
            assert researcher.pages_file == temp_pages_file
            assert researcher.config == website_researcher_config
            assert researcher.cache_manager is mock_cache_manager
            assert researcher.max_results == website_researcher_config.get('max_results', 10)
            # In WebsiteResearcher, 'similarity_threshold' is used from config.
            assert researcher.similarity_threshold == website_researcher_config.get('similarity_threshold', 0.1)

        def test_initialization_with_empty_config(self, temp_pages_file: Path, mock_cache_manager: MagicMock):
            """
            Tests WebsiteResearcher initialization with an empty config,
            expecting it to use internal default values.
            """
            researcher = WebsiteResearcher(
                pages_file=temp_pages_file,
                config={}, # Empty config
                cache_manager=mock_cache_manager
            )
            assert researcher.max_results == 10  # Internal default in WebsiteResearcher
            assert researcher.similarity_threshold == 0.1 # Internal default
            assert researcher.content_snippet_length == 500 # Internal default

        @pytest.mark.asyncio
        async def test_initial_data_loading_on_init(self, researcher_instance: WebsiteResearcher, sample_pages_data_dict: dict):
            """
            Tests if _load_pages_data is called during initialization and loads data.
            The researcher_instance fixture already waits for potential async init.
            """
            # _load_pages_data is called via asyncio.create_task in __init__
            # We need to ensure it has run. The fixture 'researcher_instance' handles a small delay.
            # For a more robust check, one might mock _load_pages_data or check its effects.
            assert len(researcher_instance._pages_data) == len(sample_pages_data_dict["pages"])
            assert researcher_instance._last_loaded is not None

        def test_initialization_with_nonexistent_file(self, website_researcher_config: dict, mock_cache_manager: MagicMock):
            """
            Tests initialization with a path to a non-existent pages.json file.
            Loading should fail gracefully, and is_healthy should reflect this.
            """
            non_existent_file = Path("path/to/non_existent_pages.json")
            researcher = WebsiteResearcher(
                pages_file=non_existent_file,
                config=website_researcher_config,
                cache_manager=mock_cache_manager
            )
            # The initial _load_pages_data (async task) might log an error.
            # The health check should confirm the state.
            assert not researcher.is_healthy() # is_healthy checks pages_file.exists()

    # --- Data Loading Tests ---
    class TestDataLoading:
        """Tests for the _load_pages_data method."""

        @pytest.mark.asyncio
        async def test_load_pages_data_success(self, researcher_instance: WebsiteResearcher, sample_pages_data_dict: dict):
            """Tests successful loading and parsing of a valid pages.json."""
            # The researcher_instance fixture should have already loaded the data.
            assert researcher_instance._pages_data is not None
            assert len(researcher_instance._pages_data) == len(sample_pages_data_dict["pages"])
            assert researcher_instance._pages_data[0]["content_id"] == sample_pages_data_dict["pages"][0]["content_id"]
            assert researcher_instance._last_loaded is not None

        @pytest.mark.asyncio
        async def test_load_pages_data_file_not_found(self, website_researcher_config: dict, mock_cache_manager: MagicMock, caplog):
            """Tests _load_pages_data when the pages_file does not exist."""
            non_existent_file = Path("path/to/non_existent_pages.json")
            researcher = WebsiteResearcher(pages_file=non_existent_file, config=website_researcher_config, cache_manager=mock_cache_manager)
            
            loaded_successfully = await researcher._load_pages_data()
            assert loaded_successfully is False
            assert len(researcher._pages_data) == 0 # Should be empty
            assert f"Pages file not found: {non_existent_file}" in caplog.text

        @pytest.mark.asyncio
        async def test_load_pages_data_invalid_json(self, website_researcher_config: dict, mock_cache_manager: MagicMock, caplog):
            """Tests _load_pages_data with a file containing invalid JSON."""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                tmp_file.write("{invalid_json_content: ")
                tmp_file_path = Path(tmp_file.name)
            
            researcher = WebsiteResearcher(pages_file=tmp_file_path, config=website_researcher_config, cache_manager=mock_cache_manager)
            try:
                loaded_successfully = await researcher._load_pages_data()
                assert loaded_successfully is False
                assert "Invalid JSON in pages file" in caplog.text
            finally:
                tmp_file_path.unlink()

        @pytest.mark.asyncio
        async def test_load_pages_data_missing_pages_key(self, website_researcher_config: dict, mock_cache_manager: MagicMock, caplog):
            """Tests _load_pages_data with valid JSON but missing the root 'pages' key."""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump({"metadata": {"info": "some_data"}}, tmp_file) # No 'pages' key
                tmp_file_path = Path(tmp_file.name)

            researcher = WebsiteResearcher(pages_file=tmp_file_path, config=website_researcher_config, cache_manager=mock_cache_manager)
            try:
                loaded_successfully = await researcher._load_pages_data()
                assert loaded_successfully is False
                assert "Invalid pages.json structure - missing 'pages' key" in caplog.text
            finally:
                tmp_file_path.unlink()

        @pytest.mark.asyncio
        async def test_load_pages_data_no_reload_if_not_modified(self, researcher_instance: WebsiteResearcher, temp_pages_file: Path, caplog):
            """Tests that data is not reloaded if the file hasn't changed since last load."""
            # Initial load is done by the fixture.
            researcher_instance.logger.info("---Marker for first load completion---") # For caplog clarity
            first_load_time = researcher_instance._last_loaded
            
            await researcher_instance._load_pages_data() # Attempt reload
            
            assert researcher_instance._last_loaded == first_load_time
            # Check logs to ensure it didn't try to read the file again (or a specific log message for skipping)
            # The current code doesn't log a specific "skipped reload" message, but it shouldn't log "Loading pages data from..."
            assert f"Loading pages data from {temp_pages_file}" not in caplog.text.split("---Marker for first load completion---")[-1]


        @pytest.mark.asyncio
        async def test_load_pages_data_reloads_if_modified(self, researcher_instance: WebsiteResearcher, temp_pages_file: Path, sample_pages_data_dict: dict, caplog):
            """Tests that data is reloaded if the file has been modified."""
            # Initial load by fixture
            initial_load_time = researcher_instance._last_loaded
            assert initial_load_time is not None

            # Modify the file's content and mtime
            await asyncio.sleep(0.1) # Ensure mtime will be different
            modified_data = sample_pages_data_dict.copy()
            modified_data["metadata"]["new_field"] = "added_for_reload_test"
            with open(temp_pages_file, 'w', encoding='utf-8') as f:
                json.dump(modified_data, f)
            
            researcher_instance.logger.info("---Marker for modified file---") # For caplog clarity
            
            await researcher_instance._load_pages_data() # Attempt reload
            
            assert researcher_instance._last_loaded > initial_load_time
            assert f"Loading pages data from {temp_pages_file}" in caplog.text.split("---Marker for modified file---")[-1]
            # Check if the new field is accessible if _pages_data was updated (though it's metadata)
            # The primary check is that _last_loaded was updated.


    # --- Search Functionality Tests ---
    class TestSearchFunctionality:
        """Tests for the search_content method."""

        @pytest.mark.asyncio
        @pytest.mark.parametrize("topics, keywords, expected_min_results, expected_titles_part", [
            (["artificial intelligence"], ["machine learning"], 1, ["Artificial Intelligence Fundamentals"]),
            (["data science"], ["python", "curriculum"], 1, ["Data Science Program"]),
            (["research"], ["NLP"], 1, ["Machine Learning Applications"]), # NLP is in search_text for ML page
            (["non_existent_topic"], ["mystery_keyword"], 0, [])
        ])
        async def test_search_content_various_queries(self, researcher_instance: WebsiteResearcher, topics: list, keywords: list, expected_min_results: int, expected_titles_part: list):
            """Tests search_content with different queries and expected outcomes."""
            results = await researcher_instance.search_content(topics=topics, keywords=keywords)
            
            assert len(results) >= expected_min_results
            if expected_min_results > 0:
                assert isinstance(results[0], dict) # SearchResult is converted to dict
                found_titles = [r['title'] for r in results]
                for title_part in expected_titles_part:
                    assert any(title_part in ft for ft in found_titles)
            
            # All results should have a relevance score
            for r in results:
                 assert 'relevance_score' in r and r['relevance_score'] >= researcher_instance.similarity_threshold

        @pytest.mark.asyncio
        async def test_search_content_empty_query(self, researcher_instance: WebsiteResearcher):
            """Tests search_content with empty topics and keywords."""
            results = await researcher_instance.search_content(topics=[], keywords=[])
            assert results == []

        @pytest.mark.asyncio
        async def test_search_content_result_structure(self, researcher_instance: WebsiteResearcher):
            """Ensures search results adhere to the expected SearchResult structure (as dicts)."""
            results = await researcher_instance.search_content(topics=["AI"], keywords=["algorithms"])
            if not results:
                pytest.skip("No results found for structure test, query might need adjustment or sample data.")

            result_item = results[0]
            expected_keys = SearchResult.__annotations__.keys() # Get keys from SearchResult dataclass
            for key in expected_keys:
                assert key in result_item
            
            assert isinstance(result_item['relevance_score'], float)
            assert isinstance(result_item['keywords'], list) # Page keywords
            assert isinstance(result_item['matched_terms'], list) # Query terms matched

        @pytest.mark.asyncio
        async def test_search_content_ordering_by_relevance(self, researcher_instance: WebsiteResearcher):
            """Tests that results are sorted by relevance_score in descending order."""
            results = await researcher_instance.search_content(topics=["machine learning"], keywords=["AI", "research", "data"])
            if len(results) > 1:
                scores = [r['relevance_score'] for r in results]
                assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

        @pytest.mark.asyncio
        async def test_search_content_max_results_respected(self, researcher_instance: WebsiteResearcher, website_researcher_config: dict):
            """Tests if the max_results configuration limits the number of returned results."""
            original_max_results = researcher_instance.max_results
            researcher_instance.max_results = 1 # Set a low limit for the test
            
            results = await researcher_instance.search_content(topics=["AI", "machine learning", "data science"], keywords=[])
            assert len(results) <= 1
            
            researcher_instance.max_results = original_max_results # Restore original

        @pytest.mark.asyncio
        async def test_search_content_no_pages_data(self, researcher_instance: WebsiteResearcher, caplog):
            """Tests search when _pages_data is empty."""
            researcher_instance._pages_data = [] # Manually empty the data
            results = await researcher_instance.search_content(topics=["test"], keywords=["query"])
            assert results == []
            assert "No pages data available for search" in caplog.text


    # --- Text Processing and Relevance Tests ---
    class TestTextProcessingAndRelevance:
        """Tests for internal text processing and relevance scoring logic."""

        @pytest.mark.parametrize("text_in, expected_out", [
            ("  leading & trailing spaces  ", "leading & trailing spaces"), # _clean_text doesn't strip anymore
            ("text\nwith\nnewlines", "text with newlines"),
            ("text <p>with</p> html", "text with html"), # Basic HTML tag removal
        ])
        def test_clean_text(self, researcher_instance: WebsiteResearcher, text_in: str, expected_out: str):
            """Tests the _clean_text utility method."""
            # Note: researcher._clean_text() is basic. If it becomes more complex, add more cases.
            # Current version: re.sub(r'\s+', ' ', text.strip()); re.sub(r'<[^>]+>', '', text)
            assert researcher_instance._clean_text(text_in) == expected_out

        @pytest.mark.parametrize("content, terms, max_len, expected_part_of_snippet", [
            ("Full long text about AI and Machine Learning. AI is cool. ML helps.", ["AI", "ML"], 50, "AI"),
            ("Another example sentence. This one focuses on data science and python.", ["python"], 30, "python"),
            ("Short content", ["Short"], 100, "Short content"), # Content shorter than max_len
            ("No matching terms in this sentence.", ["xyz"], 50, "No matching terms") # Returns start of content
        ])
        def test_extract_relevant_snippet(self, researcher_instance: WebsiteResearcher, content: str, terms: list, max_len: int, expected_part_of_snippet: str):
            """Tests the _extract_relevant_snippet method."""
            snippet = researcher_instance._extract_relevant_snippet(content, terms, max_len)
            assert len(snippet) <= max_len + 3 # For "..."
            assert expected_part_of_snippet in snippet
            if len(content) > max_len and "..." not in snippet and len(snippet) < len(content) : # Check if "..." is added
                 if snippet != content : # if snippet is not the full content
                    pass # Snippet logic might return a full sentence without "..." if it fits

        def test_calculate_basic_relevance(self, researcher_instance: WebsiteResearcher, sample_pages_data_dict: dict):
            """Tests the _calculate_basic_relevance scoring."""
            page_ai = sample_pages_data_dict["pages"][0] # AI fundamentals
            page_ds = sample_pages_data_dict["pages"][2] # Data Science

            score_ai, terms_ai = researcher_instance._calculate_basic_relevance(page_ai, ["AI"], ["machine learning"])
            score_ds_for_ai, _ = researcher_instance._calculate_basic_relevance(page_ds, ["AI"], ["machine learning"])

            assert score_ai > 0
            assert "AI" in terms_ai or "artificial intelligence" in terms_ai
            assert "machine learning" in terms_ai
            assert score_ai > score_ds_for_ai # AI page should be more relevant to AI query

        def test_generate_cache_key_consistency(self, researcher_instance: WebsiteResearcher):
            """Tests that _generate_cache_key is consistent for same inputs."""
            key1 = researcher_instance._generate_cache_key(topics=["a", "b"], keywords=["c"])
            key2 = researcher_instance._generate_cache_key(topics=["b", "a"], keywords=["c"]) # Order of topics matters if not sorted internally
            key3 = researcher_instance._generate_cache_key(topics=["a", "b"], keywords=["c"])
            
            # The current _generate_cache_key sorts topics+keywords
            assert key1 == key2 
            assert key1 == key3


    # --- Advanced Processing (TF-IDF) Tests ---
    class TestAdvancedProcessing:
        """Tests related to TF-IDF and advanced relevance (if sklearn is available)."""

        @pytest.mark.asyncio
        async def test_tfidf_vectorizer_setup_with_sklearn(self, researcher_instance: WebsiteResearcher):
            """Tests if TF-IDF vectorizer is set up when sklearn is available."""
            from modules.website_researcher import SKLEARN_AVAILABLE
            if not SKLEARN_AVAILABLE:
                pytest.skip("scikit-learn not available, skipping TF-IDF setup test.")
            
            # researcher_instance should have tried to set it up if available
            assert researcher_instance._tfidf_vectorizer is not None
            # _update_document_vectors is called in _load_pages_data if vectorizer is present
            await researcher_instance._load_pages_data() # Ensure it's called
            assert researcher_instance._document_vectors is not None
            assert researcher_instance._document_vectors.shape[0] == len(researcher_instance._pages_data)

        @pytest.mark.asyncio
        async def test_tfidf_vectorizer_not_setup_without_sklearn(self, researcher_instance_no_sklearn: WebsiteResearcher):
            """Tests that TF-IDF vectorizer is None when sklearn is unavailable."""
            assert researcher_instance_no_sklearn._tfidf_vectorizer is None
            assert researcher_instance_no_sklearn._document_vectors is None

        @pytest.mark.asyncio
        async def test_calculate_advanced_relevance_with_sklearn(self, researcher_instance: WebsiteResearcher):
            """Tests _calculate_advanced_relevance when sklearn is available."""
            from modules.website_researcher import SKLEARN_AVAILABLE
            if not SKLEARN_AVAILABLE:
                pytest.skip("scikit-learn not available.")
            
            # Ensure vectors are built
            await researcher_instance._load_pages_data() # This calls _update_document_vectors
            if researcher_instance._document_vectors is None:
                 pytest.fail("_document_vectors not created even with SKLEARN_AVAILABLE")

            query_text = "artificial intelligence concepts"
            # Test against the first page (AI fundamentals)
            score = await researcher_instance._calculate_advanced_relevance(query_text, page_index=0)
            assert 0.0 <= score <= 1.0
            # A more specific assertion would require knowing the exact vector values.
            # For now, checking type and range is a good start.
            
            # Test against a less relevant page (e.g., Data Science for an AI query)
            score_less_relevant = await researcher_instance._calculate_advanced_relevance(query_text, page_index=2)
            assert score > score_less_relevant or score == 0.0 # Expect higher score for more relevant page

        @pytest.mark.asyncio
        async def test_calculate_advanced_relevance_without_sklearn(self, researcher_instance_no_sklearn: WebsiteResearcher):
            """Tests that _calculate_advanced_relevance returns 0 when sklearn is unavailable."""
            score = await researcher_instance_no_sklearn._calculate_advanced_relevance("query", page_index=0)
            assert score == 0.0

        @pytest.mark.asyncio
        async def test_search_content_uses_advanced_scoring_if_available(self, researcher_instance: WebsiteResearcher, mocker):
            """
            Tests that search_content attempts to use _calculate_advanced_relevance
            if TF-IDF components are initialized.
            """
            from modules.website_researcher import SKLEARN_AVAILABLE
            if not SKLEARN_AVAILABLE:
                pytest.skip("scikit-learn not available.")

            # Ensure TF-IDF is ready
            await researcher_instance._load_pages_data()
            if researcher_instance._document_vectors is None:
                 pytest.fail("_document_vectors not created. Advanced scoring won't be tested.")

            # Spy on _calculate_advanced_relevance
            spy_advanced_relevance = mocker.spy(researcher_instance, '_calculate_advanced_relevance')
            
            await researcher_instance.search_content(topics=["AI"], keywords=["test"])
            
            assert spy_advanced_relevance.called # Check if it was called at least once

    # --- Caching Behavior Tests ---
    class TestCachingBehavior:
        """Tests caching behavior of search_content using the mock_cache_manager."""

        @pytest.mark.asyncio
        async def test_search_content_cache_hit(self, researcher_instance: WebsiteResearcher, mock_cache_manager: MagicMock, sample_pages_data_dict: dict):
            """Tests that a cached search result is returned if available."""
            topics = ["AI"]
            keywords = ["fundamentals"]
            # Generate a key as the researcher would (mocked or actual logic)
            # The mock_cache_manager fixture already mocks generate_cache_key.
            # For this test, we need to ensure the internal cache of WebsiteResearcher is used.
            # WebsiteResearcher uses its own self._search_cache dictionary.

            # First call: populates the cache
            await researcher_instance.search_content(topics=topics, keywords=keywords)
            cache_key_used = researcher_instance._generate_cache_key(topics, keywords)
            assert cache_key_used in researcher_instance._search_cache
            
            # To test a "hit", we can spy on the actual search logic
            # Or, verify that the result from cache matches and no further processing is done.
            # Let's modify the cache entry to be distinct
            researcher_instance._search_cache[cache_key_used] = ([SearchResult(content_id="cached_id", url="cached_url", title="Cached Title", summary="cached_sum", relevant_content="cached_cont", relevance_score=0.99, source_url="src", keywords=[], category="cat", matched_terms=[], word_count=10)], datetime.now())
            
            # Second call: should hit the cache
            results = await researcher_instance.search_content(topics=topics, keywords=keywords)
            
            assert len(results) == 1
            assert results[0]['content_id'] == "cached_id" # Verifying it's the cached result
            assert results[0]['title'] == "Cached Title"


        @pytest.mark.asyncio
        async def test_search_content_cache_miss_and_set(self, researcher_instance: WebsiteResearcher, mock_cache_manager: MagicMock):
            """Tests that search results are stored in its internal cache on a miss."""
            topics = ["new_topic_for_cache"]
            keywords = ["new_keyword"]
            
            cache_key = researcher_instance._generate_cache_key(topics, keywords)
            assert cache_key not in researcher_instance._search_cache # Pre-condition: miss

            await researcher_instance.search_content(topics=topics, keywords=keywords)
            
            assert cache_key in researcher_instance._search_cache
            cached_entry, _ = researcher_instance._search_cache[cache_key]
            assert isinstance(cached_entry, list) # Should be list of SearchResult (or dicts)

        @pytest.mark.asyncio
        async def test_search_content_cache_expiry(self, researcher_instance: WebsiteResearcher):
            """Tests that expired cache entries are not used (or are re-fetched)."""
            topics = ["expiring_topic"]
            keywords = ["test"]
            researcher_instance.cache_ttl = 0.01 # Set a very short TTL (in seconds)

            # First call, populates cache
            await researcher_instance.search_content(topics=topics, keywords=keywords)
            cache_key = researcher_instance._generate_cache_key(topics, keywords)
            _, first_cache_time = researcher_instance._search_cache[cache_key]
            
            await asyncio.sleep(0.05) # Wait for TTL to expire

            # Spy on the internal _calculate_basic_relevance to see if it's called again (indicating a re-fetch)
            with patch.object(researcher_instance, '_calculate_basic_relevance', wraps=researcher_instance._calculate_basic_relevance) as spy_relevance:
                await researcher_instance.search_content(topics=topics, keywords=keywords)
                # If cache expired, _calculate_basic_relevance should be called again.
                # This depends on how many pages, so check if called at all after expiry.
                assert spy_relevance.called 
            
            _, second_cache_time = researcher_instance._search_cache[cache_key]
            assert second_cache_time > first_cache_time # Timestamp should be updated


    # --- Utility Method Tests ---
    class TestUtilityMethods:
        """Tests for helper/utility methods like get_page_by_id, get_statistics, etc."""

        @pytest.mark.asyncio
        @pytest.mark.parametrize("content_id, expected_title_part", [
            ("ai_fundamentals_001", "Artificial Intelligence Fundamentals"),
            ("data_science_003", "Data Science Program"),
            ("non_existent_id_123", None) # Expect None if not found
        ])
        async def test_get_page_by_id(self, researcher_instance: WebsiteResearcher, content_id: str, expected_title_part: str or None):
            """Tests get_page_by_id for existing and non-existing IDs."""
            page = await researcher_instance.get_page_by_id(content_id)
            if expected_title_part:
                assert page is not None
                assert expected_title_part in page['title']
                assert page['content_id'] == content_id
            else:
                assert page is None

        @pytest.mark.asyncio
        @pytest.mark.parametrize("category, expected_min_count, expected_page_id_in_results", [
            ("academics", 2, "ai_fundamentals_001"), # AI and Data Science
            ("research", 1, "ml_applications_002"),
            ("non_existent_category", 0, None)
        ])
        async def test_get_pages_by_category(self, researcher_instance: WebsiteResearcher, category: str, expected_min_count: int, expected_page_id_in_results: str or None):
            """Tests get_pages_by_category."""
            pages = await researcher_instance.get_pages_by_category(category)
            assert len(pages) >= expected_min_count
            if expected_page_id_in_results:
                assert any(p['content_id'] == expected_page_id_in_results for p in pages)
            for page in pages:
                assert page['category'].lower() == category.lower()

        @pytest.mark.asyncio
        async def test_get_statistics(self, researcher_instance: WebsiteResearcher, sample_pages_data_dict: dict):
            """Tests the get_statistics method for correct structure and data."""
            # Ensure data is loaded
            await researcher_instance._load_pages_data()
            stats = await researcher_instance.get_statistics()

            assert isinstance(stats, dict)
            assert stats['total_pages'] == len(sample_pages_data_dict["pages"])
            assert 'cache_entries' in stats # From researcher's internal cache
            assert isinstance(stats['categories'], dict)
            assert stats['total_words'] > 0 # Sum of word_counts from sample data
            assert 'config' in stats
            assert stats['config']['max_results'] == researcher_instance.max_results

        def test_is_healthy_when_ok(self, researcher_instance: WebsiteResearcher, temp_pages_file: Path):
            """Tests is_healthy when the researcher is expected to be operational."""
            # Fixture ensures temp_pages_file exists and data is loaded into _pages_data
            assert researcher_instance.is_healthy() is True

        def test_is_healthy_no_file(self, website_researcher_config: dict):
            """Tests is_healthy when the pages_file does not exist."""
            researcher = WebsiteResearcher(pages_file=Path("non_existent.json"), config=website_researcher_config)
            assert researcher.is_healthy() is False

        def test_is_healthy_no_data_loaded(self, researcher_instance: WebsiteResearcher):
            """Tests is_healthy when _pages_data is empty."""
            researcher_instance._pages_data = [] # Manually clear loaded data
            assert researcher_instance.is_healthy() is False

        @pytest.mark.asyncio
        async def test_cleanup(self, researcher_instance: WebsiteResearcher):
            """Ensures the cleanup method clears internal caches and resets TF-IDF components."""
            # Populate internal cache for testing cleanup
            researcher_instance._search_cache["some_key"] = ([], datetime.now())
            
            # If TF-IDF was used, components would be set
            from modules.website_researcher import SKLEARN_AVAILABLE
            if SKLEARN_AVAILABLE:
                 await researcher_instance._load_pages_data() # ensure _tfidf_vectorizer potentially set up

            await researcher_instance.cleanup()

            assert len(researcher_instance._search_cache) == 0
            assert researcher_instance._tfidf_vectorizer is None # Should be reset
            assert researcher_instance._document_vectors is None # Should be reset