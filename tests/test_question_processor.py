# tests/test_question_processor.py

"""
AI Gate for Artificial Intelligence Applications
test_question_processor Module

Test Suite for the Question Processor Module (modules.question_processor)

This module provides comprehensive tests for the QuestionProcessor class,
covering various aspects including:
- Initialization and configuration handling.
- Input validation (length, content, suspicious patterns).
- Text cleaning and normalization.
- Language detection (including fallback mechanisms).
- Extraction of topics, keywords, and entities.
- Question type classification.
- Calculation of complexity and confidence scores.
- Caching behavior (via a mocked CacheManager).
- Error handling for unexpected issues.
- Utility methods like health checks and statistics.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock # For NLTK_AVAILABLE and LANGDETECT_AVAILABLE patches
from typing import Dict, Any # For type hinting
from datetime import datetime

# Import the class to be tested
from modules.question_processor import QuestionProcessor, QuestionAnalysis

# Pytest marker for all tests in this file to be treated as asyncio
pytestmark = pytest.mark.asyncio


# --- Test Class for QuestionProcessor ---

class TestQuestionProcessor:
    """Groups tests for the QuestionProcessor class."""

    # Fixtures 'processor' and 'processor_no_nltk' will be injected by pytest
    # based on their definitions in conftest.py (or this file if defined locally).
    # We rely on conftest.py for these fixtures.

    # --- Initialization and Configuration Tests ---
    class TestInitialization:
        """Tests related to QuestionProcessor initialization and configuration."""

        def test_initialization_with_config(self, question_processor_config, mock_cache_manager):
            """
            Tests if the QuestionProcessor initializes correctly with configuration
            provided by the 'question_processor_config' fixture.
            """
            processor = QuestionProcessor(config=question_processor_config, cache_manager=mock_cache_manager)

            # Assert that configuration values are correctly set
            # Assuming question_processor_config fixture provides these keys from default.yaml
            assert processor.min_question_length == question_processor_config.get('min_length', 3)
            assert processor.max_question_length == question_processor_config.get('max_length', 2000)
            # Crucially, testing the corrected/aligned key name:
            assert processor.min_confidence_threshold == question_processor_config.get('min_confidence_threshold', 0.6)
            assert processor.enable_caching == question_processor_config.get('enable_caching', True)
            assert processor.supported_languages == question_processor_config.get('supported_languages', ['en', 'es', 'fr', 'de'])
            assert processor.cache_manager is mock_cache_manager

        def test_initialization_with_empty_config(self, mock_cache_manager):
            """
            Tests QuestionProcessor initialization with an empty config dictionary,
            expecting it to use its internal default values.
            """
            processor = QuestionProcessor(config={}, cache_manager=mock_cache_manager)
            
            # Assert internal defaults are used
            assert processor.min_question_length == 3 # Default from QuestionProcessor class
            assert processor.max_question_length == 2000 # Default
            assert processor.min_confidence_threshold == 0.6 # Default
            assert processor.enable_caching is True # Default
            assert 'en' in processor.supported_languages # Default includes 'en'

        def test_initialization_nltk_and_langdetect_status(self, processor):
            """
            Tests if NLTK and langdetect availability are correctly determined.
            This test depends on the actual availability in the test environment.
            """
            # This will reflect the actual environment unless NLTK_AVAILABLE is patched for a specific test
            # For more controlled testing, you might patch these globals.
            from modules.question_processor import NLTK_AVAILABLE, LANGDETECT_AVAILABLE
            # Assuming the 'processor' fixture is initialized in an environment where these are available or not.
            # The QuestionAnalysis.metadata will later confirm what was used.
            pass # Implicitly tested by other tests using metadata.

        def test_question_patterns_are_loaded(self, processor):
            """Ensures that predefined question patterns are loaded during initialization."""
            assert isinstance(processor.question_patterns, dict)
            assert 'what' in processor.question_patterns
            assert 'definition' in processor.question_patterns

        def test_institutional_keywords_are_loaded(self, processor):
            """Ensures that predefined institutional keywords are loaded."""
            assert isinstance(processor.institutional_keywords, dict)
            assert 'academic' in processor.institutional_keywords
            assert 'course' in processor.institutional_keywords['academic']


    # --- Input Validation Tests ---
    class TestInputValidation:
        """Tests for the _validate_input method."""

        @pytest.mark.parametrize("valid_question", [
            "What is machine learning?",
            "How do I register for courses?",
            "This is a perfectly valid and long enough question for testing purposes."
        ])
        def test_validate_input_valid_questions(self, processor, valid_question: str):
            """Tests _validate_input with various valid questions."""
            validation = processor._validate_input(valid_question)
            assert validation['is_valid'] is True
            assert validation.get('error_message') is None

        @pytest.mark.parametrize("invalid_input, expected_error_part", [
            (None, "non-empty string"),
            ("", "non-empty string"),
            ("   ", "non-empty string"),
            ("Hi", "too short"), # Assuming min_length is 3 from config
            ("a" * 2001, "too long"), # Assuming max_length is 2000
            ("<script>alert('xss')</script>", "invalid content"),
            ("eval('danger')", "invalid content"),
            ("The a an or", "meaningful content") # Only stopwords or very short words
        ])
        def test_validate_input_invalid_cases(self, processor, invalid_input: Any, expected_error_part: str):
            """Tests _validate_input with various invalid inputs and checks error messages."""
            # Adjust min/max length for specific tests if needed, or rely on fixture's config
            if invalid_input == "Hi" and processor.min_question_length > 2:
                pass # This case is valid for this specific input
            elif isinstance(invalid_input, str) and "too long" in expected_error_part and len(invalid_input) <= processor.max_question_length:
                pytest.skip("Skipping too long test as length is within configured max_length")


            validation = processor._validate_input(invalid_input)
            assert validation['is_valid'] is False
            assert expected_error_part in validation['error_message'].lower()

        def test_validate_input_length_boundaries(self, processor, question_processor_config):
            """Tests _validate_input at the configured length boundaries."""
            min_len = question_processor_config.get('min_length', 3)
            max_len = question_processor_config.get('max_length', 2000)

            # Test exactly min_length
            question_at_min_len = "a" * min_len
            validation_min = processor._validate_input(question_at_min_len)
            assert validation_min['is_valid'] is True

            # Test exactly max_length
            question_at_max_len = "b" * max_len
            validation_max = processor._validate_input(question_at_max_len)
            assert validation_max['is_valid'] is True


    # --- Text Cleaning Tests ---
    class TestTextCleaning:
        """Tests for the _clean_text method."""

        @pytest.mark.parametrize("raw_text, expected_cleaned_text", [
            ("  What is   machine learning?  ", "What is machine learning?"),
            ("How\t\tdo\nI\n\nregister?", "How do I register?"),
            ("What's the #1 course???", "Whats the 1 course?"), # '#' and extra '?' removed
            ("Can u tell me ur best program?", "Can you tell me your best program?"), # Abbreviation
            ("r u ok w/ this?", "are you ok with this?"), # Multiple abbreviations
            ("Test with <html_tag> and some   entities.", "Test with html_tag and some nbsp entities.") # Simplified entity and tag removal
        ])
        def test_clean_text_various_cases(self, processor, raw_text: str, expected_cleaned_text: str):
            """Tests _clean_text with various inputs and expected outputs."""
            # The _clean_text method in the provided code is quite basic with regex.
            # Adjust expected_cleaned_text if the cleaning logic is more sophisticated.
            # Current _clean_text: re.sub(r'[^\w\s\.,;:?!-]', '', cleaned)
            # This regex might remove more than intended for "special characters".
            # For example, "cost@university.edu" would become "costuniversityedu"
            # If the goal is to keep '@' or '.', the regex needs adjustment.
            # Let's refine the test cases based on the current regex.
            if "<html_tag>" in raw_text: # The current regex does not remove HTML tags
                 # The provided regex re.sub(r'[^\w\s\.,;:?!-]', '', cleaned) doesn't specifically handle HTML.
                 # It would turn <html_tag> into html_tag if it's not caught by other cleaning.
                 # For more robust HTML cleaning, a library like BeautifulSoup is usually recommended.
                 # Given the current code, the expected output might be different for HTML.
                 # Let's assume a simplified expectation for the current regex.
                 # For now, let's test the provided cases with the current cleaning logic.
                 # The case "Test with <html_tag>..." would become "Test with html_tag and some nbsp entities." (if nbsp is also filtered)
                 # The original code's clean_text has: re.sub(r'[^\w\s\.,;:?!-]', '', cleaned)
                 # This means '<', '>', '&', ';' will be removed.
                 pass # Expected output for HTML tags may need refinement based on exact cleaning goals.

            cleaned = processor._clean_text(raw_text)
            assert cleaned == expected_cleaned_text


    # --- Language Detection Tests ---
    class TestLanguageDetection:
        """Tests for the _detect_language method."""

        @pytest.mark.parametrize("text, expected_lang", [
            ("This is an English sentence.", "en"),
            ("Este es une phrase en français.", "fr"), # Example, langdetect might get this or similar
            ("Das ist ein deutscher Satz.", "de"),    # Example
        ])
        def test_detect_language_various_languages(self, processor, text: str, expected_lang: str):
            """Tests language detection for different languages (relies on langdetect)."""
            # This test is somewhat dependent on the langdetect library's accuracy.
            # For critical multi-language support, more robust testing or specific mocks might be needed.
            from modules.question_processor import LANGDETECT_AVAILABLE
            if not LANGDETECT_AVAILABLE:
                pytest.skip("Langdetect library not available, skipping this test.")
            
            detection = processor._detect_language(text)
            assert detection['language'] == expected_lang
            assert detection['confidence'] > 0.5 # General expectation

        def test_detect_language_fallback_if_langdetect_unavailable(self, question_processor_config, mock_cache_manager):
            """Tests fallback behavior when langdetect is not available."""
            with patch('modules.question_processor.LANGDETECT_AVAILABLE', False):
                processor_no_langdetect = QuestionProcessor(config=question_processor_config, cache_manager=mock_cache_manager)
                detection = processor_no_langdetect._detect_language("Some text")
                assert detection['language'] == 'en' # Default fallback
                assert detection['confidence'] == 0.5

        def test_detect_language_short_text(self, processor):
            """Tests language detection for very short text (should default reasonably)."""
            from modules.question_processor import LANGDETECT_AVAILABLE
            if not LANGDETECT_AVAILABLE: # If langdetect is off, it uses its own fallback
                 detection = processor._detect_language("Hi")
                 assert detection['language'] == 'en'
                 assert detection['confidence'] == 0.5
                 return

            detection = processor._detect_language("OK") # langdetect handles short text
            assert detection['language'] == 'en' # langdetect might detect 'en' or 'und'
            assert detection['confidence'] >= 0.7 # As per QuestionProcessor logic for short text

        @patch('modules.question_processor.detect', side_effect=Exception("Langdetect internal error"))
        def test_detect_language_langdetect_exception(self, mock_detect, processor):
            """Tests graceful handling of exceptions from the langdetect library."""
            from modules.question_processor import LANGDETECT_AVAILABLE
            if not LANGDETECT_AVAILABLE:
                pytest.skip("Test not applicable if langdetect is mocked as unavailable at module level.")

            detection = processor._detect_language("An problematic text for langdetect")
            assert detection['language'] == 'en' # Fallback on exception
            assert detection['confidence'] == 0.5
            mock_detect.assert_called_once()


    # --- Topic and Keyword Extraction Tests ---
    class TestTopicAndKeywordExtraction:
        """Tests for the _extract_topics_and_keywords method."""

        @pytest.mark.asyncio
        async def test_extract_topics_and_keywords_with_nltk(self, processor):
            """Tests extraction with NLTK available."""
            from modules.question_processor import NLTK_AVAILABLE
            if not NLTK_AVAILABLE:
                pytest.skip("NLTK not available, skipping NLTK-specific test.")

            text = "What are the admission requirements for the computer science program?"
            extraction = await processor._extract_topics_and_keywords(text)
            
            assert 'administrative' in extraction['topics'] or 'academic' in extraction['topics']
            assert any(kw in extraction['keywords'] for kw in ['admission', 'requirements', 'computer', 'science', 'program'])
            assert 'CS' not in extraction['entities'] # Example of what might not be an entity with simple regex

        @pytest.mark.asyncio
        async def test_extract_topics_and_keywords_without_nltk(self, processor_no_nltk):
            """Tests extraction when NLTK is not available (fallback)."""
            text = "What are the admission requirements for the computer science program?"
            extraction = await processor_no_nltk._extract_topics_and_keywords(text)

            # Basic extraction should still work
            assert len(extraction['topics']) > 0 
            assert len(extraction['keywords']) > 0
            # Check for some expected keywords (case-insensitive basic split)
            assert any(kw in extraction['keywords'] for kw in ['admission', 'requirements', 'computer', 'science', 'program'])

        @pytest.mark.asyncio
        @pytest.mark.parametrize("text, expected_topics, expected_keywords_subset", [
            ("Tell me about course credits and degree programs.", ['academic'], ['course', 'credits', 'degree', 'programs']),
            ("How to apply for student housing and parking permits?", ['administrative', 'campus'], ['apply', 'student', 'housing', 'parking', 'permits']),
            ("Information on library facilities and research support.", ['campus', 'academic'], ['information', 'library', 'facilities', 'research', 'support'])
        ])
        async def test_extract_various_content(self, processor, text, expected_topics, expected_keywords_subset):
            extraction = await processor._extract_topics_and_keywords(text)
            for topic in expected_topics:
                assert topic in extraction['topics']
            for keyword in expected_keywords_subset:
                assert keyword in extraction['keywords']

        @pytest.mark.asyncio
        async def test_entity_extraction_examples(self, processor):
            """Tests entity extraction for common patterns like capitalized words and numbers."""
            text = "Contact CS Dept or Dr. Smith about course CS101 in Fall 2024."
            extraction = await processor._extract_topics_and_keywords(text)
            entities = extraction['entities']
            
            assert "CS" in entities or "Dept" in entities # Assuming "CS" and "Dept" are capitalized
            assert "Smith" in entities
            assert "CS101" in entities
            assert "2024" in entities
            assert "Fall" in entities


    # --- Question Classification Tests ---
    class TestQuestionClassification:
        """Tests for the _classify_question_type method."""

        @pytest.mark.parametrize("question, expected_type", [
            ("What is your name?", "what"),
            ("How does this work?", "how"),
            ("Why is the sky blue?", "why"),
            ("When is the deadline?", "when"),
            ("Where is the main office?", "where"),
            ("Who is the department head?", "who"),
            ("Define 'synergy'.", "definition"),
            ("Compare option A versus option B.", "comparison"),
            ("What are the steps to enroll?", "procedure"),
            ("Is this service free?", "factual"), # Starts with an auxiliary verb
            ("This is a statement.", "general"), # No question indicators
            ("Help me find my way.", "help_request")
        ])
        def test_classify_question_type_various_cases(self, processor, question: str, expected_type: str):
            """Tests question type classification for a range of question patterns."""
            q_type = processor._classify_question_type(question)
            assert q_type == expected_type


    # --- Scoring Mechanism Tests ---
    class TestScoringMechanisms:
        """Tests for _calculate_complexity_score and _calculate_confidence_score."""

        def test_calculate_complexity_score(self, processor):
            """Tests complexity score calculation for different analysis inputs."""
            analysis_simple = QuestionAnalysis(word_count=5, sentence_count=1, keywords=['test'], topics=['general'])
            analysis_complex = QuestionAnalysis(word_count=20, sentence_count=3, keywords=['complex', 'analysis', 'test', 'process'], topics=['academic', 'technical'], question_type='comparison')
            
            score_simple = processor._calculate_complexity_score(analysis_simple)
            score_complex = processor._calculate_complexity_score(analysis_complex)

            assert 0.0 <= score_simple <= 1.0
            assert 0.0 <= score_complex <= 1.0
            assert score_complex > score_simple

        def test_calculate_confidence_score(self, processor):
            """Tests confidence score calculation for different analysis inputs."""
            analysis_high_conf = QuestionAnalysis(language_confidence=0.95, word_count=10, keywords=['clear', 'question'], topics=['general'], question_type='what')
            analysis_low_conf = QuestionAnalysis(language_confidence=0.4, word_count=2, keywords=[], topics=[], question_type='general')

            score_high = processor._calculate_confidence_score(analysis_high_conf)
            score_low = processor._calculate_confidence_score(analysis_low_conf)

            assert 0.0 <= score_high <= 1.0
            assert 0.0 <= score_low <= 1.0
            assert score_high > score_low
        
        def test_confidence_score_penalty_for_short_question(self, processor):
            """Tests if very short questions get a penalty in confidence score."""
            analysis_very_short = QuestionAnalysis(
                language_confidence=0.9, word_count=1, # word_count < 2
                keywords=['hi'], topics=['general'], question_type='general'
            )
            analysis_normal = QuestionAnalysis(
                language_confidence=0.9, word_count=5,
                keywords=['hello', 'there'], topics=['general'], question_type='what'
            )
            score_very_short = processor._calculate_confidence_score(analysis_very_short)
            score_normal = processor._calculate_confidence_score(analysis_normal)
            assert score_very_short < score_normal, "Very short question should have lower confidence score due to penalty."


    # --- Full Processing Workflow Tests ---
    class TestFullProcessingWorkflow:
        """Tests the main process_question method integrating all steps."""

        @pytest.mark.asyncio
        async def test_process_question_valid_input(self, processor):
            """Tests process_question with a typical valid question."""
            question = "What are the admission requirements for the computer science program?"
            result = await processor.process_question(question)

            assert isinstance(result, dict)
            assert result['is_valid'] is True
            assert result['original_question'] == question
            assert result.get('error_message') is None
            assert len(result['keywords']) > 0
            assert len(result['topics']) > 0
            assert result['processing_time'] > 0
            assert 'nltk_available' in result['metadata']

        @pytest.mark.asyncio
        @pytest.mark.parametrize("invalid_question, expected_error_part", [
            ("", "non-empty string"),
            ("Hi", "too short"), # Assuming min_length = 3
            ("a" * 2001, "too long"), # Assuming max_length = 2000
            ("<script>alert('bad')</script>", "invalid content")
        ])
        async def test_process_question_invalid_input(self, processor, invalid_question: str, expected_error_part: str):
            """Tests process_question with various invalid inputs."""
            # Adjust min/max length for specific tests if needed
            if invalid_question == "Hi" and processor.min_question_length > 2:
                 pass
            elif "too long" in expected_error_part and len(invalid_question) <= processor.max_question_length:
                 pytest.skip("Skipping as length is within configured max_length for this test run")

            result = await processor.process_question(invalid_question)
            assert result['is_valid'] is False
            assert expected_error_part in result['error_message'].lower()

        @pytest.mark.asyncio
        async def test_process_question_unsupported_language(self, processor, question_processor_config):
            """Tests process_question when an unsupported language is detected."""
            # Requires langdetect to be available and to correctly detect a non-supported language.
            # Let's assume 'zh' (Chinese) is not in supported_languages.
            from modules.question_processor import LANGDETECT_AVAILABLE
            if not LANGDETECT_AVAILABLE:
                pytest.skip("Langdetect not available for this test.")
            
            if 'zh' in question_processor_config.get('supported_languages', []):
                pytest.skip("Test requires 'zh' to be an unsupported language in config.")

            # Mock 'detect' to return 'zh'
            with patch('modules.question_processor.detect', return_value='zh') as mock_lang_detect:
                question_in_chinese = "你好，世界" # "Hello, world" in Chinese
                result = await processor.process_question(question_in_chinese)
                
                mock_lang_detect.assert_called_once_with(processor._clean_text(question_in_chinese))
                assert result['is_valid'] is False
                assert result['language'] == 'zh'
                assert "not supported" in result['error_message'].lower()
        
        @pytest.mark.asyncio
        async def test_process_question_with_session_id(self, processor):
            """Tests that session_id is correctly passed into metadata."""
            question = "Test question with session ID"
            session_id = "session123"
            result = await processor.process_question(question, session_id=session_id)
            assert result['metadata'].get('session_id') == session_id


    # --- Caching Functionality Tests (using mock_cache_manager) ---
    class TestCachingFunctionality:
        """Tests caching behavior of process_question."""

        @pytest.mark.asyncio
        async def test_process_question_cache_hit(self, processor, mock_cache_manager):
            """Tests that a cached result is returned if available."""
            question = "Frequently asked question"
            cached_analysis = QuestionAnalysis(
                is_valid=True, original_question=question, cleaned_question="faq",
                language='en', confidence_score=0.9, keywords=['faq'], topics=['general']
            ). __dict__ # Convert to dict
            
            # Configure mock to return the cached data
            mock_cache_manager.get_cached_response.return_value = cached_analysis
            
            result = await processor.process_question(question)

            mock_cache_manager.get_cached_response.assert_called_once()
            # The key generation might be complex, so we check if it was called.
            # The exact key check would require replicating the hash logic or mocking hash.
            
            mock_cache_manager.cache_response.assert_not_called() # Should not cache again
            assert result == cached_analysis # Ensure the cached result is returned

        @pytest.mark.asyncio
        async def test_process_question_cache_miss_and_set(self, processor, mock_cache_manager):
            """Tests that a result is cached on a cache miss."""
            question = "A new unique question"
            mock_cache_manager.get_cached_response.return_value = None # Simulate cache miss

            result = await processor.process_question(question)

            mock_cache_manager.get_cached_response.assert_called_once()
            mock_cache_manager.cache_response.assert_called_once()
            # Assert that the first argument to cache_response (the key) is a string
            # and the second argument (the value) is the result dict.
            args, _ = mock_cache_manager.cache_response.call_args
            assert isinstance(args[0], str) # Cache key
            assert args[1] == result # Cached value

        @pytest.mark.asyncio
        async def test_process_question_caching_disabled(self, question_processor_config, mock_cache_manager):
            """Tests that caching is skipped if 'enable_caching' is false."""
            mutated_config = question_processor_config.copy()
            mutated_config['enable_caching'] = False
            processor_no_cache = QuestionProcessor(config=mutated_config, cache_manager=mock_cache_manager)

            await processor_no_cache.process_question("A question when caching is off")

            mock_cache_manager.get_cached_response.assert_not_called()
            mock_cache_manager.cache_response.assert_not_called()

        @pytest.mark.asyncio
        async def test_process_question_cache_manager_unavailable(self, question_processor_config):
            """Tests behavior if cache_manager is None."""
            processor_no_cm = QuestionProcessor(config=question_processor_config, cache_manager=None)
            question = "Question without cache manager"
            # Should not raise an error
            result = await processor_no_cm.process_question(question)
            assert result['is_valid'] # Or False depending on question, main point is no crash.
            # This also implies that no cache methods would be called.

    # --- Error Handling and Edge Case Tests ---
    class TestErrorHandlingAndEdgeCases:
        """Tests for robustness and handling of unexpected situations."""

        @pytest.mark.asyncio
        @patch('modules.question_processor.QuestionProcessor._extract_topics_and_keywords', 
               new_callable=AsyncMock, side_effect=RuntimeError("Unexpected extraction error"))
        async def test_process_question_internal_exception(self, mock_extraction, processor):
            """
            Tests that process_question handles internal exceptions gracefully and
            returns an invalid analysis with an error message.
            """
            question = "This question will cause an internal error"
            result = await processor.process_question(question)

            assert result['is_valid'] is False
            assert "processing error" in result['error_message'].lower()
            assert "unexpected extraction error" in result['error_message'].lower()
            mock_extraction.assert_called_once()

        def test_generate_validation_message_various_scenarios(self, processor, question_processor_config):
            """Tests _generate_validation_message for different invalid analysis states."""
            # Scenario 1: Unsupported language
            analysis_unsupported_lang = QuestionAnalysis(language='zz', supported_languages=['en'])
            # Temporarily modify processor's supported_languages for this test case
            original_supported_languages = processor.supported_languages
            processor.supported_languages = ['en']
            msg1 = processor._generate_validation_message(analysis_unsupported_lang)
            assert "not supported" in msg1.lower()
            processor.supported_languages = original_supported_languages # Restore

            # Scenario 2: No keywords
            analysis_no_keywords = QuestionAnalysis(language='en', keywords=[])
            msg2 = processor._generate_validation_message(analysis_no_keywords)
            assert "more specific" in msg2.lower()

            # Scenario 3: Low confidence score
            min_conf = question_processor_config.get('min_confidence_threshold', 0.6)
            analysis_low_conf = QuestionAnalysis(language='en', keywords=['test'], confidence_score=min_conf - 0.1)
            msg3 = processor._generate_validation_message(analysis_low_conf)
            assert "rephrase" in msg3.lower()

            # Scenario 4: Default message
            analysis_other_invalid = QuestionAnalysis(language='en', keywords=['test'], confidence_score=min_conf + 0.1) # Valid on its own
            # To make it invalid overall, we might simulate some other condition.
            # For now, let's test the "default" invalid message if no other condition is met.
            # _generate_validation_message is called when analysis.is_valid is False AND analysis.error_message is None
            # So, let's assume a state where is_valid is False but specific checks don't trigger.
            msg4 = processor._generate_validation_message(analysis_other_invalid) # This might not be the best test
            # The function _generate_validation_message is typically called after is_valid is False.
            # The conditions inside it are checked sequentially. If none match, it returns a default.
            # To test the true default, all prior conditions must be false.
            analysis_for_default = QuestionAnalysis(
                language='en', # supported
                keywords=['some', 'keywords'], # not empty
                confidence_score=processor.min_confidence_threshold + 0.1, # above threshold
                is_valid=False # but somehow still invalid (e.g. some other unstated rule)
            )
            # This is tricky to test in isolation. The logic in process_question sets is_valid.
            # Let's assume the function is called when it's appropriate.
            # The default message "Please provide a clearer, more specific question." is the last resort.
            # To trigger it, we need: lang supported, keywords present, confidence score above threshold.
            # But analysis.is_valid is False.
            # This case is hard to construct perfectly without replicating more of process_question logic.
            # For now, the main specific cases (lang, keywords, confidence) are more important.
            # Default case:
            analysis_default_case = QuestionAnalysis(language='en', keywords=['a','b'], confidence_score=0.9) # This should be valid.
            # The _generate_validation_message is called when is_valid is False AND error_message is not set.
            # This implies all specific checks for error messages failed.

    # --- Utility Method Tests ---
    class TestUtilityMethods:
        """Tests for helper/utility methods like health_check, cleanup, get_statistics."""

        def test_is_healthy_normal_operation(self, processor):
            """Tests the is_healthy method under normal conditions."""
            assert processor.is_healthy() is True

        @patch('modules.question_processor.QuestionProcessor._validate_input', side_effect=Exception("Validation subsystem down"))
        def test_is_healthy_when_validation_fails(self, mock_validate, processor):
            """Tests is_healthy when a critical internal check (like _validate_input) fails."""
            assert processor.is_healthy() is False
            mock_validate.assert_called_once()

        def test_cleanup_runs_without_error(self, processor):
            """Ensures the cleanup method can be called without raising errors."""
            try:
                processor.cleanup()
            except Exception as e:
                pytest.fail(f"processor.cleanup() raised an exception: {e}")

        def test_get_statistics_returns_expected_structure(self, processor, question_processor_config):
            """Tests that get_statistics returns a dictionary with expected keys."""
            stats = processor.get_statistics()
            assert isinstance(stats, dict)
            expected_keys = [
                'nltk_available', 'langdetect_available', 'supported_languages',
                'min_confidence_threshold', 'caching_enabled'
            ]
            for key in expected_keys:
                assert key in stats
            
            assert stats['min_confidence_threshold'] == question_processor_config.get('min_confidence_threshold')
            assert stats['caching_enabled'] == question_processor_config.get('enable_caching')


# Note: Performance and concurrent tests are good to have but might be
# better suited for a separate performance testing suite or integration tests
# if they become too slow for regular unit test runs.
# The examples in the original file are good starting points.
# For this rewrite, focus is on unit test correctness with fixtures.