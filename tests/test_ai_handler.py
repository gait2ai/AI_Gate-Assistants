# tests/test_ai_handler.py

"""
AI Gate for Artificial Intelligence Applications
test_ai_handler Module

Test Suite for the AI Handler Module (modules.ai_handler)

This module provides comprehensive tests for the AIHandler class, ensuring
its robustness in managing communication with AI models via OpenRouter.
It covers aspects including:
- Initialization with various configurations.
- Model selection and fallback logic.
- API request construction and execution (mocked).
- Response validation and formatting.
- Error handling (API errors, timeouts, rate limits).
- Caching behavior (via a mocked CacheManager).
- Statistics tracking and health checks.
- Session management and cleanup.
"""

import pytest
import asyncio
import os
import json
from unittest.mock import patch, AsyncMock, MagicMock, call
from datetime import datetime, timedelta

# Import the class to be tested
from modules.ai_handler import AIHandler, ModelInfo, ModelStatus, RequestMetrics

# Pytest marker for all tests in this file to be treated as asyncio
pytestmark = pytest.mark.asyncio


# --- Main Test Class for AIHandler ---

class TestAIHandler:
    """Groups tests for the AIHandler class."""

    @pytest.fixture
    def ai_handler_instance(self, ai_handler_config: dict, mock_cache_manager: MagicMock) -> AIHandler:
        """
        Provides an AIHandler instance initialized with configuration from conftest.py
        and a mock cache manager. OPENROUTER_API_KEY is also patched.
        """
        # Patch a test API key for initialization
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test_api_key_for_ai_handler'}):
            handler = AIHandler(config=ai_handler_config, cache_manager=mock_cache_manager)
            # AIHandler initializes its session in __init__ if self.session is None or closed.
            # No explicit wait needed here unless __init__ itself does complex async tasks
            # that are not awaited internally, which is not the case here for session creation.
            return handler

    @pytest.fixture
    async def initialized_ai_handler(self, ai_handler_instance: AIHandler) -> AIHandler:
        """
        Provides an AIHandler instance and ensures its aiohttp session is properly
        set up for tests that make mocked API calls, and cleans up afterwards.
        """
        # _ensure_session is called before each API request.
        # For testing, we can call it explicitly if needed, but most tests will mock
        # _make_api_request or the session's post method directly.
        # The main purpose here is to ensure cleanup.
        yield ai_handler_instance
        await ai_handler_instance.cleanup() # Ensure session is closed after tests


    # --- Initialization and Configuration Tests ---
    class TestInitialization:
        """Tests related to AIHandler initialization and configuration."""

        def test_initialization_with_config(self, ai_handler_instance: AIHandler, ai_handler_config: dict, mock_cache_manager: MagicMock):
            """
            Tests if AIHandler initializes correctly with configuration from
            the 'ai_handler_config' fixture.
            """
            assert ai_handler_instance.config == ai_handler_config
            assert ai_handler_instance.cache_manager is mock_cache_manager
            assert ai_handler_instance.api_key == 'test_api_key_for_ai_handler'
            
            # Check if models are loaded based on config (primary_model, fallback_models, or models)
            expected_model_names = []
            if "primary_model" in ai_handler_config and ai_handler_config["primary_model"]:
                expected_model_names.append(ai_handler_config["primary_model"])
                if "fallback_models" in ai_handler_config:
                    expected_model_names.extend(ai_handler_config.get("fallback_models", []))
            elif "models" in ai_handler_config: # Legacy
                 expected_model_names.extend(ai_handler_config.get("models", []))
            
            if not expected_model_names: # If config was empty, check defaults
                assert len(ai_handler_instance.models) > 0 # Should have internal defaults
            else:
                assert len(ai_handler_instance.models) == len(expected_model_names)
                assert [m.name for m in ai_handler_instance.models] == expected_model_names

            assert ai_handler_instance.timeout == ai_handler_config.get('timeout', 30)

        def test_initialization_no_api_key(self, ai_handler_config: dict, mock_cache_manager: MagicMock, caplog):
            """Tests initialization when OPENROUTER_API_KEY is not set."""
            with patch.dict(os.environ, {}, clear=True): # Clear all env vars, including patched ones
                handler = AIHandler(config=ai_handler_config, cache_manager=mock_cache_manager)
                assert handler.api_key is None
                assert "OPENROUTER_API_KEY not found" in caplog.text

        def test_initialization_no_models_in_config_uses_defaults(self, mock_cache_manager: MagicMock, caplog):
            """
            Tests that AIHandler uses its internal default models if the 'models',
            'primary_model', and 'fallback_models' keys are missing from the config.
            """
            empty_model_config = {"base_url": "test_url"} # Config without any model definition
            handler = AIHandler(config=empty_model_config, cache_manager=mock_cache_manager)
            assert len(handler.models) > 0 # Check that some models are loaded
            assert "No models configured in YAML, using embedded defaults" in caplog.text or \
                   "No valid model configuration found, using embedded defaults" in caplog.text


    # --- Model Management Tests ---
    class TestModelManagement:
        """Tests for model selection, metrics, and status updates."""

        def test_get_available_model_selects_first_healthy(self, ai_handler_instance: AIHandler):
            """Tests that the first available model is selected."""
            ai_handler_instance.models[0].status = ModelStatus.AVAILABLE
            ai_handler_instance.models[1].status = ModelStatus.AVAILABLE
            
            selected_model = ai_handler_instance._get_available_model()
            assert selected_model is not None
            assert selected_model.name == ai_handler_instance.models[0].name

        def test_get_available_model_skips_failed(self, ai_handler_instance: AIHandler):
            """Tests that failed models are skipped."""
            ai_handler_instance.models[0].status = ModelStatus.FAILED
            ai_handler_instance.models[0].consecutive_failures = ai_handler_instance.max_consecutive_failures
            ai_handler_instance.models[1].status = ModelStatus.AVAILABLE
            
            selected_model = ai_handler_instance._get_available_model()
            assert selected_model is not None
            assert selected_model.name == ai_handler_instance.models[1].name

        def test_get_available_model_skips_rate_limited(self, ai_handler_instance: AIHandler):
            """Tests that rate-limited models are skipped until reset time."""
            ai_handler_instance.models[0].status = ModelStatus.RATE_LIMITED
            ai_handler_instance.models[0].rate_limit_reset = datetime.now() + timedelta(minutes=5)
            ai_handler_instance.models[1].status = ModelStatus.AVAILABLE
            
            selected_model = ai_handler_instance._get_available_model()
            assert selected_model is not None
            assert selected_model.name == ai_handler_instance.models[1].name

        def test_get_available_model_rate_limit_expired(self, ai_handler_instance: AIHandler):
            """Tests that a rate-limited model becomes available after reset time."""
            ai_handler_instance.models[0].status = ModelStatus.RATE_LIMITED
            ai_handler_instance.models[0].rate_limit_reset = datetime.now() - timedelta(minutes=1) # Expired
            
            selected_model = ai_handler_instance._get_available_model()
            assert selected_model is not None
            # It should pick the first model if its rate_limit_reset is in the past.
            assert selected_model.name == ai_handler_instance.models[0].name

        def test_get_available_model_all_unavailable_resets_first(self, ai_handler_instance: AIHandler):
            """
            Tests that if all models are marked unavailable (failed/rate-limited past max retries),
            the first model is reset and returned as an emergency fallback.
            """
            for model in ai_handler_instance.models:
                model.status = ModelStatus.FAILED
                model.consecutive_failures = ai_handler_instance.max_consecutive_failures
            
            selected_model = ai_handler_instance._get_available_model()
            assert selected_model is not None
            assert selected_model.name == ai_handler_instance.models[0].name
            assert selected_model.status == ModelStatus.AVAILABLE
            assert selected_model.consecutive_failures == 0

        def test_update_model_metrics_on_success(self, ai_handler_instance: AIHandler):
            """Tests updating model metrics after a successful API call."""
            model_to_test = ai_handler_instance.models[0]
            initial_success_count = model_to_test.successful_requests
            
            ai_handler_instance._update_model_metrics(model_to_test, success=True, response_time=0.5, error=None)
            
            assert model_to_test.successful_requests == initial_success_count + 1
            assert model_to_test.consecutive_failures == 0
            assert model_to_test.status == ModelStatus.AVAILABLE
            assert model_to_test.last_error is None

        def test_update_model_metrics_on_failure(self, ai_handler_instance: AIHandler):
            """Tests updating model metrics after a failed API call."""
            model_to_test = ai_handler_instance.models[0]
            initial_failure_count = model_to_test.failed_requests
            initial_consecutive_failures = model_to_test.consecutive_failures

            error_message = "API Error 500"
            ai_handler_instance._update_model_metrics(model_to_test, success=False, response_time=0.1, error=error_message)
            
            assert model_to_test.failed_requests == initial_failure_count + 1
            assert model_to_test.consecutive_failures == initial_consecutive_failures + 1
            assert model_to_test.last_error == error_message
            if model_to_test.consecutive_failures >= ai_handler_instance.max_consecutive_failures:
                assert model_to_test.status == ModelStatus.FAILED

        def test_reset_model_failures_specific_model(self, ai_handler_instance: AIHandler):
            """Tests resetting failure counts for a specific model."""
            model_to_reset = ai_handler_instance.models[0]
            model_to_reset.status = ModelStatus.FAILED
            model_to_reset.consecutive_failures = 5
            model_to_reset.last_error = "Previous error"

            ai_handler_instance.reset_model_failures(model_name=model_to_reset.name)
            
            assert model_to_reset.status == ModelStatus.AVAILABLE
            assert model_to_reset.consecutive_failures == 0
            assert model_to_reset.last_error is None


    # --- Request Payload Building Tests ---
    class TestRequestPayloadBuilding:
        """Tests for the _build_request_payload method."""

        def test_build_payload_structure(self, ai_handler_instance: AIHandler):
            """Tests the basic structure of the generated payload."""
            user_msg = "Hello AI"
            system_prompt = "You are helpful."
            model_name = "test-model/one"
            payload = ai_handler_instance._build_request_payload(user_msg, system_prompt, model_name)

            assert payload['model'] == model_name
            assert isinstance(payload['messages'], list)
            assert len(payload['messages']) == 2
            assert payload['messages'][0]['role'] == 'system'
            assert payload['messages'][0]['content'] == system_prompt
            assert payload['messages'][1]['role'] == 'user'
            assert payload['messages'][1]['content'] == user_msg
            assert payload['max_tokens'] == ai_handler_instance.max_tokens
            assert payload['temperature'] == ai_handler_instance.temperature

        def test_build_payload_no_system_prompt(self, ai_handler_instance: AIHandler):
            """Tests payload building when the system_prompt is empty."""
            user_msg = "Hello AI"
            payload = ai_handler_instance._build_request_payload(user_msg, "", "test-model/one")
            assert len(payload['messages']) == 1
            assert payload['messages'][0]['role'] == 'user'

        def test_build_payload_optional_params(self, ai_handler_instance: AIHandler, ai_handler_config: dict):
            """Tests inclusion of optional parameters like top_p if configured."""
            # Modify config for this test
            ai_handler_instance.config['top_p'] = 0.95
            ai_handler_instance.top_p = 0.95 # Also update the instance attribute if it's read directly

            payload = ai_handler_instance._build_request_payload("User msg", "System prompt", "model")
            assert payload.get('top_p') == 0.95

            # Cleanup: remove the modified config to not affect other tests using the same instance/config fixture
            del ai_handler_instance.config['top_p']
            ai_handler_instance.top_p = ai_handler_config.get('top_p') # Reset from original config


    # --- Response Validation and Formatting Tests ---
    class TestResponseValidationAndFormatting:
        """Tests for _validate_response and _format_response methods."""

        @pytest.mark.parametrize("response_text, is_valid_expected, error_part_expected", [
            ("This is a perfectly fine and long enough response.", True, None),
            ("", False, "empty response"),
            ("Short", False, "too short"), # Assumes min_response_length > 5
            ("I apologize, I am unable to help.", False, "unable to help"),
            ("error occurred during processing", False, "error or inability"),
            ("test test test test test test test test test test test test test test test", False, "repetitive"), # Test repetition
        ])
        def test_validate_response(self, ai_handler_instance: AIHandler, response_text: str, is_valid_expected: bool, error_part_expected: str or None):
            """Tests _validate_response with various response texts."""
            # Adjust min_length for specific test case if needed
            if response_text == "Short" and ai_handler_instance.min_response_length <= 5:
                pass # This specific case might pass if min_response_length is low

            is_valid, error_msg = ai_handler_instance._validate_response(response_text)
            assert is_valid == is_valid_expected
            if error_part_expected:
                assert error_part_expected in error_msg.lower()
            else:
                assert error_msg is None

        @pytest.mark.parametrize("raw_response, preserve_markdown, expected_formatted", [
            ("  Extra   spaces  \n\n\n  here.  ", False, "Extra spaces\n\nhere."),
            ("Markdown: **bold** and *italic*.", False, "Markdown: bold and italic."),
            ("Markdown: **bold** and *italic*.", True, "Markdown: **bold** and *italic*."),
        ])
        def test_format_response(self, ai_handler_instance: AIHandler, raw_response: str, preserve_markdown: bool, expected_formatted: str):
            """Tests _format_response with and without markdown preservation."""
            ai_handler_instance.preserve_markdown = preserve_markdown
            formatted = ai_handler_instance._format_response(raw_response)
            assert formatted == expected_formatted


    # --- Mocked API Interaction Tests ---
    class TestMockedAPIInteraction:
        """Tests the _make_api_request and generate_response methods with mocked aiohttp calls."""

        async def mock_aiohttp_post(self, status_code: int, json_response_data: dict = None, text_response: str = None, raise_exception=None):
            """Helper to create a mock aiohttp.ClientResponse."""
            mock_resp = AsyncMock(spec=aiohttp.ClientResponse)
            mock_resp.status = status_code
            if json_response_data is not None:
                mock_resp.json = AsyncMock(return_value=json_response_data)
            if text_response is not None: # For non-JSON error responses
                mock_resp.text = AsyncMock(return_value=text_response)
            if raise_exception:
                mock_resp.json.side_effect = raise_exception # Or mock_resp.text.side_effect

            # Mock the context manager part
            mock_session_post = AsyncMock()
            mock_session_post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_post.return_value.__aexit__ = AsyncMock(return_value=None)
            return mock_session_post

        @pytest.mark.asyncio
        async def test_make_api_request_successful(self, initialized_ai_handler: AIHandler):
            """Tests a successful _make_api_request call."""
            model_info = initialized_ai_handler.models[0]
            payload = {"model": model_info.name, "messages": [{"role": "user", "content": "Hi"}]}
            api_response_data = {"choices": [{"message": {"content": "Hello there!"}}], "usage": {"total_tokens": 5}}
            
            mock_post_method = await self.mock_aiohttp_post(200, api_response_data)
            
            with patch.object(initialized_ai_handler.session, 'post', mock_post_method):
                success, text, metadata = await initialized_ai_handler._make_api_request(payload, model_info)

            assert success is True
            assert text == "Hello there!"
            assert metadata['model_used'] == model_info.name
            assert metadata['tokens_used'] == 5
            mock_post_method.assert_called_once_with(initialized_ai_handler.base_url, json=payload)

        @pytest.mark.asyncio
        @pytest.mark.parametrize("status_code, error_json, expected_error_part", [
            (429, {"error": {"message": "Rate limit exceeded"}}, "rate limit exceeded"),
            (400, {"error": {"message": "Bad request"}}, "bad request"),
            (500, {"error": {"message": "Internal server error"}}, "internal server error"),
        ])
        async def test_make_api_request_http_errors(self, initialized_ai_handler: AIHandler, status_code: int, error_json: dict or None, expected_error_part: str):
            """Tests _make_api_request handling various HTTP error codes."""
            model_info = initialized_ai_handler.models[0]
            payload = {"model": model_info.name, "messages": []}
            
            mock_post_method = await self.mock_aiohttp_post(status_code, json_response_data=error_json, text_response=error_json.get("error",{}).get("message") if error_json else "Error")
            
            with patch.object(initialized_ai_handler.session, 'post', mock_post_method):
                success, text, _ = await initialized_ai_handler._make_api_request(payload, model_info)
            
            assert success is False
            assert expected_error_part in text.lower()
            if status_code == 429:
                assert model_info.status == ModelStatus.RATE_LIMITED

        @pytest.mark.asyncio
        @pytest.mark.parametrize("exception_to_raise, expected_error_part", [
            (asyncio.TimeoutError, "timeout"),
            (aiohttp.ClientConnectorError(MagicMock(), OSError("Connection failed")), "connection error"),
            (json.JSONDecodeError("msg", "doc", 0), "unexpected error"), # If parsing non-error response fails
        ])
        async def test_make_api_request_network_exceptions(self, initialized_ai_handler: AIHandler, exception_to_raise, expected_error_part: str):
            """Tests _make_api_request handling network and client-side exceptions."""
            model_info = initialized_ai_handler.models[0]
            payload = {"model": model_info.name, "messages": []}

            mock_post_method = AsyncMock(side_effect=exception_to_raise)
            
            with patch.object(initialized_ai_handler.session, 'post', mock_post_method):
                success, text, _ = await initialized_ai_handler._make_api_request(payload, model_info)
            
            assert success is False
            assert expected_error_part in text.lower()

        @pytest.mark.asyncio
        async def test_generate_response_successful_flow(self, initialized_ai_handler: AIHandler, mock_cache_manager: MagicMock):
            """Tests the full generate_response flow for a successful case."""
            user_message = "Tell me a joke."
            system_prompt = "Be a funny assistant."
            api_response_data = {"choices": [{"message": {"content": "Why did the scarecrow win an award? Because he was outstanding in his field!"}}]}
            
            mock_post_method = await self.mock_aiohttp_post(200, api_response_data)
            
            with patch.object(initialized_ai_handler.session, 'post', mock_post_method):
                response = await initialized_ai_handler.generate_response(user_message, system_prompt)

            assert response == "Why did the scarecrow win an award? Because he was outstanding in his field!"
            mock_cache_manager.get_cached_response.assert_called_once() # Cache miss
            mock_cache_manager.cache_response.assert_called_once() # Should cache the successful response
            assert initialized_ai_handler.metrics.successful_requests == 1

        @pytest.mark.asyncio
        async def test_generate_response_fallback_and_retry_logic(self, initialized_ai_handler: AIHandler, mock_cache_manager: MagicMock):
            """
            Tests generate_response with model fallback: first model fails, second succeeds.
            """
            user_message = "Query for fallback test."
            system_prompt = "System prompt."
            
            # Mock responses: first fails, second succeeds
            mock_resp_fail = AsyncMock(spec=aiohttp.ClientResponse)
            mock_resp_fail.status = 500 # Simulate server error for first model
            mock_resp_fail.json = AsyncMock(return_value={"error": {"message": "Server error"}})

            mock_resp_success = AsyncMock(spec=aiohttp.ClientResponse)
            mock_resp_success.status = 200
            mock_resp_success.json = AsyncMock(return_value={"choices": [{"message": {"content": "Success from fallback model!"}}]})

            # Context managers for each response
            cm_fail = AsyncMock()
            cm_fail.__aenter__ = AsyncMock(return_value=mock_resp_fail)
            cm_fail.__aexit__ = AsyncMock(return_value=None)

            cm_success = AsyncMock()
            cm_success.__aenter__ = AsyncMock(return_value=mock_resp_success)
            cm_success.__aexit__ = AsyncMock(return_value=None)

            # Configure the session.post mock to return these in sequence
            mock_session_post = MagicMock(side_effect=[cm_fail, cm_success]) # Use MagicMock for side_effect list

            with patch.object(initialized_ai_handler.session, 'post', mock_session_post):
                 # Patch asyncio.sleep to speed up retries in test
                with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                    response = await initialized_ai_handler.generate_response(user_message, system_prompt)

            assert response == "Success from fallback model!"
            assert mock_session_post.call_count == 2 # Called for primary and one fallback
            assert mock_sleep.called # Retry delay should have been called
            assert initialized_ai_handler.models[0].status == ModelStatus.AVAILABLE # Or FAILED if max_consecutive_failures is 1
            assert initialized_ai_handler.models[1].status == ModelStatus.AVAILABLE
            assert initialized_ai_handler.metrics.successful_requests == 1
            assert initialized_ai_handler.metrics.requests_by_model.get(initialized_ai_handler.models[0].name, 0) == 1
            assert initialized_ai_handler.metrics.requests_by_model.get(initialized_ai_handler.models[1].name, 0) == 1


        @pytest.mark.asyncio
        async def test_generate_response_all_models_fail_returns_fallback_text(self, initialized_ai_handler: AIHandler, mock_cache_manager: MagicMock, ai_handler_config: dict):
            """
            Tests generate_response returning a configured fallback text when all models fail.
            """
            mock_resp_fail = AsyncMock(spec=aiohttp.ClientResponse)
            mock_resp_fail.status = 503 # Simulate server error for all models
            mock_resp_fail.json = AsyncMock(return_value={"error": {"message": "Service unavailable"}})
            
            cm_fail = AsyncMock()
            cm_fail.__aenter__ = AsyncMock(return_value=mock_resp_fail)
            cm_fail.__aexit__ = AsyncMock(return_value=None)

            mock_session_post = MagicMock(return_value=cm_fail) # All calls to post will return this failing CM

            with patch.object(initialized_ai_handler.session, 'post', mock_session_post):
                with patch('asyncio.sleep', new_callable=AsyncMock): # Speed up retries
                    response = await initialized_ai_handler.generate_response("User Q", "System P")
            
            expected_fallback_responses = ai_handler_config.get('fallback_responses', [])
            assert response in expected_fallback_responses
            assert initialized_ai_handler.metrics.failed_requests == 1
            # All models should have been tried and marked as failed (or hit max_consecutive_failures)
            assert mock_session_post.call_count == len(initialized_ai_handler.models) * initialized_ai_handler.max_retries # Max retries for each model


    # --- Statistics and Health Check Tests ---
    class TestStatisticsAndHealth:
        """Tests for statistics gathering and health checks."""

        @pytest.mark.asyncio
        async def test_get_statistics_structure_and_initial_values(self, initialized_ai_handler: AIHandler):
            """Tests the structure and initial values of get_statistics."""
            stats = await initialized_ai_handler.get_statistics()
            assert 'total_requests' in stats and stats['total_requests'] == 0
            assert 'successful_requests' in stats and stats['successful_requests'] == 0
            assert 'models' in stats and len(stats['models']) == len(initialized_ai_handler.models)
            for model_stat in stats['models']:
                assert 'name' in model_stat
                assert 'status' in model_stat

        @pytest.mark.asyncio
        async def test_get_statistics_after_activity(self, initialized_ai_handler: AIHandler):
            """Simulates some activity and checks if statistics reflect it."""
            # Simulate a successful request on the first model
            initialized_ai_handler.metrics.total_requests = 1
            initialized_ai_handler.metrics.successful_requests = 1
            initialized_ai_handler.metrics.requests_by_model[initialized_ai_handler.models[0].name] = 1
            initialized_ai_handler.models[0].requests_made = 1
            initialized_ai_handler.models[0].successful_requests = 1
            initialized_ai_handler.models[0].total_tokens = 50

            stats = await initialized_ai_handler.get_statistics()
            assert stats['total_requests'] == 1
            assert stats['successful_requests'] == 1
            assert stats['models'][0]['name'] == initialized_ai_handler.models[0].name
            assert stats['models'][0]['successful_requests'] == 1
            assert stats['models'][0]['total_tokens'] == 50

        def test_is_healthy_true_when_ok(self, initialized_ai_handler: AIHandler):
            """Tests is_healthy returns True for a healthy handler."""
            assert initialized_ai_handler.is_healthy() is True

        def test_is_healthy_false_no_api_key(self, ai_handler_config: dict, mock_cache_manager: MagicMock):
            """Tests is_healthy returns False if API key is missing."""
            with patch.dict(os.environ, {}, clear=True):
                handler_no_key = AIHandler(ai_handler_config, mock_cache_manager)
            assert handler_no_key.is_healthy() is False

        @pytest.mark.asyncio
        async def test_is_healthy_false_session_closed(self, initialized_ai_handler: AIHandler):
            """Tests is_healthy returns False if the aiohttp session is closed."""
            await initialized_ai_handler.session.close() # Manually close the session
            assert initialized_ai_handler.is_healthy() is False

        def test_is_healthy_false_no_available_models(self, initialized_ai_handler: AIHandler):
            """Tests is_healthy returns False if no models are currently available."""
            for model in initialized_ai_handler.models:
                model.status = ModelStatus.FAILED
            assert initialized_ai_handler.is_healthy() is False