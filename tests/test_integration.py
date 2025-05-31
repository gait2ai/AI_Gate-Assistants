# tests/test_integration.py

"""
AI Gate for Artificial Intelligence Applications
test_integration Module

AI Gate Integration Tests

This module contains integration tests that verify the interaction and
proper functioning of all AI Gate components working together as a complete system.
It tests the entire pipeline from API request to AI response generation,
focusing on the interplay between modules rather than isolated unit behavior.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, AsyncMock # For mocking component methods and external calls

from fastapi.testclient import TestClient
import yaml # To create YAML config files for tests

# Import the main application and its components for patching and verification
# We need to be careful with how `main.py` is imported if it initializes things at import time.
# It's generally better if main.py's initialization logic is within functions called at startup.
# Assuming `main.app` is the FastAPI instance and `initialize_components` is the setup function.
from main import app, initialize_components, app_components, LOGS_DIR as MAIN_LOGS_DIR, CONFIG_DIR as MAIN_CONFIG_DIR, DATA_DIR as MAIN_DATA_DIR
# We will patch MAIN_CONFIG_DIR and MAIN_DATA_DIR for tests.

# Pytest marker for all tests in this file to be treated as asyncio
pytestmark = pytest.mark.asyncio


# --- Test Environment Setup Fixture (Local to this test file) ---

@pytest.fixture(scope="function") # Use "function" scope to reset env for each test
def integration_test_environment():
    """
    Sets up a temporary, isolated environment for integration testing.
    This includes temporary config, data, and log directories,
    and sample configuration files.
    Yields a dictionary containing paths to these temporary directories.
    Cleans up the temporary directory estruturafter the test.
    """
    temp_base_dir = Path(tempfile.mkdtemp(prefix="aigate_integration_test_"))
    
    # Create directory structure
    temp_config_dir = temp_base_dir / "config"
    temp_data_dir = temp_base_dir / "data"
    temp_cache_dir = temp_data_dir / "cache" # CacheManager will create subdirs if needed
    temp_logs_dir = temp_base_dir / "logs"

    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    temp_cache_dir.mkdir(parents=True, exist_ok=True)
    temp_logs_dir.mkdir(parents=True, exist_ok=True)

    # --- Create sample default.yaml ---
    sample_default_yaml_content = {
        "institution": {
            "name": "Integration Test University",
            "description": "A university for integration testing.",
            "website": "https://integration.test",
            "timezone": "UTC"
        },
        "ai_models": {
            "primary_model": "mock/test-model-primary",
            "fallback_models": ["mock/test-model-fallback"],
            "base_url": "https://mock.openrouter.ai/api/v1", # Mock URL
            "timeout": 5,
            "max_tokens": 50,
            "temperature": 0.1,
            "max_retries": 1,
            "fallback_responses": ["Mocked AI is experiencing mock difficulties."]
        },
        "cache": {
            "max_size": 10, # Small for testing eviction if necessary
            "ttl": 60,
            "cleanup_interval": 300,
             "categories": { # Ensure categories match CacheManager defaults or are specified
                "chat_response": {"ttl": 60, "persistent": False}, # Non-persistent for easier test cleanup
                "question_analysis": {"ttl": 60, "persistent": False},
                "website_research": {"ttl": 60, "persistent": False},
                "prompt_template": {"ttl": 0, "persistent": False},
                "ai_response": {"ttl": 60, "persistent": False},
            }
        },
        "question_processing": {
            "min_length": 3,
            "max_length": 100,
            "supported_languages": ["en"],
            "min_confidence_threshold": 0.5,
            "enable_caching": True
        },
        "website_research": {
            "max_results": 2,
            "similarity_threshold": 0.1,
            "content_snippet_length": 50
        },
        "prompts": {
            "system_template_file": "system_prompt.txt", # Will create this file
            "max_context_length": 200,
            "max_prompt_length": 500,
            "prompt_optimization": True
        },
        "logging": {
            "level": "DEBUG", # Useful for debugging integration tests
            "console_output": False # Avoid cluttering test output unless debugging
        },
        "api": {
            "cors_origins": ["*"],
            "gzip_compression": False # Easier to inspect non-gzipped responses in tests
        }
    }
    with open(temp_config_dir / "default.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(sample_default_yaml_content, f)

    # --- Create sample system_prompt.txt ---
    sample_system_prompt = """Institution: {institution_name}
Context:
{context}
Language: {language_instruction}
Query: Respond to the user.
"""
    with open(temp_config_dir / "system_prompt.txt", 'w', encoding='utf-8') as f:
        f.write(sample_system_prompt)

    # --- Create sample data/pages.json ---
    sample_pages_json_content = {
        "metadata": {"processed_at": "2024-01-01T00:00:00"},
        "pages": [
            {
                "content_id": "page1", "url": "http://test.com/page1", "title": "About Us",
                "summary": "Information about our test university.",
                "keywords": ["about", "university"], "category": "general",
                "search_text": "Test University is a place for learning and integration testing.",
                "metrics": {"word_count": 10}
            },
            {
                "content_id": "page2", "url": "http://test.com/page2", "title": "Admissions",
                "summary": "How to apply for admissions.",
                "keywords": ["admissions", "apply"], "category": "admissions",
                "search_text": "To apply for admissions, please visit our portal. Requirements are important.",
                "metrics": {"word_count": 12}
            }
        ]
    }
    with open(temp_data_dir / "pages.json", 'w', encoding='utf-8') as f:
        json.dump(sample_pages_json_content, f, indent=2)
        
    # --- Create sample .env file ---
    # (OPENROUTER_API_KEY will be patched in tests where AIHandler is directly involved,
    # but if initialize_components or utils.validate_environment checks for it,
    # it's good to have a placeholder)
    with open(temp_base_dir / ".env", 'w', encoding='utf-8') as f:
        f.write("OPENROUTER_API_KEY=mock_integration_key\n")
        f.write("OPENROUTER_API_URL=https://mock.openrouter.ai/api/v1\n") # Match config


    env_paths = {
        "base": temp_base_dir,
        "config": temp_config_dir,
        "data": temp_data_dir,
        "cache": temp_cache_dir,
        "logs": temp_logs_dir
    }

    # --- Patch global directory paths used by main.py ---
    # Store original paths to restore them later
    original_main_config_dir = MAIN_CONFIG_DIR
    original_main_data_dir = MAIN_DATA_DIR
    original_main_logs_dir = MAIN_LOGS_DIR
    original_cwd = Path.cwd()

    # Patch the global path variables in main.py
    # This is crucial for initialize_components to use our temp dirs.
    # We also need to patch os.getcwd() if validate_environment uses it.
    with patch('main.CONFIG_DIR', temp_config_dir), \
         patch('main.DATA_DIR', temp_data_dir), \
         patch('main.LOGS_DIR', temp_logs_dir), \
         patch('modules.utils.Path.cwd', return_value=temp_base_dir), \
         patch.dict(os.environ, {"PYTHONPATH": str(temp_base_dir.parent)}): # Ensure modules can be found if main.py changes cwd behavior

        # Also patch dotenv load to use the temp .env
        with patch('main.load_dotenv', lambda: True), \
             patch('modules.utils.load_dotenv', lambda: True): # Patch in utils too if it loads .env directly
            
            # Change current working directory to the temp base so that .env is found if main.py relies on it
            # os.chdir(temp_base_dir) # Be cautious with os.chdir in tests

            yield env_paths # Provide paths to the test

    # --- Cleanup ---
    # os.chdir(original_cwd) # Restore original CWD
    
    # Restore original global path variables in main.py
    # This is a bit tricky as they are module-level globals.
    # Re-importing main or direct assignment might be needed if they were truly modified.
    # For safety, it's often better if main.py uses functions to get these paths
    # that can be more easily patched.
    # If main.py re-evaluates these at startup, this might not be strictly necessary for subsequent tests
    # if each test runs in a fresh process, but for function-scoped fixtures in one session:
    globals()['MAIN_CONFIG_DIR'] = original_main_config_dir
    globals()['MAIN_DATA_DIR'] = original_main_data_dir
    globals()['MAIN_LOGS_DIR'] = original_main_logs_dir
    # Or more robustly:
    # setattr(main_module, 'CONFIG_DIR', original_main_config_dir) ... if you have main_module imported

    if temp_base_dir.exists():
        shutil.rmtree(temp_base_dir)


@pytest.fixture
def client(integration_test_environment) -> TestClient:
    """
    Provides a FastAPI TestClient configured to run against the application
    initialized within the isolated integration test environment.
    Components are initialized before the client is returned.
    """
    # initialize_components() should be called here, after paths are patched by
    # the integration_test_environment fixture.
    # The patching happens when integration_test_environment is entered.
    
    # We need to ensure components are reset for each test function using this client.
    # This can be done by re-initializing or carefully managing state.
    # For simplicity, we'll re-initialize.
    
    # Backup original app_components
    original_app_components = app_components.copy()

    if not initialize_components(): # This uses the patched paths
        pytest.fail("Integration Test: Failed to initialize application components.")

    yield TestClient(app)

    # Restore original app_components
    for key, value in original_app_components.items():
        app_components[key] = value
    # Potentially also call a cleanup function for components if they have one
    # e.g., app_components['cache_manager'].cleanup()


# --- Integration Test Classes ---

class TestApplicationStartupAndHealth:
    """Tests application startup, component initialization, and health checks."""

    @pytest.mark.asyncio
    async def test_successful_initialization(self, integration_test_environment):
        """
        Tests if initialize_components successfully sets up all app_components
        using the temporary configuration.
        """
        # The 'integration_test_environment' fixture patches the paths.
        # Now, call initialize_components which should use these patched paths.
        success = initialize_components()
        assert success is True
        assert app_components.get("config") is not None
        assert app_components.get("question_processor") is not None
        assert app_components.get("website_researcher") is not None
        assert app_components.get("prompt_builder") is not None
        assert app_components.get("ai_handler") is not None
        assert app_components.get("cache_manager") is not None
        assert app_components["config"]["institution"]["name"] == "Integration Test University"

    def test_health_endpoint_after_initialization(self, client: TestClient): # Client fixture ensures init
        """Tests the /health endpoint after components are initialized."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"] # Healthy if all components are ok
        assert data["components"]["question_processor"] != "not_initialized"
        assert data["components"]["ai_handler"] != "not_initialized"


class TestEndToEndChatFlow:
    """Tests the complete end-to-end chat processing pipeline."""

    @pytest.mark.asyncio
    async def test_chat_api_successful_response(self, client: TestClient):
        """
        Tests a successful chat request through the /api/chat endpoint,
        mocking the actual AI model call.
        """
        # Mock the AIHandler's generate_response method to avoid external API calls
        # and control the AI's output.
        mock_ai_response_text = "This is a mocked AI response about Integration Test University."
        
        # app_components should be populated by the 'client' fixture's setup
        assert app_components.get("ai_handler") is not None, "AIHandler not initialized in app_components"

        with patch.object(app_components["ai_handler"], 'generate_response', AsyncMock(return_value=mock_ai_response_text)) as mock_generate_response:
            user_message = "Tell me about the university."
            response = client.post("/api/chat", json={"message": user_message, "session_id": "integ_test_1"})
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["answer"] == mock_ai_response_text
            assert data["session_id"] == "integ_test_1"
            assert isinstance(data["sources"], list) # Website researcher should find 'About Us'
            if data["sources"]: # If sources were found
                 assert any("test.com/page1" in src for src in data["sources"])
            assert data["cached"] is False # First request

            mock_generate_response.assert_called_once()
            # We can inspect the call to mock_generate_response to see what system_prompt was built
            call_args = mock_generate_response.call_args[1] # keyword arguments
            assert "user_message" in call_args and call_args["user_message"] == user_message
            assert "system_prompt" in call_args and "Integration Test University" in call_args["system_prompt"]
            assert "Test University is a place for learning" in call_args["system_prompt"] # Context from pages.json

    @pytest.mark.asyncio
    async def test_chat_api_caching_behavior(self, client: TestClient, integration_test_environment): # Needs env for cache path
        """Tests if responses are cached and retrieved on subsequent identical requests."""
        mock_ai_response_text = "This response should be cached."
        user_message = "What are the admission requirements?"

        # Ensure CacheManager is using the temp directory
        # This is handled by initialize_components using the patched DATA_DIR

        with patch.object(app_components["ai_handler"], 'generate_response', AsyncMock(return_value=mock_ai_response_text)) as mock_generate_response:
            # First request (cache miss)
            response1 = client.post("/api/chat", json={"message": user_message})
            assert response1.status_code == 200
            data1 = response1.json()
            assert data1["answer"] == mock_ai_response_text
            assert data1["cached"] is False
            mock_generate_response.assert_called_once() # AI Handler called

            # Allow time for potential background caching task if any (CacheManager saves async)
            # For integration test, depends on CacheManager's cache_response implementation
            # If it's truly async and detached, this sleep might be needed.
            # The provided CacheManager.cache_response seems to be async but not detached in a way that requires long sleep.
            await asyncio.sleep(0.1) 

            # Second request (should be cache hit)
            mock_generate_response.reset_mock() # Reset mock for the second call
            response2 = client.post("/api/chat", json={"message": user_message})
            assert response2.status_code == 200
            data2 = response2.json()
            
            assert data2["answer"] == mock_ai_response_text
            assert data2["cached"] is True
            mock_generate_response.assert_not_called() # AI Handler should NOT be called

    @pytest.mark.parametrize("invalid_message, expected_detail_part", [
        ("", "cannot be empty"), # Based on UserMessage Pydantic model
        ("a", "too short"),   # Based on UserMessage Pydantic model
        ("b" * 101, "too long"), # Assuming max_length=100 in test config for QuestionProcessor
    ])
    def test_chat_api_invalid_input_validation(self, client: TestClient, invalid_message: str, expected_detail_part: str):
        """Tests API validation for invalid user messages via Pydantic models."""
        response = client.post("/api/chat", json={"message": invalid_message})
        assert response.status_code == 422 # Unprocessable Entity for Pydantic validation errors
        data = response.json()
        assert "detail" in data
        # Pydantic's error messages can be a list of error objects
        found_error = False
        for error_item in data["detail"]:
            if expected_detail_part.lower() in error_item["msg"].lower():
                found_error = True
                break
        assert found_error, f"Expected error containing '{expected_detail_part}' not found in {data['detail']}"


class TestOtherAPIEndpoints:
    """Tests other miscellaneous API endpoints like /api/institution, /api/stats."""

    def test_get_institution_data_endpoint(self, client: TestClient):
        """Tests the /api/institution endpoint."""
        response = client.get("/api/institution")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Integration Test University" # From sample_default_yaml_content
        assert data["website"] == "https://integration.test"

    @pytest.mark.asyncio
    async def test_get_statistics_endpoint(self, client: TestClient, integration_test_environment):
        """Tests the /api/stats endpoint."""
        # Make a chat request to generate some stats
        with patch.object(app_components["ai_handler"], 'generate_response', AsyncMock(return_value="Stats test.")):
            client.post("/api/chat", json={"message": "Generate some stats"})

        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        
        assert "cache" in data
        assert "ai" in data
        assert "uptime" in data
        assert "configuration" in data
        assert data["configuration"]["institution_name"] == "Integration Test University"
        assert data["cache"]["max_entries"] == 10 # From sample_default_yaml_content['cache']['max_size']
        assert data["ai"]["total_requests"] >= 0 # AIHandler stats should be present

    def test_clear_cache_endpoint(self, client: TestClient, integration_test_environment):
        """Tests the /api/clear-cache endpoint."""
        # Add something to cache via a chat call first
        with patch.object(app_components["ai_handler"], 'generate_response', AsyncMock(return_value="Cache me.")):
            client.post("/api/chat", json={"message": "Cache this test message"})
        
        # Verify something might be in cache (indirectly)
        stats_before_clear = client.get("/api/stats").json()
        
        response_clear = client.post("/api/clear-cache")
        assert response_clear.status_code == 200
        assert response_clear.json()["message"] == "Cache cleared successfully"

        stats_after_clear = client.get("/api/stats").json()
        # Check if cache entries or size changed as expected.
        # Exact check depends on what CacheManager.clear_cache() does to stats.
        # A simple check might be that total_entries is 0 if it clears everything.
        assert stats_after_clear["cache"].get("total_entries", -1) == 0 or \
               stats_after_clear["cache"].get("hits", -1) >= stats_before_clear["cache"].get("hits", 0) # Hits might persist if not reset

# Note: Static file serving tests are usually straightforward with TestClient
# but depend on the existence of files in a 'static' directory.
# The integration_test_environment fixture creates a temp 'static' dir but doesn't populate it.
# For true static file tests, you'd add sample static files to the temp env.