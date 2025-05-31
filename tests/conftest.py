# tests/conftest.py
"""
AI Gate for Artificial Intelligence Applications
Pytest Configuration and Fixtures (conftest.py)

This module defines shared fixtures for the AI Gate test suite. These fixtures
provide a centralized way to manage test configurations, mock objects, and
other resources needed by various test modules.

Key responsibilities of this conftest.py:
- Loading and providing application configuration (from default.yaml).
- Offering module-specific configuration slices (e.g., for QuestionProcessor, AIHandler).
- Supplying common mock objects (e.g., for CacheManager).
- Setting up the asyncio event loop for asynchronous tests.
- Providing core path utilities (e.g., project root, config directory).

By centralizing these components, this file helps in writing cleaner, more
maintainable, and consistent tests across the AI Gate application.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import asyncio # For event_loop fixture

# --- Core Path Fixtures ---

@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Provides the absolute path to the project's root directory.
    Assumes conftest.py is in the 'tests/' directory at the project root.
    """
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def config_dir(project_root: Path) -> Path:
    """Provides the absolute path to the 'config/' directory."""
    return project_root / "config"

@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Provides the absolute path to the 'data/' directory."""
    return project_root / "data"

# --- Main Configuration Loading Fixture ---

@pytest.fixture(scope="session")
def default_config_data(config_dir: Path) -> dict:
    """
    Loads the content of 'default.yaml' once per test session.
    
    Raises:
        pytest.fail: If 'default.yaml' is not found or cannot be parsed.
    """
    default_yaml_path = config_dir / "default.yaml"
    if not default_yaml_path.exists():
        pytest.fail(
            f"Configuration file 'default.yaml' not found at: {default_yaml_path}\n"
            "Ensure 'config/default.yaml' exists in the project root."
        )
    
    try:
        with open(default_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                pytest.fail(
                    f"Content of 'default.yaml' at {default_yaml_path} is not a valid YAML dictionary."
                )
            return config
    except yaml.YAMLError as e:
        pytest.fail(
            f"Error parsing 'default.yaml' at {default_yaml_path}: {e}"
        )
    except Exception as e:
        pytest.fail(
            f"An unexpected error occurred while reading 'default.yaml' at {default_yaml_path}: {e}"
        )
    return {} # Should not be reached if pytest.fail is called

# --- Module-Specific Configuration Fixtures ---

@pytest.fixture
def question_processor_config(default_config_data: dict) -> dict:
    """Provides the 'question_processing' section from the default configuration."""
    config = default_config_data.get("question_processing", {})
    # Ensure 'min_confidence_threshold' is used as per prior discussion.
    # If 'min_confidence' was present, it should have been 'min_confidence_threshold'
    # or QuestionProcessor code updated to expect 'min_confidence_threshold'.
    # This fixture assumes 'default.yaml' and QuestionProcessor code are aligned on 'min_confidence_threshold'.
    if not config:
        pytest.warning(
            "No 'question_processing' section found in default.yaml. "
            "QuestionProcessor tests might use default hardcoded values."
        )
    return config

@pytest.fixture
def website_researcher_config(default_config_data: dict) -> dict:
    """Provides the 'website_research' section from the default configuration."""
    config = default_config_data.get("website_research", {})
    if not config:
        pytest.warning(
            "No 'website_research' section found in default.yaml. "
            "WebsiteResearcher tests might use default hardcoded values."
        )
    return config

@pytest.fixture
def prompt_builder_templates_config(default_config_data: dict) -> dict:
    """Provides the 'prompts' section from the default configuration (for PromptBuilder templates)."""
    config = default_config_data.get("prompts", {})
    if not config:
        pytest.warning(
            "No 'prompts' section found in default.yaml. "
            "PromptBuilder tests might use default hardcoded values or embedded templates."
        )
    return config

@pytest.fixture
def institution_data_config(default_config_data: dict) -> dict:
    """Provides the 'institution' section from the default configuration (for PromptBuilder)."""
    config = default_config_data.get("institution", {})
    if not config:
        pytest.warning(
            "No 'institution' section found in default.yaml. "
            "PromptBuilder and other components might use default hardcoded values."
        )
    return config

@pytest.fixture
def ai_handler_config(default_config_data: dict) -> dict:
    """Provides the 'ai_models' section from the default configuration."""
    config = default_config_data.get("ai_models", {})
    if not config:
        pytest.warning(
            "No 'ai_models' section found in default.yaml. "
            "AIHandler tests might use default hardcoded model lists and settings."
        )
    return config

@pytest.fixture
def cache_manager_config(default_config_data: dict) -> dict:
    """Provides the 'cache' section from the default configuration."""
    config = default_config_data.get("cache", {})
    if not config:
        pytest.warning(
            "No 'cache' section found in default.yaml. "
            "CacheManager tests might use default hardcoded values for TTL, size etc."
        )
    return config

@pytest.fixture
def logging_config(default_config_data: dict) -> dict:
    """Provides the 'logging' section from the default configuration (for test_utils.py)."""
    config = default_config_data.get("logging", {})
    if not config:
        pytest.warning(
            "No 'logging' section found in default.yaml. "
            "Logging utility tests might use default hardcoded values."
        )
    return config

@pytest.fixture
def api_config(default_config_data: dict) -> dict:
    """Provides the 'api' section from the default configuration (for test_main.py if needed)."""
    config = default_config_data.get("api", {})
    if not config:
        pytest.warning(
            "No 'api' section found in default.yaml. "
            "API related tests might use default hardcoded values."
        )
    return config

# --- Common Mock Fixtures ---

@pytest.fixture
def mock_cache_manager():
    """
    Provides a generic mock for the CacheManager.
    Individual test modules can customize its behavior if needed.
    """
    cache_manager = Mock(name="MockCacheManager")
    
    # Common methods used by other modules that might need mocking
    cache_manager.get = AsyncMock(return_value=None)
    cache_manager.set = AsyncMock(return_value=True)
    cache_manager.get_cached_response = AsyncMock(return_value=None)
    cache_manager.cache_response = AsyncMock(return_value=True)
    cache_manager.get_cached_question_analysis = AsyncMock(return_value=None)
    cache_manager.cache_question_analysis = AsyncMock(return_value=True)
    cache_manager.get_cached_website_research = AsyncMock(return_value=None)
    cache_manager.cache_website_research = AsyncMock(return_value=True)
    cache_manager.get_cached_prompt_template = AsyncMock(return_value=None)
    cache_manager.cache_prompt_template = AsyncMock(return_value=True)
    cache_manager.generate_cache_key = Mock(side_effect=lambda *args, **kwargs: f"mock_cache_key_for_{args}_{kwargs.get('category','general')}")
    cache_manager.clear_cache = AsyncMock()
    cache_manager.invalidate_prompt_templates = AsyncMock(return_value=True)
    cache_manager.get_statistics = AsyncMock(return_value={
        "hits": 0, "misses": 0, "total_entries": 0, "size_bytes": 0
    })
    cache_manager.is_healthy = Mock(return_value=True)

    return cache_manager

# --- Event Loop Fixture for Async Tests ---

@pytest.fixture(scope="session")
def event_loop():
    """
    Creates an instance of the default event loop for the test session.
    This is necessary for running asyncio tests with pytest.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()