# AI Gate Test Suite 🧪

[![Testing Framework](https://img.shields.io/badge/Testing-pytest-blue.svg)](https://docs.pytest.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Async Support](https://img.shields.io/badge/async-pytest--asyncio-green.svg)](https://github.com/pytest-dev/pytest-asyncio)

> **Comprehensive test suite for the AI Gate conversational interface application**

This document provides a complete guide to understanding, running, and maintaining the AI Gate test suite. The tests ensure reliability, maintainability, and correctness of all application components through unit tests, integration tests, and end-to-end API testing.

## 📋 Table of Contents

- [Test Strategy Overview](#-test-strategy-overview)
- [Prerequisites & Dependencies](#-prerequisites--dependencies)
- [Quick Start](#-quick-start)
- [Test Configuration](#-test-configuration)
- [Test Modules Overview](#-test-modules-overview)
- [Running Tests](#-running-tests)
- [Fixtures Reference](#-fixtures-reference)
- [Writing New Tests](#-writing-new-tests)
- [Troubleshooting](#-troubleshooting)

## 🎯 Test Strategy Overview

The AI Gate test suite employs a multi-layered testing approach:

### Testing Layers

| Layer | Purpose | Tools | Coverage |
|-------|---------|-------|----------|
| **Unit Tests** | Test individual functions/methods in isolation | `pytest`, `unittest.mock` | Each module's core functionality |
| **Integration Tests** | Verify component interactions | `pytest`, mocked dependencies | Cross-module workflows |
| **API Tests** | End-to-end testing via FastAPI endpoints | `TestClient`, mocked external services | Complete request-response cycles |

### Testing Philosophy

- **Isolation**: External dependencies are mocked to ensure consistent, fast tests
- **Consistency**: Shared fixtures provide uniform test configuration
- **Reliability**: No external API calls or network dependencies during testing
- **Coverage**: All critical paths and edge cases are tested

## 📦 Prerequisites & Dependencies

### Core Requirements

```bash
# Essential testing dependencies
pytest>=7.4.0,<8.3.0
pytest-asyncio>=0.21.0,<0.24.0
PyYAML>=6.0,<6.1
python-dotenv>=1.0.0,<1.1.0
```

### Optional Dependencies

```bash
# For enhanced testing capabilities
pytest-cov>=4.1.0,<5.1.0        # Coverage reports
httpx>=0.26.0,<0.28.0           # Alternative HTTP client testing
```

### Application Dependencies (Required for Full Testing)

```bash
# NLP features testing
langdetect>=1.0.9,<1.1.0        # Language detection tests
nltk>=3.8.0,<3.9.0              # Advanced NLP tests
scikit-learn>=1.3.0,<1.5.0      # TF-IDF processing tests
psutil>=5.9.0,<5.10.0           # System validation tests
```

## 🚀 Quick Start

### Running All Tests

```bash
# From project root directory
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=modules --cov-report=html
```

### Running Specific Tests

```bash
# Single test file
pytest tests/test_question_processor.py

# Specific test function
pytest tests/test_ai_handler.py::test_generate_response_success

# Tests matching pattern
pytest -k "test_validation"
```

### Running Tests by Category

```bash
# Unit tests only
pytest tests/test_*.py -k "not integration"

# Integration tests only
pytest tests/test_integration.py

# Async tests only
pytest tests/ -k "async"
```

## ⚙️ Test Configuration

### Configuration Architecture

The test suite uses a centralized configuration system through `tests/conftest.py`:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_*.py               # Individual test modules
└── temp_*/                 # Temporary test data (created during tests)
```

### Key Configuration Sources

1. **`config/default.yaml`** - Base application configuration
2. **Environment Variables** - `AIGATE_*` prefixed variables
3. **Temporary Configs** - Created per test for isolation
4. **Mock Objects** - Pre-configured mocks for external dependencies

## 📁 Test Modules Overview

### Core Test Files

| Test File | Purpose | Key Areas Tested |
|-----------|---------|------------------|
| **`conftest.py`** | Shared fixtures and configuration | Path helpers, config loading, mock objects |
| **`test_utils.py`** | Utility functions testing | Logging, config merging, environment validation |
| **`test_question_processor.py`** | Input processing and validation | Text cleaning, language detection, topic extraction |
| **`test_website_researcher.py`** | Knowledge base search | Content retrieval, relevance scoring, TF-IDF |
| **`test_prompt_builder.py`** | Dynamic prompt construction | Template loading, context formatting, optimization |
| **`test_ai_handler.py`** | AI model communication | API requests, fallback logic, error handling |
| **`test_cache_manager.py`** | Caching system | TTL expiry, LRU eviction, persistence |
| **`test_integration.py`** | End-to-end API testing | Complete workflows, error handling, performance |

### Detailed Test Coverage

#### 🔧 `test_utils.py`
```python
✅ Logging setup and configuration
✅ Multi-file config merging (default.yaml + institution.yaml + local.yaml)
✅ Environment variable override (AIGATE_* variables)
✅ Environment validation (Python version, dependencies, permissions)
✅ Helper functions (hashing, sanitization, formatting)
```

#### 📝 `test_question_processor.py`
```python
✅ Input validation (length, content, suspicious patterns)
✅ Text cleaning and normalization
✅ Language detection (with/without langdetect)
✅ Topic and keyword extraction (with/without NLTK)
✅ Question classification and complexity scoring
✅ Caching integration
```

#### 🔍 `test_website_researcher.py`
```python
✅ Knowledge base loading and reloading
✅ Content search and relevance scoring
✅ Advanced text processing (TF-IDF when available)
✅ Result formatting and pagination
✅ Category-based filtering
✅ Performance statistics
```

#### 🏗️ `test_prompt_builder.py`
```python
✅ Template loading (embedded vs. file-based)
✅ Context integration (institution data, research results)
✅ Language-specific instructions
✅ Prompt optimization and truncation
✅ Template validation
```

#### 🤖 `test_ai_handler.py`
```python
✅ Model configuration and fallback logic
✅ API request payload construction
✅ Response validation and formatting
✅ Error handling (rate limits, timeouts, API errors)
✅ Retry mechanisms and caching
```

#### 🗄️ `test_cache_manager.py`
```python
✅ Cache operations (get, set, delete)
✅ TTL expiry and LRU eviction
✅ Category-specific configurations
✅ Persistent storage operations
✅ Cache statistics and health checks
```

#### 🌐 `test_integration.py`
```python
✅ Application startup and component initialization
✅ End-to-end chat workflow via /api/chat
✅ API endpoint functionality (/health, /api/institution, /api/stats)
✅ Cross-component error handling
✅ Performance under load
```

## 🏃‍♂️ Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=modules

# Generate HTML coverage report
pytest --cov=modules --cov-report=html

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

### Advanced Options

```bash
# Stop on first failure
pytest -x

# Show local variables in tracebacks
pytest -l

# Run only failed tests from last run
pytest --lf

# Run tests matching pattern
pytest -k "processor and not integration"

# Run with specific markers
pytest -m "unit_test"
```

### Performance Testing

```bash
# Run with timing information
pytest --durations=10

# Profile test execution
pytest --profile

# Memory usage tracking (requires memory-profiler)
pytest --memprof
```

## 🔧 Fixtures Reference

### Path Fixtures
```python
project_root    # Absolute path to project root
config_dir      # Path to config/ directory
data_dir        # Path to data/ directory
```

### Configuration Fixtures
```python
default_config_data              # Full YAML configuration
question_processor_config        # QuestionProcessor settings
website_researcher_config        # WebsiteResearcher settings
prompt_builder_templates_config  # PromptBuilder templates
institution_data_config          # Institution-specific data
ai_handler_config               # AIHandler settings
cache_manager_config            # CacheManager settings
logging_config                  # Logging configuration
api_config                      # API settings
```

### Mock Fixtures
```python
mock_cache_manager    # Pre-configured CacheManager mock
event_loop           # Asyncio event loop for async tests
```

### Temporary Environment Fixtures
```python
temp_test_env_for_utils              # Utils testing environment
temp_pages_file                      # Temporary pages.json
temp_config_dir_with_custom_prompt   # Custom prompt template testing
integration_test_environment         # Complete isolated test environment
```

## ✍️ Writing New Tests

### Test Structure Template

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
from modules.your_module import YourClass

class TestYourClass:
    """Test suite for YourClass functionality."""
    
    def test_initialization(self, your_module_config):
        """Test proper initialization."""
        instance = YourClass(config=your_module_config)
        assert instance.some_property == expected_value
    
    @pytest.mark.asyncio
    async def test_async_method(self, your_module_config, mock_cache_manager):
        """Test asynchronous methods."""
        instance = YourClass(config=your_module_config, cache=mock_cache_manager)
        result = await instance.async_method()
        assert result is not None
    
    @patch('modules.your_module.external_dependency')
    def test_with_mocked_dependency(self, mock_external, your_module_config):
        """Test with mocked external dependencies."""
        mock_external.return_value = "mocked_result"
        instance = YourClass(config=your_module_config)
        result = instance.method_using_external()
        assert result == "expected_result"
```

### Best Practices

1. **Use Descriptive Names**: Test names should clearly describe what they test
2. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
3. **Mock External Dependencies**: Use `@patch` for external services, file system, etc.
4. **Test Edge Cases**: Include boundary conditions and error scenarios
5. **Use Fixtures**: Leverage shared fixtures for consistent test data
6. **Keep Tests Independent**: Each test should be able to run in isolation

### Adding New Fixtures

```python
# In conftest.py
@pytest.fixture
def your_custom_fixture(project_root):
    """Provide custom test data or configuration."""
    # Setup
    test_data = create_test_data()
    yield test_data
    # Cleanup (if needed)
    cleanup_test_data()
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Configuration Loading Errors
```python
# Error: Config file not found
# Solution: Ensure config/default.yaml exists and is valid YAML

# Check fixture in conftest.py
def test_config_loading(default_config_data):
    assert default_config_data is not None
    assert 'modules' in default_config_data
```

#### Async Test Failures
```python
# Error: RuntimeError: no running event loop
# Solution: Use @pytest.mark.asyncio decorator

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

#### Mock Configuration Issues
```python
# Error: Mock not called as expected
# Solution: Verify mock setup and call patterns

@patch('modules.ai_handler.aiohttp.ClientSession.post')
def test_api_call(self, mock_post):
    mock_post.return_value.__aenter__.return_value.status = 200
    # Test code here
    mock_post.assert_called_once()
```

#### Import Errors
```python
# Error: ModuleNotFoundError
# Solution: Check PYTHONPATH and ensure __init__.py files exist

# Run tests from project root
cd ai-gate/
export PYTHONPATH=$PWD:$PYTHONPATH
pytest
```

### Debugging Tips

1. **Use `-s` flag**: See print statements during test execution
2. **Use `--pdb`**: Drop into debugger on failures
3. **Check fixtures**: Verify fixture values with print statements
4. **Isolate tests**: Run single tests to identify issues
5. **Check mocks**: Verify mock calls with `.assert_called_with()`

### Performance Issues

```bash
# Identify slow tests
pytest --durations=0

# Profile memory usage
pytest --memprof

# Run tests in parallel
pytest -n auto
```

## 📊 Coverage Reports

### Generating Coverage Reports

```bash
# Terminal coverage report
pytest --cov=modules

# HTML coverage report
pytest --cov=modules --cov-report=html
open htmlcov/index.html

# XML coverage report (for CI/CD)
pytest --cov=modules --cov-report=xml
```

### Coverage Targets

- **Unit Tests**: Aim for >90% line coverage
- **Integration Tests**: Focus on critical user paths
- **Branch Coverage**: Test all conditional branches

---

## 🎯 Summary

The AI Gate test suite provides comprehensive coverage of all application components through:

- **Modular Design**: Each component tested in isolation
- **Shared Configuration**: Centralized fixtures for consistency
- **Multiple Test Types**: Unit, integration, and API tests
- **Mock Strategy**: External dependencies safely mocked
- **Easy Execution**: Simple commands for various test scenarios

For questions or issues with the test suite, please refer to the main project documentation or create an issue in the repository.

---

<div align="center">

**Built with ❤️ for reliable AI Gate functionality**

[🧪 View Test Coverage](htmlcov/index.html) • [📝 Main Documentation](../README.md) • [🐛 Report Issues](https://github.com/yourusername/ai-gate/issues)

</div>