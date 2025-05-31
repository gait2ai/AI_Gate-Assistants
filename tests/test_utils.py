# tests/test_utils.py

"""
AI Gate for Artificial Intelligence Applications
test_utils Module

Test Suite for the Utility Module (modules.utils)

This module provides tests for common utility functions used throughout
the AI Gate system, including logging setup, configuration management,
environment validation, and other shared helper functions.
"""

import pytest
import asyncio
import os
import json
import yaml
import logging
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock, call # Added call
from datetime import datetime

# Import functions and classes to be tested
from modules.utils import (
    setup_logging,
    load_config,
    validate_environment,
    generate_hash,
    sanitize_filename,
    format_bytes,
    format_duration,
    LogLevel, # If you want to test LogLevel enum itself
    SystemInfo, # If you want to test SystemInfo dataclass
    _get_system_info # Test private function if necessary and possible
)

# Pytest marker for all tests in this file to be treated as asyncio if any test needs it
# pytestmark = pytest.mark.asyncio # Not strictly needed if all tests are sync

# --- Fixtures Local to This Test File ---

@pytest.fixture
def temp_test_env_for_utils():
    """
    Creates a temporary directory structure for utils testing,
    including config, data, logs, and static folders.
    Yields the base temporary directory path.
    Cleans up afterwards.
    """
    base_dir = Path(tempfile.mkdtemp(prefix="test_utils_env_"))
    
    # Create common directories that validate_environment might check
    (base_dir / "config").mkdir()
    (base_dir / "data").mkdir()
    (base_dir / "logs").mkdir()
    (base_dir / "static").mkdir()
    (base_dir / "modules").mkdir() # Added as per validate_environment checks

    # Create dummy required files for validate_environment
    with open(base_dir / "config" / "default.yaml", "w") as f:
        yaml.dump({"app_name": "TestAppFromDefault"}, f)
    with open(base_dir / "config" / "institution.yaml", "w") as f:
        yaml.dump({"institution": {"name": "TestInstitutionFromInst"}}, f)
    with open(base_dir / "data" / "pages.json", "w") as f:
        json.dump({"pages": []}, f)
        
    yield base_dir
    
    shutil.rmtree(base_dir)


@pytest.fixture
def temp_logs_dir() -> Path:
    """Creates a temporary directory for log files and cleans up afterwards."""
    log_dir = Path(tempfile.mkdtemp(prefix="test_logs_"))
    yield log_dir
    shutil.rmtree(log_dir)

@pytest.fixture
def temp_config_dir_for_load_config() -> Path:
    """
    Creates a temporary config directory with sample YAML files for testing load_config.
    Cleans up afterwards.
    """
    config_path = Path(tempfile.mkdtemp(prefix="test_load_config_"))
    
    # default.yaml
    default_content = {
        "setting1": "default_value1",
        "nested": {"keyA": "default_A", "keyB": "default_B"},
        "logging": {"level": "INFO"}
    }
    with open(config_path / "default.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(default_content, f)

    # institution.yaml
    institution_content = {
        "setting1": "institution_override1", # Overrides default
        "institution_specific": "inst_value",
        "nested": {"keyB": "institution_B_override", "keyC": "institution_C"} # Deep merge
    }
    with open(config_path / "institution.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(institution_content, f)
        
    # local.yaml (optional)
    local_content = {
        "local_setting": "local_val",
        "nested": {"keyA": "local_A_override"}
    }
    with open(config_path / "local.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(local_content, f)
        
    yield config_path
    
    shutil.rmtree(config_path)


# --- Test Classes ---

class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_setup_logging_creates_files(self, temp_logs_dir: Path, caplog):
        """Tests that log files are created in the specified directory."""
        logger = setup_logging(logs_dir=temp_logs_dir, log_level="DEBUG", console_output=False)
        
        assert (temp_logs_dir / "ai_gate.log").exists()
        assert (temp_logs_dir / "ai_gate_errors.log").exists()
        assert "Logging initialized" in caplog.text
        assert logger.name == "ai_gate"
        assert logger.level == logging.DEBUG

    def test_setup_logging_invalid_level_raises_error_and_falls_back(self, temp_logs_dir: Path, caplog):
        """Tests that an invalid log level raises ValueError then falls back to basicConfig."""
        # setup_logging catches the ValueError and uses basicConfig as fallback
        logger = setup_logging(logs_dir=temp_logs_dir, log_level="INVALID_LEVEL")
        
        # Check for fallback logging messages
        assert "Failed to setup logging: Invalid log level: INVALID_LEVEL" in caplog.text
        assert "Using basic logging configuration as fallback" in caplog.text
        assert logger.level == logging.INFO # Default for basicConfig if not set, or what basicConfig defaults to.
                                           # The fallback in setup_logging sets INFO for the logger.

    def test_setup_logging_console_output(self, temp_logs_dir: Path, capsys):
        """Tests console output enabling/disabling."""
        # Console output enabled (default)
        setup_logging(logs_dir=temp_logs_dir, log_level="INFO", console_output=True)
        logging.getLogger("ai_gate").info("Test console message enabled")
        captured_stdout_enabled = capsys.readouterr().out
        assert "Test console message enabled" in captured_stdout_enabled

        # Console output disabled
        logging.getLogger("ai_gate").handlers.clear() # Clear handlers from previous setup
        setup_logging(logs_dir=temp_logs_dir, log_level="INFO", console_output=False)
        logging.getLogger("ai_gate").info("Test console message disabled")
        captured_stdout_disabled = capsys.readouterr().out
        assert "Test console message disabled" not in captured_stdout_disabled

    def test_setup_logging_file_rotation(self, temp_logs_dir: Path):
        """
        Basic test for file rotation setup. True rotation is harder to unit test
        without writing large amounts of data. We check handler types.
        """
        logger = setup_logging(
            logs_dir=temp_logs_dir,
            max_file_size=100, # Very small for testing setup
            backup_count=2,
            console_output=False
        )
        
        file_handler_found = False
        error_handler_found = False
        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                if "errors" in handler.baseFilename:
                    error_handler_found = True
                    assert handler.maxBytes == 100
                    assert handler.backupCount == 2
                else:
                    file_handler_found = True
                    assert handler.maxBytes == 100
                    assert handler.backupCount == 2
        
        assert file_handler_found
        assert error_handler_found


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_merges_correctly(self, temp_config_dir_for_load_config: Path):
        """
        Tests that configurations from default.yaml, institution.yaml, and local.yaml
        are loaded and merged correctly, with later files overriding earlier ones.
        """
        # Mock environment variables to test AIGATE_ overrides
        test_env_vars = {
            "AIGATE_SETTING1": "env_override_top_level",
            "AIGATE_NESTED_KEYA": "env_override_nested_A",
            "AIGATE_NEW_SECTION_NEW_KEY": "env_new_value",
            "AIGATE_LOGGING_LEVEL": "DEBUG" # Override logging level
        }
        # Temporarily patch os.environ for this test
        with patch.dict(os.environ, test_env_vars):
            # Mock _validate_config_structure to avoid needing all sections for this specific test
            # Or ensure your test YAMLs create a structure that passes validation.
            # For this test, let's assume the test YAMLs might not have all required sections.
            with patch('modules.utils._validate_config_structure') as mock_validate_structure:
                config = load_config(config_dir=temp_config_dir_for_load_config)

        mock_validate_structure.assert_called_once()

        # Check top-level overrides (default -> institution -> env)
        assert config['setting1'] == "env_override_top_level"
        
        # Check institution-specific addition
        assert config['institution_specific'] == "inst_value"
        
        # Check local-specific addition
        assert config['local_setting'] == "local_val"
        
        # Check nested merging and overrides (default -> institution -> local -> env)
        assert config['nested']['keyA'] == "env_override_nested_A" # env overrides local
        assert config['nested']['keyB'] == "institution_B_override" # institution overrides default
        assert config['nested']['keyC'] == "institution_C" # added by institution
        
        # Check environment variable creating new section/key
        assert config['new_section']['new_key'] == "env_new_value"

        # Check env override for logging level
        assert config['logging']['level'] == "DEBUG"

        # Check metadata
        assert '_metadata' in config
        assert 'loaded_at' in config['_metadata']
        assert str(temp_config_dir_for_load_config) in config['_metadata']['config_dir']
        assert len(config['_metadata']['sources']) >= 3 # default, institution, local, env

    def test_load_config_missing_required_files_raises_error(self, temp_config_dir_for_load_config: Path):
        """
        Tests that FileNotFoundError is raised (and then caught by ValueError in load_config)
        if default.yaml or institution.yaml is missing.
        """
        # Test missing default.yaml
        (temp_config_dir_for_load_config / "default.yaml").unlink()
        with pytest.raises(ValueError, match="Failed to load configuration: Required default configuration not found"):
            load_config(config_dir=temp_config_dir_for_load_config)
        
        # Recreate default, remove institution
        with open(temp_config_dir_for_load_config / "default.yaml", 'w') as f: yaml.dump({}, f)
        (temp_config_dir_for_load_config / "institution.yaml").unlink()
        with pytest.raises(ValueError, match="Failed to load configuration: Required institution configuration not found"):
            load_config(config_dir=temp_config_dir_for_load_config)

    def test_load_config_invalid_yaml_raises_error(self, temp_config_dir_for_load_config: Path):
        """Tests that an error is raised if a YAML file is malformed."""
        with open(temp_config_dir_for_load_config / "default.yaml", 'w') as f:
            f.write("setting1: value_ok\ninvalid_yaml: [missing_bracket")
        
        with pytest.raises(ValueError, match="Failed to load config file"):
            load_config(config_dir=temp_config_dir_for_load_config)

    def test_load_config_handles_empty_yaml_files(self, temp_config_dir_for_load_config: Path):
        """Tests that empty YAML files are handled as empty dicts and don't break loading."""
        with open(temp_config_dir_for_load_config / "local.yaml", 'w') as f:
            f.write("") # Empty file
        
        # Should load without error, merging other files
        with patch('modules.utils._validate_config_structure'): # Avoid validation for this specific case
            config = load_config(config_dir=temp_config_dir_for_load_config)
        assert 'local_setting' not in config # Empty local.yaml means no 'local_setting'

    def test_validate_config_structure_missing_sections(self, default_config_data):
        """
        Tests _validate_config_structure for missing sections.
        This tests the helper directly.
        """
        from modules.utils import _validate_config_structure
        
        # A config missing 'ai_models' and 'prompts'
        incomplete_config = {
            'institution': {}, 'cache': {}, 'question_processing': {}, 'website_research': {}
        }
        with pytest.raises(ValueError, match="Missing required configuration sections: \\['ai_models', 'prompts'\\]"):
            _validate_config_structure(incomplete_config)

        # A complete enough config (using default_config_data from conftest)
        # This assumes default_config_data from conftest.py has all required sections.
        try:
            _validate_config_structure(default_config_data)
        except ValueError:
            pytest.fail("_validate_config_structure raised ValueError on a supposedly complete config.")


class TestValidateEnvironment:
    """Tests for the validate_environment function."""

    @patch('modules.utils.sys.version_info', (3, 7, 0)) # Simulate older Python
    def test_validate_environment_python_version_too_low(self, temp_test_env_for_utils: Path, caplog):
        """Tests Python version check (too low)."""
        with patch.dict(os.environ, {}, clear=True): # No env vars
            with patch('modules.utils.os.access', return_value=True): # Assume dir access
                 with patch('pathlib.Path.exists', return_value=True): # Assume files/dirs exist
                    assert validate_environment() is False # Should fail due to Python version
        assert "Python 3.8+ required, found 3.7" in caplog.text

    @patch('modules.utils.sys.version_info', (3, 8, 5)) # Simulate compatible Python
    def test_validate_environment_missing_env_vars(self, temp_test_env_for_utils: Path, caplog):
        """Tests check for missing required environment variables."""
        # Clear OPENROUTER_API_KEY and OPENROUTER_API_URL
        with patch.dict(os.environ, {"OTHER_VAR": "value"}, clear=True):
            with patch('modules.utils.os.access', return_value=True):
                 with patch('pathlib.Path.exists', return_value=True):
                    assert validate_environment() is False
        assert "Required environment variable missing: OPENROUTER_API_KEY" in caplog.text
        assert "Required environment variable missing: OPENROUTER_API_URL" in caplog.text

    @patch('modules.utils.sys.version_info', (3, 9, 0))
    @patch('modules.utils.os.access', return_value=False) # Simulate insufficient permissions
    def test_validate_environment_insufficient_dir_permissions(self, mock_access, temp_test_env_for_utils: Path, caplog):
        """Tests check for directory permissions."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "key", "OPENROUTER_API_URL": "url"}):
            with patch('pathlib.Path.exists', return_value=True): # Assume files/dirs exist
                assert validate_environment() is False
        assert "Insufficient permissions for directory" in caplog.text

    @patch('modules.utils.sys.version_info', (3, 9, 0))
    @patch('modules.utils.os.access', return_value=True)
    @patch('pathlib.Path.exists', side_effect=lambda p: "pages.json" not in str(p)) # Simulate pages.json missing
    def test_validate_environment_missing_required_file(self, mock_exists, temp_test_env_for_utils: Path, caplog):
        """Tests check for missing required files (e.g., data/pages.json)."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "key", "OPENROUTER_API_URL": "url"}):
            assert validate_environment() is False
        assert "Required file missing: data/pages.json" in caplog.text # Path is relative to CWD mock

    @patch('modules.utils.sys.version_info', (3, 9, 0))
    @patch('modules.utils.os.access', return_value=True)
    @patch('pathlib.Path.exists', return_value=True)
    @patch('modules.utils.requests.get') # Mock requests.get
    def test_validate_environment_openrouter_api_failure(self, mock_requests_get, temp_test_env_for_utils: Path, caplog):
        """Tests OpenRouter API connectivity check failure."""
        mock_response = MagicMock()
        mock_response.status_code = 401 # Unauthorized
        mock_requests_get.return_value = mock_response
        
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "key", "OPENROUTER_API_URL": "url"}):
            assert validate_environment() is True # API failure is a warning, not a hard fail for validate_environment
        assert "OpenRouter API test failed with status 401" in caplog.text
        mock_requests_get.assert_called_once()

    @patch('modules.utils.sys.version_info', (3, 9, 0))
    @patch('modules.utils.os.access', return_value=True)
    @patch('pathlib.Path.exists', return_value=True)
    @patch('modules.utils.requests.get')
    def test_validate_environment_all_checks_pass(self, mock_requests_get, temp_test_env_for_utils: Path, caplog):
        """Tests validate_environment when all checks are expected to pass."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        # Mock psutil if used for system info to avoid platform differences
        with patch('modules.utils.psutil.virtual_memory') as mock_vm, \
             patch('modules.utils.psutil.disk_usage') as mock_disk:
            mock_vm.return_value.total = 4 * (1024**3) # 4GB RAM
            mock_disk.return_value.free = 100 * (1024**3) # 100GB Free Disk

            with patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "valid_key",
                "OPENROUTER_API_URL": "https://valid.url/api/v1",
                "DEBUG": "true", "LOG_LEVEL": "DEBUG" # Optional vars
            }):
                assert validate_environment() is True
        
        assert "Environment validation passed successfully" in caplog.text
        assert "OpenRouter API connectivity test passed" in caplog.text
        assert "System Info - Python: 3.9" in caplog.text # Part of the version


class TestHelperFunctions:
    """Tests for miscellaneous helper functions."""

    @pytest.mark.parametrize("data, algorithm, expected_hash_start", [
        ("hello world", "md5", "5eb63bbbe01eeed093cb22bb8f5acdc3"), # Full MD5
        ("hello world", "sha256", "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"), # Full SHA256
        ("Test Data 123!", "sha1", "f1ade96def4c7acab5515357a819367aa752bd66"), # Full SHA1
    ])
    def test_generate_hash(self, data: str, algorithm: str, expected_hash_start: str):
        """Tests generate_hash with different algorithms."""
        # For algorithms like sha256, comparing start might be enough if full hash is too long for parametrize
        generated_hash = generate_hash(data, algorithm)
        assert generated_hash.startswith(expected_hash_start)
        if len(expected_hash_start) == len(generated_hash): # If full hash is provided
            assert generated_hash == expected_hash_start


    def test_generate_hash_invalid_algorithm(self):
        """Tests generate_hash with an unsupported algorithm."""
        with pytest.raises(ValueError, match="Unsupported hash algorithm: invalid_algo"):
            generate_hash("data", "invalid_algo")

    @pytest.mark.parametrize("filename, expected_sanitized", [
        ("file*name?.txt", "file_name_.txt"),
        ("  My Document / Version 2.docx  ", "My Document _ Version 2.docx"),
        ("<script>alert.js</script>", "_script_alert.js__script_"), # Behavior of current regex
        ("file_with_very_long_name_" + ("a"*300) + ".txt", "file_with_very_long_name_" + ("a"* (255-30-4)) + ".txt"), # Max length test
        (".hiddenfile", "_hiddenfile"), # Leading dot handling
        ("", "untitled"), # Empty filename
    ])
    def test_sanitize_filename(self, filename: str, expected_sanitized: str):
        """Tests sanitize_filename with various inputs."""
        # The max_length check in sanitize_filename might need adjustment if original filename is shorter than ext
        # Current sanitize_filename: available_length = max_length - len(ext)
        # If name[:available_length] is empty, it might result in just ".ext"
        # This test assumes the current logic.
        sanitized = sanitize_filename(filename)

        if "very_long_name" in filename: # Special case for length check
             assert len(sanitized) <= 255
             assert sanitized.endswith(".txt")
             assert sanitized.startswith("file_with_very_long_name_")
        else:
            assert sanitized == expected_sanitized


    @pytest.mark.parametrize("bytes_val, expected_format", [
        (100, "100.0 B"),
        (2048, "2.0 KB"),
        (1024 * 1024 * 1.5, "1.5 MB"),
        (1024**3 * 2.345, "2.3 GB"),
        (1024**4 * 5, "5.0 TB"),
    ])
    def test_format_bytes(self, bytes_val: int, expected_format: str):
        """Tests format_bytes for human-readable byte formatting."""
        assert format_bytes(bytes_val) == expected_format

    @pytest.mark.parametrize("seconds, expected_duration", [
        (0.5, "500ms"),
        (30.25, "30.3s"),
        (125, "2m 5s"), # 2 minutes and 5 seconds
        (3661, "1h 1m"), # 1 hour and 1 minute (and 1 second, but format is H M)
    ])
    def test_format_duration(self, seconds: float, expected_duration: str):
        """Tests format_duration for human-readable time duration."""
        assert format_duration(seconds) == expected_duration