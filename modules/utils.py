"""
AI Gate for Artificial Intelligence Applications
Utility functions and helper classes

This module provides common utility functions used throughout the AI Gate system,
including logging setup, configuration management, environment validation,
and other shared functionality.
"""

import os
import json
import yaml
import logging
import logging.handlers
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import hashlib
import re
import sys
import platform
from dataclasses import dataclass
from enum import Enum

# Third-party imports
import requests
from dotenv import load_dotenv


class LogLevel(Enum):
    """Enumeration for log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ConfigFormat(Enum):
    """Enumeration for configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    YML = "yml"


@dataclass
class SystemInfo:
    """System information dataclass."""
    python_version: str
    platform: str
    architecture: str
    processor: str
    memory_gb: float
    disk_space_gb: float


def setup_logging(
    logs_dir: Path,
    log_level: str = "INFO",
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup comprehensive logging configuration for the application.
    
    Args:
        logs_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_file_size: Maximum size of each log file in bytes
        backup_count: Number of backup log files to keep
        console_output: Whether to output logs to console
        log_format: Custom log format string
        
    Returns:
        logging.Logger: Configured logger instance
        
    Raises:
        ValueError: If log_level is invalid
        OSError: If logs directory cannot be created
    """
    try:
        # Validate log level
        if log_level.upper() not in [level.value for level in LogLevel]:
            raise ValueError(f"Invalid log level: {log_level}")
        
        # Ensure logs directory exists
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Default log format
        if log_format is None:
            log_format = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "[%(filename)s:%(lineno)d] - %(message)s"
            )
        
        # Create formatter
        formatter = logging.Formatter(
            fmt=log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Get root logger
        logger = logging.getLogger("ai_gate")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=logs_dir / "ai_gate.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Error-specific file handler
        error_handler = logging.handlers.RotatingFileHandler(
            filename=logs_dir / "ai_gate_errors.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            
            # Simpler format for console
            console_formatter = logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        # Log successful setup
        logger.info(f"Logging initialized - Level: {log_level}, Directory: {logs_dir}")
        logger.info(f"Log files: ai_gate.log (all), ai_gate_errors.log (errors only)")
        
        return logger
        
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger("ai_gate")
        logger.error(f"Failed to setup logging: {e}")
        logger.info("Using basic logging configuration as fallback")
        return logger


def load_config(config_dir: Path) -> Dict[str, Any]:
    """
    Load and merge application configuration from multiple sources.
    
    Loads configuration in the following order (later configs override earlier ones):
    1. config/default.yaml - Default configuration
    2. config/institution.yaml - Institution-specific configuration
    3. config/local.yaml - Local overrides (optional, gitignored)
    4. Environment variables (prefixed with AIGATE_)
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        Dict[str, Any]: Merged configuration dictionary
        
    Raises:
        FileNotFoundError: If required configuration files are missing
        ValueError: If configuration files contain invalid data
    """
    try:
        config_dir = Path(config_dir)
        merged_config = {}
        
        # Load default configuration (required)
        default_config_path = config_dir / "default.yaml"
        if not default_config_path.exists():
            raise FileNotFoundError(f"Required default configuration not found: {default_config_path}")
        
        merged_config = _load_config_file(default_config_path)
        
        # Load institution configuration (required)
        institution_config_path = config_dir / "institution.yaml"
        if not institution_config_path.exists():
            raise FileNotFoundError(f"Required institution configuration not found: {institution_config_path}")
        
        institution_config = _load_config_file(institution_config_path)
        merged_config = _deep_merge_dict(merged_config, institution_config)
        
        # Load local configuration (optional)
        local_config_path = config_dir / "local.yaml"
        if local_config_path.exists():
            local_config = _load_config_file(local_config_path)
            merged_config = _deep_merge_dict(merged_config, local_config)
        
        # Override with environment variables
        env_overrides = _load_env_overrides()
        if env_overrides:
            merged_config = _deep_merge_dict(merged_config, env_overrides)
        
        # Validate configuration structure
        _validate_config_structure(merged_config)
        
        # Add runtime metadata
        merged_config['_metadata'] = {
            'loaded_at': datetime.now().isoformat(),
            'config_dir': str(config_dir),
            'sources': _get_config_sources(config_dir)
        }
        
        return merged_config
        
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")


def validate_environment() -> bool:
    """
    Validate that the application environment is properly configured.
    
    Checks for:
    - Required environment variables
    - Python version compatibility
    - Required directories and permissions
    - External service connectivity
    - System resources
    
    Returns:
        bool: True if environment is valid, False otherwise
    """
    logger = logging.getLogger("ai_gate")
    validation_errors = []
    warnings = []
    
    try:
        logger.info("Starting environment validation...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            validation_errors.append(
                f"Python 3.8+ required, found {python_version.major}.{python_version.minor}"
            )
        elif python_version < (3, 9):
            warnings.append(
                f"Python 3.9+ recommended, found {python_version.major}.{python_version.minor}"
            )
        
        # Check required environment variables
        required_env_vars = [
            "OPENROUTER_API_KEY",
            "OPENROUTER_API_URL"
        ]
        
        for var in required_env_vars:
            if not os.getenv(var):
                validation_errors.append(f"Required environment variable missing: {var}")
        
        # Check optional but important environment variables
        optional_env_vars = {
            "DEBUG": "false",
            "LOG_LEVEL": "INFO",
            "CACHE_MAX_SIZE": "1000",
            "CACHE_TTL": "3600"
        }
        
        for var, default in optional_env_vars.items():
            if not os.getenv(var):
                warnings.append(f"Optional environment variable not set: {var} (using default: {default})")
        
        # Check directory structure and permissions
        base_dir = Path.cwd()
        required_dirs = [
            "config",
            "data",
            "static",
            "logs",
            "modules"
        ]
        
        for dir_name in required_dirs:
            dir_path = base_dir / dir_name
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created missing directory: {dir_path}")
                except OSError as e:
                    validation_errors.append(f"Cannot create required directory {dir_path}: {e}")
            elif not os.access(dir_path, os.R_OK | os.W_OK):
                validation_errors.append(f"Insufficient permissions for directory: {dir_path}")
        
        # Check required files
        required_files = [
            "config/default.yaml",
            "config/institution.yaml",
            "data/pages.json"
        ]
        
        for file_path in required_files:
            full_path = base_dir / file_path
            if not full_path.exists():
                validation_errors.append(f"Required file missing: {file_path}")
            elif not os.access(full_path, os.R_OK):
                validation_errors.append(f"Cannot read required file: {file_path}")
        
        # Test OpenRouter API connectivity
        api_key = os.getenv("OPENROUTER_API_KEY")
        api_url = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
        
        if api_key:
            try:
                test_url = f"{api_url.rstrip('/')}/models"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://ai-gate.local",
                    "X-Title": "AI Gate Application"
                }
                
                response = requests.get(test_url, headers=headers, timeout=10)
                if response.status_code != 200:
                    warnings.append(f"OpenRouter API test failed with status {response.status_code}")
                else:
                    logger.info("OpenRouter API connectivity test passed")
                    
            except requests.RequestException as e:
                warnings.append(f"OpenRouter API connectivity test failed: {e}")
        
        # Check system resources
        system_info = _get_system_info()
        
        if system_info.memory_gb < 1.0:
            warnings.append(f"Low system memory: {system_info.memory_gb:.1f}GB (2GB+ recommended)")
        
        if system_info.disk_space_gb < 1.0:
            warnings.append(f"Low disk space: {system_info.disk_space_gb:.1f}GB")
        
        # Log validation results
        if validation_errors:
            logger.error("Environment validation failed:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            return False
        
        if warnings:
            logger.warning("Environment validation passed with warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("Environment validation passed successfully")
        
        # Log system information
        logger.info(f"System Info - Python: {system_info.python_version}, "
                   f"Platform: {system_info.platform}, Memory: {system_info.memory_gb:.1f}GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Environment validation error: {e}")
        return False


def generate_hash(data: str, algorithm: str = "sha256") -> str:
    """
    Generate hash for given data.
    
    Args:
        data: String data to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        
    Returns:
        str: Hexadecimal hash string
        
    Raises:
        ValueError: If algorithm is not supported
    """
    try:
        if algorithm not in ["md5", "sha1", "sha256", "sha512"]:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode('utf-8'))
        return hash_obj.hexdigest()
        
    except Exception as e:
        raise ValueError(f"Failed to generate hash: {e}")


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    
    # Trim whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure not empty
    if not sanitized:
        sanitized = "untitled"
    
    # Truncate if too long
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        available_length = max_length - len(ext)
        sanitized = name[:available_length] + ext
    
    return sanitized


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable format.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        str: Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:.0f}m {remaining_seconds:.0f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {remaining_minutes:.0f}m"


# Private helper functions

def _load_config_file(file_path: Path) -> Dict[str, Any]:
    """Load configuration from a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif file_path.suffix.lower() == '.json':
                return json.load(f) or {}
            else:
                raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    except Exception as e:
        raise ValueError(f"Failed to load config file {file_path}: {e}")


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


def _load_env_overrides() -> Dict[str, Any]:
    """Load configuration overrides from environment variables."""
    overrides = {}
    prefix = "AIGATE_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Convert AIGATE_SECTION_OPTION to nested dict
            config_key = key[len(prefix):].lower()
            key_parts = config_key.split('_')
            
            # Try to convert value to appropriate type
            parsed_value = _parse_env_value(value)
            
            # Create nested dictionary structure
            current_dict = overrides
            for part in key_parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            
            current_dict[key_parts[-1]] = parsed_value
    
    return overrides


def _parse_env_value(value: str) -> Union[str, int, float, bool, List[str]]:
    """Parse environment variable value to appropriate type."""
    # Boolean values
    if value.lower() in ['true', 'yes', '1', 'on']:
        return True
    elif value.lower() in ['false', 'no', '0', 'off']:
        return False
    
    # Numeric values
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass
    
    # List values (comma-separated)
    if ',' in value:
        return [item.strip() for item in value.split(',')]
    
    # String value
    return value


def _validate_config_structure(config: Dict[str, Any]) -> None:
    """Validate configuration structure has required sections."""
    required_sections = [
        'institution',
        'ai_models',
        'cache',
        'question_processing',
        'website_research',
        'prompts'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)
    
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")


def _get_config_sources(config_dir: Path) -> List[str]:
    """Get list of configuration sources that were loaded."""
    sources = []
    
    # Check which config files exist
    config_files = [
        "default.yaml",
        "institution.yaml",
        "local.yaml"
    ]
    
    for filename in config_files:
        file_path = config_dir / filename
        if file_path.exists():
            sources.append(str(file_path))
    
    # Add environment variables if any AIGATE_ vars exist
    env_vars = [key for key in os.environ.keys() if key.startswith("AIGATE_")]
    if env_vars:
        sources.append(f"Environment variables ({len(env_vars)} vars)")
    
    return sources


def _get_system_info() -> SystemInfo:
    """Get system information."""
    try:
        import psutil
        
        # Get memory info
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.total / (1024**3)
        
        # Get disk space
        disk_info = psutil.disk_usage('/')
        disk_space_gb = disk_info.free / (1024**3)
        
    except ImportError:
        # Fallback without psutil
        memory_gb = 0.0
        disk_space_gb = 0.0
    
    return SystemInfo(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=platform.system(),
        architecture=platform.machine(),
        processor=platform.processor() or "Unknown",
        memory_gb=memory_gb,
        disk_space_gb=disk_space_gb
    )
