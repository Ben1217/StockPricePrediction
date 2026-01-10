"""
Config Loader Module
Load and access configuration settings
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from .logger import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

_config_cache = {}


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file

    Parameters
    ----------
    config_path : str
        Path to configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    global _config_cache
    
    if config_path in _config_cache:
        return _config_cache[config_path]
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        _config_cache[config_path] = config
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def get_config_value(key_path: str, default: Any = None, config: dict = None) -> Any:
    """
    Get a configuration value using dot notation

    Parameters
    ----------
    key_path : str
        Dot-separated path to config value (e.g., 'models.xgboost.params.max_depth')
    default : Any
        Default value if key not found
    config : dict, optional
        Configuration dictionary (loads default if None)

    Returns
    -------
    Any
        Configuration value

    Examples
    --------
    >>> get_config_value('models.default')
    'xgboost'
    >>> get_config_value('portfolio.optimization.method')
    'max_sharpe'
    """
    if config is None:
        config = load_config()
    
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def get_env(key: str, default: str = None) -> Optional[str]:
    """
    Get environment variable

    Parameters
    ----------
    key : str
        Environment variable name
    default : str, optional
        Default value if not set

    Returns
    -------
    str or None
        Environment variable value
    """
    return os.getenv(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    """Get environment variable as integer"""
    try:
        return int(os.getenv(key, default))
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get environment variable as float"""
    try:
        return float(os.getenv(key, default))
    except ValueError:
        return default
