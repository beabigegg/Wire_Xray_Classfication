"""
Configuration management module.

This module handles loading and validation of application configuration
from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


class Config:
    """
    Application configuration manager.

    Loads configuration from YAML file and provides type-safe access
    to configuration values.
    """

    DEFAULT_CONFIG = {
        'application': {
            'name': 'Wire Loop Annotation Tool',
            'version': '1.0.0',
            'author': 'Unknown'
        },
        'paths': {
            'database': 'annotations.db',
            'data_dir': 'data',
            'labels_dir': 'labels',
            'models_dir': 'models',
            'runs_dir': 'runs'
        },
        'image': {
            'width': 1004,
            'height': 1004,
            'supported_formats': ['.png', '.jpg', '.jpeg', '.bmp']
        },
        'annotation': {
            'view_types': ['TOP', 'SIDE'],
            'defect_types': ['PASS', '沖線', '晃動', '碰觸'],
            'default_view': 'TOP',
            'default_defect': 'PASS',
            'default_annotator': 'unknown'
        },
        'gui': {
            'window_title': 'Wire Loop Annotation Tool',
            'window_width': 1600,
            'window_height': 900,
            'canvas_bg_color': '#2b2b2b',
            'bbox_color': '#00ff00',
            'bbox_selected_color': '#ff0000',
            'bbox_line_width': 2,
            'label_font_size': 10,
            'image_list_filter': 'All',
            'image_list_sort': 'Name'
        },
        'shortcuts': {
            'draw': 'W',
            'next': 'D',
            'prev': 'A',
            'save': 'S',
            'delete': 'Delete',
            'cancel': 'Escape',
            'open': 'Ctrl+O',
            'undo': 'Ctrl+Z',
            'quit': 'Ctrl+Q'
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default config.

        Raises:
            ConfigError: If config file cannot be loaded
        """
        self._config = self.DEFAULT_CONFIG.copy()
        self._config_path = config_path or 'config.yaml'

        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        elif config_path:
            raise ConfigError(f"Config file not found: {config_path}")

        self._validate()

    def _load_from_file(self, config_path: str):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Raises:
            ConfigError: If file cannot be loaded or parsed
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)

            if user_config:
                self._merge_config(self._config, user_config)

        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse config file: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load config file: {e}")

    def _merge_config(self, base: Dict, override: Dict):
        """
        Recursively merge override config into base config.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _validate(self):
        """
        Validate configuration values.

        Raises:
            ConfigError: If configuration is invalid
        """
        # Validate image dimensions
        if self.get('image.width') != 1004 or self.get('image.height') != 1004:
            raise ConfigError("Image dimensions must be 1004x1004")

        # Validate view types
        view_types = self.get('annotation.view_types', [])
        if not view_types or not all(isinstance(v, str) for v in view_types):
            raise ConfigError("Invalid view_types configuration")

        # Validate defect types
        defect_types = self.get('annotation.defect_types', [])
        if not defect_types or not all(isinstance(d, str) for d in defect_types):
            raise ConfigError("Invalid defect_types configuration")

        # Validate default values
        default_view = self.get('annotation.default_view')
        if default_view not in view_types:
            raise ConfigError(f"default_view '{default_view}' not in view_types")

        default_defect = self.get('annotation.default_defect')
        if default_defect not in defect_types:
            raise ConfigError(f"default_defect '{default_defect}' not in defect_types")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., 'image.width')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config.get('image.width')
            1004
            >>> config.get('gui.window_title')
            'Wire Loop Annotation Tool'
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., 'annotation.default_annotator')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, config_path: Optional[str] = None):
        """
        Save current configuration to YAML file.

        Args:
            config_path: Path to output YAML file. If None, uses stored path.

        Raises:
            ConfigError: If file cannot be written
        """
        path = config_path or self._config_path
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

        except Exception as e:
            raise ConfigError(f"Failed to save config file: {e}")

    def to_dict(self) -> Dict:
        """
        Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    @property
    def image_width(self) -> int:
        """Get image width."""
        return self.get('image.width')

    @property
    def image_height(self) -> int:
        """Get image height."""
        return self.get('image.height')

    @property
    def view_types(self) -> list:
        """Get list of view types."""
        return self.get('annotation.view_types', [])

    @property
    def defect_types(self) -> list:
        """Get list of defect types."""
        return self.get('annotation.defect_types', [])

    @property
    def database_path(self) -> str:
        """Get database file path."""
        return self.get('paths.database')

    @property
    def data_dir(self) -> str:
        """Get data directory path."""
        return self.get('paths.data_dir')

    @property
    def labels_dir(self) -> str:
        """Get labels directory path."""
        return self.get('paths.labels_dir')

    @property
    def default_annotator(self) -> str:
        """Get default annotator name."""
        return self.get('annotation.default_annotator', 'unknown')


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Path to YAML config file (optional)

    Returns:
        Config instance

    Raises:
        ConfigError: If config cannot be loaded
    """
    return Config(config_path)


def create_default_config(output_path: str):
    """
    Create a default configuration file.

    Args:
        output_path: Path to output YAML file

    Raises:
        ConfigError: If file cannot be written
    """
    config = Config()
    config.save(output_path)
