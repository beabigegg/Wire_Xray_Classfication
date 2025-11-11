"""
Training configuration management module.

Handles loading, validation, and management of training configurations
for different model types.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


@dataclass
class YOLOConfig:
    """Configuration for YOLO detection model training."""
    model_name: str = 'yolov8m'  # yolov8n, yolov8s, yolov8m (upgraded for accuracy)
    epochs: int = 100
    imgsz: int = 1004
    batch: int = 12  # Reduced for larger model
    lr0: float = 0.01
    patience: int = 20
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    workers: int = 4
    pretrained: bool = True
    optimizer: str = 'SGD'
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    save_period: int = 10
    val_interval: int = 1


@dataclass
class ViewClassifierConfig:
    """Configuration for view classifier training."""
    model_name: str = 'resnet50'  # Upgraded from resnet18 for higher accuracy
    num_classes: int = 2
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    device: str = 'auto'
    num_workers: int = 4
    early_stopping_patience: int = 10
    save_best_only: bool = True
    optimizer: str = 'Adam'
    scheduler: str = 'ReduceLROnPlateau'
    scheduler_params: Dict = None

    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {'mode': 'max', 'factor': 0.5, 'patience': 5}


@dataclass
class DefectClassifierConfig:
    """Configuration for defect classifier training."""
    model_name: str = 'efficientnet_b3'  # Upgraded from b0 for higher accuracy
    num_classes: int = 4
    epochs: int = 100
    batch_size: int = 12  # Reduced for larger model
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    device: str = 'auto'
    num_workers: int = 4
    early_stopping_patience: int = 15
    save_best_only: bool = True
    optimizer: str = 'Adam'
    scheduler: str = 'ReduceLROnPlateau'
    scheduler_params: Dict = None
    # Class imbalance handling
    use_class_weights: bool = True
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # Augmentation
    pass_augmentation_factor: int = 20
    heavy_augmentation_classes: list = None
    # Sampling
    use_balanced_sampling: bool = True
    samples_per_class_per_batch: int = 4

    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {'mode': 'max', 'factor': 0.5, 'patience': 5}
        if self.heavy_augmentation_classes is None:
            self.heavy_augmentation_classes = ['PASS']


class ConfigManager:
    """
    Training configuration manager.

    Handles loading, saving, and validation of training configurations
    for all model types.
    """

    DEFAULT_CONFIGS = {
        'yolo': YOLOConfig(),
        'view': ViewClassifierConfig(),
        'defect': DefectClassifierConfig()
    }

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path('configs')
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_config(
        self,
        model_type: str,
        config_path: Optional[str] = None
    ) -> Any:
        """
        Load configuration for a model type.

        Args:
            model_type: Type of model ('yolo', 'view', 'defect')
            config_path: Optional path to config file, if None uses default

        Returns:
            Configuration object (YOLOConfig, ViewClassifierConfig, or DefectClassifierConfig)

        Raises:
            ConfigError: If configuration cannot be loaded
        """
        if model_type not in self.DEFAULT_CONFIGS:
            raise ConfigError(f"Unknown model type: {model_type}")

        # Start with default config
        config_class = type(self.DEFAULT_CONFIGS[model_type])
        config_dict = asdict(self.DEFAULT_CONFIGS[model_type])

        # Load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)

                # Merge user config
                if user_config:
                    config_dict.update(user_config)

            except Exception as e:
                raise ConfigError(f"Failed to load config from {config_path}: {e}")

        # Create config object
        try:
            config = config_class(**config_dict)
        except TypeError as e:
            raise ConfigError(f"Invalid configuration parameters: {e}")

        return config

    def save_config(
        self,
        config: Any,
        model_type: str,
        config_name: Optional[str] = None
    ) -> str:
        """
        Save configuration to file.

        Args:
            config: Configuration object
            model_type: Type of model ('yolo', 'view', 'defect')
            config_name: Optional custom config name

        Returns:
            Path to saved config file

        Raises:
            ConfigError: If configuration cannot be saved
        """
        if config_name:
            filename = f"{config_name}.yaml"
        else:
            filename = f"{model_type}_config.yaml"

        filepath = self.config_dir / filename

        try:
            config_dict = asdict(config)

            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

            return str(filepath)

        except Exception as e:
            raise ConfigError(f"Failed to save config to {filepath}: {e}")

    def create_default_configs(self):
        """Create default configuration files for all model types."""
        for model_type, config in self.DEFAULT_CONFIGS.items():
            self.save_config(config, model_type)

        print(f"Default configurations created in {self.config_dir}")

    def validate_config(
        self,
        config: Any,
        model_type: str
    ) -> tuple[bool, list[str]]:
        """
        Validate configuration parameters.

        Args:
            config: Configuration object
            model_type: Type of model

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        # Common validation
        if hasattr(config, 'epochs') and config.epochs <= 0:
            errors.append("epochs must be positive")

        if hasattr(config, 'batch_size') and config.batch_size <= 0:
            errors.append("batch_size must be positive")
        elif hasattr(config, 'batch') and config.batch <= 0:
            errors.append("batch must be positive")

        if hasattr(config, 'learning_rate') and (config.learning_rate <= 0 or config.learning_rate > 1):
            errors.append("learning_rate must be in (0, 1]")

        # Model-specific validation
        if model_type == 'yolo':
            if config.imgsz != 1004:
                errors.append("imgsz must be 1004 for this dataset")
            if config.model_name not in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']:
                errors.append(f"Unknown YOLO model: {config.model_name}")

        elif model_type == 'view':
            if config.num_classes != 2:
                errors.append("num_classes must be 2 for view classifier")

        elif model_type == 'defect':
            if config.num_classes != 4:
                errors.append("num_classes must be 4 for defect classifier")
            if config.pass_augmentation_factor < 1:
                errors.append("pass_augmentation_factor must be >= 1")

        return len(errors) == 0, errors

    def get_config_dict(self, config: Any) -> Dict:
        """
        Convert configuration object to dictionary.

        Args:
            config: Configuration object

        Returns:
            Configuration as dictionary
        """
        return asdict(config)

    def from_dict(
        self,
        config_dict: Dict,
        model_type: str
    ) -> Any:
        """
        Create configuration object from dictionary.

        Args:
            config_dict: Configuration dictionary
            model_type: Type of model

        Returns:
            Configuration object

        Raises:
            ConfigError: If model type is unknown
        """
        if model_type == 'yolo':
            return YOLOConfig(**config_dict)
        elif model_type == 'view':
            return ViewClassifierConfig(**config_dict)
        elif model_type == 'defect':
            return DefectClassifierConfig(**config_dict)
        else:
            raise ConfigError(f"Unknown model type: {model_type}")

    def override_from_args(
        self,
        config: Any,
        args_dict: Dict[str, Any]
    ) -> Any:
        """
        Override configuration with command-line arguments.

        Args:
            config: Base configuration object
            args_dict: Dictionary of argument overrides

        Returns:
            Updated configuration object
        """
        config_dict = asdict(config)

        for key, value in args_dict.items():
            if value is not None and key in config_dict:
                config_dict[key] = value

        return type(config)(**config_dict)

    def config_to_json(self, config: Any) -> str:
        """
        Convert configuration to JSON string.

        Args:
            config: Configuration object

        Returns:
            JSON string
        """
        return json.dumps(asdict(config), indent=2)

    def config_from_json(self, json_str: str, model_type: str) -> Any:
        """
        Create configuration from JSON string.

        Args:
            json_str: JSON configuration string
            model_type: Type of model

        Returns:
            Configuration object
        """
        config_dict = json.loads(json_str)
        return self.from_dict(config_dict, model_type)


# Default configuration templates
def get_default_yolo_config() -> YOLOConfig:
    """Get default YOLO configuration."""
    return YOLOConfig()


def get_default_view_config() -> ViewClassifierConfig:
    """Get default view classifier configuration."""
    return ViewClassifierConfig()


def get_default_defect_config() -> DefectClassifierConfig:
    """Get default defect classifier configuration."""
    return DefectClassifierConfig()


# Test if run directly
if __name__ == "__main__":
    print("Testing configuration manager...")

    manager = ConfigManager(config_dir='test_configs')

    # Create default configs
    manager.create_default_configs()

    # Load and validate
    for model_type in ['yolo', 'view', 'defect']:
        config = manager.load_config(model_type)
        is_valid, errors = manager.validate_config(config, model_type)

        print(f"\n{model_type.upper()} Config:")
        print(f"  Valid: {is_valid}")
        if errors:
            print(f"  Errors: {errors}")
        print(f"  Config: {manager.config_to_json(config)[:200]}...")

    print("\nConfiguration manager test complete!")
