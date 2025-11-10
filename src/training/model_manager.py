"""
Model version management module.

Handles saving, loading, and versioning of trained models with metadata.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import torch


class ModelManagerError(Exception):
    """Exception raised for model management errors."""
    pass


class ModelManager:
    """
    Model version manager.

    Manages model checkpoints, versioning, and metadata for all model types.
    Creates organized directory structure and tracks model performance.
    """

    def __init__(self, models_dir: str = 'models'):
        """
        Initialize model manager.

        Args:
            models_dir: Base directory for model storage
        """
        self.models_dir = Path(models_dir)
        self._create_directory_structure()

    def _create_directory_structure(self):
        """Create organized directory structure for models."""
        for model_type in ['detection', 'view', 'defect']:
            (self.models_dir / model_type).mkdir(parents=True, exist_ok=True)

    def generate_version_string(
        self,
        metrics: Optional[Dict[str, float]] = None,
        prefix: str = 'v1'
    ) -> str:
        """
        Generate version string with timestamp and metrics.

        Args:
            metrics: Optional dictionary of metrics to include in version
            prefix: Version prefix (default: 'v1')

        Returns:
            Version string (e.g., 'v1_20250106_143022_map0.85')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = f"{prefix}_{timestamp}"

        if metrics:
            # Add key metric to version string
            if 'map50' in metrics:
                version += f"_map{metrics['map50']:.2f}"
            elif 'accuracy' in metrics:
                version += f"_acc{metrics['accuracy']:.2f}"
            elif 'balanced_accuracy' in metrics:
                version += f"_bacc{metrics['balanced_accuracy']:.2f}"

        return version

    def get_model_path(
        self,
        model_type: str,
        version: str,
        extension: str = '.pt'
    ) -> Path:
        """
        Get full path for a model file.

        Args:
            model_type: Type of model ('detection', 'view', 'defect')
            version: Version string
            extension: File extension (default: '.pt')

        Returns:
            Path object for model file
        """
        filename = f"{version}{extension}"
        return self.models_dir / model_type / filename

    def save_model(
        self,
        model: Any,
        model_type: str,
        model_name: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        version_prefix: str = 'v1',
        additional_info: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Save model with metadata and versioning.

        Args:
            model: Model object to save (PyTorch model or YOLO model)
            model_type: Type of model ('detection', 'view', 'defect')
            model_name: Name of the model architecture
            metrics: Dictionary of evaluation metrics
            config: Training configuration
            version_prefix: Version prefix
            additional_info: Optional additional metadata

        Returns:
            Dictionary with saved file paths and version info

        Raises:
            ModelManagerError: If saving fails
        """
        try:
            # Generate version
            version = self.generate_version_string(metrics, version_prefix)

            # Get save paths
            model_path = self.get_model_path(model_type, version)
            metadata_path = model_path.with_suffix('.json')

            # Save model
            if hasattr(model, 'save'):
                # YOLO model
                model.save(str(model_path))
            elif hasattr(model, 'state_dict'):
                # PyTorch model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_name': model_name,
                    'metrics': metrics,
                    'config': config
                }, model_path)
            else:
                raise ModelManagerError(f"Unknown model type: {type(model)}")

            # Save metadata
            metadata = {
                'version': version,
                'model_type': model_type,
                'model_name': model_name,
                'metrics': metrics,
                'config': config,
                'saved_at': datetime.now().isoformat(),
                'model_path': str(model_path),
                'file_size_mb': model_path.stat().st_size / (1024 * 1024)
            }

            if additional_info:
                metadata['additional_info'] = additional_info

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            return {
                'version': version,
                'model_path': str(model_path),
                'metadata_path': str(metadata_path)
            }

        except Exception as e:
            raise ModelManagerError(f"Failed to save model: {e}")

    def load_model(
        self,
        model_path: str,
        model_class: Optional[Any] = None,
        device: str = 'cpu'
    ) -> tuple[Any, Dict]:
        """
        Load model and metadata.

        Args:
            model_path: Path to model file
            model_class: Optional model class for PyTorch models
            device: Device to load model on

        Returns:
            Tuple of (model, metadata)

        Raises:
            ModelManagerError: If loading fails
        """
        try:
            model_path = Path(model_path)
            metadata_path = model_path.with_suffix('.json')

            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Load model
            if model_path.suffix == '.pt':
                if metadata.get('model_type') == 'detection':
                    # YOLO model
                    from ultralytics import YOLO
                    model = YOLO(str(model_path))
                else:
                    # PyTorch model
                    checkpoint = torch.load(model_path, map_location=device)

                    if model_class is not None:
                        model = model_class
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.to(device)
                    else:
                        model = checkpoint

            else:
                raise ModelManagerError(f"Unsupported model format: {model_path.suffix}")

            return model, metadata

        except Exception as e:
            raise ModelManagerError(f"Failed to load model: {e}")

    def list_models(
        self,
        model_type: Optional[str] = None,
        sort_by: str = 'date',
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        List available model versions.

        Args:
            model_type: Filter by model type (optional)
            sort_by: Sort criteria ('date', 'metric', 'size')
            limit: Maximum number of results

        Returns:
            List of model information dictionaries
        """
        models = []

        # Determine which directories to search
        if model_type:
            search_dirs = [self.models_dir / model_type]
        else:
            search_dirs = [
                self.models_dir / 'detection',
                self.models_dir / 'view',
                self.models_dir / 'defect'
            ]

        # Find all model files
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for model_path in search_dir.glob('*.pt'):
                metadata_path = model_path.with_suffix('.json')

                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                else:
                    metadata = {
                        'version': model_path.stem,
                        'model_type': model_path.parent.name,
                        'model_path': str(model_path)
                    }

                models.append(metadata)

        # Sort
        if sort_by == 'date':
            models.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
        elif sort_by == 'metric':
            # Sort by first available metric
            def get_metric_value(m):
                metrics = m.get('metrics', {})
                for key in ['map50', 'accuracy', 'balanced_accuracy']:
                    if key in metrics:
                        return metrics[key]
                return 0
            models.sort(key=get_metric_value, reverse=True)
        elif sort_by == 'size':
            models.sort(key=lambda x: x.get('file_size_mb', 0), reverse=True)

        # Limit results
        if limit:
            models = models[:limit]

        return models

    def delete_model(
        self,
        model_path: str,
        delete_metadata: bool = True
    ) -> bool:
        """
        Delete a model and optionally its metadata.

        Args:
            model_path: Path to model file
            delete_metadata: Whether to delete metadata file too

        Returns:
            True if deletion was successful

        Raises:
            ModelManagerError: If deletion fails
        """
        try:
            model_path = Path(model_path)

            if model_path.exists():
                model_path.unlink()

            if delete_metadata:
                metadata_path = model_path.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()

            return True

        except Exception as e:
            raise ModelManagerError(f"Failed to delete model: {e}")

    def get_best_model(
        self,
        model_type: str,
        metric: str = 'auto'
    ) -> Optional[Dict]:
        """
        Get the best model of a type based on metrics.

        Args:
            model_type: Type of model
            metric: Metric to optimize ('auto', 'map50', 'accuracy', 'balanced_accuracy')

        Returns:
            Model metadata dictionary or None if no models found
        """
        models = self.list_models(model_type=model_type)

        if not models:
            return None

        # Auto-select metric based on model type
        if metric == 'auto':
            if model_type == 'detection':
                metric = 'map50'
            elif model_type == 'view':
                metric = 'accuracy'
            elif model_type == 'defect':
                metric = 'balanced_accuracy'

        # Find best model
        best_model = None
        best_value = -1

        for model in models:
            metrics = model.get('metrics', {})
            if metric in metrics:
                value = metrics[metric]
                if value > best_value:
                    best_value = value
                    best_model = model

        return best_model

    def compare_models(
        self,
        model_paths: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple models side-by-side.

        Args:
            model_paths: List of model file paths

        Returns:
            Comparison dictionary with metrics
        """
        comparison = {
            'models': [],
            'metrics': {}
        }

        # Load metadata for each model
        for model_path in model_paths:
            metadata_path = Path(model_path).with_suffix('.json')

            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {'model_path': model_path}

            comparison['models'].append(metadata)

            # Collect all metric names
            for metric_name in metadata.get('metrics', {}).keys():
                if metric_name not in comparison['metrics']:
                    comparison['metrics'][metric_name] = []

        # Build metric comparison table
        for metric_name in comparison['metrics'].keys():
            for model_metadata in comparison['models']:
                value = model_metadata.get('metrics', {}).get(metric_name, None)
                comparison['metrics'][metric_name].append(value)

        return comparison

    def export_model(
        self,
        model_path: str,
        export_format: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export model to different format (ONNX, TorchScript, etc.).

        Args:
            model_path: Path to model file
            export_format: Export format ('onnx', 'torchscript', 'tflite')
            output_path: Optional output path

        Returns:
            Path to exported model

        Raises:
            ModelManagerError: If export fails
        """
        try:
            # Load model
            model, metadata = self.load_model(model_path)

            if output_path is None:
                output_path = Path(model_path).with_suffix(f'.{export_format}')

            # Export based on format
            if export_format == 'onnx':
                if hasattr(model, 'export'):
                    # YOLO model
                    model.export(format='onnx', path=output_path)
                else:
                    # PyTorch model
                    dummy_input = torch.randn(1, 3, 1004, 1004)
                    torch.onnx.export(
                        model, dummy_input, output_path,
                        opset_version=11,
                        input_names=['input'],
                        output_names=['output']
                    )

            elif export_format == 'torchscript':
                scripted_model = torch.jit.script(model)
                scripted_model.save(output_path)

            else:
                raise ModelManagerError(f"Unsupported export format: {export_format}")

            return str(output_path)

        except Exception as e:
            raise ModelManagerError(f"Failed to export model: {e}")

    def cleanup_old_models(
        self,
        model_type: str,
        keep_best_n: int = 5,
        metric: str = 'auto'
    ) -> int:
        """
        Clean up old models, keeping only the best N.

        Args:
            model_type: Type of model
            keep_best_n: Number of best models to keep
            metric: Metric to determine "best"

        Returns:
            Number of models deleted
        """
        models = self.list_models(model_type=model_type, sort_by='metric')

        if len(models) <= keep_best_n:
            return 0

        # Delete excess models
        deleted_count = 0
        for model in models[keep_best_n:]:
            model_path = model.get('model_path')
            if model_path:
                self.delete_model(model_path, delete_metadata=True)
                deleted_count += 1

        return deleted_count


# Test if run directly
if __name__ == "__main__":
    print("Testing model manager...")

    manager = ModelManager(models_dir='test_models')

    # Create dummy model and save
    dummy_model = torch.nn.Linear(10, 2)
    metrics = {'accuracy': 0.95, 'loss': 0.1}
    config = {'epochs': 50, 'batch_size': 32}

    result = manager.save_model(
        dummy_model,
        model_type='view',
        model_name='resnet18',
        metrics=metrics,
        config=config
    )

    print(f"Saved model: {result}")

    # List models
    models = manager.list_models(model_type='view')
    print(f"\nFound {len(models)} view models")

    # Get best model
    best = manager.get_best_model('view', metric='accuracy')
    if best:
        print(f"\nBest view model: {best['version']} (accuracy: {best['metrics']['accuracy']})")

    print("\nModel manager test complete!")
