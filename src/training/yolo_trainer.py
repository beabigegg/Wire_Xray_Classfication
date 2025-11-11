"""
YOLO detection trainer wrapper.

This module wraps Ultralytics YOLOv8 for detection training.
Most functionality is handled by the Ultralytics library.
"""

from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. YOLO training will not work.")
    YOLO = None

from src.training.config_manager import YOLOConfig
from src.training.model_manager import ModelManager
from src.core.database import Database


class YOLOTrainer:
    """
    YOLOv8 detection trainer wrapper.

    This trainer wraps the Ultralytics YOLO library for wire defect detection.
    Most training logic is handled by YOLO's built-in training pipeline.
    """

    def __init__(
        self,
        config: YOLOConfig,
        db: Database,
        models_dir: str = 'models',
        model_type: str = 'detection'
    ):
        """
        Initialize YOLO trainer.

        Args:
            config: YOLO training configuration
            db: Database instance
            models_dir: Directory for model storage
            model_type: Model type ('detection', 'detection_top', 'detection_side')
        """
        if YOLO is None:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )

        self.config = config
        self.db = db
        self.model_type = model_type
        self.model_manager = ModelManager(models_dir=models_dir)

        # Model
        self.model = None

    def _load_model(self):
        """Load YOLO model (pretrained or from scratch)."""
        if self.config.pretrained:
            # Load pretrained model
            model_path = f'{self.config.model_name}.pt'
            print(f"Loading pretrained model: {model_path}")
            self.model = YOLO(model_path)
        else:
            # Load architecture only (train from scratch)
            model_yaml = f'{self.config.model_name}.yaml'
            print(f"Loading model architecture: {model_yaml}")
            self.model = YOLO(model_yaml)

    def train(
        self,
        data_yaml_path: str,
        output_dir: str = 'runs/detection'
    ) -> str:
        """
        Train YOLOv8 detection model.

        Args:
            data_yaml_path: Path to YOLO data.yaml configuration file
            output_dir: Output directory for training runs

        Returns:
            Path to best model checkpoint

        Raises:
            FileNotFoundError: If data_yaml_path does not exist
            RuntimeError: If training fails
        """
        # Validate data.yaml exists
        if not Path(data_yaml_path).exists():
            raise FileNotFoundError(f"Data configuration not found: {data_yaml_path}")

        # Load model
        self._load_model()

        # Setup output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'yolo_{timestamp}'

        # Auto-detect device for YOLO (YOLO doesn't accept 'auto')
        import torch
        if self.config.device == 'auto':
            device = '0' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.config.device

        # Add training run to database
        config_json = json.dumps({
            'model_name': self.config.model_name,
            'epochs': self.config.epochs,
            'imgsz': self.config.imgsz,
            'batch': self.config.batch,
            'lr0': self.config.lr0,
            'patience': self.config.patience,
            'optimizer': self.config.optimizer,
            'device': device
        })
        run_id = self.db.add_training_run(
            model_type='detection',
            config_json=config_json
        )

        # Training parameters
        train_args = {
            'data': data_yaml_path,
            'epochs': self.config.epochs,
            'imgsz': self.config.imgsz,
            'batch': self.config.batch,
            'lr0': self.config.lr0,
            'patience': self.config.patience,
            'device': device,
            'workers': self.config.workers,
            'optimizer': self.config.optimizer,
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'warmup_epochs': self.config.warmup_epochs,
            'save_period': self.config.save_period,
            'val': True,
            'project': output_dir,
            'name': run_name,
            'exist_ok': True,
            'verbose': True,
            'plots': True  # Generate training plots
        }

        # Print training configuration
        print(f"\n{'='*70}")
        print(f"Starting YOLO Detection Training")
        print(f"{'='*70}")
        print(f"Model: {self.config.model_name}")
        print(f"Data: {data_yaml_path}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Image size: {self.config.imgsz}")
        print(f"Batch size: {self.config.batch}")
        print(f"Device: {device}")
        print(f"Patience: {self.config.patience}")
        print(f"{'='*70}\n")

        try:
            # Train model (Ultralytics handles everything)
            results = self.model.train(**train_args)

            # Get best model path (saved by YOLO)
            run_dir = Path(output_dir) / run_name
            best_model_path = run_dir / 'weights' / 'best.pt'

            if not best_model_path.exists():
                raise RuntimeError("Best model not found after training")

            # Load metrics from results
            # YOLO saves metrics in results object
            final_metrics = {
                'map50': float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
                'map50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0.0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0.0))
            }

            print(f"\n{'='*70}")
            print(f"Training Completed Successfully!")
            print(f"{'='*70}")
            print(f"mAP@0.5: {final_metrics['map50']:.4f}")
            print(f"mAP@0.5-0.95: {final_metrics['map50_95']:.4f}")
            print(f"Precision: {final_metrics['precision']:.4f}")
            print(f"Recall: {final_metrics['recall']:.4f}")

            if final_metrics['map50'] > 0.85:
                print("\n✓ SUCCESS: Model meets target mAP@0.5 > 0.85!")
            else:
                print(f"\n⚠ WARNING: mAP@0.5 {final_metrics['map50']:.4f} below target 0.85")

            print(f"{'='*70}\n")

            # Copy best model to our models directory
            saved_model = self.model_manager.save_model(
                model=self.model,
                model_type=self.model_type,  # Use dynamic model_type (detection, detection_top, detection_side)
                model_name=self.config.model_name,
                metrics=final_metrics,
                config={
                    'model_name': self.config.model_name,
                    'epochs': self.config.epochs,
                    'imgsz': self.config.imgsz,
                    'batch': self.config.batch
                },
                version_prefix='v1',
                additional_info={
                    'yolo_run_dir': str(run_dir),
                    'original_best_path': str(best_model_path)
                }
            )

            print(f"Model saved to: {saved_model['model_path']}")

            # Update database
            self.db.update_training_run(
                run_id=run_id,
                status='completed',
                final_metrics_json=json.dumps(final_metrics)
            )

            return saved_model['model_path']

        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            self.db.update_training_run(
                run_id=run_id,
                status='failed',
                error_message=str(e)
            )
            raise

    def validate(
        self,
        model_path: str,
        data_yaml_path: str
    ) -> Dict[str, float]:
        """
        Validate a trained model.

        Args:
            model_path: Path to trained model
            data_yaml_path: Path to YOLO data.yaml

        Returns:
            Dictionary with validation metrics
        """
        # Load model
        model = YOLO(model_path)

        # Run validation
        results = model.val(data=data_yaml_path, imgsz=self.config.imgsz)

        metrics = {
            'map50': float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
            'map50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
            'precision': float(results.results_dict.get('metrics/precision(B)', 0.0)),
            'recall': float(results.results_dict.get('metrics/recall(B)', 0.0))
        }

        return metrics

    def export_model(
        self,
        model_path: str,
        format: str = 'onnx'
    ) -> str:
        """
        Export model to different format.

        Args:
            model_path: Path to trained model
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)

        Returns:
            Path to exported model
        """
        model = YOLO(model_path)
        export_path = model.export(format=format)
        return export_path


# Test if run directly
if __name__ == "__main__":
    print("Testing YOLO trainer...")

    # Check if ultralytics is available
    if YOLO is None:
        print("Ultralytics not installed, skipping test")
    else:
        # Create dummy configuration
        config = YOLOConfig(
            model_name='yolov8n',
            epochs=1,
            batch=4,
            imgsz=1004
        )

        # Create dummy database
        db = Database(':memory:')

        # Create trainer
        trainer = YOLOTrainer(config, db)

        print(f"Model: {config.model_name}")
        print(f"Epochs: {config.epochs}")
        print(f"Image size: {config.imgsz}")
        print(f"Batch size: {config.batch}")

    print("\nYOLO trainer test complete!")
