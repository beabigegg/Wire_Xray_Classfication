"""
Checkpoint Manager for Training State Persistence.

This module provides atomic checkpoint save/load functionality with integrity
verification to support pause/resume training.
"""

import os
import shutil
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages training checkpoints with atomic saves and integrity verification.

    Features:
    - Atomic checkpoint saving (write to temp, then rename)
    - Integrity verification before committing
    - Automatic cleanup of old checkpoints
    - Separate checkpoints per model type
    """

    def __init__(self, checkpoint_dir: str = "checkpoints", retention_days: int = 7):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            retention_days: Days to keep cancelled training checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.retention_days = retention_days

    def save_checkpoint(
        self,
        model_type: str,
        epoch: int,
        total_epochs: int,
        model_state: Dict,
        optimizer_state: Dict,
        scheduler_state: Optional[Dict],
        metrics: Dict[str, Any],
        config: Dict[str, Any]
    ) -> bool:
        """
        Save training checkpoint atomically.

        Args:
            model_type: Type of model (detection/view/defect)
            epoch: Current epoch number
            total_epochs: Total epochs planned
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict (optional)
            metrics: Training metrics dict
            config: Training configuration

        Returns:
            True if save successful, False otherwise
        """
        checkpoint_path = self._get_checkpoint_path(model_type)
        temp_path = checkpoint_path.with_suffix('.tmp')

        try:
            checkpoint_data = {
                'model_type': model_type,
                'epoch': epoch,
                'total_epochs': total_epochs,
                'model_state': model_state,
                'optimizer_state': optimizer_state,
                'scheduler_state': scheduler_state,
                'metrics': metrics,
                'config': config,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }

            # Save to temporary file
            torch.save(checkpoint_data, temp_path)
            logger.info(f"Saved checkpoint to temporary file: {temp_path}")

            # Verify integrity by loading
            loaded_data = torch.load(temp_path, map_location='cpu')
            if not self._verify_checkpoint(loaded_data):
                raise ValueError("Checkpoint verification failed")

            # Atomic rename (replaces existing checkpoint)
            shutil.move(str(temp_path), str(checkpoint_path))
            logger.info(f"Checkpoint committed: {checkpoint_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def load_checkpoint(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for given model type.

        Args:
            model_type: Type of model (detection/view/defect)

        Returns:
            Checkpoint data dict, or None if not found/invalid
        """
        checkpoint_path = self._get_checkpoint_path(model_type)

        if not checkpoint_path.exists():
            logger.info(f"No checkpoint found for {model_type}")
            return None

        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

            if not self._verify_checkpoint(checkpoint_data):
                logger.error("Checkpoint verification failed during load")
                return None

            logger.info(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}/{checkpoint_data['total_epochs']}")
            return checkpoint_data

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def delete_checkpoint(self, model_type: str) -> bool:
        """
        Delete checkpoint for given model type.

        Args:
            model_type: Type of model (detection/view/defect)

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self._get_checkpoint_path(model_type)

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
            return True
        return False

    def has_checkpoint(self, model_type: str) -> bool:
        """Check if checkpoint exists for model type."""
        return self._get_checkpoint_path(model_type).exists()

    def get_checkpoint_info(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get checkpoint metadata without loading full state.

        Args:
            model_type: Type of model

        Returns:
            Dict with epoch, timestamp, metrics, or None if no checkpoint
        """
        checkpoint_path = self._get_checkpoint_path(model_type)

        if not checkpoint_path.exists():
            return None

        try:
            data = torch.load(checkpoint_path, map_location='cpu')
            return {
                'model_type': data.get('model_type'),
                'epoch': data.get('epoch'),
                'total_epochs': data.get('total_epochs'),
                'timestamp': data.get('timestamp'),
                'metrics': data.get('metrics', {})
            }
        except Exception as e:
            logger.error(f"Failed to get checkpoint info: {e}")
            return None

    def cleanup_old_checkpoints(self):
        """Delete checkpoints older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for checkpoint_path in self.checkpoint_dir.glob("*.ckpt"):
            try:
                # Get modification time
                mtime = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)

                if mtime < cutoff_date:
                    checkpoint_path.unlink()
                    logger.info(f"Cleaned up old checkpoint: {checkpoint_path.name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {checkpoint_path}: {e}")

    def _get_checkpoint_path(self, model_type: str) -> Path:
        """Get checkpoint file path for model type."""
        filename = f"{model_type}_latest.ckpt"
        return self.checkpoint_dir / filename

    def _verify_checkpoint(self, data: Dict) -> bool:
        """
        Verify checkpoint data integrity.

        Args:
            data: Loaded checkpoint data

        Returns:
            True if valid, False otherwise
        """
        required_keys = [
            'model_type', 'epoch', 'total_epochs',
            'model_state', 'optimizer_state', 'config'
        ]

        for key in required_keys:
            if key not in data:
                logger.error(f"Checkpoint missing required key: {key}")
                return False

        # Verify model state is a valid state dict
        if not isinstance(data['model_state'], dict):
            logger.error("Invalid model_state format")
            return False

        # Verify optimizer state
        if not isinstance(data['optimizer_state'], dict):
            logger.error("Invalid optimizer_state format")
            return False

        return True
