"""
Training utility functions.

Provides device detection, metrics calculation, and helper functions
for the training pipeline.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, balanced_accuracy_score,
    classification_report
)
import psutil
import platform


class DeviceManager:
    """Manage training device (GPU/CPU) selection and configuration."""

    @staticmethod
    def get_available_device(preferred: str = 'auto') -> str:
        """
        Get available training device.

        Args:
            preferred: Preferred device ('auto', 'cuda', 'cpu')

        Returns:
            Device string ('cuda' or 'cpu')
        """
        if preferred == 'cpu':
            return 'cpu'

        if preferred == 'auto' or preferred == 'cuda':
            if torch.cuda.is_available():
                return 'cuda'

        return 'cpu'

    @staticmethod
    def get_device_info() -> Dict:
        """
        Get detailed device information.

        Returns:
            Dictionary with device information
        """
        info = {
            'platform': platform.system(),
            'cpu_count': psutil.cpu_count(),
            'ram_total_gb': psutil.virtual_memory().total / (1024 ** 3),
            'ram_available_gb': psutil.virtual_memory().available / (1024 ** 3),
            'cuda_available': torch.cuda.is_available()
        }

        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        return info

    @staticmethod
    def get_optimal_batch_size(device: str, model_type: str) -> int:
        """
        Get optimal batch size based on device.

        Args:
            device: Device string ('cuda' or 'cpu')
            model_type: Type of model ('detection', 'view', 'defect')

        Returns:
            Recommended batch size
        """
        if device == 'cuda':
            # GPU batch sizes
            if model_type == 'detection':
                return 16
            elif model_type == 'view':
                return 32
            elif model_type == 'defect':
                return 16
        else:
            # CPU batch sizes (smaller)
            if model_type == 'detection':
                return 4
            elif model_type == 'view':
                return 8
            elif model_type == 'defect':
                return 4

        return 8  # Default

    @staticmethod
    def get_optimal_workers(device: str) -> int:
        """
        Get optimal number of data loader workers.

        Args:
            device: Device string

        Returns:
            Recommended number of workers
        """
        if device == 'cuda':
            return min(4, psutil.cpu_count() // 2)
        else:
            return min(2, psutil.cpu_count() // 2)

    @staticmethod
    def check_memory_available(required_gb: float) -> bool:
        """
        Check if sufficient memory is available.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if sufficient memory available
        """
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        return available_gb >= required_gb

    @staticmethod
    def monitor_gpu_memory() -> Optional[Dict]:
        """
        Monitor GPU memory usage.

        Returns:
            Dictionary with GPU memory stats or None if no GPU
        """
        if not torch.cuda.is_available():
            return None

        return {
            'allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
            'reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3),
            'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024 ** 3)
        }


class MetricsCalculator:
    """Calculate and aggregate training metrics."""

    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Calculate comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional list of class names

        Returns:
            Dictionary with all metrics
        """
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'confusion_matrix': cm.tolist()
        }

        # Per-class metrics
        if class_names is None:
            class_names = [f'class_{i}' for i in range(len(precision))]

        for i, class_name in enumerate(class_names):
            metrics[f'{class_name}_precision'] = float(precision[i])
            metrics[f'{class_name}_recall'] = float(recall[i])
            metrics[f'{class_name}_f1'] = float(f1[i])
            metrics[f'{class_name}_support'] = int(support[i])

        return metrics

    @staticmethod
    def calculate_pass_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pass_class_idx: int = 0
    ) -> Dict:
        """
        Calculate specific metrics for PASS class.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            pass_class_idx: Index of PASS class

        Returns:
            Dictionary with PASS-specific metrics
        """
        # Binary mask for PASS class
        pass_true = (y_true == pass_class_idx)
        pass_pred = (y_pred == pass_class_idx)

        # True/False Positives/Negatives
        tp = np.sum(pass_true & pass_pred)
        fp = np.sum(~pass_true & pass_pred)
        fn = np.sum(pass_true & ~pass_pred)
        tn = np.sum(~pass_true & ~pass_pred)

        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            'pass_precision': float(precision),
            'pass_recall': float(recall),
            'pass_f1': float(f1),
            'pass_specificity': float(specificity),
            'pass_tp': int(tp),
            'pass_fp': int(fp),
            'pass_fn': int(fn),
            'pass_tn': int(tn)
        }

    @staticmethod
    def format_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str]
    ) -> str:
        """
        Format confusion matrix as readable string.

        Args:
            cm: Confusion matrix
            class_names: List of class names

        Returns:
            Formatted string
        """
        # Calculate column widths
        max_name_len = max(len(name) for name in class_names)
        max_val = cm.max()
        val_width = len(str(max_val))

        # Header
        header = " " * (max_name_len + 2) + "Predicted\n"
        header += " " * (max_name_len + 2)
        header += "  ".join(f"{name:{max_name_len}}" for name in class_names) + "\n"

        # Rows
        rows = []
        for i, true_class in enumerate(class_names):
            if i == 0:
                row = f"{'Actual':{max_name_len}}  "
            else:
                row = f"{'':{max_name_len}}  "

            row += f"{true_class:{max_name_len}}  "
            row += "  ".join(f"{cm[i, j]:{val_width}}" for j in range(len(class_names)))
            rows.append(row)

        return header + "\n".join(rows)

    @staticmethod
    def print_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str]
    ):
        """
        Print detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        """
        print("\n" + "=" * 70)
        print("Classification Report")
        print("=" * 70)

        # Overall metrics
        metrics = MetricsCalculator.calculate_classification_metrics(
            y_true, y_pred, class_names
        )

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Macro Precision:   {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:      {metrics['macro_recall']:.4f}")
        print(f"  Macro F1:          {metrics['macro_f1']:.4f}")

        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 70)

        for class_name in class_names:
            precision = metrics[f'{class_name}_precision']
            recall = metrics[f'{class_name}_recall']
            f1 = metrics[f'{class_name}_f1']
            support = metrics[f'{class_name}_support']

            print(f"{class_name:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

        # Confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        print(f"\nConfusion Matrix:")
        print(MetricsCalculator.format_confusion_matrix(cm, class_names))

        print("=" * 70 + "\n")


class TrainingMonitor:
    """Monitor training progress and calculate ETA."""

    def __init__(self, total_epochs: int):
        """
        Initialize training monitor.

        Args:
            total_epochs: Total number of epochs
        """
        self.total_epochs = total_epochs
        self.epoch_times = []
        self.start_time = None

    def start(self):
        """Start monitoring."""
        import time
        self.start_time = time.time()

    def epoch_complete(self):
        """Record epoch completion."""
        import time
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.epoch_times.append(elapsed)
            self.start_time = time.time()

    def get_eta(self, current_epoch: int) -> str:
        """
        Get estimated time remaining.

        Args:
            current_epoch: Current epoch number

        Returns:
            Formatted ETA string
        """
        if not self.epoch_times:
            return "Calculating..."

        avg_time_per_epoch = np.mean(self.epoch_times[-10:])  # Use last 10 epochs
        remaining_epochs = self.total_epochs - current_epoch
        eta_seconds = avg_time_per_epoch * remaining_epochs

        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def get_progress_percentage(self, current_epoch: int) -> float:
        """
        Get training progress percentage.

        Args:
            current_epoch: Current epoch number

        Returns:
            Progress percentage (0-100)
        """
        return (current_epoch / self.total_epochs) * 100


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_value: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == 'max':
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# Test if run directly
if __name__ == "__main__":
    print("Testing training utilities...")

    # Test device manager
    print("\nDevice Information:")
    device_info = DeviceManager.get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")

    device = DeviceManager.get_available_device('auto')
    print(f"\nSelected device: {device}")

    batch_size = DeviceManager.get_optimal_batch_size(device, 'view')
    print(f"Optimal batch size for view classifier: {batch_size}")

    # Test metrics calculator
    print("\nTesting metrics calculation...")
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2, 0, 1, 3, 3])
    class_names = ['PASS', '沖線', '晃動', '碰觸']

    metrics = MetricsCalculator.calculate_classification_metrics(y_true, y_pred, class_names)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")

    pass_metrics = MetricsCalculator.calculate_pass_class_metrics(y_true, y_pred, pass_class_idx=0)
    print(f"PASS Recall: {pass_metrics['pass_recall']:.4f}")

    # Test training monitor
    print("\nTesting training monitor...")
    monitor = TrainingMonitor(total_epochs=100)
    monitor.start()

    import time
    time.sleep(0.1)
    monitor.epoch_complete()

    eta = monitor.get_eta(current_epoch=1)
    print(f"ETA after 1 epoch: {eta}")

    print("\nTraining utilities test complete!")
