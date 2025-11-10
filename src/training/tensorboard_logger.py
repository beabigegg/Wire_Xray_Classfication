"""
TensorBoard Logger Utility for consistent logging across all trainers.

This module provides helper functions for logging metrics, images, and
visualizations to TensorBoard during training.
"""

import logging
import numpy as np
import torch
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
except ImportError:
    plt = None
    matplotlib = None

try:
    import seaborn as sns
except ImportError:
    sns = None

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """
    Utility class for TensorBoard logging with consistent formatting.

    Features:
    - Smart logging configuration (scalars every epoch, images every N epochs)
    - Confusion matrix plotting
    - Prediction grid visualization
    - Histogram logging
    - Optional async figure logging for performance
    """

    def __init__(
        self,
        log_dir: str,
        image_log_frequency: int = 10,
        histogram_log_frequency: int = 10
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
            image_log_frequency: Log images every N epochs
            histogram_log_frequency: Log histograms every N epochs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.image_log_frequency = image_log_frequency
        self.histogram_log_frequency = histogram_log_frequency

        logger.info(f"TensorBoard logging to: {self.log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value.

        Args:
            tag: Name of the scalar (e.g., 'train/loss')
            value: Scalar value
            step: Training step/epoch
        """
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        Log multiple scalars under a main tag.

        Args:
            main_tag: Main category (e.g., 'metrics')
            tag_scalar_dict: Dict of {tag: value}
            step: Training step/epoch
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """
        Log histogram of tensor values.

        Args:
            tag: Name of the histogram
            values: Tensor to histogram
            step: Training step/epoch
        """
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """
        Log an image.

        Args:
            tag: Name of the image
            image: Image tensor (C, H, W) or (H, W)
            step: Training step/epoch
        """
        self.writer.add_image(tag, image, step)

    def log_figure(self, tag: str, figure: Any, step: int):
        """
        Log a matplotlib figure.

        Args:
            tag: Name of the figure
            figure: Matplotlib figure object
            step: Training step/epoch
        """
        self.writer.add_figure(tag, figure, step, close=True)

    def should_log_images(self, epoch: int) -> bool:
        """
        Check if images should be logged at this epoch.

        Args:
            epoch: Current epoch number

        Returns:
            True if images should be logged
        """
        return epoch % self.image_log_frequency == 0

    def should_log_histograms(self, epoch: int) -> bool:
        """
        Check if histograms should be logged at this epoch.

        Args:
            epoch: Current epoch number

        Returns:
            True if histograms should be logged
        """
        return epoch % self.histogram_log_frequency == 0

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: tuple = (10, 8)
) -> Optional[Any]:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array (n_classes, n_classes)
        class_names: List of class names
        title: Plot title
        normalize: Whether to normalize to percentages
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object, or None if matplotlib not available
    """
    if plt is None or sns is None:
        logger.warning("matplotlib or seaborn not available, cannot plot confusion matrix")
        return None

    try:
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Percentage' if normalize else 'Count'}
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)

        plt.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")
        return None


def create_prediction_grid(
    images: torch.Tensor,
    true_labels: List[str],
    pred_labels: List[str],
    confidences: Optional[List[float]] = None,
    max_images: int = 16,
    grid_cols: int = 4,
    figsize_per_image: tuple = (3, 3)
) -> Optional[Any]:
    """
    Create a grid of prediction images with labels.

    Args:
        images: Tensor of images (N, C, H, W) or (N, H, W)
        true_labels: List of true label strings
        pred_labels: List of predicted label strings
        confidences: Optional list of prediction confidences
        max_images: Maximum number of images to show
        grid_cols: Number of columns in grid
        figsize_per_image: Size of each image subplot

    Returns:
        Matplotlib figure object, or None if matplotlib not available
    """
    if plt is None:
        logger.warning("matplotlib not available, cannot create prediction grid")
        return None

    try:
        # Limit number of images
        n_images = min(len(images), max_images)
        images = images[:n_images]
        true_labels = true_labels[:n_images]
        pred_labels = pred_labels[:n_images]
        if confidences:
            confidences = confidences[:n_images]

        # Calculate grid dimensions
        grid_rows = (n_images + grid_cols - 1) // grid_cols

        # Create figure
        figsize = (figsize_per_image[0] * grid_cols, figsize_per_image[1] * grid_rows)
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)

        # Flatten axes for easier iteration
        if grid_rows == 1:
            axes = [axes] if grid_cols == 1 else axes
        else:
            axes = axes.flatten() if grid_cols > 1 else axes

        # Plot each image
        for idx in range(n_images):
            ax = axes[idx] if isinstance(axes, (list, np.ndarray)) else axes

            # Convert image to numpy
            img = images[idx]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()

            # Handle different image formats
            if img.ndim == 3:
                # (C, H, W) -> (H, W, C)
                if img.shape[0] == 1:
                    img = img[0]  # Grayscale
                elif img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))  # RGB

            # Display image
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)

            # Set title with labels
            true_label = true_labels[idx]
            pred_label = pred_labels[idx]
            is_correct = true_label == pred_label

            if confidences and idx < len(confidences):
                title = f"True: {true_label}\nPred: {pred_label} ({confidences[idx]:.2%})"
            else:
                title = f"True: {true_label}\nPred: {pred_label}"

            # Color code title (green if correct, red if wrong)
            color = 'green' if is_correct else 'red'
            ax.set_title(title, fontsize=10, color=color)
            ax.axis('off')

        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Failed to create prediction grid: {e}")
        return None


def log_model_parameters(
    logger: TensorBoardLogger,
    model: torch.nn.Module,
    step: int
):
    """
    Log model parameter histograms.

    Args:
        logger: TensorBoardLogger instance
        model: PyTorch model
        step: Training step/epoch
    """
    try:
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Log parameter values
                logger.log_histogram(f'parameters/{name}', param.data, step)

                # Log gradients
                logger.log_histogram(f'gradients/{name}', param.grad, step)

    except Exception as e:
        logger.error(f"Failed to log model parameters: {e}")


def log_learning_rate(
    logger: TensorBoardLogger,
    optimizer: torch.optim.Optimizer,
    step: int,
    tag: str = 'learning_rate'
):
    """
    Log current learning rate(s).

    Args:
        logger: TensorBoardLogger instance
        optimizer: PyTorch optimizer
        step: Training step/epoch
        tag: Tag for the scalar
    """
    try:
        # Get learning rates from all parameter groups
        lrs = [group['lr'] for group in optimizer.param_groups]

        if len(lrs) == 1:
            logger.log_scalar(tag, lrs[0], step)
        else:
            for idx, lr in enumerate(lrs):
                logger.log_scalar(f'{tag}/group_{idx}', lr, step)

    except Exception as e:
        logger.error(f"Failed to log learning rate: {e}")


# Example usage
if __name__ == "__main__":
    print("Testing TensorBoardLogger...")

    # Test logger creation
    logger_instance = TensorBoardLogger(
        log_dir='runs/test',
        image_log_frequency=5,
        histogram_log_frequency=10
    )

    # Test scalar logging
    logger_instance.log_scalar('test/loss', 0.5, step=1)
    logger_instance.log_scalars('test/metrics', {'accuracy': 0.9, 'precision': 0.85}, step=1)

    # Test frequency checks
    assert logger_instance.should_log_images(5)
    assert not logger_instance.should_log_images(3)
    assert logger_instance.should_log_histograms(10)
    assert not logger_instance.should_log_histograms(7)

    # Test confusion matrix plotting
    if plt is not None and sns is not None:
        cm = np.array([[50, 2, 3], [1, 45, 4], [2, 3, 40]])
        class_names = ['Class A', 'Class B', 'Class C']
        fig = plot_confusion_matrix(cm, class_names, normalize=True)
        if fig:
            logger_instance.log_figure('test/confusion_matrix', fig, step=1)
            print("Confusion matrix plotted successfully")

    # Test prediction grid
    if plt is not None:
        dummy_images = torch.randn(8, 3, 64, 64)
        true_labels = ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B']
        pred_labels = ['A', 'B', 'C', 'C', 'B', 'A', 'C', 'A']
        confidences = [0.95, 0.87, 0.65, 0.78, 0.92, 0.88, 0.94, 0.71]

        fig = create_prediction_grid(
            dummy_images, true_labels, pred_labels, confidences,
            max_images=8, grid_cols=4
        )
        if fig:
            logger_instance.log_figure('test/predictions', fig, step=1)
            print("Prediction grid created successfully")

    logger_instance.close()

    print("\nTensorBoardLogger test complete!")
    print("\nFeatures implemented:")
    print("  - Scalar logging (single and multiple)")
    print("  - Histogram logging")
    print("  - Image and figure logging")
    print("  - Smart logging frequency control")
    print("  - Confusion matrix plotting with seaborn")
    print("  - Prediction grid visualization")
    print("  - Model parameter and gradient logging")
    print("  - Learning rate logging")
    print("  - Context manager support")
