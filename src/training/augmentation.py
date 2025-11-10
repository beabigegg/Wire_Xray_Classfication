"""
Data augmentation module using albumentations.

Provides standard and heavy augmentation pipelines for training,
with special handling for the PASS class imbalance.
"""

import albumentations as A
import numpy as np
from typing import Dict, Tuple, Optional, List
import cv2


class AugmentationPipeline:
    """
    Data augmentation pipeline manager.

    Provides different augmentation strategies for YOLO detection,
    view classification, and defect classification with special
    handling for minority classes.
    """

    def __init__(self, image_size: int = 1004):
        """
        Initialize augmentation pipeline.

        Args:
            image_size: Target image size (default: 1004)
        """
        self.image_size = image_size

    def get_yolo_augmentation(
        self,
        training: bool = True
    ) -> A.Compose:
        """
        Get augmentation pipeline for YOLO detection training.

        Args:
            training: If True, returns training augmentation, else validation

        Returns:
            Albumentations Compose object with bbox-aware transforms
        """
        if training:
            return A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussianBlur(
                    blur_limit=(3, 7),
                    p=0.3
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                # Note: No resize needed as images are already 1004x1004
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))
        else:
            # Validation: no augmentation
            return A.Compose([
                # Identity transform
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))

    def get_classifier_augmentation(
        self,
        training: bool = True,
        heavy: bool = False
    ) -> A.Compose:
        """
        Get augmentation pipeline for classification training.

        Args:
            training: If True, returns training augmentation
            heavy: If True, applies heavy augmentation for minority class

        Returns:
            Albumentations Compose object
        """
        if not training:
            # Validation: only resize if needed
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if heavy:
            # Heavy augmentation for PASS class (minority class)
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.8
                ),
                A.GaussianBlur(blur_limit=(3, 9), p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.2,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.7
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=20,
                    max_width=20,
                    min_holes=1,
                    min_height=5,
                    min_width=5,
                    fill_value=0,
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Standard augmentation
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def augment_image(
        self,
        image: np.ndarray,
        pipeline: A.Compose,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        class_label: Optional[int] = None
    ) -> Dict:
        """
        Apply augmentation to an image.

        Args:
            image: Input image as numpy array
            pipeline: Augmentation pipeline
            bbox: Optional bounding box in YOLO format (x_center, y_center, width, height)
            class_label: Optional class label for bbox

        Returns:
            Dictionary with augmented image and transformed bbox (if provided)
        """
        if bbox is not None and class_label is not None:
            # Apply augmentation with bbox
            augmented = pipeline(
                image=image,
                bboxes=[bbox],
                class_labels=[class_label]
            )
            return {
                'image': augmented['image'],
                'bboxes': augmented['bboxes'],
                'class_labels': augmented['class_labels']
            }
        else:
            # Apply augmentation without bbox
            augmented = pipeline(image=image)
            return {'image': augmented['image']}

    def generate_augmented_samples(
        self,
        image: np.ndarray,
        augmentation_factor: int,
        heavy: bool = False
    ) -> List[np.ndarray]:
        """
        Generate multiple augmented versions of an image.

        Args:
            image: Input image as numpy array
            augmentation_factor: Number of augmented samples to generate
            heavy: Whether to use heavy augmentation

        Returns:
            List of augmented images
        """
        pipeline = self.get_classifier_augmentation(training=True, heavy=heavy)

        augmented_images = []
        for _ in range(augmentation_factor):
            aug_result = self.augment_image(image, pipeline)
            augmented_images.append(aug_result['image'])

        return augmented_images

    def visualize_augmentation(
        self,
        image: np.ndarray,
        num_samples: int = 4,
        heavy: bool = False,
        save_path: Optional[str] = None
    ):
        """
        Visualize augmentation effects (for debugging).

        Args:
            image: Input image
            num_samples: Number of augmented samples to show
            heavy: Whether to use heavy augmentation
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt

        pipeline = self.get_classifier_augmentation(training=True, heavy=heavy)

        # Create grid
        fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 3))

        # Original image
        axes[0].imshow(image if len(image.shape) == 3 else image, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        # Augmented images
        for i in range(num_samples):
            aug_result = self.augment_image(image, pipeline)
            aug_image = aug_result['image']

            # Denormalize for visualization
            if aug_image.max() <= 1.0:
                # Assuming normalized image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                aug_image = aug_image * std + mean
                aug_image = np.clip(aug_image, 0, 1)

            axes[i + 1].imshow(aug_image if len(aug_image.shape) == 3 else aug_image, cmap='gray')
            axes[i + 1].set_title(f'Aug {i+1}')
            axes[i + 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()


def get_class_weights(class_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate class weights for balanced training.

    Args:
        class_counts: Dictionary mapping class names to sample counts

    Returns:
        Dictionary mapping class names to weights
    """
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)

    weights = {}
    for class_name, count in class_counts.items():
        # Inverse frequency weighting
        # weight = total_samples / (num_classes * count)
        # For severe imbalance, use log-based weighting
        weights[class_name] = np.log1p(total_samples / count) + 1.0

    return weights


def calculate_augmentation_factor(
    class_counts: Dict[str, int],
    target_balance_ratio: float = 0.5
) -> Dict[str, int]:
    """
    Calculate augmentation factor needed for each class.

    Args:
        class_counts: Dictionary mapping class names to sample counts
        target_balance_ratio: Target ratio of minority class to majority class

    Returns:
        Dictionary mapping class names to augmentation factors
    """
    max_count = max(class_counts.values())
    target_count = max_count * target_balance_ratio

    aug_factors = {}
    for class_name, count in class_counts.items():
        if count < target_count:
            aug_factors[class_name] = int(np.ceil(target_count / count))
        else:
            aug_factors[class_name] = 1  # No augmentation needed

    return aug_factors


class BalancedBatchSampler:
    """
    Custom sampler to ensure balanced batches during training.

    Ensures that each batch contains samples from all classes,
    particularly important for minority classes like PASS.
    """

    def __init__(
        self,
        class_indices: Dict[str, List[int]],
        batch_size: int,
        samples_per_class: Optional[int] = None
    ):
        """
        Initialize balanced batch sampler.

        Args:
            class_indices: Dictionary mapping class names to list of sample indices
            batch_size: Total batch size
            samples_per_class: Number of samples per class in each batch
        """
        self.class_indices = class_indices
        self.batch_size = batch_size
        self.num_classes = len(class_indices)

        if samples_per_class is None:
            self.samples_per_class = batch_size // self.num_classes
        else:
            self.samples_per_class = samples_per_class

        # Adjust batch size to be divisible
        self.actual_batch_size = self.samples_per_class * self.num_classes

    def __iter__(self):
        """Generate balanced batches."""
        # Prepare indices for each class
        class_iters = {}
        for class_name, indices in self.class_indices.items():
            # Shuffle indices for this class
            shuffled = indices.copy()
            np.random.shuffle(shuffled)
            class_iters[class_name] = iter(shuffled)

        # Determine number of batches
        min_samples = min(len(indices) for indices in self.class_indices.values())
        num_batches = min_samples // self.samples_per_class

        for _ in range(num_batches):
            batch = []
            for class_name in self.class_indices.keys():
                try:
                    for _ in range(self.samples_per_class):
                        batch.append(next(class_iters[class_name]))
                except StopIteration:
                    # Restart iterator if exhausted
                    shuffled = self.class_indices[class_name].copy()
                    np.random.shuffle(shuffled)
                    class_iters[class_name] = iter(shuffled)
                    for _ in range(self.samples_per_class):
                        batch.append(next(class_iters[class_name]))

            # Shuffle batch to mix classes
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        """Return number of batches."""
        min_samples = min(len(indices) for indices in self.class_indices.values())
        return min_samples // self.samples_per_class


# Test augmentation if run directly
if __name__ == "__main__":
    print("Testing augmentation pipeline...")

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (1004, 1004, 3), dtype=np.uint8)

    pipeline = AugmentationPipeline()

    # Test YOLO augmentation
    yolo_aug = pipeline.get_yolo_augmentation(training=True)
    result = pipeline.augment_image(
        dummy_image,
        yolo_aug,
        bbox=(0.5, 0.5, 0.3, 0.2),
        class_label=0
    )
    print(f"YOLO augmentation result: {result['image'].shape}")
    print(f"Transformed bbox: {result['bboxes']}")

    # Test classifier augmentation
    clf_aug = pipeline.get_classifier_augmentation(training=True, heavy=False)
    result = pipeline.augment_image(dummy_image, clf_aug)
    print(f"Classifier augmentation result: {result['image'].shape}")

    # Test class weights
    class_counts = {'PASS': 6, '沖線': 40, '晃動': 58, '碰觸': 68}
    weights = get_class_weights(class_counts)
    print(f"\nClass weights: {weights}")

    # Test augmentation factors
    aug_factors = calculate_augmentation_factor(class_counts, target_balance_ratio=0.3)
    print(f"Augmentation factors: {aug_factors}")

    print("\nAugmentation module test complete!")
