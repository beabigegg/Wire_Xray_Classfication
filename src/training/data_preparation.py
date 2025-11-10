"""
Data preparation module for training pipeline.

Handles train/val split, YOLO format export, and dataset preparation
for view and defect classifiers.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import random
from collections import defaultdict
import sqlite3
import numpy as np
from PIL import Image

# Import wire pairing utilities
from src.core.pairing_utils import (
    extract_wire_id,
    group_images_by_wire,
    validate_pairing
)


class DataPreparationError(Exception):
    """Exception raised for data preparation errors."""
    pass


class DataPreparator:
    """
    Data preparation for training pipeline.

    Handles stratified train/val split and exports data in various formats
    for different model types (YOLO detection, view classifier, defect classifier).
    """

    def __init__(self, db_path: str, random_seed: int = 42):
        """
        Initialize data preparator.

        Args:
            db_path: Path to annotations database
            random_seed: Random seed for reproducibility
        """
        self.db_path = db_path
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def stratified_split(
        self,
        val_ratio: float = 0.2,
        stratify_by: str = 'defect_type'
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Perform stratified train/validation split.

        Args:
            val_ratio: Ratio of validation set (default: 0.2)
            stratify_by: Field to stratify by ('defect_type' or 'view_type')

        Returns:
            Tuple of (train_annotations, val_annotations)

        Raises:
            DataPreparationError: If split cannot be performed
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all annotations with image info
            cursor.execute("""
                SELECT
                    a.id, a.image_id, a.bbox_x, a.bbox_y, a.bbox_width, a.bbox_height,
                    a.view_type, a.defect_type, a.confidence, a.annotator,
                    i.filename, i.filepath, i.width, i.height
                FROM annotations a
                JOIN images i ON a.image_id = i.id
                ORDER BY a.id
            """)
            all_annotations = [dict(row) for row in cursor.fetchall()]
            conn.close()

            if not all_annotations:
                raise DataPreparationError("No annotations found in database")

            # Group by stratification field
            groups = defaultdict(list)
            for ann in all_annotations:
                groups[ann[stratify_by]].append(ann)

            # Perform stratified split
            train_set = []
            val_set = []

            for group_key, items in groups.items():
                n_val = max(1, int(len(items) * val_ratio))  # At least 1 in val
                n_train = len(items) - n_val

                # Shuffle within group
                shuffled = items.copy()
                random.shuffle(shuffled)

                val_set.extend(shuffled[:n_val])
                train_set.extend(shuffled[n_val:])

            return train_set, val_set

        except sqlite3.Error as e:
            raise DataPreparationError(f"Database error during split: {e}")

    def stratified_split_with_pairing(
        self,
        val_ratio: float = 0.2,
        stratify_by: str = 'defect_type'
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Perform stratified train/validation split while preserving TOP/SIDE wire pairs.

        This method ensures that both TOP and SIDE views of the same wire are kept
        in the same split (train or val), preventing data leakage.

        Args:
            val_ratio: Ratio of validation set (default: 0.2)
            stratify_by: Field to stratify by ('defect_type' or 'view_type')

        Returns:
            Tuple of (train_annotations, val_annotations)

        Raises:
            DataPreparationError: If split cannot be performed

        Note:
            For wire pairing to work, filenames must follow the format:
            {wire_id}_TOP.jpg and {wire_id}_SIDE.jpg
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all annotations with image info
            cursor.execute("""
                SELECT
                    a.id, a.image_id, a.bbox_x, a.bbox_y, a.bbox_width, a.bbox_height,
                    a.view_type, a.defect_type, a.confidence, a.annotator,
                    i.filename, i.filepath, i.width, i.height
                FROM annotations a
                JOIN images i ON a.image_id = i.id
                ORDER BY a.id
            """)
            all_annotations = [dict(row) for row in cursor.fetchall()]
            conn.close()

            if not all_annotations:
                raise DataPreparationError("No annotations found in database")

            # Group annotations by wire_id
            wire_groups = defaultdict(list)  # wire_id -> list of annotations
            unpaired_annotations = []

            for ann in all_annotations:
                wire_id = extract_wire_id(ann['filename'])
                if wire_id:
                    wire_groups[wire_id].append(ann)
                else:
                    # Handle annotations without valid wire_id format
                    unpaired_annotations.append(ann)

            # Validate pairing (warn about incomplete pairs)
            if unpaired_annotations:
                print(f"Warning: {len(unpaired_annotations)} annotations without valid wire_id format")

            # Check that each wire has both TOP and SIDE
            complete_wire_ids = []
            incomplete_wire_ids = []
            mismatched_labels = []

            for wire_id, anns in wire_groups.items():
                views = {ann['view_type'] for ann in anns}
                if 'TOP' in views and 'SIDE' in views:
                    # Check if TOP and SIDE have consistent defect labels
                    top_anns = [a for a in anns if a['view_type'] == 'TOP']
                    side_anns = [a for a in anns if a['view_type'] == 'SIDE']

                    top_defect = top_anns[0]['defect_type'] if top_anns else None
                    side_defect = side_anns[0]['defect_type'] if side_anns else None

                    if top_defect != side_defect:
                        mismatched_labels.append(wire_id)
                        print(f"Warning: Wire {wire_id} has mismatched labels - TOP: {top_defect}, SIDE: {side_defect}")
                        print(f"  → Will use TOP label ({top_defect}) for stratification")

                    complete_wire_ids.append(wire_id)
                else:
                    incomplete_wire_ids.append(wire_id)
                    print(f"Warning: Wire {wire_id} has incomplete pairing (views: {views})")

            # Group complete wires by stratification key
            # For stratification, we use the label from either view (should be same for paired wires)
            stratified_wires = defaultdict(list)

            for wire_id in complete_wire_ids:
                anns = wire_groups[wire_id]
                # Use the stratification field from first annotation (assumes same label for both views)
                strat_key = anns[0][stratify_by]
                stratified_wires[strat_key].append(wire_id)

            # Perform stratified split at wire level
            train_wire_ids = set()
            val_wire_ids = set()

            for strat_key, wire_ids in stratified_wires.items():
                n_val = max(1, int(len(wire_ids) * val_ratio))  # At least 1 wire in val
                n_train = len(wire_ids) - n_val

                # Shuffle within group
                shuffled_wires = wire_ids.copy()
                random.shuffle(shuffled_wires)

                val_wire_ids.update(shuffled_wires[:n_val])
                train_wire_ids.update(shuffled_wires[n_val:])

            # Collect annotations based on wire assignment
            train_set = []
            val_set = []

            for wire_id, anns in wire_groups.items():
                if wire_id in train_wire_ids:
                    train_set.extend(anns)  # Add all annotations for this wire (TOP + SIDE)
                elif wire_id in val_wire_ids:
                    val_set.extend(anns)

            # Handle unpaired annotations using old method
            if unpaired_annotations:
                print(f"Warning: Using individual split for {len(unpaired_annotations)} unpaired annotations")
                groups = defaultdict(list)
                for ann in unpaired_annotations:
                    groups[ann[stratify_by]].append(ann)

                for group_key, items in groups.items():
                    n_val = max(1, int(len(items) * val_ratio))
                    shuffled = items.copy()
                    random.shuffle(shuffled)
                    val_set.extend(shuffled[:n_val])
                    train_set.extend(shuffled[n_val:])

            print(f"Stratified split with pairing:")
            print(f"  Total wires: {len(complete_wire_ids)} complete, {len(incomplete_wire_ids)} incomplete")
            print(f"  Train wires: {len(train_wire_ids)} ({len(train_set)} annotations)")
            print(f"  Val wires: {len(val_wire_ids)} ({len(val_set)} annotations)")

            return train_set, val_set

        except sqlite3.Error as e:
            raise DataPreparationError(f"Database error during split: {e}")

    def export_yolo_format(
        self,
        train_annotations: List[Dict],
        val_annotations: List[Dict],
        output_dir: str,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Export annotations to YOLO format for detection training.

        Args:
            train_annotations: Training annotations
            val_annotations: Validation annotations
            output_dir: Output directory for YOLO dataset
            class_names: List of class names (defect types)

        Returns:
            Dictionary with paths to dataset components

        Raises:
            DataPreparationError: If export fails
        """
        try:
            output_path = Path(output_dir)

            # Create directory structure
            train_images_dir = output_path / "images" / "train"
            val_images_dir = output_path / "images" / "val"
            train_labels_dir = output_path / "labels" / "train"
            val_labels_dir = output_path / "labels" / "val"

            for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Get class names from data if not provided
            if class_names is None:
                class_names = sorted(set(
                    ann['defect_type'] for ann in train_annotations + val_annotations
                ))

            class_to_idx = {name: idx for idx, name in enumerate(class_names)}

            # Process training set
            self._export_yolo_split(
                train_annotations, train_images_dir, train_labels_dir, class_to_idx
            )

            # Process validation set
            self._export_yolo_split(
                val_annotations, val_images_dir, val_labels_dir, class_to_idx
            )

            # Create data.yaml
            data_yaml_path = output_path / "data.yaml"
            yaml_content = {
                'path': str(output_path.absolute()),
                'train': 'images/train',
                'val': 'images/val',
                'nc': len(class_names),
                'names': class_names
            }

            with open(data_yaml_path, 'w', encoding='utf-8') as f:
                for key, value in yaml_content.items():
                    if isinstance(value, list):
                        f.write(f"{key}: {value}\n")
                    else:
                        f.write(f"{key}: {value}\n")

            return {
                'dataset_path': str(output_path),
                'data_yaml': str(data_yaml_path),
                'train_images': str(train_images_dir),
                'val_images': str(val_images_dir),
                'train_labels': str(train_labels_dir),
                'val_labels': str(val_labels_dir),
                'class_names': class_names
            }

        except Exception as e:
            raise DataPreparationError(f"Failed to export YOLO format: {e}")

    def _export_yolo_split(
        self,
        annotations: List[Dict],
        images_dir: Path,
        labels_dir: Path,
        class_to_idx: Dict[str, int]
    ):
        """
        Export a single split (train or val) to YOLO format.

        Args:
            annotations: List of annotations
            images_dir: Directory for images
            labels_dir: Directory for labels
            class_to_idx: Mapping from class name to index
        """
        for ann in annotations:
            # Copy image
            src_image = Path(ann['filepath'])
            # Use img_{image_id}_{filename} format for consistency
            image_id = ann["image_id"]
            dst_image = images_dir / f"img_{image_id}_{ann['filename']}"

            if src_image.exists():
                shutil.copy2(src_image, dst_image)
            else:
                print(f"Warning: Image not found: {src_image}")
                continue

            # Create YOLO label file
            # YOLO format: class x_center y_center width height (normalized)
            img_width = ann['width']
            img_height = ann['height']

            x_center = (ann['bbox_x'] + ann['bbox_width'] / 2) / img_width
            y_center = (ann['bbox_y'] + ann['bbox_height'] / 2) / img_height
            width = ann['bbox_width'] / img_width
            height = ann['bbox_height'] / img_height

            class_idx = class_to_idx[ann['defect_type']]

            # Use img_{image_id}_{stem}.txt format to ensure uniqueness (OpenSpec requirement)
            # This prevents label file conflicts when images with same name exist in different directories
            image_id = ann['image_id']
            stem = Path(ann['filename']).stem
            label_file = labels_dir / f"img_{image_id}_{stem}.txt"
            with open(label_file, 'w') as f:
                f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def export_classifier_dataset(
        self,
        train_annotations: List[Dict],
        val_annotations: List[Dict],
        output_dir: str,
        classifier_type: str = 'view',
        crop_boxes: bool = True
    ) -> Dict[str, str]:
        """
        Export dataset for view or defect classifier.

        Args:
            train_annotations: Training annotations
            val_annotations: Validation annotations
            output_dir: Output directory for classifier dataset
            classifier_type: 'view' or 'defect'
            crop_boxes: Whether to crop bounding boxes or use full images

        Returns:
            Dictionary with paths to dataset components

        Raises:
            DataPreparationError: If export fails
        """
        try:
            output_path = Path(output_dir)

            # Determine class field
            class_field = 'view_type' if classifier_type == 'view' else 'defect_type'

            # Get class names
            class_names = sorted(set(
                ann[class_field] for ann in train_annotations + val_annotations
            ))

            # Create directory structure
            for split in ['train', 'val']:
                for class_name in class_names:
                    class_dir = output_path / split / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)

            # Export training set
            train_stats = self._export_classifier_split(
                train_annotations, output_path / "train", class_field, crop_boxes
            )

            # Export validation set
            val_stats = self._export_classifier_split(
                val_annotations, output_path / "val", class_field, crop_boxes
            )

            # Create info.json
            info = {
                'classifier_type': classifier_type,
                'class_names': class_names,
                'num_classes': len(class_names),
                'crop_boxes': crop_boxes,
                'train_stats': train_stats,
                'val_stats': val_stats
            }

            info_path = output_path / "info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

            return {
                'dataset_path': str(output_path),
                'train_dir': str(output_path / "train"),
                'val_dir': str(output_path / "val"),
                'info_path': str(info_path),
                'class_names': class_names,
                'train_stats': train_stats,
                'val_stats': val_stats
            }

        except Exception as e:
            raise DataPreparationError(f"Failed to export classifier dataset: {e}")

    def _export_classifier_split(
        self,
        annotations: List[Dict],
        split_dir: Path,
        class_field: str,
        crop_boxes: bool
    ) -> Dict[str, int]:
        """
        Export a single split for classifier.

        Args:
            annotations: List of annotations
            split_dir: Directory for this split (train or val)
            class_field: Field to use for class labels
            crop_boxes: Whether to crop bounding boxes

        Returns:
            Statistics dictionary with counts per class
        """
        stats = defaultdict(int)

        for idx, ann in enumerate(annotations):
            class_name = ann[class_field]
            class_dir = split_dir / class_name

            try:
                # Load image
                img_path = Path(ann['filepath'])
                if not img_path.exists():
                    print(f"Warning: Image not found: {img_path}")
                    continue

                img = Image.open(img_path)

                # Crop if requested
                if crop_boxes:
                    bbox = (
                        int(ann['bbox_x']),
                        int(ann['bbox_y']),
                        int(ann['bbox_x'] + ann['bbox_width']),
                        int(ann['bbox_y'] + ann['bbox_height'])
                    )
                    img = img.crop(bbox)

                # Save image
                output_filename = f"{Path(ann['filename']).stem}_{idx:04d}.png"
                output_path = class_dir / output_filename
                img.save(output_path)

                stats[class_name] += 1

            except Exception as e:
                print(f"Warning: Failed to process {ann['filename']}: {e}")
                continue

        return dict(stats)

    def get_dataset_statistics(
        self,
        train_annotations: List[Dict],
        val_annotations: List[Dict]
    ) -> Dict:
        """
        Generate statistics about the split dataset.

        Args:
            train_annotations: Training annotations
            val_annotations: Validation annotations

        Returns:
            Statistics dictionary
        """
        def get_distribution(annotations, field):
            dist = defaultdict(int)
            for ann in annotations:
                dist[ann[field]] += 1
            return dict(dist)

        train_view_dist = get_distribution(train_annotations, 'view_type')
        val_view_dist = get_distribution(val_annotations, 'view_type')
        train_defect_dist = get_distribution(train_annotations, 'defect_type')
        val_defect_dist = get_distribution(val_annotations, 'defect_type')

        stats = {
            'total_train': len(train_annotations),
            'total_val': len(val_annotations),
            'split_ratio': f"{len(train_annotations)}/{len(val_annotations)}",
            'train_view_distribution': train_view_dist,
            'val_view_distribution': val_view_dist,
            'train_defect_distribution': train_defect_dist,
            'val_defect_distribution': val_defect_dist
        }

        return stats

    def validate_minimum_samples(
        self,
        annotations: List[Dict],
        min_samples_per_class: int = 1
    ) -> Tuple[bool, List[str]]:
        """
        Validate that each class has minimum number of samples.

        Args:
            annotations: List of annotations
            min_samples_per_class: Minimum samples required per class

        Returns:
            Tuple of (is_valid, list of issues)
        """
        view_counts = defaultdict(int)
        defect_counts = defaultdict(int)

        for ann in annotations:
            view_counts[ann['view_type']] += 1
            defect_counts[ann['defect_type']] += 1

        issues = []

        for view, count in view_counts.items():
            if count < min_samples_per_class:
                issues.append(f"View type '{view}' has only {count} samples (min: {min_samples_per_class})")

        for defect, count in defect_counts.items():
            if count < min_samples_per_class:
                issues.append(f"Defect type '{defect}' has only {count} samples (min: {min_samples_per_class})")

        return len(issues) == 0, issues

    def _filter_by_view(
        self,
        annotations: List[Dict],
        view_type: str
    ) -> List[Dict]:
        """
        Filter annotations by view type.

        Args:
            annotations: List of annotations
            view_type: 'TOP' or 'SIDE'

        Returns:
            Filtered list of annotations
        """
        return [ann for ann in annotations if ann['view_type'] == view_type]

    def prepare_full_pipeline(
        self,
        output_base_dir: str,
        val_ratio: float = 0.2,
        stratify_by: str = 'defect_type',
        preserve_wire_pairs: bool = True,
        view_aware: bool = True
    ) -> Dict:
        """
        Prepare all datasets for complete training pipeline.

        Args:
            output_base_dir: Base directory for all outputs
            val_ratio: Validation set ratio
            stratify_by: Field to stratify by
            preserve_wire_pairs: If True, keeps TOP/SIDE pairs together (default: True)
            view_aware: If True, creates separate datasets for TOP/SIDE views (default: True)

        Returns:
            Dictionary with all dataset paths and statistics

        Raises:
            DataPreparationError: If preparation fails
        """
        base_path = Path(output_base_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        # Perform split (with or without wire pairing)
        if preserve_wire_pairs:
            print("Using wire-aware stratified split to preserve TOP/SIDE pairs")
            train_annotations, val_annotations = self.stratified_split_with_pairing(val_ratio, stratify_by)
        else:
            print("Using standard stratified split (may split wire pairs)")
            train_annotations, val_annotations = self.stratified_split(val_ratio, stratify_by)

        # Validate
        is_valid, issues = self.validate_minimum_samples(val_annotations, min_samples_per_class=1)
        if not is_valid:
            print("Warning: Validation set has classes with very few samples:")
            for issue in issues:
                print(f"  - {issue}")

        # Prepare dataset info
        complete_info = {
            'preparation_info': {
                'val_ratio': val_ratio,
                'stratify_by': stratify_by,
                'random_seed': self.random_seed,
                'preserve_wire_pairs': preserve_wire_pairs,
                'view_aware': view_aware
            },
            'statistics': self.get_dataset_statistics(train_annotations, val_annotations)
        }

        if view_aware:
            print("Exporting VIEW-aware datasets (separate TOP/SIDE models)")

            # Filter annotations by view
            train_top = self._filter_by_view(train_annotations, 'TOP')
            train_side = self._filter_by_view(train_annotations, 'SIDE')
            val_top = self._filter_by_view(val_annotations, 'TOP')
            val_side = self._filter_by_view(val_annotations, 'SIDE')

            print(f"  TOP: {len(train_top)} train, {len(val_top)} val")
            print(f"  SIDE: {len(train_side)} train, {len(val_side)} val")

            # Export YOLO detection datasets (view-specific)
            yolo_top_info = self.export_yolo_format(
                train_top,
                val_top,
                str(base_path / "yolo_detection_top")
            )

            yolo_side_info = self.export_yolo_format(
                train_side,
                val_side,
                str(base_path / "yolo_detection_side")
            )

            # Export view classifier dataset (using full images, all views)
            view_info = self.export_classifier_dataset(
                train_annotations,
                val_annotations,
                str(base_path / "view_classifier"),
                classifier_type='view',
                crop_boxes=False  # VIEW uses full image, not cropped
            )

            # Export defect classifier datasets (view-specific)
            defect_top_info = self.export_classifier_dataset(
                train_top,
                val_top,
                str(base_path / "defect_classifier_top"),
                classifier_type='defect',
                crop_boxes=True
            )

            defect_side_info = self.export_classifier_dataset(
                train_side,
                val_side,
                str(base_path / "defect_classifier_side"),
                classifier_type='defect',
                crop_boxes=True
            )

            # Store all dataset info
            complete_info.update({
                'yolo_detection_top': yolo_top_info,
                'yolo_detection_side': yolo_side_info,
                'view_classifier': view_info,
                'defect_classifier_top': defect_top_info,
                'defect_classifier_side': defect_side_info
            })

        else:
            print("Exporting unified datasets (single model for both views)")

            # Export YOLO detection dataset (unified)
            yolo_info = self.export_yolo_format(
                train_annotations,
                val_annotations,
                str(base_path / "yolo_detection")
            )

            # Export view classifier dataset (using full images)
            view_info = self.export_classifier_dataset(
                train_annotations,
                val_annotations,
                str(base_path / "view_classifier"),
                classifier_type='view',
                crop_boxes=False  # VIEW uses full image, not cropped
            )

            # Export defect classifier dataset (unified)
            defect_info = self.export_classifier_dataset(
                train_annotations,
                val_annotations,
                str(base_path / "defect_classifier"),
                classifier_type='defect',
                crop_boxes=True
            )

            # Store all dataset info
            complete_info.update({
                'yolo_detection': yolo_info,
                'view_classifier': view_info,
                'defect_classifier': defect_info
            })

        # Save complete info
        info_path = base_path / "preparation_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(complete_info, f, indent=2, ensure_ascii=False)

        print(f"Dataset preparation complete. Info saved to: {info_path}")
        return complete_info
