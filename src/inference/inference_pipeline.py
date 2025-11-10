"""
VIEW-aware three-stage inference pipeline.

Pipeline stages:
1. View Classification (full image) - Classify as TOP or SIDE
2. YOLO Detection (view-specific model) - Find wire loop bounding box
3. Defect Classification (view-specific model) - Classify as PASS/沖線/晃動/碰觸
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import numpy as np

from .model_loader import ModelLoader
from .preprocessor import ImagePreprocessor
from src.core.pairing_utils import (
    extract_wire_id,
    find_pair_filename,
    combine_pair_predictions
)

logger = logging.getLogger(__name__)


class InferencePipeline:
    """VIEW-aware three-stage inference pipeline for Wire Loop X-ray classification."""

    def __init__(
        self,
        model_loader: ModelLoader,
        confidence_threshold: float = 0.5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize VIEW-aware inference pipeline.

        Args:
            model_loader: ModelLoader instance with loaded VIEW-aware models
            confidence_threshold: Minimum confidence for YOLO detections
            device: Device to use (if None, uses model_loader's device)
        """
        self.model_loader = model_loader
        self.confidence_threshold = confidence_threshold
        self.device = device or model_loader.device

        # Load all VIEW-aware models
        (self.view_model,
         self.yolo_top_model,
         self.yolo_side_model,
         self.defect_top_model,
         self.defect_side_model) = model_loader.load_models()

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()

        logger.info("VIEW-aware inference pipeline initialized")

    def infer_single_image(self, image_path: str) -> Dict:
        """
        Run complete VIEW-aware three-stage inference on a single image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary containing inference results:
            {
                'image_name': str,
                'success': bool,
                'error': Optional[str],
                'detections': List[Dict],  # All detections found
                'primary_result': Optional[Dict],  # Main result used for classification
                'processing_time': float
            }
        """
        start_time = time.time()
        image_path = Path(image_path)

        try:
            # Load and validate image
            image = self.preprocessor.load_image(str(image_path))

            # Stage 1: View Classification (using full image)
            view_label, view_confidence = self._run_view_classification(image)
            logger.debug(f"View classified as: {view_label} (confidence: {view_confidence:.3f})")

            # Stage 2: YOLO Detection (using view-specific model)
            detections = self._run_yolo_detection(image, str(image_path), view_label)

            if not detections:
                # No detection found
                result = {
                    'image_name': image_path.name,
                    'success': True,
                    'error': None,
                    'detections': [],
                    'primary_result': {
                        'bbox': None,
                        'bbox_confidence': 0.0,
                        'view': view_label,
                        'view_confidence': view_confidence,
                        'defect': 'no_detection',
                        'defect_confidence': 0.0
                    },
                    'processing_time': time.time() - start_time
                }
                logger.info(f"No detection found for {image_path.name}")
                return result

            # Use highest confidence detection for defect classification
            primary_detection = detections[0]
            bbox = primary_detection['bbox']

            # Crop detected region for defect classification
            cropped_image = self.preprocessor.crop_bbox(image, bbox)

            # Stage 3: Defect Classification (using view-specific model)
            defect_label, defect_confidence = self._run_defect_classification(cropped_image, view_label)

            # Build result
            primary_result = {
                'bbox': bbox,
                'bbox_confidence': primary_detection['confidence'],
                'view': view_label,
                'view_confidence': view_confidence,
                'defect': defect_label,
                'defect_confidence': defect_confidence
            }

            result = {
                'image_name': image_path.name,
                'success': True,
                'error': None,
                'detections': detections,
                'primary_result': primary_result,
                'processing_time': time.time() - start_time
            }

            logger.info(
                f"Inference completed for {image_path.name}: "
                f"View={view_label}({view_confidence:.3f}), "
                f"Defect={defect_label}({defect_confidence:.3f})"
            )

            return result

        except Exception as e:
            logger.error(f"Inference failed for {image_path.name}: {e}")
            return {
                'image_name': image_path.name,
                'success': False,
                'error': str(e),
                'detections': [],
                'primary_result': None,
                'processing_time': time.time() - start_time
            }

    def infer_wire_pair(
        self,
        top_image_path: str,
        side_image_path: str,
        combination_strategy: str = "worst_case"
    ) -> Dict:
        """
        Run inference on paired TOP/SIDE images and combine results.

        Args:
            top_image_path: Path to TOP view image
            side_image_path: Path to SIDE view image
            combination_strategy: How to combine results ("worst_case" or "confidence")

        Returns:
            Dictionary containing combined results
        """
        start_time = time.time()
        top_path = Path(top_image_path)
        side_path = Path(side_image_path)

        try:
            # Extract wire_id from filename
            wire_id_top = extract_wire_id(top_path.name)
            wire_id_side = extract_wire_id(side_path.name)

            if wire_id_top != wire_id_side:
                raise ValueError(
                    f"Wire ID mismatch: TOP={wire_id_top}, SIDE={wire_id_side}. "
                    "TOP and SIDE images must be from the same wire."
                )

            wire_id = wire_id_top

            # Run inference on both views
            logger.info(f"Running inference on wire pair: {wire_id}")
            top_result = self.infer_single_image(top_image_path)
            side_result = self.infer_single_image(side_image_path)

            # Check if both inferences succeeded
            if not top_result['success'] or not side_result['success']:
                error_msg = []
                if not top_result['success']:
                    error_msg.append(f"TOP: {top_result['error']}")
                if not side_result['success']:
                    error_msg.append(f"SIDE: {side_result['error']}")

                return {
                    'wire_id': wire_id,
                    'success': False,
                    'error': "; ".join(error_msg),
                    'top_result': top_result,
                    'side_result': side_result,
                    'combined_result': None,
                    'processing_time': time.time() - start_time
                }

            # Extract primary results
            top_primary = top_result['primary_result']
            side_primary = side_result['primary_result']

            # Check for no detection cases
            if top_primary['defect'] == 'no_detection' or side_primary['defect'] == 'no_detection':
                return {
                    'wire_id': wire_id,
                    'success': True,
                    'error': 'No detection in one or both views',
                    'top_result': top_result,
                    'side_result': side_result,
                    'combined_result': {
                        'defect_type': 'no_detection',
                        'defect_confidence': 0.0,
                        'decisive_view': None
                    },
                    'processing_time': time.time() - start_time
                }

            # Combine predictions
            combined = combine_pair_predictions(
                {
                    'defect_type': top_primary['defect'],
                    'defect_confidence': top_primary['defect_confidence']
                },
                {
                    'defect_type': side_primary['defect'],
                    'defect_confidence': side_primary['defect_confidence']
                },
                strategy=combination_strategy
            )

            result = {
                'wire_id': wire_id,
                'success': True,
                'error': None,
                'top_result': top_result,
                'side_result': side_result,
                'combined_result': combined,
                'processing_time': time.time() - start_time
            }

            logger.info(
                f"Paired inference completed for wire {wire_id}: "
                f"TOP={top_primary['defect']}({top_primary['defect_confidence']:.3f}), "
                f"SIDE={side_primary['defect']}({side_primary['defect_confidence']:.3f}), "
                f"Combined={combined['defect_type']}({combined['defect_confidence']:.3f}) "
                f"[{combined['decisive_view']}]"
            )

            return result

        except Exception as e:
            logger.error(f"Paired inference failed: {e}")
            return {
                'wire_id': wire_id if 'wire_id' in locals() else 'unknown',
                'success': False,
                'error': str(e),
                'top_result': None,
                'side_result': None,
                'combined_result': None,
                'processing_time': time.time() - start_time
            }

    def infer_batch_with_pairing(
        self,
        image_dir: str,
        combination_strategy: str = "worst_case"
    ) -> List[Dict]:
        """
        Run inference on all wire pairs in a directory.

        Args:
            image_dir: Directory containing TOP and SIDE images
            combination_strategy: How to combine results

        Returns:
            List of wire pair results
        """
        image_dir_path = Path(image_dir)

        # Find all TOP images
        top_images = list(image_dir_path.glob("*_TOP.*"))

        results = []

        for top_image in top_images:
            # Find corresponding SIDE image
            side_filename = find_pair_filename(top_image.name)

            if not side_filename:
                logger.warning(f"Invalid filename format: {top_image.name}")
                continue

            side_image = image_dir_path / side_filename

            if not side_image.exists():
                logger.warning(f"Missing SIDE pair for {top_image.name}")
                continue

            # Run paired inference
            result = self.infer_wire_pair(
                str(top_image),
                str(side_image),
                combination_strategy
            )

            results.append(result)

        logger.info(f"Batch inference completed: {len(results)} wire pairs processed")
        return results

    def _run_yolo_detection(
        self,
        image: np.ndarray,
        image_path: str,
        view: str
    ) -> List[Dict]:
        """
        Run VIEW-specific YOLO detection.

        Args:
            image: Image array (H, W, C)
            image_path: Path to image (for YOLO)
            view: View type ('TOP' or 'SIDE')

        Returns:
            List of detections, sorted by confidence (highest first)
            Each detection: {'bbox': (x1, y1, x2, y2), 'confidence': float}
        """
        # Get view-specific YOLO model
        yolo_model = self.model_loader.get_yolo_model(view)

        # Run YOLO inference
        results = yolo_model(image_path, verbose=False)

        detections = []

        # Extract detections
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    confidence = float(box.conf[0])
                    if confidence >= self.confidence_threshold:
                        # Get bbox coordinates (x1, y1, x2, y2)
                        xyxy = box.xyxy[0].cpu().numpy()
                        bbox = tuple(int(x) for x in xyxy)

                        detections.append({
                            'bbox': bbox,
                            'confidence': confidence
                        })

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        logger.debug(f"YOLO {view} found {len(detections)} detections")
        return detections

    def _run_view_classification(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Run view classification (full image).

        Args:
            image: Full image array (H, W, C)

        Returns:
            Tuple of (class_label, confidence)
        """
        # Preprocess image
        input_tensor = self.preprocessor.preprocess_for_classifier(image)
        input_tensor = input_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.view_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        class_idx = predicted.item()
        confidence_value = confidence.item()
        class_label = self.model_loader.get_view_class_name(class_idx)

        return class_label, confidence_value

    def _run_defect_classification(self, image: np.ndarray, view: str) -> Tuple[str, float]:
        """
        Run VIEW-specific defect classification.

        Args:
            image: Cropped image array (H, W, C)
            view: View type ('TOP' or 'SIDE')

        Returns:
            Tuple of (class_label, confidence)
        """
        # Get view-specific defect model
        defect_model = self.model_loader.get_defect_model(view)

        # Preprocess image
        input_tensor = self.preprocessor.preprocess_for_classifier(image)
        input_tensor = input_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = defect_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        class_idx = predicted.item()
        confidence_value = confidence.item()
        class_label = self.model_loader.get_defect_class_name(class_idx)

        return class_label, confidence_value

    def get_pipeline_info(self) -> Dict:
        """
        Get pipeline configuration info.

        Returns:
            Dictionary with pipeline info
        """
        return {
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'view_classes': self.model_loader.view_classes,
            'defect_classes': self.model_loader.defect_classes,
            'architecture': 'VIEW-aware (separate models for TOP/SIDE)'
        }
