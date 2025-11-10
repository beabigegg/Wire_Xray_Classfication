"""
Batch processor for inference.

Handles processing multiple images efficiently with progress tracking.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from .inference_pipeline import InferencePipeline
from .preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes multiple images in batches."""

    def __init__(
        self,
        pipeline: InferencePipeline,
        chunk_size: int = 16
    ):
        """
        Initialize batch processor.

        Args:
            pipeline: InferencePipeline instance
            chunk_size: Number of images to process in each chunk (for memory management)
        """
        self.pipeline = pipeline
        self.chunk_size = chunk_size
        self.preprocessor = ImagePreprocessor()

    def process_directory(
        self,
        directory_path: str,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Process all images in a directory.

        Args:
            directory_path: Path to directory containing images
            show_progress: Whether to show progress bar

        Returns:
            List of inference results for all images
        """
        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Find all image files
        image_files = self._scan_directory(directory)

        if not image_files:
            logger.warning(f"No image files found in {directory}")
            return []

        logger.info(f"Found {len(image_files)} images to process")

        # Process images
        results = []

        # Use tqdm for progress bar if requested
        iterator = tqdm(image_files, desc="Processing images", unit="img") if show_progress else image_files

        for image_path in iterator:
            try:
                result = self.pipeline.infer_single_image(str(image_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
                results.append({
                    'image_name': image_path.name,
                    'success': False,
                    'error': str(e),
                    'detections': [],
                    'primary_result': None,
                    'processing_time': 0.0
                })

        # Log summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")

        return results

    def process_file_list(
        self,
        file_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Process a list of image files.

        Args:
            file_paths: List of image file paths
            show_progress: Whether to show progress bar

        Returns:
            List of inference results
        """
        logger.info(f"Processing {len(file_paths)} images from file list")

        results = []

        # Use tqdm for progress bar if requested
        iterator = tqdm(file_paths, desc="Processing images", unit="img") if show_progress else file_paths

        for image_path in iterator:
            try:
                result = self.pipeline.infer_single_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'image_name': Path(image_path).name,
                    'success': False,
                    'error': str(e),
                    'detections': [],
                    'primary_result': None,
                    'processing_time': 0.0
                })

        # Log summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")

        return results

    def _scan_directory(self, directory: Path) -> List[Path]:
        """
        Scan directory for image files.

        Args:
            directory: Directory path

        Returns:
            List of image file paths
        """
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp'}
        image_files = []

        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                image_files.append(file_path)

        # Sort by filename
        image_files.sort()

        return image_files

    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        Calculate statistics from batch results.

        Args:
            results: List of inference results

        Returns:
            Dictionary with statistics
        """
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total - successful

        # Calculate detection statistics
        detected = sum(1 for r in results if r['success'] and r['primary_result'] and r['primary_result']['bbox'])
        no_detection = successful - detected

        # Calculate average processing time
        total_time = sum(r['processing_time'] for r in results)
        avg_time = total_time / total if total > 0 else 0

        # View distribution (for successful detections)
        view_dist = {}
        defect_dist = {}
        for r in results:
            if r['success'] and r['primary_result'] and r['primary_result']['view'] != 'no_detection':
                view = r['primary_result']['view']
                defect = r['primary_result']['defect']
                view_dist[view] = view_dist.get(view, 0) + 1
                defect_dist[defect] = defect_dist.get(defect, 0) + 1

        return {
            'total_images': total,
            'successful': successful,
            'failed': failed,
            'detected': detected,
            'no_detection': no_detection,
            'total_processing_time': total_time,
            'average_processing_time': avg_time,
            'view_distribution': view_dist,
            'defect_distribution': defect_dist
        }
