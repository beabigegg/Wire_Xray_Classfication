"""
Result exporter for inference results.

Exports results in CSV and JSON formats.
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class ResultExporter:
    """Exports inference results to various formats."""

    @staticmethod
    def export_csv(results: List[Dict], output_path: str):
        """
        Export results to CSV format.

        CSV format:
        image_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2, view, view_confidence, defect, defect_confidence, success, error

        Args:
            results: List of inference results
            output_path: Path to output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'image_name',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'view', 'view_confidence',
                'defect', 'defect_confidence',
                'success', 'error'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header
            writer.writeheader()

            # Write results
            for result in results:
                if result['success'] and result['primary_result']:
                    primary = result['primary_result']
                    bbox = primary['bbox']

                    if bbox:
                        row = {
                            'image_name': result['image_name'],
                            'bbox_x1': bbox[0],
                            'bbox_y1': bbox[1],
                            'bbox_x2': bbox[2],
                            'bbox_y2': bbox[3],
                            'view': primary['view'],
                            'view_confidence': f"{primary['view_confidence']:.4f}",
                            'defect': primary['defect'],
                            'defect_confidence': f"{primary['defect_confidence']:.4f}",
                            'success': 'True',
                            'error': ''
                        }
                    else:
                        # No detection
                        row = {
                            'image_name': result['image_name'],
                            'bbox_x1': '',
                            'bbox_y1': '',
                            'bbox_x2': '',
                            'bbox_y2': '',
                            'view': 'no_detection',
                            'view_confidence': '0.0000',
                            'defect': 'no_detection',
                            'defect_confidence': '0.0000',
                            'success': 'True',
                            'error': ''
                        }
                else:
                    # Failed inference
                    row = {
                        'image_name': result['image_name'],
                        'bbox_x1': '',
                        'bbox_y1': '',
                        'bbox_x2': '',
                        'bbox_y2': '',
                        'view': '',
                        'view_confidence': '',
                        'defect': '',
                        'defect_confidence': '',
                        'success': 'False',
                        'error': result.get('error', 'Unknown error')
                    }

                writer.writerow(row)

        logger.info(f"Exported {len(results)} results to CSV: {output_path}")

    @staticmethod
    def export_json(results: List[Dict], output_path: str, indent: int = 2):
        """
        Export results to JSON format.

        Args:
            results: List of inference results
            output_path: Path to output JSON file
            indent: JSON indentation level
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to JSON-serializable format
        json_results = []
        for result in results:
            json_result = {
                'image_name': result['image_name'],
                'success': result['success'],
                'processing_time': result['processing_time']
            }

            if result['success']:
                if result['primary_result']:
                    primary = result['primary_result']
                    json_result['bbox'] = {
                        'x1': primary['bbox'][0] if primary['bbox'] else None,
                        'y1': primary['bbox'][1] if primary['bbox'] else None,
                        'x2': primary['bbox'][2] if primary['bbox'] else None,
                        'y2': primary['bbox'][3] if primary['bbox'] else None,
                        'confidence': primary['bbox_confidence']
                    } if primary['bbox'] else None

                    json_result['view'] = {
                        'label': primary['view'],
                        'confidence': primary['view_confidence']
                    }

                    json_result['defect'] = {
                        'label': primary['defect'],
                        'confidence': primary['defect_confidence']
                    }

                # Include all detections
                json_result['all_detections'] = [
                    {
                        'bbox': {'x1': d['bbox'][0], 'y1': d['bbox'][1], 'x2': d['bbox'][2], 'y2': d['bbox'][3]},
                        'confidence': d['confidence']
                    }
                    for d in result['detections']
                ]
            else:
                json_result['error'] = result.get('error', 'Unknown error')

            json_results.append(json_result)

        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_results, jsonfile, indent=indent, ensure_ascii=False)

        logger.info(f"Exported {len(results)} results to JSON: {output_path}")

    @staticmethod
    def export_summary(results: List[Dict], statistics: Dict, output_path: str):
        """
        Export summary statistics to text file.

        Args:
            results: List of inference results
            statistics: Statistics dictionary from BatchProcessor
            output_path: Path to output text file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Inference Summary\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total images: {statistics['total_images']}\n")
            f.write(f"Successful: {statistics['successful']}\n")
            f.write(f"Failed: {statistics['failed']}\n")
            f.write(f"Detected: {statistics['detected']}\n")
            f.write(f"No detection: {statistics['no_detection']}\n\n")

            f.write(f"Total processing time: {statistics['total_processing_time']:.2f} seconds\n")
            f.write(f"Average processing time: {statistics['average_processing_time']:.3f} seconds/image\n\n")

            # View distribution
            f.write("View Distribution:\n")
            for view, count in statistics['view_distribution'].items():
                percentage = (count / statistics['detected'] * 100) if statistics['detected'] > 0 else 0
                f.write(f"  {view}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            # Defect distribution
            f.write("Defect Distribution:\n")
            for defect, count in statistics['defect_distribution'].items():
                percentage = (count / statistics['detected'] * 100) if statistics['detected'] > 0 else 0
                f.write(f"  {defect}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            # Failed images
            if statistics['failed'] > 0:
                f.write("Failed Images:\n")
                for result in results:
                    if not result['success']:
                        f.write(f"  {result['image_name']}: {result.get('error', 'Unknown error')}\n")

        logger.info(f"Exported summary to: {output_path}")
