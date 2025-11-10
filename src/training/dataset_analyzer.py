"""
Dataset analyzer for Wire Loop annotation data.

Provides statistics, validation, and insights about the annotation dataset.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import json


class DatasetAnalyzer:
    """Analyze annotation dataset and provide statistics."""

    def __init__(self, db_path: str):
        """
        Initialize analyzer.

        Args:
            db_path: Path to annotations database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def get_overall_statistics(self) -> Dict:
        """
        Get overall dataset statistics.

        Returns:
            Dictionary with overall statistics
        """
        cursor = self.conn.cursor()

        # Total counts
        cursor.execute("SELECT COUNT(*) FROM images")
        total_images = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM annotations")
        total_annotations = cursor.fetchone()[0]

        # View type distribution
        cursor.execute("""
            SELECT view_type, COUNT(*) as count
            FROM annotations
            GROUP BY view_type
            ORDER BY count DESC
        """)
        view_distribution = {row['view_type']: row['count'] for row in cursor.fetchall()}

        # Defect type distribution
        cursor.execute("""
            SELECT defect_type, COUNT(*) as count
            FROM annotations
            GROUP BY defect_type
            ORDER BY count DESC
        """)
        defect_distribution = {row['defect_type']: row['count'] for row in cursor.fetchall()}

        # BBox size statistics
        cursor.execute("""
            SELECT
                AVG(bbox_width) as avg_width,
                AVG(bbox_height) as avg_height,
                MIN(bbox_width) as min_width,
                MIN(bbox_height) as min_height,
                MAX(bbox_width) as max_width,
                MAX(bbox_height) as max_height
            FROM annotations
        """)
        bbox_stats = dict(cursor.fetchone())

        return {
            'total_images': total_images,
            'total_annotations': total_annotations,
            'view_distribution': view_distribution,
            'defect_distribution': defect_distribution,
            'bbox_statistics': bbox_stats,
            'annotations_per_image': total_annotations / total_images if total_images > 0 else 0
        }

    def get_class_balance_report(self) -> Dict:
        """
        Generate class balance report for training.

        Returns:
            Dictionary with balance analysis
        """
        stats = self.get_overall_statistics()

        # Analyze view balance
        view_dist = stats['view_distribution']
        view_total = sum(view_dist.values())
        view_balance = {
            k: {
                'count': v,
                'percentage': (v / view_total * 100) if view_total > 0 else 0,
                'is_balanced': abs(v / view_total - 0.5) < 0.1 if view_total > 0 else False
            }
            for k, v in view_dist.items()
        }

        # Analyze defect balance
        defect_dist = stats['defect_distribution']
        defect_total = sum(defect_dist.values())
        expected_percentage = 100.0 / len(defect_dist) if defect_dist else 0
        defect_balance = {
            k: {
                'count': v,
                'percentage': (v / defect_total * 100) if defect_total > 0 else 0,
                'deviation_from_balanced': abs(v / defect_total * 100 - expected_percentage) if defect_total > 0 else 0,
                'needs_augmentation': (v / defect_total * 100) < (expected_percentage * 0.5) if defect_total > 0 else False
            }
            for k, v in defect_dist.items()
        }

        return {
            'view_balance': view_balance,
            'defect_balance': defect_balance,
            'overall_balance_score': self._calculate_balance_score(defect_dist)
        }

    def _calculate_balance_score(self, distribution: Dict[str, int]) -> float:
        """
        Calculate balance score (0-1, where 1 is perfectly balanced).

        Args:
            distribution: Class distribution dictionary

        Returns:
            Balance score
        """
        if not distribution:
            return 0.0

        values = list(distribution.values())
        total = sum(values)
        if total == 0:
            return 0.0

        expected = total / len(values)
        deviations = [abs(v - expected) / expected for v in values]
        avg_deviation = sum(deviations) / len(deviations)

        # Score: 1 - normalized deviation
        return max(0.0, 1.0 - avg_deviation)

    def get_train_val_split_suggestion(self, val_ratio: float = 0.2) -> Dict:
        """
        Suggest train/validation split maintaining class balance.

        Args:
            val_ratio: Ratio of validation set (default: 0.2)

        Returns:
            Dictionary with split suggestions
        """
        cursor = self.conn.cursor()

        # Get all annotations with their labels
        cursor.execute("""
            SELECT
                i.id as image_id,
                i.filepath,
                a.view_type,
                a.defect_type
            FROM images i
            JOIN annotations a ON i.id = a.image_id
        """)
        annotations = cursor.fetchall()

        # Group by defect type
        defect_groups = {}
        for ann in annotations:
            defect = ann['defect_type']
            if defect not in defect_groups:
                defect_groups[defect] = []
            defect_groups[defect].append(dict(ann))

        # Calculate split for each class
        split_plan = {}
        for defect, items in defect_groups.items():
            val_count = int(len(items) * val_ratio)
            train_count = len(items) - val_count

            split_plan[defect] = {
                'total': len(items),
                'train': train_count,
                'val': val_count,
                'train_percentage': (train_count / len(items) * 100) if len(items) > 0 else 0,
                'val_percentage': (val_count / len(items) * 100) if len(items) > 0 else 0
            }

        return {
            'val_ratio': val_ratio,
            'split_plan': split_plan,
            'total_train': sum(s['train'] for s in split_plan.values()),
            'total_val': sum(s['val'] for s in split_plan.values())
        }

    def check_data_quality(self) -> List[Dict]:
        """
        Check for potential data quality issues.

        Returns:
            List of issues found
        """
        issues = []
        cursor = self.conn.cursor()

        # Check for very small bounding boxes
        cursor.execute("""
            SELECT i.filename, a.bbox_width, a.bbox_height
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            WHERE a.bbox_width < 50 OR a.bbox_height < 20
        """)
        small_boxes = cursor.fetchall()
        if small_boxes:
            issues.append({
                'type': 'small_bounding_boxes',
                'severity': 'warning',
                'count': len(small_boxes),
                'message': f'{len(small_boxes)} annotations have very small bounding boxes',
                'examples': [dict(b) for b in small_boxes[:3]]
            })

        # Check for out-of-bounds boxes
        cursor.execute("""
            SELECT i.filename, a.bbox_x, a.bbox_y, a.bbox_width, a.bbox_height
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            WHERE a.bbox_x < 0 OR a.bbox_y < 0
               OR (a.bbox_x + a.bbox_width) > i.width
               OR (a.bbox_y + a.bbox_height) > i.height
        """)
        oob_boxes = cursor.fetchall()
        if oob_boxes:
            issues.append({
                'type': 'out_of_bounds',
                'severity': 'error',
                'count': len(oob_boxes),
                'message': f'{len(oob_boxes)} annotations have out-of-bounds coordinates',
                'examples': [dict(b) for b in oob_boxes[:3]]
            })

        # Check for missing image files
        cursor.execute("SELECT filename, filepath FROM images")
        for row in cursor.fetchall():
            if not Path(row['filepath']).exists():
                issues.append({
                    'type': 'missing_file',
                    'severity': 'error',
                    'message': f"Image file not found: {row['filename']}",
                    'filepath': row['filepath']
                })

        # Check class imbalance (defect types)
        stats = self.get_overall_statistics()
        defect_counts = list(stats['defect_distribution'].values())
        if defect_counts:
            max_count = max(defect_counts)
            min_count = min(defect_counts)
            if max_count > min_count * 5:  # More than 5x difference
                issues.append({
                    'type': 'class_imbalance',
                    'severity': 'warning',
                    'message': f'Severe class imbalance detected (ratio: {max_count / min_count:.1f}:1)',
                    'distribution': stats['defect_distribution']
                })

        return issues

    def generate_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive dataset analysis report.

        Args:
            output_path: Optional path to save JSON report

        Returns:
            JSON report string
        """
        report = {
            'overall_statistics': self.get_overall_statistics(),
            'class_balance': self.get_class_balance_report(),
            'split_suggestion': self.get_train_val_split_suggestion(),
            'quality_issues': self.check_data_quality()
        }

        report_json = json.dumps(report, indent=2, ensure_ascii=False)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_json)

        return report_json

    def print_summary(self):
        """Print human-readable summary of dataset."""
        stats = self.get_overall_statistics()
        balance = self.get_class_balance_report()
        split = self.get_train_val_split_suggestion()
        issues = self.check_data_quality()

        print("=" * 60)
        print("Wire Loop Dataset Analysis Report")
        print("=" * 60)

        print(f"\n[Overall Statistics]")
        print(f"  Total Images: {stats['total_images']}")
        print(f"  Total Annotations: {stats['total_annotations']}")
        print(f"  Avg Annotations per Image: {stats['annotations_per_image']:.2f}")

        print(f"\n[View Type Distribution]")
        for view, count in stats['view_distribution'].items():
            pct = balance['view_balance'][view]['percentage']
            print(f"  {view:10} {count:3} ({pct:.1f}%)")

        print(f"\n[Defect Type Distribution]")
        for defect, count in stats['defect_distribution'].items():
            pct = balance['defect_balance'][defect]['percentage']
            needs_aug = balance['defect_balance'][defect]['needs_augmentation']
            warning = " [NEEDS AUGMENTATION]" if needs_aug else ""
            print(f"  {defect:10} {count:3} ({pct:.1f}%){warning}")

        print(f"\n[Bounding Box Statistics]")
        bbox = stats['bbox_statistics']
        print(f"  Average Size: {bbox['avg_width']:.1f} x {bbox['avg_height']:.1f}")
        print(f"  Min Size: {bbox['min_width']:.1f} x {bbox['min_height']:.1f}")
        print(f"  Max Size: {bbox['max_width']:.1f} x {bbox['max_height']:.1f}")

        print(f"\n[Suggested Train/Val Split (80/20)]")
        print(f"  Training Set: {split['total_train']} images")
        print(f"  Validation Set: {split['total_val']} images")

        if issues:
            print(f"\n[WARNING] Data Quality Issues: {len(issues)} found")
            for issue in issues:
                severity_icon = "[ERROR]" if issue['severity'] == 'error' else "[WARN]"
                print(f"  {severity_icon} {issue['type']}: {issue['message']}")
        else:
            print(f"\n[OK] No data quality issues found!")

        print(f"\n[Overall Balance Score] {balance['overall_balance_score']:.2f}/1.00")

        print("=" * 60)

    def close(self):
        """Close database connection."""
        self.conn.close()


if __name__ == "__main__":
    # Example usage
    analyzer = DatasetAnalyzer("annotations.db")
    analyzer.print_summary()
    analyzer.generate_report("dataset_analysis.json")
    analyzer.close()
