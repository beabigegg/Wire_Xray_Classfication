"""
Backfill TensorBoard logs for completed YOLO training runs.

This script parses results.csv from YOLO training runs and creates
TensorBoard event files for visualization.

Usage:
    python scripts/backfill_yolo_tensorboard.py runs/detection_top/yolo_20251111_094917
"""

import sys
import csv
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def backfill_tensorboard(run_dir: str):
    """
    Parse YOLO results.csv and create TensorBoard logs.

    Args:
        run_dir: Path to YOLO training run directory
    """
    run_path = Path(run_dir)
    results_csv = run_path / 'results.csv'

    if not results_csv.exists():
        print(f"[ERROR] results.csv not found at {results_csv}")
        return False

    print(f"[*] Backfilling TensorBoard logs for: {run_path.name}")
    print(f"[*] Log directory: {run_path}")

    try:
        # Create TensorBoard writer
        writer = SummaryWriter(log_dir=str(run_path))

        # Read and parse CSV
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"[*] Found {len(rows)} epochs of data")

        # Log each epoch's metrics
        for row in rows:
            epoch = int(row.get('epoch', 0))

            # Training losses
            if 'train/box_loss' in row and row['train/box_loss']:
                writer.add_scalar('train/box_loss', float(row['train/box_loss']), epoch)
            if 'train/cls_loss' in row and row['train/cls_loss']:
                writer.add_scalar('train/cls_loss', float(row['train/cls_loss']), epoch)
            if 'train/dfl_loss' in row and row['train/dfl_loss']:
                writer.add_scalar('train/dfl_loss', float(row['train/dfl_loss']), epoch)

            # Validation losses
            if 'val/box_loss' in row and row['val/box_loss']:
                writer.add_scalar('val/box_loss', float(row['val/box_loss']), epoch)
            if 'val/cls_loss' in row and row['val/cls_loss']:
                writer.add_scalar('val/cls_loss', float(row['val/cls_loss']), epoch)
            if 'val/dfl_loss' in row and row['val/dfl_loss']:
                writer.add_scalar('val/dfl_loss', float(row['val/dfl_loss']), epoch)

            # Metrics
            if 'metrics/precision(B)' in row and row['metrics/precision(B)']:
                writer.add_scalar('metrics/precision', float(row['metrics/precision(B)']), epoch)
            if 'metrics/recall(B)' in row and row['metrics/recall(B)']:
                writer.add_scalar('metrics/recall', float(row['metrics/recall(B)']), epoch)
            if 'metrics/mAP50(B)' in row and row['metrics/mAP50(B)']:
                writer.add_scalar('metrics/mAP50', float(row['metrics/mAP50(B)']), epoch)
            if 'metrics/mAP50-95(B)' in row and row['metrics/mAP50-95(B)']:
                writer.add_scalar('metrics/mAP50-95', float(row['metrics/mAP50-95(B)']), epoch)

            # Learning rates
            if 'lr/pg0' in row and row['lr/pg0']:
                writer.add_scalar('learning_rate/pg0', float(row['lr/pg0']), epoch)

        writer.flush()
        writer.close()

        print(f"[OK] Successfully logged {len(rows)} epochs to TensorBoard")
        print(f"[*] View logs with: tensorboard --logdir={run_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to backfill TensorBoard logs: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/backfill_yolo_tensorboard.py <run_directory>")
        print("\nExample:")
        print("  python scripts/backfill_yolo_tensorboard.py runs/detection_top/yolo_20251111_094917")
        print("\nOr backfill all runs:")
        print("  python scripts/backfill_yolo_tensorboard.py runs/detection_top")
        sys.exit(1)

    target_path = Path(sys.argv[1])

    if not target_path.exists():
        print(f"[ERROR] Path does not exist: {target_path}")
        sys.exit(1)

    # If directory contains subdirectories with results.csv, process all
    if target_path.is_dir():
        # Check if this is a single run directory
        if (target_path / 'results.csv').exists():
            backfill_tensorboard(str(target_path))
        else:
            # Process all subdirectories
            processed = 0
            for subdir in target_path.iterdir():
                if subdir.is_dir() and (subdir / 'results.csv').exists():
                    print(f"\n{'='*70}")
                    if backfill_tensorboard(str(subdir)):
                        processed += 1

            print(f"\n{'='*70}")
            print(f"[*] Summary: Processed {processed} training runs")
    else:
        print(f"[ERROR] Not a directory: {target_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
