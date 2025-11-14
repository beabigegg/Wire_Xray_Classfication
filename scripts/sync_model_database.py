"""
Sync Model Database Script

This script scans the models/ directory and synchronizes all model files
with the database. It will:
1. Find all .pt model files
2. Read their metadata from .json files
3. Save records to the database
4. Report sync status

Usage:
    python scripts/sync_model_database.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import Database, DatabaseError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_metrics_from_filename(filename: str, model_type: str) -> dict:
    """
    Extract metrics from filename as fallback.

    Args:
        filename: Model filename (e.g., v1_20251111_094424_acc1.00.pt)
        model_type: Type of model

    Returns:
        Dictionary of metrics
    """
    metrics = {}
    stem = filename.replace('.pt', '')

    # Extract metric from filename
    if '_map' in stem:
        # YOLO detection model (e.g., _map0.95)
        try:
            metric_str = stem.split('_map')[1]
            metrics['map50'] = float(metric_str)
        except (IndexError, ValueError):
            pass
    elif '_acc' in stem:
        # Classifier model (e.g., _acc1.00)
        try:
            metric_str = stem.split('_acc')[1]
            # Remove 'bacc' prefix if exists
            if metric_str.startswith('bacc'):
                metric_str = metric_str[4:]
            metrics['accuracy'] = float(metric_str)
        except (IndexError, ValueError):
            pass
    elif '_bacc' in stem:
        # Balanced accuracy (e.g., _bacc0.75)
        try:
            metric_str = stem.split('_bacc')[1]
            metrics['balanced_accuracy'] = float(metric_str)
        except (IndexError, ValueError):
            pass

    return metrics


def sync_models_to_database(db_path: str = 'annotations.db', models_dir: str = 'models', dry_run: bool = False):
    """
    Sync all models from filesystem to database.

    Args:
        db_path: Path to database file
        models_dir: Path to models directory
        dry_run: If True, only report what would be done without making changes
    """
    logger.info("="*80)
    logger.info("Model Database Synchronization Tool")
    logger.info("="*80)

    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made to database")

    # Initialize database
    try:
        db = Database(db_path)
        logger.info(f"✓ Connected to database: {db_path}")
    except Exception as e:
        logger.error(f"✗ Failed to connect to database: {e}")
        return False

    # Scan for model files
    models_path = Path(models_dir)
    if not models_path.exists():
        logger.error(f"✗ Models directory does not exist: {models_dir}")
        return False

    model_files = list(models_path.rglob('*.pt'))
    logger.info(f"Found {len(model_files)} model files")
    logger.info("-"*80)

    synced_count = 0
    skipped_count = 0
    failed_count = 0

    for model_file in sorted(model_files):
        # Extract info from path
        model_type = model_file.parent.name
        version = model_file.stem
        # Store filepath relative to project root for portability
        try:
            filepath = str(model_file.relative_to(models_path.parent))
        except ValueError:
            # Fallback: use absolute path
            filepath = str(model_file)

        logger.info(f"\n📦 Processing: {model_type}/{version}")

        # Check if already in database
        try:
            existing_models = db.get_model_versions(model_type=model_type)
            already_exists = any(m['version'] == version for m in existing_models)

            if already_exists:
                logger.info(f"   ⏭  Already in database - SKIPPED")
                skipped_count += 1
                continue
        except DatabaseError as e:
            logger.warning(f"   ⚠  Could not check existing models: {e}")

        # Try to load metadata from .json file
        metadata_file = model_file.with_suffix('.json')
        metrics = {}
        config = {}
        model_name = f"{model_type}_model"

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    metrics = metadata.get('metrics', {})
                    config = metadata.get('config', {})
                    model_name = metadata.get('model_name', model_name)
                logger.info(f"   ✓ Loaded metadata from {metadata_file.name}")
            except Exception as e:
                logger.warning(f"   ⚠  Failed to load metadata: {e}")
                # Fallback: extract metrics from filename
                metrics = extract_metrics_from_filename(model_file.name, model_type)
        else:
            logger.info(f"   ℹ  No metadata file found, extracting from filename")
            metrics = extract_metrics_from_filename(model_file.name, model_type)

        # Display metrics
        if metrics:
            logger.info(f"   📊 Metrics: {metrics}")
        else:
            logger.warning(f"   ⚠  No metrics available")

        # Save to database
        if not dry_run:
            try:
                model_id = db.save_model_version(
                    model_name=model_name,
                    model_type=model_type,
                    version=version,
                    filepath=filepath,
                    metrics=metrics,
                    set_active=False  # Don't auto-activate
                )
                logger.info(f"   ✅ Synced to database (ID: {model_id})")
                synced_count += 1
            except DatabaseError as e:
                logger.error(f"   ✗ Failed to save: {e}")
                failed_count += 1
        else:
            logger.info(f"   [DRY RUN] Would sync to database")
            synced_count += 1

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Synchronization Summary")
    logger.info("="*80)
    logger.info(f"Total files found:    {len(model_files)}")
    logger.info(f"✅ Newly synced:      {synced_count}")
    logger.info(f"⏭  Already in DB:     {skipped_count}")
    logger.info(f"✗ Failed:             {failed_count}")

    if not dry_run:
        logger.info("\n" + "-"*80)
        logger.info("Current database contents:")
        logger.info("-"*80)

        # Show all models in database by type
        model_types = ['view', 'detection', 'detection_top', 'detection_side',
                      'defect', 'defect_top', 'defect_side']

        for model_type in model_types:
            try:
                models = db.get_model_versions(model_type=model_type)
                if models:
                    logger.info(f"\n{model_type.upper()}:")
                    for model in models:
                        active_marker = " [ACTIVE]" if model.get('is_active') else ""
                        logger.info(f"  - {model['version']}{active_marker}")
            except DatabaseError:
                pass

    db.close()
    logger.info("\n" + "="*80)
    logger.info("✓ Synchronization complete!")
    logger.info("="*80)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Sync model files to database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--db', default='annotations.db',
                       help='Path to database file (default: annotations.db)')
    parser.add_argument('--models-dir', default='models',
                       help='Path to models directory (default: models)')

    args = parser.parse_args()

    success = sync_models_to_database(
        db_path=args.db,
        models_dir=args.models_dir,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)
