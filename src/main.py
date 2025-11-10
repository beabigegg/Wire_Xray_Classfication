"""
Main entry point for Wire Loop Annotation Tool.

This module initializes the application, database, and launches the GUI.
"""

import sys
import argparse
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.core.config import Config, ConfigError, create_default_config
from src.core.database import Database, DatabaseError
from src.gui.annotation_window import AnnotationWindow


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Wire Loop X-ray Annotation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python -m src.main

  # Run with custom config
  python -m src.main --config my_config.yaml

  # Specify custom database
  python -m src.main --database my_annotations.db

  # Set annotator name
  python -m src.main --annotator "John Doe"

  # Create default config file
  python -m src.main --create-config
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )

    parser.add_argument(
        '--database',
        type=str,
        default=None,
        help='Path to SQLite database file (default: from config or annotations.db)'
    )

    parser.add_argument(
        '--annotator',
        type=str,
        default=None,
        help='Annotator username (default: from config or "unknown")'
    )

    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default config.yaml file and exit'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Wire Loop Annotation Tool v1.0.0'
    )

    return parser.parse_args()


def create_directories(config: Config):
    """
    Create necessary directories if they don't exist.

    Args:
        config: Configuration object
    """
    dirs_to_create = [
        config.data_dir,
        config.labels_dir,
        config.get('paths.models_dir', 'models'),
        config.get('paths.runs_dir', 'runs')
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()

    # Handle config creation
    if args.create_config:
        try:
            output_path = args.config if args.config != 'config.yaml' else 'config.yaml'
            create_default_config(output_path)
            print(f"Created default configuration file: {output_path}")
            return 0
        except ConfigError as e:
            print(f"Error creating config file: {e}", file=sys.stderr)
            return 1

    # Load configuration
    try:
        if Path(args.config).exists():
            config = Config(args.config)
            print(f"Loaded configuration from: {args.config}")
        else:
            print(f"Configuration file not found: {args.config}")
            print("Using default configuration")
            config = Config()
    except ConfigError as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        print("Using default configuration")
        config = Config()

    # Override config with command line arguments
    if args.database:
        config.set('paths.database', args.database)

    if args.annotator:
        config.set('annotation.default_annotator', args.annotator)

    # Create necessary directories
    try:
        create_directories(config)
    except Exception as e:
        print(f"Warning: Failed to create directories: {e}", file=sys.stderr)

    # Initialize database
    try:
        db = Database(config.database_path)
        print(f"Initialized database: {config.database_path}")
    except DatabaseError as e:
        print(f"Error initializing database: {e}", file=sys.stderr)
        return 1

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName(config.get('application.name', 'Wire Loop Annotation Tool'))
    app.setApplicationVersion(config.get('application.version', '1.0.0'))

    # Prompt for annotator if not specified via command line
    if not args.annotator:
        from src.gui.annotator_dialog import AnnotatorDialog
        current_annotator = config.get('annotation.default_annotator', '')
        dialog = AnnotatorDialog(current_annotator=current_annotator)

        if dialog.exec() == AnnotatorDialog.DialogCode.Accepted:
            annotator = dialog.get_annotator()
            if annotator:
                config.set('annotation.default_annotator', annotator)
                config.save()
        else:
            # User cancelled, exit application
            print("Annotator selection cancelled. Exiting.")
            return 0

    # Create and show main window
    try:
        window = AnnotationWindow(config, db)
        window.show()

        print("\n" + "=" * 60)
        print("Wire Loop Annotation Tool Started")
        print("=" * 60)
        print(f"Annotator: {config.default_annotator}")
        print(f"Database: {config.database_path}")
        print(f"Data Directory: {config.data_dir}")
        print(f"Labels Directory: {config.labels_dir}")
        print("\nKeyboard Shortcuts:")
        print(f"  {config.get('shortcuts.open', 'Ctrl+O'):12} - Open folder")
        print(f"  {config.get('shortcuts.draw', 'W'):12} - Draw bounding box")
        print(f"  {config.get('shortcuts.save', 'S'):12} - Save annotations")
        print(f"  {config.get('shortcuts.delete', 'Delete'):12} - Delete selected bbox")
        print(f"  {config.get('shortcuts.next', 'D'):12} - Next image")
        print(f"  {config.get('shortcuts.prev', 'A'):12} - Previous image")
        print(f"  {config.get('shortcuts.cancel', 'Escape'):12} - Cancel drawing")
        print(f"  {config.get('shortcuts.quit', 'Ctrl+Q'):12} - Quit application")
        print("=" * 60 + "\n")

        # Run application event loop
        return app.exec()

    except Exception as e:
        print(f"Error starting application: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
