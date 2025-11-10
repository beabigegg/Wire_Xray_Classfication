"""
CLI tool for inference on Wire Loop X-ray images.

Usage:
    # Single image inference
    python infer.py --image path/to/image.png

    # Batch inference
    python infer.py --batch path/to/images/ --output results.csv

    # With custom configuration
    python infer.py --batch path/to/images/ --config inference_config.yaml --output results.json
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

from src.inference import ModelLoader, InferencePipeline, BatchProcessor, ResultExporter


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Setup logging configuration."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def print_single_result(result: dict):
    """Print single image inference result to console."""
    print("\n" + "=" * 60)
    print(f"Image: {result['image_name']}")
    print("=" * 60)

    if not result['success']:
        print(f"❌ Inference failed: {result['error']}")
        return

    if not result['primary_result'] or not result['primary_result']['bbox']:
        print("⚠️  No detection found")
        return

    primary = result['primary_result']
    bbox = primary['bbox']

    print(f"✅ Detection found")
    print(f"\nBounding Box: ({bbox[0]}, {bbox[1]}) -> ({bbox[2]}, {bbox[3]})")
    print(f"  Confidence: {primary['bbox_confidence']:.4f}")

    print(f"\nView: {primary['view']}")
    print(f"  Confidence: {primary['view_confidence']:.4f}")

    print(f"\nDefect: {primary['defect']}")
    print(f"  Confidence: {primary['defect_confidence']:.4f}")

    print(f"\nProcessing time: {result['processing_time']:.3f} seconds")

    if len(result['detections']) > 1:
        print(f"\nNote: {len(result['detections'])} detections found (showing primary)")


def print_batch_summary(results: list, statistics: dict):
    """Print batch processing summary."""
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    print(f"Total images: {statistics['total_images']}")
    print(f"✅ Successful: {statistics['successful']}")
    print(f"❌ Failed: {statistics['failed']}")
    print(f"🎯 Detected: {statistics['detected']}")
    print(f"⚠️  No detection: {statistics['no_detection']}")

    print(f"\n⏱️  Total time: {statistics['total_processing_time']:.2f}s")
    print(f"⏱️  Average time: {statistics['average_processing_time']:.3f}s per image")

    if statistics['view_distribution']:
        print("\n📊 View Distribution:")
        for view, count in statistics['view_distribution'].items():
            print(f"  {view}: {count}")

    if statistics['defect_distribution']:
        print("\n📊 Defect Distribution:")
        for defect, count in statistics['defect_distribution'].items():
            print(f"  {defect}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Inference tool for Wire Loop X-ray classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python infer.py --image data/sample.png

  # Batch inference with CSV output
  python infer.py --batch data/images/ --output results.csv

  # Batch inference with JSON output
  python infer.py --batch data/images/ --output results.json

  # Using custom configuration
  python infer.py --batch data/images/ --config inference_config.yaml --output results.csv

  # Verbose output
  python infer.py --image data/sample.png --verbose

  # Quiet mode (errors only)
  python infer.py --batch data/images/ --output results.csv --quiet
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to single image file')
    input_group.add_argument('--batch', type=str, help='Path to directory containing images')

    # Output options
    parser.add_argument('--output', type=str, help='Output file path (CSV or JSON)')
    parser.add_argument('--summary', type=str, help='Summary output file path (TXT)')

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='inference_config.yaml',
        help='Path to configuration file (default: inference_config.yaml)'
    )

    # Model paths (override config)
    parser.add_argument('--yolo-model', type=str, help='Path to YOLO model')
    parser.add_argument('--view-model', type=str, help='Path to View classifier model')
    parser.add_argument('--defect-model', type=str, help='Path to Defect classifier model')

    # Inference parameters
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--confidence', type=float, help='YOLO confidence threshold')

    # Output options
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode (errors only)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger(__name__)

    # Load configuration
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.info("No configuration file found, using defaults")

    def find_latest_model(model_dir: str, extension: str = '.pt') -> str:
        """Find the latest model file in a directory."""
        from pathlib import Path
        model_path = Path(model_dir)
        if not model_path.exists():
            return f"{model_dir}/best{extension}"
        model_files = list(model_path.glob(f'*{extension}'))
        if not model_files:
            return f"{model_dir}/best{extension}"
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(model_files[0])

    # Override config with command-line arguments
    model_paths = config.get('models', {})
    yolo_path = args.yolo_model or model_paths.get('detection') or find_latest_model('models/detection')
    view_path = args.view_model or model_paths.get('view_classifier') or find_latest_model('models/view')
    defect_path = args.defect_model or model_paths.get('defect_classifier') or find_latest_model('models/defect')

    inference_config = config.get('inference', {})
    device = args.device or inference_config.get('device', 'auto')
    confidence = args.confidence or inference_config.get('confidence_threshold', 0.5)

    # Initialize models
    try:
        logger.info("Initializing models...")
        model_loader = ModelLoader(
            yolo_path=yolo_path,
            view_classifier_path=view_path,
            defect_classifier_path=defect_path,
            device=device
        )

        pipeline = InferencePipeline(
            model_loader=model_loader,
            confidence_threshold=confidence
        )

        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        sys.exit(1)

    # Single image mode
    if args.image:
        try:
            result = pipeline.infer_single_image(args.image)

            # Print result
            if not args.quiet:
                print_single_result(result)

            # Save result if output specified
            if args.output:
                output_path = Path(args.output)
                if output_path.suffix.lower() == '.json':
                    ResultExporter.export_json([result], args.output)
                else:
                    ResultExporter.export_csv([result], args.output)
                print(f"\n💾 Result saved to: {args.output}")

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            sys.exit(1)

    # Batch mode
    elif args.batch:
        try:
            batch_processor = BatchProcessor(pipeline)

            # Process directory
            results = batch_processor.process_directory(
                args.batch,
                show_progress=not args.quiet
            )

            # Calculate statistics
            statistics = batch_processor.get_statistics(results)

            # Print summary
            if not args.quiet:
                print_batch_summary(results, statistics)

            # Save results if output specified
            if args.output:
                output_path = Path(args.output)
                if output_path.suffix.lower() == '.json':
                    ResultExporter.export_json(results, args.output)
                else:
                    ResultExporter.export_csv(results, args.output)
                print(f"\n💾 Results saved to: {args.output}")

            # Save summary if specified
            if args.summary:
                ResultExporter.export_summary(results, statistics, args.summary)
                print(f"💾 Summary saved to: {args.summary}")

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
