# Inference Pipeline Specification

**Capability:** inference-pipeline
**Status:** ADDED
**Change:** add-inference-system

---

## Overview

The inference pipeline provides three-stage inference capabilities for Wire Loop X-ray images using trained models (YOLO detection → View classification → Defect classification). It supports single image and batch processing with result export functionality.

---

## ADDED Requirements

### Requirement: Model Loading and Management

The system SHALL load and manage three trained models with proper device placement.

#### Scenario: Load YOLO detection model
- **WHEN** system initializes with path to YOLO `.pt` file
- **THEN** YOLO model is loaded successfully
- **AND** model is placed on available GPU device (if available)
- **AND** model is set to evaluation mode

#### Scenario: Load View classifier model
- **WHEN** system initializes with path to View classifier `.pth` file
- **THEN** ResNet18 model is loaded with correct architecture
- **AND** model weights are loaded from checkpoint
- **AND** model is placed on same device as YOLO model
- **AND** model is set to evaluation mode

#### Scenario: Load Defect classifier model
- **WHEN** system initializes with path to Defect classifier `.pth` file
- **THEN** EfficientNet-B0 model is loaded with correct architecture
- **AND** model weights are loaded from checkpoint
- **AND** model is placed on same device as other models
- **AND** model is set to evaluation mode

#### Scenario: Handle missing model files
- **WHEN** any model file path does not exist
- **THEN** clear error message is raised indicating which model file is missing
- **AND** error message includes the expected file path
- **AND** system does not proceed with inference

#### Scenario: Automatic device selection
- **WHEN** system initializes without explicit device specification
- **THEN** CUDA device is selected if GPU is available
- **AND** all models are placed on the same device
- **AND** device information is logged
- **AND** CPU fallback is used if no GPU available

---

### Requirement: Single Image Inference Pipeline

The system SHALL perform three-stage inference on a single 1004x1004 X-ray image.

#### Scenario: Complete three-stage inference
- **WHEN** valid 1004x1004 image is provided
- **THEN** Stage 1 (YOLO detection) produces bounding box coordinates
- **AND** Stage 2 (View classification) produces TOP or SIDE label with confidence
- **AND** Stage 3 (Defect classification) produces one of [PASS, 沖線, 晃動, 碰觸] with confidence
- **AND** total inference time is < 2 seconds on CPU or < 500ms on GPU
- **AND** all results are returned in structured format

#### Scenario: Handle no detection
- **WHEN** YOLO detection finds no objects (confidence below threshold)
- **THEN** result indicates "no_detection"
- **AND** view and defect classification are skipped
- **AND** result includes empty bounding box
- **AND** result includes null values for view and defect

#### Scenario: Handle multiple detections
- **WHEN** YOLO detection finds multiple bounding boxes
- **THEN** highest confidence detection is selected
- **AND** view and defect classification run on that detection
- **AND** result includes all detected bounding boxes with confidence scores
- **AND** result indicates which detection was used for classification

#### Scenario: Handle invalid image size
- **WHEN** image dimensions are not 1004x1004
- **THEN** error message indicates expected dimensions
- **AND** inference is not performed
- **AND** error includes actual image dimensions

#### Scenario: Handle corrupted image
- **WHEN** image file cannot be decoded
- **THEN** clear error message is raised
- **AND** error includes image file path
- **AND** inference is not performed

---

### Requirement: Image Preprocessing

The system SHALL preprocess images correctly for each model stage.

#### Scenario: Preprocess for YOLO detection
- **WHEN** image is loaded for YOLO inference
- **THEN** image is converted to RGB if needed
- **AND** image is resized to YOLO input size (640 or 1004)
- **AND** pixel values are normalized to [0, 1]
- **AND** image is converted to tensor format
- **AND** batch dimension is added

#### Scenario: Preprocess for View classifier
- **WHEN** detected region is cropped for view classification
- **THEN** image is resized to 224x224
- **AND** ImageNet normalization is applied (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **AND** image is converted to tensor format
- **AND** batch dimension is added

#### Scenario: Preprocess for Defect classifier
- **WHEN** detected region is cropped for defect classification
- **THEN** image is resized to 224x224
- **AND** ImageNet normalization is applied
- **AND** image is converted to tensor format
- **AND** batch dimension is added

#### Scenario: Handle grayscale images
- **WHEN** input image is grayscale (single channel)
- **THEN** image is converted to 3-channel RGB by replicating the single channel
- **AND** preprocessing continues normally

---

### Requirement: Batch Processing

The system SHALL efficiently process multiple images with progress tracking.

#### Scenario: Process batch of images
- **WHEN** directory path with multiple images is provided
- **THEN** all valid image files (PNG, JPG, BMP) are discovered
- **AND** images are processed sequentially or in small batches
- **AND** progress bar shows current progress and ETA
- **AND** results are collected for all images

#### Scenario: Memory-efficient batch processing
- **WHEN** processing large number of images (100+)
- **THEN** images are processed in chunks to avoid OOM
- **AND** chunk size is configurable
- **AND** memory usage stays reasonable (< 4GB on GPU)
- **AND** all images are processed successfully

#### Scenario: Handle individual image failures in batch
- **WHEN** some images in batch fail to process
- **THEN** processing continues for remaining images
- **AND** failed images are logged with error details
- **AND** results include error status for failed images
- **AND** batch processing completes for all valid images

#### Scenario: Progress tracking
- **WHEN** batch processing is running
- **THEN** progress bar shows percentage complete
- **AND** progress bar shows number of images processed
- **AND** progress bar shows estimated time remaining
- **AND** progress updates in real-time

---

### Requirement: Result Structure and Export

The system SHALL export structured results in multiple formats.

#### Scenario: CSV export format
- **WHEN** results are exported to CSV
- **THEN** CSV includes columns: image_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2, view, view_confidence, defect, defect_confidence
- **AND** bounding box coordinates are in pixel units
- **AND** confidence scores are formatted to 4 decimal places
- **AND** CSV includes header row
- **AND** file is saved with UTF-8 encoding

#### Scenario: JSON export format
- **WHEN** results are exported to JSON
- **THEN** JSON structure includes array of image results
- **AND** each result includes: image_name, bbox (dict with x1,y1,x2,y2), view (dict with label and confidence), defect (dict with label and confidence), processing_time
- **AND** JSON is formatted with indentation for readability
- **AND** file is saved with UTF-8 encoding

#### Scenario: Handle no detection in export
- **WHEN** image has no detection
- **THEN** CSV row includes empty strings for bbox coordinates
- **AND** CSV row includes "no_detection" for view and defect
- **AND** confidence scores are 0.0000
- **AND** JSON result includes null values for bbox, view, and defect

#### Scenario: Export with multiple detections
- **WHEN** image has multiple detections
- **THEN** CSV includes one row per detection
- **AND** image_name is repeated for each detection
- **AND** JSON includes array of detections per image
- **AND** primary detection is marked in JSON

---

### Requirement: CLI Interface

The system SHALL provide command-line interface for inference operations.

#### Scenario: Single image inference mode
- **WHEN** CLI is invoked with `--image <path>` argument
- **THEN** single image is processed
- **AND** results are printed to console in formatted text
- **AND** results optionally saved to file if `--output` specified
- **AND** exit code is 0 on success, non-zero on error

#### Scenario: Batch inference mode
- **WHEN** CLI is invoked with `--batch <directory>` argument
- **THEN** all images in directory are processed
- **AND** progress bar is displayed
- **AND** results are saved to output file (CSV or JSON based on extension)
- **AND** summary statistics are printed (total processed, failures, average time)

#### Scenario: Configuration file support
- **WHEN** CLI is invoked with `--config <yaml_path>` argument
- **THEN** configuration is loaded from YAML file
- **AND** config includes model paths, device, confidence thresholds, batch size
- **AND** CLI arguments override config file values
- **AND** default config path is `inference_config.yaml` if exists

#### Scenario: Verbose output mode
- **WHEN** CLI is invoked with `--verbose` flag
- **THEN** detailed logging is enabled
- **AND** model loading steps are logged
- **AND** per-image inference times are logged
- **AND** device information is logged
- **AND** configuration values are logged

#### Scenario: Quiet output mode
- **WHEN** CLI is invoked with `--quiet` flag
- **THEN** only essential output is displayed
- **AND** progress bar is hidden
- **AND** only errors and final summary are shown

#### Scenario: Help documentation
- **WHEN** CLI is invoked with `--help` flag
- **THEN** comprehensive help text is displayed
- **AND** all arguments are documented with descriptions
- **AND** usage examples are included
- **AND** output format examples are shown

---

### Requirement: Configuration Management

The system SHALL support flexible configuration for inference parameters.

#### Scenario: Load configuration from YAML
- **WHEN** configuration file path is provided
- **THEN** YAML file is parsed successfully
- **AND** model paths are loaded from config
- **AND** confidence thresholds are loaded from config
- **AND** batch size is loaded from config
- **AND** device preference is loaded from config

#### Scenario: Default configuration values
- **WHEN** no configuration file is provided
- **THEN** default model paths are used (models/detection/best.pt, models/view/best.pth, models/defect/best.pth)
- **AND** default confidence threshold is 0.5 for YOLO
- **AND** default batch size is 16
- **AND** device is auto-detected (GPU if available)

#### Scenario: Configuration validation
- **WHEN** configuration is loaded
- **THEN** model paths are validated to exist
- **AND** numeric parameters are validated to be in valid range
- **AND** device value is validated to be "cuda", "cpu", or "auto"
- **AND** validation errors are reported with specific issues

#### Scenario: Template configuration generation
- **WHEN** user requests configuration template
- **THEN** template YAML file is generated with all parameters
- **AND** each parameter includes comments explaining its purpose
- **AND** default values are provided as examples

---

### Requirement: Error Handling and Logging

The system SHALL handle errors gracefully and provide informative logging.

#### Scenario: Model loading failure
- **WHEN** model file is corrupted or incompatible
- **THEN** specific error is raised identifying the model
- **AND** error message suggests checking model file integrity
- **AND** error includes full file path
- **AND** system exits cleanly without crash

#### Scenario: CUDA out of memory
- **WHEN** GPU memory is exhausted during batch processing
- **THEN** error is caught and logged
- **AND** suggestion to reduce batch size is provided
- **AND** suggestion to use CPU is provided
- **AND** system attempts to recover or exits cleanly

#### Scenario: Invalid image format
- **WHEN** image file has unsupported format
- **THEN** warning is logged with image filename
- **AND** image is skipped in batch processing
- **AND** processing continues for other images
- **AND** error is included in final results

#### Scenario: Logging levels
- **WHEN** system is running
- **THEN** INFO level logs key operations (model loading, batch start/end)
- **AND** DEBUG level logs detailed steps (preprocessing, inference times)
- **AND** WARNING level logs recoverable issues (skipped images)
- **AND** ERROR level logs critical failures
- **AND** log level is configurable via CLI or config file

---

### Requirement: Performance Requirements

The system SHALL meet specified performance criteria.

#### Scenario: Single image inference speed on GPU
- **WHEN** inference runs on CUDA-enabled GPU
- **THEN** total inference time (all 3 stages) is < 500ms per image
- **AND** YOLO detection takes < 100ms
- **AND** View classification takes < 200ms
- **AND** Defect classification takes < 200ms

#### Scenario: Single image inference speed on CPU
- **WHEN** inference runs on CPU
- **THEN** total inference time is < 2 seconds per image
- **AND** inference completes without timeout

#### Scenario: Batch processing throughput
- **WHEN** processing batch of 100 images on GPU
- **THEN** average time per image is < 500ms
- **AND** total processing time is < 2 minutes
- **AND** memory usage is stable (no memory leaks)

#### Scenario: Model loading time
- **WHEN** models are loaded at initialization
- **THEN** all three models load within 10 seconds on GPU
- **AND** all three models load within 30 seconds on CPU
- **AND** loading time is logged

---

### Requirement: Testing and Validation

The system SHALL include comprehensive tests with high coverage.

#### Scenario: Unit test coverage
- **WHEN** pytest is run with coverage report
- **THEN** overall coverage is > 80%
- **AND** model_loader.py has > 80% coverage
- **AND** inference_pipeline.py has > 80% coverage
- **AND** batch_processor.py has > 80% coverage
- **AND** result_exporter.py has > 80% coverage

#### Scenario: Integration test for full pipeline
- **WHEN** integration test runs with sample image
- **THEN** all three stages complete successfully
- **AND** result structure is validated
- **AND** confidence scores are within [0, 1] range
- **AND** bounding box coordinates are valid

#### Scenario: Test with various image formats
- **WHEN** tests run with PNG, JPEG, BMP images
- **THEN** all formats are processed successfully
- **AND** results are consistent across formats
- **AND** no format-specific errors occur

#### Scenario: Test error handling
- **WHEN** tests provide invalid inputs
- **THEN** appropriate exceptions are raised
- **AND** error messages are informative
- **AND** system does not crash
- **AND** resources are cleaned up properly

---

## Dependencies

### Internal Dependencies
- `src/training/model_manager.py` - For model architecture definitions
- `src/core/image_utils.py` - For image loading and preprocessing utilities

### External Dependencies
- PyTorch >= 2.0.1
- TorchVision >= 0.15.2
- Ultralytics (YOLOv8)
- Pillow
- PyYAML
- tqdm
- pytest (for testing)
- pytest-cov (for coverage)

---

## Configuration Schema

```yaml
# inference_config.yaml
models:
  detection: "models/detection/best.pt"
  view_classifier: "models/view/best.pth"
  defect_classifier: "models/defect/best.pth"

inference:
  device: "auto"  # "auto", "cuda", "cpu"
  confidence_threshold: 0.5  # YOLO detection threshold
  batch_size: 16  # For batch processing
  image_size: 1004  # Expected input size

output:
  format: "csv"  # "csv" or "json"
  output_dir: "results"
  save_annotated_images: false

logging:
  level: "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
  file: "inference.log"
```

---

## Success Criteria Mapping

This spec directly supports all success criteria defined in proposal.md:

1. ✅ **Load three trained models** - REQ-INF-001
2. ✅ **Single image three-stage inference** - REQ-INF-002, REQ-INF-003
3. ✅ **Batch processing 100+ images** - REQ-INF-004
4. ✅ **Performance targets** - REQ-INF-009
5. ✅ **CSV export with required fields** - REQ-INF-005
6. ✅ **Unit test coverage > 80%** - REQ-INF-010

---

## Notes

- All image coordinates use pixel units with origin at top-left (0,0)
- Confidence scores are probabilities in range [0, 1]
- YOLO detection uses standard COCO format for bounding boxes (x1, y1, x2, y2)
- View classifier outputs: ["TOP", "SIDE"]
- Defect classifier outputs: ["PASS", "沖線", "晃動", "碰觸"]
