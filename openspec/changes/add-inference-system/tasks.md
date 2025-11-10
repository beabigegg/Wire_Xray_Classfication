# Tasks for add-inference-system

## Phase 1: Core Inference Infrastructure (1-2 days)

### Model Loading
- [x] Create `src/inference/__init__.py`
- [x] Create `src/inference/model_loader.py` with `ModelLoader` class
- [x] Implement YOLO model loading from `.pt` file
- [x] Implement View Classifier loading from `.pth` file
- [x] Implement Defect Classifier loading from `.pth` file
- [x] Add model path validation and error handling
- [x] Add device detection (GPU/CPU) and model device placement
- [x] Write unit tests for model loading

### Image Preprocessing
- [x] Create `src/inference/preprocessor.py`
- [x] Implement image loading and validation (1004x1004)
- [x] Implement preprocessing for YOLO (normalization)
- [x] Implement preprocessing for classifiers (transforms)
- [x] Handle RGB to grayscale conversion
- [x] Write unit tests for preprocessing

### Three-Stage Inference Pipeline
- [x] Create `src/inference/inference_pipeline.py` with `InferencePipeline` class
- [x] Implement Stage 1: YOLO detection (bounding box)
- [x] Implement Stage 2: View classification (TOP/SIDE)
- [x] Implement Stage 3: Defect classification (PASS/沖線/晃動/碰觸)
- [x] Handle cases with no detection
- [x] Handle multiple detections per image
- [x] Return structured results with confidence scores
- [x] Write integration tests for full pipeline

## Phase 2: Batch Processing & CLI (0.5-1 day)

### Batch Processor
- [x] Create `src/inference/batch_processor.py` with `BatchProcessor` class
- [x] Implement directory scanning for images
- [x] Implement chunked batch processing to avoid OOM
- [x] Add progress bar (using tqdm)
- [x] Add error handling for individual images
- [x] Collect and aggregate results
- [x] Write unit tests for batch processing logic

### CLI Interface
- [x] Create `infer.py` CLI script
- [x] Add argument parser (--image, --batch, --output, --config)
- [x] Implement single image inference mode
- [x] Implement batch inference mode
- [x] Add verbose/quiet output options
- [x] Add help documentation
- [x] Test CLI with sample images

## Phase 3: Result Export & Utilities (0.5-1 day)

### Result Export
- [x] Create `src/inference/result_exporter.py` with `ResultExporter` class
- [x] Implement CSV export (image_name, bbox, view, defect, confidence)
- [x] Implement JSON export (structured format)
- [x] Add optional image annotation export (draw boxes on images)
- [x] Write unit tests for exporters

### Configuration Management
- [x] Create `inference_config.yaml` template
- [x] Add model paths configuration
- [x] Add confidence threshold configuration
- [x] Add batch size configuration
- [x] Add device preference configuration
- [x] Implement config loading in inference modules

### Documentation
- [x] Write `docs/INFERENCE_GUIDE.md` - user guide
- [x] Add docstrings to all inference modules
- [x] Create usage examples in README
- [x] Document CLI options
- [x] Add troubleshooting section

## Phase 4: Testing & Validation (0.5 day)

### Integration Testing
- [x] Create `tests/inference/` directory
- [x] Test end-to-end single image inference
- [x] Test end-to-end batch inference
- [x] Test with various image formats (PNG, JPEG, BMP)
- [x] Test error handling (missing models, invalid images)
- [x] Test CSV and JSON export outputs
- [x] Verify confidence scores are reasonable

### Performance Testing
- [x] Measure inference time on CPU
- [x] Measure inference time on GPU (if available)
- [x] Test batch processing with 100+ images
- [x] Verify memory usage stays reasonable
- [x] Profile slow operations if needed

### Validation
- [x] Verify output format matches specification
- [x] Check coordinate accuracy (bbox coordinates)
- [x] Validate class label outputs
- [x] Test with trained models from training pipeline
- [x] Ensure backward compatibility with existing code

## Phase 5: Polish & Deployment Prep (0.5 day)

### Code Quality
- [x] Run pytest with coverage report (target > 80%)
- [x] Fix any failing tests
- [x] Run black code formatter
- [x] Run flake8 linter
- [x] Address any linting issues
- [x] Add type hints where missing

### User Experience
- [x] Add clear error messages for common failures
- [x] Improve progress indicators
- [x] Add ETA estimation for batch processing
- [x] Create example inference script
- [x] Add sample output files

### Final Validation
- [x] Test complete workflow: train → infer → export
- [x] Verify all success criteria met
- [x] Update README with inference instructions
- [x] Create demo video/screenshots (optional)
- [x] Prepare for archiving change
