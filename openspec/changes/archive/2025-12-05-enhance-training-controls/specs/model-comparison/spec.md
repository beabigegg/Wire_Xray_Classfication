# model-comparison Specification

## Purpose
Enable side-by-side comparison of multiple trained models to identify the best performing model for deployment.

## ADDED Requirements

### Requirement: Model Selection for Comparison
The system SHALL allow users to select 2-4 models of the same type for comparison.

#### Scenario: Open model comparison dialog
- **WHEN** user opens annotation window
- **AND** clicks menu: "Training" → "Compare Models..."
- **THEN** system opens Model Comparison Dialog
- **AND** dialog shows list of all trained models grouped by type:
  - Detection Models (YOLO)
  - View Classifier Models
  - Defect Classifier Models
- **AND** each model shows:
  - Model name (auto-generated: model_type_YYYYMMDD_HHMMSS)
  - Training date
  - Best metric value (mAP for detection, balanced_accuracy for defect, accuracy for view)
  - Epoch count
  - Status badge (🟢 Active or ⚪ Inactive)

#### Scenario: Select models for comparison
- **WHEN** user views model list in comparison dialog
- **THEN** each model has a checkbox for selection
- **AND** checkboxes are disabled if model type doesn't match first selected model
- **AND** user can select minimum 2 models, maximum 4 models
- **AND** "Compare" button is enabled only when 2-4 models are selected
- **AND** status label shows "Select 2-4 models to compare (currently selected: N)"

#### Scenario: Handle insufficient models
- **WHEN** user opens model comparison dialog
- **AND** model type has < 2 trained models
- **THEN** system shows message:
  - "Not enough models to compare."
  - "Train at least 2 models of the same type to use comparison."
- **AND** "Compare" button is disabled

### Requirement: Metrics Comparison Table
The system SHALL display a comparison table showing metrics for all selected models.

#### Scenario: Compare detection models (YOLO)
- **WHEN** user selects 2-4 detection models and clicks "Compare"
- **THEN** system displays comparison table with columns:
  - Model Name
  - Training Date
  - Epochs
  - mAP@0.5 (primary metric)
  - mAP@0.5:0.95
  - Precision
  - Recall
  - Box Loss
  - Inference Time (ms/image)
- **AND** rows are sorted by mAP@0.5 descending (best first)
- **AND** best value in each column is highlighted in green bold
- **AND** worst value in each column is highlighted in red
- **AND** active model row has 🟢 badge

#### Scenario: Compare view classifier models
- **WHEN** user selects 2-4 view classifier models and clicks "Compare"
- **THEN** system displays comparison table with columns:
  - Model Name
  - Training Date
  - Epochs
  - Overall Accuracy (primary metric)
  - 正視 Precision
  - 正視 Recall
  - 正視 F1
  - 俯視 Precision
  - 俯視 Recall
  - 俯視 F1
  - Inference Time (ms/image)
- **AND** rows are sorted by overall accuracy descending (best first)
- **AND** best value in each column is highlighted in green bold
- **AND** worst value in each column is highlighted in red

#### Scenario: Compare defect classifier models
- **WHEN** user selects 2-4 defect classifier models and clicks "Compare"
- **THEN** system displays comparison table with columns:
  - Model Name
  - Training Date
  - Epochs
  - Balanced Accuracy (primary metric)
  - Overall Accuracy
  - PASS Recall (critical metric, highlighted)
  - PASS Precision
  - PASS F1
  - Avg Defect Recall (avg of 沖線, 晃動, 碰觸)
  - Avg Defect Precision
  - Inference Time (ms/image)
- **AND** rows are sorted by balanced accuracy descending
- **AND** PASS recall column has warning icon ⚠️ if value < 90%
- **AND** best balanced accuracy is highlighted in green bold
- **AND** worst balanced accuracy is highlighted in red

### Requirement: Delta Calculation and Visualization
The system SHALL compute and display differences between models.

#### Scenario: Show metric deltas
- **WHEN** comparison table is displayed
- **THEN** for each metric, system shows delta relative to best model:
  - Absolute delta: "+0.03" or "-0.05"
  - Relative delta: "(+3.2%)" or "(-5.1%)"
  - Up/down arrow: "↑" for improvement, "↓" for regression
- **AND** best model (baseline) shows "—" for deltas
- **AND** deltas are color-coded:
  - Green for improvements (within 2% of best)
  - Yellow for minor regressions (2-5% worse than best)
  - Red for major regressions (>5% worse than best)

#### Scenario: Example delta display
- **WHEN** comparing 3 defect models with balanced accuracies: 0.92, 0.88, 0.85
- **THEN** comparison table shows:
  - Model A: 0.92 (baseline, no delta shown)
  - Model B: 0.88 **-0.04** (↓-4.3%) in yellow
  - Model C: 0.85 **-0.07** (↓-7.6%) in red

### Requirement: Model Recommendation
The system SHALL provide automatic recommendation based on comprehensive analysis.

#### Scenario: Generate recommendation for detection models
- **WHEN** comparison table is displayed for detection models
- **THEN** system shows recommendation panel with reasoning:
  - "**Recommended Model**: model_detection_20251107_143022"
  - "**Reasoning**:"
  - "✓ Highest mAP@0.5 (0.87 vs 0.83 runner-up)"
  - "✓ Acceptable inference speed (42ms vs 38ms fastest)"
  - "✓ Good recall (0.85) - minimizes missed defects"
  - "⚠ Slightly higher box loss than Model B (consider if precision is critical)"
- **AND** user can click "Set as Active" button to activate recommended model

#### Scenario: Generate recommendation for defect models
- **WHEN** comparison table is displayed for defect models
- **THEN** system shows recommendation panel with reasoning:
  - "**Recommended Model**: model_defect_20251107_150030"
  - "**Reasoning**:"
  - "✓ Highest balanced accuracy (0.92 vs 0.88 runner-up)"
  - "✓ **PASS recall 95%** - excellent minority class detection"
  - "✓ Low false negative rate for PASS class (3 vs 8 in Model B)"
  - "✓ Acceptable inference speed (35ms)"
- **AND** PASS recall is emphasized as critical decision factor

#### Scenario: Handle trade-offs in recommendation
- **WHEN** no model is clearly superior across all metrics
- **THEN** system shows trade-off analysis:
  - "**Recommendation**: Model A (if prioritizing accuracy) OR Model C (if prioritizing speed)"
  - "**Trade-off Analysis**:"
  - "• Model A: Best accuracy (0.92) but slower (50ms)"
  - "• Model C: Acceptable accuracy (0.89) and fastest (30ms)"
  - "• Model B: Middle ground - not recommended"
- **AND** user can choose based on deployment requirements

### Requirement: Additional Comparison Views
The system SHALL provide supplementary visualizations for deeper analysis.

#### Scenario: View confusion matrices side-by-side
- **WHEN** user clicks "View Confusion Matrices" button
- **THEN** system opens new dialog showing confusion matrices in grid layout:
  - 2 models: 1x2 grid (side-by-side)
  - 3 models: 1x3 grid (horizontal row)
  - 4 models: 2x2 grid
- **AND** each confusion matrix shows:
  - Model name as title
  - Class labels on axes
  - Cell colors from white (low) to dark blue (high)
  - Cell annotations with counts and percentages
- **AND** matrices use same color scale for fair comparison

#### Scenario: View TensorBoard runs together
- **WHEN** user clicks "View TensorBoard" button
- **THEN** system launches TensorBoard with logdir covering all selected model runs
- **AND** TensorBoard opens in browser showing comparative graphs:
  - Loss curves overlaid for all models
  - Accuracy curves overlaid for all models
  - Learning rate schedules overlaid
- **AND** runs are color-coded and labeled in legend

### Requirement: Set Active Model
The system SHALL allow activating a model directly from comparison dialog.

#### Scenario: Activate selected model
- **WHEN** user clicks "Set as Active" button next to a model
- **THEN** system updates database: sets is_active=1 for selected model
- **AND** sets is_active=0 for all other models of same type
- **AND** updates comparison table to show 🟢 badge next to newly active model
- **AND** shows confirmation: "Model [name] is now active for [model_type]"
- **AND** annotation window's inference system automatically uses new active model

#### Scenario: Prevent accidental activation of inferior model
- **WHEN** user attempts to activate a model that is NOT the recommended model
- **AND** metric difference is > 5%
- **THEN** system shows confirmation dialog:
  - "The selected model has significantly lower performance than the recommended model."
  - "Selected: 0.88 balanced accuracy"
  - "Recommended: 0.92 balanced accuracy (↑+4.5%)"
  - "Are you sure you want to activate this model?"
  - [Yes, Activate] [Cancel]

### Requirement: ModelComparator Backend
The system SHALL provide a backend class for metrics comparison logic.

#### Scenario: Load model metadata for comparison
- **WHEN** ModelComparator.compare_models(model_ids) is called
- **THEN** system loads metadata from database:
  - model_versions table: model_path, model_type, created_at, is_active, hyperparameters
  - training_history table: final metrics (mAP, accuracy, loss, etc.)
- **AND** validates all models are of same type
- **AND** raises ValueError if models are mixed types

#### Scenario: Compute metric deltas
- **WHEN** ModelComparator._compute_deltas(metrics_list) is called
- **THEN** system identifies best value for each metric
- **AND** computes absolute delta: current_value - best_value
- **AND** computes relative delta: (delta / best_value) * 100
- **AND** returns structured delta data:
  ```python
  {
      'model_A': {'mAP': {'value': 0.87, 'delta_abs': 0, 'delta_rel': 0, 'is_best': True}},
      'model_B': {'mAP': {'value': 0.83, 'delta_abs': -0.04, 'delta_rel': -4.6, 'is_best': False}}
  }
  ```

#### Scenario: Rank models by primary metric
- **WHEN** ModelComparator._rank_models(metrics_list, model_type) is called
- **THEN** system selects primary metric based on model type:
  - Detection: mAP@0.5
  - View Classifier: overall_accuracy
  - Defect Classifier: balanced_accuracy
- **AND** sorts models by primary metric descending
- **AND** returns ranked list with rank numbers [1, 2, 3, 4]

#### Scenario: Generate recommendation with reasoning
- **WHEN** ModelComparator._generate_recommendation(ranked_models, model_type) is called
- **THEN** system analyzes top 2 models
- **AND** IF top model has > 5% lead in primary metric, recommends top model
- **AND** IF top 2 models are within 5%, considers secondary factors:
  - Detection: precision vs recall trade-off
  - View: per-class F1 balance
  - Defect: PASS recall (critical priority)
- **AND** returns recommendation object:
  ```python
  {
      'recommended_model_id': 123,
      'reasoning': [
          "✓ Highest balanced accuracy (0.92 vs 0.88 runner-up)",
          "✓ PASS recall 95% - excellent minority class detection",
          "⚠ Slightly slower inference (35ms vs 30ms fastest)"
      ]
  }
  ```

## Success Criteria

1. **Usability**: Users can compare 2-4 models in < 30 seconds
2. **Clarity**: Deltas are clearly visible and color-coded
3. **Accuracy**: Recommendation matches expert manual analysis in 90%+ cases
4. **Performance**: Comparison table loads in < 2 seconds for 4 models
5. **Integration**: Setting active model immediately affects annotation window inference
6. **Visualization**: Confusion matrices are readable and use consistent color scales
7. **Decision Support**: Recommendation provides clear, actionable reasoning
