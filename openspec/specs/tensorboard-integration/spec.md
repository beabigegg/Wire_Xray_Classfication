# tensorboard-integration Specification

## Purpose
TBD - created by archiving change enhance-training-controls. Update Purpose after archive.
## Requirements
### Requirement: Smart Logging Configuration
The system SHALL intelligently adjust logging frequency based on training characteristics.

#### Scenario: Adjust logging for short training runs
- **WHEN** training runs for < 20 epochs
- **THEN** system logs images every 5 epochs (instead of 10)
- **AND** logs confusion matrix every 2 epochs (instead of 5)
- **AND** ensures at least 3 image logging events during training

#### Scenario: Adjust logging for long training runs
- **WHEN** training runs for > 100 epochs
- **THEN** system logs images every 20 epochs (instead of 10)
- **AND** logs confusion matrix every 10 epochs (instead of 5)
- **AND** reduces overhead for long training sessions

#### Scenario: Async figure logging (optional performance optimization)
- **WHEN** trainer logs matplotlib figures (confusion matrix, prediction grid)
- **THEN** system MAY use background thread for figure creation
- **AND** training loop continues without blocking
- **AND** figure is logged asynchronously when ready
- **AND** memory is freed after logging completes

### Requirement: PASS Class Special Monitoring
For Defect Classifier, the system SHALL provide dedicated monitoring for the PASS class (minority class).

#### Scenario: PASS class metrics dashboard
- **WHEN** Defect Classifier training is running
- **THEN** TensorBoard shows dedicated PASS class metrics:
  - PASS/recall (most critical - should be ≥ 95%)
  - PASS/precision (should be ≥ 90%)
  - PASS/f1_score (harmonic mean)
  - PASS/false_negatives (count of missed PASS samples)
  - PASS/false_positives (count of wrongly flagged defects)
- **AND** training dialog shows PASS recall in status bar
- **AND** IF PASS recall drops below 90%, system shows warning

#### Scenario: PASS class sample visualization
- **WHEN** system logs PASS class samples every 10 epochs
- **THEN** creates two separate grids:
  - Correct PASS predictions (8 samples)
  - Incorrect PASS predictions (8 samples with predicted defect labels)
- **AND** highlights false negatives (PASS predicted as defect) in red border
- **AND** shows confidence scores for each prediction

