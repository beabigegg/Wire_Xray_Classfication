# inference-pipeline Specification Delta

## ADDED Requirements

### Requirement: View Classifier Label Mapping Consistency
The inference pipeline SHALL use view class labels that exactly match the training order.

#### Scenario: Consistent label mapping between training and inference
- **WHEN** View classifier model is loaded for inference
- **THEN** the system uses class order `["TOP", "SIDE"]` (TOP=0, SIDE=1)
- **AND** this matches the training order defined in `view_classifier.py`
- **AND** inference results correctly map class indices to class names

#### Scenario: Correct view classification output
- **WHEN** inference is performed on an image with TOP view
- **THEN** the system returns view_label="TOP"
- **WHEN** inference is performed on an image with SIDE view
- **THEN** the system returns view_label="SIDE"
