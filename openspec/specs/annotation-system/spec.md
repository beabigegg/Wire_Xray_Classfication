# annotation-system Specification

## Purpose
TBD - created by archiving change add-core-annotation-system. Update Purpose after archive.
## Requirements
### Requirement: X-ray Image Loading
The system SHALL load X-ray images in PNG, JPEG, and BMP formats with automatic format detection and validation.

#### Scenario: Load valid X-ray image
- **WHEN** a valid image file path is provided (PNG/JPEG/BMP format, 1004x1004 pixels)
- **THEN** the system loads the image successfully
- **AND** converts 24-bit RGB to 8-bit grayscale automatically
- **AND** returns a numpy array with shape (1004, 1004)

#### Scenario: Reject invalid image size
- **WHEN** an image with dimensions other than 1004x1004 is provided
- **THEN** the system raises a ValidationError
- **AND** displays an error message indicating the expected size

#### Scenario: Handle corrupted image file
- **WHEN** a corrupted or unreadable image file is provided
- **THEN** the system raises an ImageLoadError
- **AND** logs the error details for debugging

#### Scenario: Handle missing file
- **WHEN** a non-existent file path is provided
- **THEN** the system raises a FileNotFoundError
- **AND** provides a clear error message with the attempted path

### Requirement: Annotation Data Storage
The system SHALL store all annotation data in a SQLite database with referential integrity and support for CRUD operations.

#### Scenario: Save new annotation
- **WHEN** a user creates a new bounding box annotation with view type and defect type
- **THEN** the system saves the annotation to the database
- **AND** assigns a unique annotation ID
- **AND** records the annotator username and timestamp
- **AND** maintains the foreign key relationship to the image

#### Scenario: Retrieve annotations for image
- **WHEN** an image is loaded in the GUI
- **THEN** the system retrieves all annotations associated with that image
- **AND** returns annotations with bounding box coordinates, view type, and defect type
- **AND** orders annotations by creation time

#### Scenario: Update existing annotation
- **WHEN** a user modifies an existing annotation (bbox position or labels)
- **THEN** the system updates the annotation record in the database
- **AND** updates the `updated_at` timestamp
- **AND** preserves the original creation metadata

#### Scenario: Delete annotation
- **WHEN** a user deletes an annotation
- **THEN** the system removes the annotation record from the database
- **AND** CASCADE deletes if the parent image is deleted
- **AND** updates the annotation count

#### Scenario: Query annotations by criteria
- **WHEN** filtering annotations by view type or defect type
- **THEN** the system returns only matching annotations
- **AND** supports multiple filter criteria combined with AND logic
- **AND** returns results ordered by timestamp

### Requirement: YOLO Format Support
The system SHALL read and write YOLO format annotation files with correct coordinate normalization.

#### Scenario: Read YOLO annotation file
- **WHEN** a YOLO format .txt file exists for an image
- **THEN** the system parses the file successfully
- **AND** converts normalized coordinates (0-1) to pixel coordinates
- **AND** handles multiple annotations (one per line)
- **AND** validates coordinate ranges

#### Scenario: Write YOLO annotation file
- **WHEN** saving annotations for an image
- **THEN** the system writes a .txt file in YOLO format
- **AND** converts pixel coordinates to normalized coordinates (0-1)
- **AND** uses the format: `<class_id> <x_center> <y_center> <width> <height>`
- **AND** creates one line per annotation

#### Scenario: Handle multi-object annotations
- **WHEN** an image has multiple wire loop annotations
- **THEN** the system writes one line per object in the YOLO file
- **AND** each line contains the correct class ID and coordinates
- **AND** preserves annotation order

#### Scenario: Coordinate normalization
- **WHEN** converting between pixel and normalized coordinates
- **THEN** the system uses the formula: `norm = pixel / image_dimension`
- **AND** ensures all normalized coordinates are in range [0, 1]
- **AND** handles center-based bounding box format correctly

### Requirement: Bounding Box Annotation GUI
The system SHALL provide a graphical interface for drawing and editing bounding box annotations with mouse and keyboard controls.

#### Scenario: Draw new bounding box
- **WHEN** a user presses 'W' and drags the mouse on the canvas
- **THEN** a bounding box is drawn from the start point to the current mouse position
- **AND** the box is displayed with a colored rectangle outline
- **AND** the box coordinates are captured in pixel space

#### Scenario: Edit existing bounding box
- **WHEN** a user clicks on an existing bounding box
- **THEN** the box is selected and highlighted
- **AND** the user can drag corner handles to resize the box
- **AND** the user can drag the box center to move it
- **AND** coordinate updates are reflected in real-time

#### Scenario: Delete bounding box
- **WHEN** a user selects a bounding box and presses Delete key
- **THEN** the bounding box is removed from the canvas
- **AND** the corresponding annotation is deleted from the database
- **AND** the YOLO file is updated

#### Scenario: Display annotation labels
- **WHEN** a bounding box is displayed on the canvas
- **THEN** a label is shown near the box with view type and defect type
- **AND** the label uses a readable font and contrasting color
- **AND** the label follows the box when moved

#### Scenario: Zoom and pan
- **WHEN** a user scrolls the mouse wheel
- **THEN** the canvas zooms in or out centered on the mouse position
- **AND** bounding boxes scale proportionally
- **AND** coordinate calculations remain accurate after zoom

#### Scenario: Navigate images with keyboard
- **WHEN** a user presses 'D' key
- **THEN** the next image in the list is loaded
- **AND** annotations for the current image are saved automatically
- **AND** annotations for the next image are loaded and displayed
- **AND** the image list selection is updated

### Requirement: View and Defect Type Selection
The system SHALL provide UI controls for selecting view type (TOP/SIDE) and defect type (PASS/沖線/晃動/碰觸) for each annotation.

#### Scenario: Select view type
- **WHEN** a user selects a bounding box
- **THEN** the properties panel displays radio buttons for TOP and SIDE
- **AND** the current view type is pre-selected
- **AND** changing the selection updates the annotation immediately

#### Scenario: Select defect type
- **WHEN** a user selects a bounding box
- **THEN** the properties panel displays radio buttons for the four defect types
- **AND** the current defect type is pre-selected
- **AND** changing the selection updates the annotation immediately

#### Scenario: Apply labels to new annotation
- **WHEN** a user finishes drawing a new bounding box
- **THEN** the system prompts for view type and defect type selection
- **AND** default values (TOP, PASS) are pre-selected
- **AND** the user can change selections before confirming

#### Scenario: Validate label selection
- **WHEN** attempting to save an annotation
- **THEN** the system verifies that both view type and defect type are selected
- **AND** prevents saving if either label is missing
- **AND** displays a validation error message

### Requirement: Image List Management
The system SHALL display a list of all images in the workspace with annotation status indicators.

#### Scenario: Load image folder
- **WHEN** a user opens a folder containing X-ray images
- **THEN** the system scans for PNG/JPEG/BMP files
- **AND** displays all found images in the image list
- **AND** shows annotation count for each image
- **AND** marks annotated images with a checkmark

#### Scenario: Display annotation progress
- **WHEN** the image list is displayed
- **THEN** each image shows its annotation count (e.g., "3 annotations")
- **AND** the total progress is shown at the bottom (e.g., "25 / 100 annotated")
- **AND** unannotated images are visually distinct

#### Scenario: Filter images
- **WHEN** a user applies a filter (e.g., "Show only unannotated")
- **THEN** the image list displays only matching images
- **AND** the filter state is preserved during the session
- **AND** the total count updates to reflect filtered results

#### Scenario: Sort images
- **WHEN** a user selects a sort option (filename, date, annotation count)
- **THEN** the image list is reordered accordingly
- **AND** the sort order is preserved during the session

### Requirement: Annotation Validation
The system SHALL validate all annotation data before saving to ensure correctness and prevent invalid data.

#### Scenario: Validate bounding box coordinates
- **WHEN** saving an annotation
- **THEN** the system verifies that x, y, width, height are all positive numbers
- **AND** verifies that the bounding box is within image boundaries
- **AND** rejects annotations with zero or negative dimensions

#### Scenario: Validate view and defect types
- **WHEN** saving an annotation
- **THEN** the system verifies that view type is either "TOP" or "SIDE"
- **AND** verifies that defect type is one of "PASS", "沖線", "晃動", "碰觸"
- **AND** rejects annotations with invalid or missing types

#### Scenario: Prevent duplicate annotations
- **WHEN** attempting to create an annotation with identical coordinates
- **THEN** the system checks for existing annotations at the same location
- **AND** warns the user about the potential duplicate
- **AND** allows the user to confirm or cancel

### Requirement: Auto-save and Data Persistence
The system SHALL automatically save annotation progress to prevent data loss.

#### Scenario: Auto-save on image change
- **WHEN** a user navigates to a different image (next/previous)
- **THEN** the system automatically saves all annotations for the current image
- **AND** writes updates to both the database and YOLO file
- **AND** displays a brief confirmation message

#### Scenario: Save on application exit
- **WHEN** a user closes the application
- **THEN** the system saves all pending changes
- **AND** ensures database transactions are committed
- **AND** closes all file handles properly

#### Scenario: Recover from crash
- **WHEN** the application is restarted after an unexpected crash
- **THEN** the system loads the last saved state from the database
- **AND** no annotations are lost that were saved to the database
- **AND** displays a message about recovered data if applicable

### Requirement: Error Handling and User Feedback
The system SHALL provide clear error messages and status feedback for all operations.

#### Scenario: Display operation status
- **WHEN** any significant operation is performed (load, save, delete)
- **THEN** the status bar displays a descriptive message
- **AND** success operations show a brief confirmation (e.g., "Saved successfully")
- **AND** errors show a persistent message with action suggestions

#### Scenario: Handle database errors
- **WHEN** a database operation fails (connection, query, transaction)
- **THEN** the system logs the full error details
- **AND** displays a user-friendly error message
- **AND** suggests corrective actions (e.g., "Check file permissions")
- **AND** allows the user to retry or cancel

#### Scenario: Handle file I/O errors
- **WHEN** file read or write operations fail
- **THEN** the system provides specific error information (permission denied, disk full, etc.)
- **AND** prevents data corruption by rolling back incomplete operations
- **AND** logs errors for debugging

### Requirement: Keyboard Shortcuts
The system SHALL provide keyboard shortcuts for all common operations to improve annotation efficiency.

#### Scenario: Core annotation shortcuts
- **WHEN** the user presses defined shortcut keys
- **THEN** the following operations are triggered:
  - `W`: Start drawing mode
  - `D`: Next image
  - `A`: Previous image
  - `S`: Save current annotations
  - `Delete`: Delete selected annotation
  - `Esc`: Cancel current operation
  - `Ctrl+O`: Open folder
  - `Ctrl+Z`: Undo last action
  - `Ctrl+Q`: Quit application

#### Scenario: Shortcut key conflicts
- **WHEN** multiple shortcut keys are pressed simultaneously
- **THEN** the system prioritizes based on context
- **AND** prevents conflicting actions from executing
- **AND** provides visual feedback for the executed action

#### Scenario: Display shortcut hints
- **WHEN** the user hovers over a button or menu item
- **THEN** a tooltip displays the associated keyboard shortcut
- **AND** the help menu lists all available shortcuts

### Requirement: Multi-Object Support
The system SHALL support multiple wire loop annotations per image to accommodate future multi-wire products.

#### Scenario: Annotate multiple wire loops
- **WHEN** an image contains multiple wire loops
- **THEN** the user can create multiple bounding box annotations
- **AND** each annotation has independent view and defect type labels
- **AND** all annotations are saved correctly to database and YOLO file

#### Scenario: Select and edit specific annotation
- **WHEN** multiple annotations overlap or are close together
- **THEN** the user can click to select a specific annotation
- **AND** the selected annotation is highlighted distinctly
- **AND** only the selected annotation can be edited

#### Scenario: YOLO format with multiple objects
- **WHEN** writing YOLO file for image with multiple annotations
- **THEN** each annotation is written on a separate line
- **AND** the correct class ID is used for each annotation
- **AND** the file format follows YOLO specification

### Requirement: Annotator Tracking
The system SHALL record the username of the person who created or modified each annotation for audit purposes.

#### Scenario: Record annotator on creation
- **WHEN** a new annotation is created
- **THEN** the system records the current user's username
- **AND** stores it in the `annotator` field
- **AND** includes the timestamp in `created_at`

#### Scenario: Track modification history
- **WHEN** an annotation is modified
- **THEN** the system updates the `updated_at` timestamp
- **AND** preserves the original annotator information
- **AND** allows querying annotations by annotator

#### Scenario: Configure annotator identity
- **WHEN** the application starts
- **THEN** the system prompts for or loads the annotator username
- **AND** uses this identity for all subsequent annotations
- **AND** allows changing the annotator identity in settings

