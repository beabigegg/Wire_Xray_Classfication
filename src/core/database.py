"""
Database module for annotation storage.

This module handles all database operations for storing images,
annotations, and model metadata using SQLite.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class DatabaseError(Exception):
    """Exception raised for database operation errors."""
    pass


class Database:
    """
    SQLite database manager for annotation system.

    Handles CRUD operations for images, annotations, and model versions.
    """

    def __init__(self, db_path: str = "annotations.db"):
        """
        Initialize database connection and create tables if needed.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to connect to database: {e}")

    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            cursor = self.conn.cursor()

            # Images table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL UNIQUE,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Annotations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    bbox_x REAL NOT NULL,
                    bbox_y REAL NOT NULL,
                    bbox_width REAL NOT NULL,
                    bbox_height REAL NOT NULL,
                    view_type TEXT NOT NULL,
                    defect_type TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    annotator TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
                )
            """)

            # Model versions table (for future use)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    metrics_json TEXT,
                    is_active BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Training history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    model_version_id INTEGER,
                    config_json TEXT,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    final_metrics_json TEXT,
                    error_message TEXT,
                    FOREIGN KEY (model_version_id) REFERENCES model_versions(id) ON DELETE SET NULL
                )
            """)

            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_annotations_image_id
                ON annotations(image_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_annotations_view_type
                ON annotations(view_type)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_annotations_defect_type
                ON annotations(defect_type)
            """)

            self.conn.commit()

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to create tables: {e}")

    def add_image(self, filename: str, filepath: str, width: int, height: int) -> int:
        """
        Add a new image record to the database.

        Args:
            filename: Image filename
            filepath: Full path to image file
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            ID of the newly created image record

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO images (filename, filepath, width, height)
                VALUES (?, ?, ?, ?)
            """, (filename, filepath, width, height))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Image already exists (based on filepath), return existing ID
            existing = self.get_image_by_filepath(filepath)
            if existing:
                return existing['id']
            # Fallback: should not happen, but handle gracefully
            raise DatabaseError(f"Image exists but cannot be retrieved: {filepath}")
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to add image: {e}")

    def get_image_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get image record by filename.

        DEPRECATED: Use get_image_by_filepath instead to avoid conflicts
        when multiple folders contain files with the same name.

        Args:
            filename: Image filename

        Returns:
            Dictionary with image data or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM images WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get image: {e}")

    def get_image_by_filepath(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Get image record by full filepath.

        This is the preferred method to avoid conflicts when different folders
        contain images with the same filename (e.g., 1.jpg in multiple folders).

        Args:
            filepath: Full path to the image file

        Returns:
            Dictionary with image data or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM images WHERE filepath = ?", (filepath,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get image: {e}")

    def get_image_by_id(self, image_id: int) -> Optional[Dict[str, Any]]:
        """
        Get image record by ID.

        Args:
            image_id: Image ID

        Returns:
            Dictionary with image data or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get image: {e}")

    def add_annotation(
        self,
        image_id: int,
        bbox: Dict[str, float],
        view_type: str,
        defect_type: str,
        annotator: str = "unknown",
        confidence: float = 1.0
    ) -> int:
        """
        Add a new annotation record.

        Args:
            image_id: ID of the associated image
            bbox: Bounding box dict with keys: x, y, width, height (pixel coordinates)
            view_type: View type (TOP or SIDE)
            defect_type: Defect type (PASS, 沖線, 晃動, 碰觸)
            annotator: Username of the annotator
            confidence: Confidence score (0-1)

        Returns:
            ID of the newly created annotation record

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO annotations
                (image_id, bbox_x, bbox_y, bbox_width, bbox_height,
                 view_type, defect_type, confidence, annotator)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_id,
                bbox['x'], bbox['y'], bbox['width'], bbox['height'],
                view_type, defect_type, confidence, annotator
            ))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to add annotation: {e}")

    def get_annotations(self, image_id: int) -> List[Dict[str, Any]]:
        """
        Get all annotations for a specific image.

        Args:
            image_id: ID of the image

        Returns:
            List of annotation dictionaries

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM annotations
                WHERE image_id = ?
                ORDER BY created_at
            """, (image_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get annotations: {e}")

    def update_annotation(self, annotation_id: int, **kwargs) -> bool:
        """
        Update an existing annotation.

        Args:
            annotation_id: ID of the annotation to update
            **kwargs: Fields to update (bbox_x, bbox_y, bbox_width, bbox_height,
                     view_type, defect_type, confidence)

        Returns:
            True if update was successful, False if annotation not found

        Raises:
            DatabaseError: If the operation fails
        """
        if not kwargs:
            return False

        try:
            # Build dynamic UPDATE query
            set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
            values = list(kwargs.values())

            # Always update the updated_at timestamp
            set_clause += ", updated_at = CURRENT_TIMESTAMP"

            cursor = self.conn.cursor()
            cursor.execute(f"""
                UPDATE annotations
                SET {set_clause}
                WHERE id = ?
            """, values + [annotation_id])
            self.conn.commit()

            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to update annotation: {e}")

    def delete_annotation(self, annotation_id: int) -> bool:
        """
        Delete an annotation.

        Args:
            annotation_id: ID of the annotation to delete

        Returns:
            True if deletion was successful, False if annotation not found

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to delete annotation: {e}")

    def get_annotation_count(self, image_id: int) -> int:
        """
        Get the number of annotations for an image.

        Args:
            image_id: ID of the image

        Returns:
            Number of annotations

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM annotations WHERE image_id = ?",
                (image_id,)
            )
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get annotation count: {e}")

    def get_all_images(self) -> List[Dict[str, Any]]:
        """
        Get all image records.

        Returns:
            List of image dictionaries

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM images ORDER BY filename")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get all images: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics:
                - total_images: Total number of images
                - annotated_images: Number of images with at least one annotation
                - total_annotations: Total number of annotations
                - annotations_by_view: Count by view type
                - annotations_by_defect: Count by defect type

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()

            stats = {}

            # Total images
            cursor.execute("SELECT COUNT(*) FROM images")
            stats['total_images'] = cursor.fetchone()[0]

            # Annotated images
            cursor.execute("""
                SELECT COUNT(DISTINCT image_id) FROM annotations
            """)
            stats['annotated_images'] = cursor.fetchone()[0]

            # Total annotations
            cursor.execute("SELECT COUNT(*) FROM annotations")
            stats['total_annotations'] = cursor.fetchone()[0]

            # By view type
            cursor.execute("""
                SELECT view_type, COUNT(*) as count
                FROM annotations
                GROUP BY view_type
            """)
            stats['annotations_by_view'] = {row[0]: row[1] for row in cursor.fetchall()}

            # By defect type
            cursor.execute("""
                SELECT defect_type, COUNT(*) as count
                FROM annotations
                GROUP BY defect_type
            """)
            stats['annotations_by_defect'] = {row[0]: row[1] for row in cursor.fetchall()}

            return stats
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get statistics: {e}")

    # Training history methods
    def add_training_run(
        self,
        model_type: str,
        config_json: str,
        model_version_id: Optional[int] = None
    ) -> int:
        """
        Add a new training run record.

        Args:
            model_type: Type of model (detection, view, defect)
            config_json: JSON string of training configuration
            model_version_id: Optional ID of the model version created

        Returns:
            ID of the newly created training run record

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO training_history (model_type, config_json, model_version_id)
                VALUES (?, ?, ?)
            """, (model_type, config_json, model_version_id))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to add training run: {e}")

    def update_training_run(
        self,
        run_id: int,
        status: Optional[str] = None,
        final_metrics_json: Optional[str] = None,
        error_message: Optional[str] = None,
        model_version_id: Optional[int] = None
    ) -> bool:
        """
        Update a training run record.

        Args:
            run_id: ID of the training run to update
            status: Training status (running, completed, failed, cancelled)
            final_metrics_json: JSON string of final metrics
            error_message: Error message if training failed
            model_version_id: ID of the model version created

        Returns:
            True if update was successful, False if run not found

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            updates = []
            values = []

            if status is not None:
                updates.append("status = ?")
                values.append(status)
                if status in ('completed', 'failed', 'cancelled'):
                    updates.append("end_time = CURRENT_TIMESTAMP")

            if final_metrics_json is not None:
                updates.append("final_metrics_json = ?")
                values.append(final_metrics_json)

            if error_message is not None:
                updates.append("error_message = ?")
                values.append(error_message)

            if model_version_id is not None:
                updates.append("model_version_id = ?")
                values.append(model_version_id)

            if not updates:
                return False

            cursor = self.conn.cursor()
            query = f"UPDATE training_history SET {', '.join(updates)} WHERE id = ?"
            values.append(run_id)
            cursor.execute(query, values)
            self.conn.commit()

            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to update training run: {e}")

    def get_training_history(
        self,
        model_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get training history records.

        Args:
            model_type: Filter by model type (optional)
            limit: Maximum number of records to return (optional)

        Returns:
            List of training history dictionaries

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()

            query = "SELECT * FROM training_history"
            params = []

            if model_type:
                query += " WHERE model_type = ?"
                params.append(model_type)

            query += " ORDER BY start_time DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get training history: {e}")

    # Model version methods
    def save_model_version(
        self,
        model_name: str,
        model_type: str,
        version: str,
        filepath: str,
        metrics: Optional[Dict[str, Any]] = None,
        set_active: bool = False
    ) -> int:
        """
        Save a new model version record.

        Args:
            model_name: Name of the model
            model_type: Type of model (detection, view, defect)
            version: Version string (e.g., "v1_20250106_123456")
            filepath: Path to the saved model file
            metrics: Dictionary of model metrics
            set_active: Whether to set this as the active model

        Returns:
            ID of the newly created model version record

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()

            metrics_json = json.dumps(metrics) if metrics else None

            # If set_active, deactivate other models of the same type
            if set_active:
                cursor.execute("""
                    UPDATE model_versions
                    SET is_active = 0
                    WHERE model_type = ?
                """, (model_type,))

            cursor.execute("""
                INSERT INTO model_versions
                (model_name, model_type, version, filepath, metrics_json, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_name, model_type, version, filepath, metrics_json, 1 if set_active else 0))

            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to save model version: {e}")

    def get_active_model(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the active model version for a specific model type.

        Args:
            model_type: Type of model (detection, view, defect)

        Returns:
            Dictionary with model version data or None if not found

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM model_versions
                WHERE model_type = ? AND is_active = 1
                ORDER BY created_at DESC
                LIMIT 1
            """, (model_type,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get active model: {e}")

    def set_active_model(self, model_id: int) -> bool:
        """
        Set a model version as active.

        Args:
            model_id: ID of the model version to activate

        Returns:
            True if successful, False if model not found

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()

            # Get model type first
            cursor.execute("SELECT model_type FROM model_versions WHERE id = ?", (model_id,))
            row = cursor.fetchone()
            if not row:
                return False

            model_type = row[0]

            # Deactivate all models of this type
            cursor.execute("""
                UPDATE model_versions
                SET is_active = 0
                WHERE model_type = ?
            """, (model_type,))

            # Activate the selected model
            cursor.execute("""
                UPDATE model_versions
                SET is_active = 1
                WHERE id = ?
            """, (model_id,))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to set active model: {e}")

    def get_model_versions(
        self,
        model_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get model version records.

        Args:
            model_type: Filter by model type (optional)
            limit: Maximum number of records to return (optional)

        Returns:
            List of model version dictionaries

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()

            query = "SELECT * FROM model_versions"
            params = []

            if model_type:
                query += " WHERE model_type = ?"
                params.append(model_type)

            query += " ORDER BY created_at DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get model versions: {e}")

    def delete_model_version(self, model_id: int) -> bool:
        """
        Delete a model version record.

        Args:
            model_id: ID of the model version to delete

        Returns:
            True if deletion was successful, False if model not found

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM model_versions WHERE id = ?", (model_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to delete model version: {e}")

    def get_trained_models(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get all trained models of a specific type.

        Args:
            model_type: Type of model (detection, view, defect)

        Returns:
            List of dictionaries containing model information with all metrics.
            Each dict includes: id, model_name, timestamp, model_path, and all metrics
            from the training run (accuracy, mAP, etc.)

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            cursor = self.conn.cursor()

            # Join training_history with model_versions to get complete information
            cursor.execute("""
                SELECT
                    th.id,
                    mv.model_name,
                    th.start_time as timestamp,
                    mv.filepath as model_path,
                    th.final_metrics_json,
                    mv.metrics_json,
                    th.status
                FROM training_history th
                LEFT JOIN model_versions mv ON th.model_version_id = mv.id
                WHERE th.model_type = ? AND th.status = 'completed'
                ORDER BY th.start_time DESC
            """, (model_type,))

            rows = cursor.fetchall()
            models = []

            for row in rows:
                model_dict = dict(row)

                # Parse metrics JSON if available
                import json
                if model_dict.get('final_metrics_json'):
                    try:
                        metrics = json.loads(model_dict['final_metrics_json'])
                        model_dict.update(metrics)
                    except:
                        pass

                if model_dict.get('metrics_json'):
                    try:
                        metrics = json.loads(model_dict['metrics_json'])
                        model_dict.update(metrics)
                    except:
                        pass

                # Remove JSON fields (already parsed)
                model_dict.pop('final_metrics_json', None)
                model_dict.pop('metrics_json', None)
                model_dict.pop('status', None)

                models.append(model_dict)

            return models
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get trained models: {e}")

    def get_active_models(self) -> dict:
        """
        Get paths to active model files for each model type.

        Returns:
            Dictionary with keys mapping to model file paths:
            - 'view': View classifier (TOP/SIDE)
            - 'detection_top': YOLO for TOP view
            - 'detection_side': YOLO for SIDE view
            - 'defect_top': Defect classifier for TOP view
            - 'defect_side': Defect classifier for SIDE view

            For backward compatibility, also includes:
            - 'detection': Falls back to detection_top if exists
            - 'defect': Falls back to defect_top if exists

            Returns None for types without active models.
        """
        cursor = self.conn.cursor()

        active_models = {}

        # Define all model types (new VIEW-aware architecture)
        model_types = [
            'view',
            'detection_top',
            'detection_side',
            'defect_top',
            'defect_side'
        ]

        for model_type in model_types:
            cursor.execute("""
                SELECT filepath FROM model_versions
                WHERE model_type = ? AND is_active = 1
                LIMIT 1
            """, (model_type,))
            result = cursor.fetchone()
            active_models[model_type] = result['filepath'] if result else None

        # Backward compatibility: if old 'detection' model exists, use it for both views
        cursor.execute("""
            SELECT filepath FROM model_versions
            WHERE model_type = 'detection' AND is_active = 1
            LIMIT 1
        """)
        result = cursor.fetchone()
        if result:
            old_detection_path = result['filepath']
            # Use old detection model for both views if new models don't exist
            if not active_models.get('detection_top'):
                active_models['detection_top'] = old_detection_path
            if not active_models.get('detection_side'):
                active_models['detection_side'] = old_detection_path
            # Also provide under old key for backward compatibility
            active_models['detection'] = old_detection_path

        # Backward compatibility: if old 'defect' model exists, use it for both views
        cursor.execute("""
            SELECT filepath FROM model_versions
            WHERE model_type = 'defect' AND is_active = 1
            LIMIT 1
        """)
        result = cursor.fetchone()
        if result:
            old_defect_path = result['filepath']
            # Use old defect model for both views if new models don't exist
            if not active_models.get('defect_top'):
                active_models['defect_top'] = old_defect_path
            if not active_models.get('defect_side'):
                active_models['defect_side'] = old_defect_path
            # Also provide under old key for backward compatibility
            active_models['defect'] = old_defect_path

        return active_models

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
