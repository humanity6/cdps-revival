"""
Face Recognition Service Wrapper
Wraps the existing face detection system for API integration
"""
import sys
import os
import cv2
import numpy as np
import asyncio
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
import base64

# Add face detection module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'face detection'))

try:
    # Import face detection modules from the correct path
    import sys
    import os
    face_path = os.path.join(os.path.dirname(__file__), '..', '..', 'face detection')
    abs_face_path = os.path.abspath(face_path)
    
    # Store original config module to restore later
    original_config = sys.modules.get('config')
    
    if abs_face_path not in sys.path:
        sys.path.insert(0, abs_face_path)  # Use insert(0) to prioritize face detection path
    
    # Import face config with explicit path to avoid conflicts
    import importlib.util
    face_config_path = os.path.join(abs_face_path, "config.py")
    face_config_spec = importlib.util.spec_from_file_location("face_config", face_config_path)
    face_config_module = importlib.util.module_from_spec(face_config_spec)
    face_config_spec.loader.exec_module(face_config_module)
    face_config = face_config_module
    
    # Temporarily add the face config to sys.modules as 'config' so imports will find it
    sys.modules['config'] = face_config_module
    
    # Load performance monitor first
    perf_spec = importlib.util.spec_from_file_location("performance_monitor", os.path.join(abs_face_path, "performance_monitor.py"))
    perf_module = importlib.util.module_from_spec(perf_spec)
    perf_spec.loader.exec_module(perf_module)
    
    # Import face detection modules with explicit module loading
    face_system_spec = importlib.util.spec_from_file_location("face_detection_system", os.path.join(abs_face_path, "face_detection_system.py"))
    face_system_module = importlib.util.module_from_spec(face_system_spec)
    face_system_spec.loader.exec_module(face_system_module)
    FaceDetectionSystem = face_system_module.FaceDetectionSystem
    
    main_spec = importlib.util.spec_from_file_location("face_main", os.path.join(abs_face_path, "main.py"))
    main_module = importlib.util.module_from_spec(main_spec)
    main_spec.loader.exec_module(main_module)
    EnhancedFaceDetectionSystem = main_module.EnhancedFaceDetectionSystem
    
    # Restore original config module to prevent conflicts
    if original_config is not None:
        sys.modules['config'] = original_config
    elif 'config' in sys.modules:
        del sys.modules['config']
    
except Exception as e:
    logging.warning(f"Could not import face detection modules: {e}")
    FaceDetectionSystem = None
    EnhancedFaceDetectionSystem = None
    face_config = None
    
    # Restore original config module even on error
    if 'original_config' in locals() and original_config is not None:
        sys.modules['config'] = original_config
    elif 'config' in sys.modules:
        del sys.modules['config']

from models.detection_models import FaceDetection, FaceResponse, BoundingBox

logger = logging.getLogger(__name__)

class FaceService:
    def __init__(self):
        self.enabled = True
        self.face_system = None
        self.config = face_config
        self.last_error = None
        
        try:
            if EnhancedFaceDetectionSystem and face_config:
                self.face_system = EnhancedFaceDetectionSystem(enable_performance_monitor=False)
                logger.info("Face Recognition Service initialized successfully")
            elif FaceDetectionSystem:
                self.face_system = FaceDetectionSystem()
                logger.info("Basic Face Recognition Service initialized successfully")
            else:
                self.enabled = False
                logger.warning("Face Recognition Service disabled - missing dependencies")
        except Exception as e:
            self.enabled = False
            self.last_error = str(e)
            logger.error(f"Failed to initialize Face Recognition Service: {e}")
    
    async def detect_faces(self, image: np.ndarray, recognition_tolerance: float = 0.5) -> FaceResponse:
        """
        Detect and recognize faces in an image
        """
        start_time = time.time()
        
        if not self.enabled:
            return FaceResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=0,
                error="Face Recognition Service is disabled"
            )
        
        try:
            # Update tolerance if provided
            if self.config and hasattr(self.config, 'RECOGNITION_TOLERANCE'):
                original_tolerance = self.config.RECOGNITION_TOLERANCE
                self.config.RECOGNITION_TOLERANCE = recognition_tolerance
            
            # Detect faces using the face system
            face_locations, face_names, face_categories = self.face_system.detect_faces_in_frame(image)
            
            # Restore original tolerance
            if self.config and hasattr(self.config, 'RECOGNITION_TOLERANCE'):
                self.config.RECOGNITION_TOLERANCE = original_tolerance
            
            detections = []
            for location, name, category in zip(face_locations, face_names, face_categories):
                top, right, bottom, left = location
                
                bbox = BoundingBox(
                    x1=left,
                    y1=top,
                    x2=right,
                    y2=bottom
                )
                
                # Determine if alert should be triggered
                alert_triggered = category in ['restricted', 'criminal']
                
                face_detection = FaceDetection(
                    bbox=bbox,
                    confidence=0.8,  # Face recognition doesn't provide confidence scores
                    class_name="face",
                    person_name=name,
                    category=category,
                    alert_triggered=alert_triggered
                )
                detections.append(face_detection)
            
            processing_time = time.time() - start_time
            
            return FaceResponse(
                success=True,
                detections=detections,
                total_detections=len(detections),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Face detection failed: {e}")
            return FaceResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=processing_time,
                error=str(e)
            )
    
    async def detect_from_base64(self, image_data: str, recognition_tolerance: float = 0.5) -> FaceResponse:
        """
        Detect faces from base64 encoded image
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return FaceResponse(
                    success=False,
                    detections=[],
                    total_detections=0,
                    processing_time=0,
                    error="Invalid image data"
                )
            
            return await self.detect_faces(image, recognition_tolerance)
            
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return FaceResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=0,
                error=f"Image decoding failed: {str(e)}"
            )
    
    def reload_known_faces(self) -> bool:
        """
        Reload known faces from directories
        """
        try:
            if self.face_system and hasattr(self.face_system, 'load_known_faces'):
                self.face_system.load_known_faces()
                logger.info("Known faces reloaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to reload known faces: {e}")
            return False
    
    def get_known_faces_count(self) -> Dict[str, int]:
        """
        Get count of known faces by category
        """
        try:
            if self.face_system:
                total_faces = len(getattr(self.face_system, 'known_face_encodings', []))
                categories = getattr(self.face_system, 'known_face_categories', [])
                
                counts = {
                    'total': total_faces,
                    'restricted': categories.count('restricted'),
                    'criminal': categories.count('criminal'),
                    'unknown': categories.count('unknown')
                }
                return counts
            return {'total': 0, 'restricted': 0, 'criminal': 0, 'unknown': 0}
        except Exception as e:
            logger.error(f"Failed to get known faces count: {e}")
            return {'total': 0, 'restricted': 0, 'criminal': 0, 'unknown': 0}
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the face recognition service
        """
        known_faces_count = self.get_known_faces_count()
        
        return {
            "enabled": self.enabled,
            "last_error": self.last_error,
            "system_loaded": self.face_system is not None,
            "known_faces_count": known_faces_count,
            "config_available": self.config is not None
        }
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        Update face recognition configuration
        """
        try:
            if self.config:
                # Update configuration based on provided parameters
                for key, value in config_updates.items():
                    if hasattr(self.config, key.upper()):
                        setattr(self.config, key.upper(), value)
                        logger.info(f"Updated face config {key.upper()} = {value}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update face config: {e}")
            return False
    
    def add_known_face(self, image_data: str, name: str, category: str = "unknown") -> bool:
        """
        Add a new known face (this would typically save to the appropriate directory)
        """
        try:
            # This is a placeholder - in a real implementation, you'd save the image
            # to the appropriate directory and reload the known faces
            logger.info(f"Would add known face: {name} in category {category}")
            return True
        except Exception as e:
            logger.error(f"Failed to add known face: {e}")
            return False