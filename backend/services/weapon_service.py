"""
Weapon Detection Service Wrapper
Wraps the existing weapon detection system for API integration
"""
import sys
import os
import cv2
import numpy as np
import asyncio
import time
import logging
from typing import List, Dict, Optional, Any
import base64
import yaml

# Add weapon detection module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'weapon'))

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import ultralytics: {e}")
    ULTRALYTICS_AVAILABLE = False

from models.detection_models import WeaponDetection, WeaponResponse, BoundingBox

logger = logging.getLogger(__name__)

class WeaponService:
    def __init__(self):
        self.enabled = True
        self.model = None
        self.config = None
        self.last_error = None
        
        try:
            if ULTRALYTICS_AVAILABLE:
                # Load configuration
                self._load_config()
                
                # Load YOLO model
                model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'weapon', 'models', 'best.pt')
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    logger.info("Weapon Detection Service initialized successfully")
                else:
                    self.enabled = False
                    self.last_error = f"Model not found at: {model_path}"
                    logger.warning(f"Weapon Detection Service disabled - {self.last_error}")
            else:
                self.enabled = False
                logger.warning("Weapon Detection Service disabled - missing dependencies")
        except Exception as e:
            self.enabled = False
            self.last_error = str(e)
            logger.error(f"Failed to initialize Weapon Detection Service: {e}")
    
    def _load_config(self):
        """Load configuration from config.yaml"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'weapon', 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
            else:
                self.config = {
                    'model': {'path': 'models/best.pt'},
                    'detection': {'confidence_threshold': 0.5}
                }
        except Exception as e:
            logger.warning(f"Failed to load weapon config: {e}")
            self.config = {
                'model': {'path': 'models/best.pt'},
                'detection': {'confidence_threshold': 0.5}
            }
    
    async def detect_weapons(self, image: np.ndarray, confidence_threshold: float = 0.5) -> WeaponResponse:
        """
        Detect weapons in an image
        """
        start_time = time.time()
        
        if not self.enabled:
            return WeaponResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=0,
                error="Weapon Detection Service is disabled"
            )
        
        try:
            # Run inference
            results = self.model(image)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        if confidence >= confidence_threshold:
                            bbox = BoundingBox(
                                x1=int(x1),
                                y1=int(y1),
                                x2=int(x2),
                                y2=int(y2)
                            )
                            
                            weapon_detection = WeaponDetection(
                                bbox=bbox,
                                confidence=confidence,
                                class_name=class_name,
                                weapon_type=class_name
                            )
                            detections.append(weapon_detection)
            
            processing_time = time.time() - start_time
            
            return WeaponResponse(
                success=True,
                detections=detections,
                total_detections=len(detections),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Weapon detection failed: {e}")
            return WeaponResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=processing_time,
                error=str(e)
            )
    
    async def detect_from_base64(self, image_data: str, confidence_threshold: float = 0.5) -> WeaponResponse:
        """
        Detect weapons from base64 encoded image
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return WeaponResponse(
                    success=False,
                    detections=[],
                    total_detections=0,
                    processing_time=0,
                    error="Invalid image data"
                )
            
            return await self.detect_weapons(image, confidence_threshold)
            
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return WeaponResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=0,
                error=f"Image decoding failed: {str(e)}"
            )
    
    async def detect_weapons_video(self, video_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect weapons in a video file
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"success": False, "error": "Could not open video file"}
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Process video
            frame_detections = []
            frame_count = 0
            detection_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 10th frame for efficiency
                if frame_count % 10 == 0:
                    result = await self.detect_weapons(frame, confidence_threshold)
                    
                    if result.success and result.detections:
                        frame_detection = {
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'detections': [
                                {
                                    'bbox': [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
                                    'weapon_type': det.weapon_type,
                                    'confidence': det.confidence
                                }
                                for det in result.detections
                            ]
                        }
                        frame_detections.append(frame_detection)
                        detection_frames += 1
            
            cap.release()
            
            return {
                'success': True,
                'total_frames': frame_count,
                'detection_frames': detection_frames,
                'duration': duration,
                'fps': fps,
                'frame_detections': frame_detections
            }
            
        except Exception as e:
            logger.error(f"Video weapon detection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_supported_weapons(self) -> List[str]:
        """
        Get list of weapon types that can be detected
        """
        try:
            if self.model and hasattr(self.model, 'names'):
                return list(self.model.names.values())
            return []
        except Exception as e:
            logger.error(f"Failed to get supported weapons: {e}")
            return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the weapon detection service
        """
        supported_weapons = self.get_supported_weapons()
        
        return {
            "enabled": self.enabled,
            "last_error": self.last_error,
            "model_loaded": self.model is not None,
            "config_loaded": self.config is not None,
            "supported_weapons": supported_weapons,
            "ultralytics_available": ULTRALYTICS_AVAILABLE
        }
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        Update weapon detection configuration
        """
        try:
            if self.config:
                # Update configuration based on provided parameters
                for key, value in config_updates.items():
                    if key in ['confidence_threshold']:
                        if 'detection' not in self.config:
                            self.config['detection'] = {}
                        self.config['detection'][key] = value
                        logger.info(f"Updated weapon config {key} = {value}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update weapon config: {e}")
            return False
    
    def create_annotated_image(self, image: np.ndarray, detections: List[WeaponDetection]) -> np.ndarray:
        """
        Create an annotated image with weapon detection boxes
        """
        try:
            annotated_image = image.copy()
            
            for detection in detections:
                bbox = detection.bbox
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw label
                label = f"{detection.weapon_type}: {detection.confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            return annotated_image
        except Exception as e:
            logger.error(f"Failed to create annotated image: {e}")
            return image