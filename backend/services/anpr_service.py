"""
ANPR Service Wrapper
Wraps the existing ANPR detection system for API integration
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

# Add ANPR module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'anpr'))

try:
    # Import ANPRConfig from backend config (which imports it from ANPR module)
    from config import ANPRConfig
    
    # Import other ANPR modules
    from anpr_system import ANPRDetectionSystem
    from plate_recognition import EnhancedPlateRecognitionEngine
    
except Exception as e:
    logging.warning(f"Could not import ANPR modules: {e}")
    ANPRDetectionSystem = None
    ANPRConfig = None
    EnhancedPlateRecognitionEngine = None

from models.detection_models import ANPRDetection, ANPRResponse, BoundingBox

logger = logging.getLogger(__name__)

class ANPRService:
    def __init__(self):
        self.enabled = True
        self.anpr_system = None
        self.config = None
        self.last_error = None
        
        try:
            if ANPRConfig and EnhancedPlateRecognitionEngine:
                self.config = ANPRConfig()
                self.plate_engine = EnhancedPlateRecognitionEngine()
                logger.info("ANPR Service initialized successfully")
            else:
                self.enabled = False
                logger.warning("ANPR Service disabled - missing dependencies")
        except Exception as e:
            self.enabled = False
            self.last_error = str(e)
            logger.error(f"Failed to initialize ANPR Service: {e}")
    
    async def detect_plates(self, image: np.ndarray, min_confidence: float = 0.5) -> ANPRResponse:
        """
        Detect license plates in an image
        """
        start_time = time.time()
        
        if not self.enabled:
            return ANPRResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=0,
                error="ANPR Service is disabled"
            )
        
        try:
            # Use the plate recognition engine directly
            detections_raw = self.plate_engine.detect_plates_in_frame(image)
            
            detections = []
            for detection in detections_raw:
                if detection['confidence'] >= min_confidence:
                    # Check if plate is red-listed (you'll need to implement this)
                    plate_number = detection['plate_number']
                    is_red_listed = self._check_red_listed(plate_number)
                    
                    bbox = BoundingBox(
                        x1=int(detection['bbox'][0]),
                        y1=int(detection['bbox'][1]),
                        x2=int(detection['bbox'][2]),
                        y2=int(detection['bbox'][3])
                    )
                    
                    anpr_detection = ANPRDetection(
                        bbox=bbox,
                        confidence=detection['confidence'],
                        class_name="license_plate",
                        plate_number=plate_number,
                        is_red_listed=is_red_listed,
                        alert_reason=self._get_alert_reason(plate_number) if is_red_listed else None
                    )
                    detections.append(anpr_detection)
            
            processing_time = time.time() - start_time
            
            return ANPRResponse(
                success=True,
                detections=detections,
                total_detections=len(detections),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"ANPR detection failed: {e}")
            return ANPRResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=processing_time,
                error=str(e)
            )
    
    async def detect_from_base64(self, image_data: str, min_confidence: float = 0.5) -> ANPRResponse:
        """
        Detect plates from base64 encoded image
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return ANPRResponse(
                    success=False,
                    detections=[],
                    total_detections=0,
                    processing_time=0,
                    error="Invalid image data"
                )
            
            return await self.detect_plates(image, min_confidence)
            
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return ANPRResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=0,
                error=f"Image decoding failed: {str(e)}"
            )
    
    def _check_red_listed(self, plate_number: str) -> bool:
        """
        Check if a plate number is red-listed
        This would typically query a database or in-memory store
        """
        # For now, return False - implement red-list checking logic
        if hasattr(self, 'anpr_system') and self.anpr_system:
            return plate_number in getattr(self.anpr_system, 'red_listed_vehicles', {})
        return False
    
    def _get_alert_reason(self, plate_number: str) -> Optional[str]:
        """
        Get the alert reason for a red-listed plate
        """
        if hasattr(self, 'anpr_system') and self.anpr_system:
            return getattr(self.anpr_system, 'red_listed_vehicles', {}).get(plate_number)
        return None
    
    def add_red_listed_vehicle(self, plate_number: str, reason: str = "Suspicious Activity") -> bool:
        """
        Add a vehicle to the red-listed vehicles
        """
        try:
            if self.anpr_system:
                return self.anpr_system.add_red_vehicle(plate_number, reason)
            return False
        except Exception as e:
            logger.error(f"Failed to add red-listed vehicle: {e}")
            return False
    
    def remove_red_listed_vehicle(self, plate_number: str) -> bool:
        """
        Remove a vehicle from the red-listed vehicles
        """
        try:
            if self.anpr_system:
                return self.anpr_system.remove_red_vehicle(plate_number)
            return False
        except Exception as e:
            logger.error(f"Failed to remove red-listed vehicle: {e}")
            return False
    
    def get_red_listed_vehicles(self) -> List[Tuple[str, str]]:
        """
        Get all red-listed vehicles
        """
        try:
            if self.anpr_system:
                return self.anpr_system.list_red_vehicles()
            return []
        except Exception as e:
            logger.error(f"Failed to get red-listed vehicles: {e}")
            return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the ANPR service
        """
        return {
            "enabled": self.enabled,
            "last_error": self.last_error,
            "config_loaded": self.config is not None,
            "engine_loaded": hasattr(self, 'plate_engine') and self.plate_engine is not None
        }
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        Update ANPR configuration
        """
        try:
            if self.config:
                # Update configuration based on provided parameters
                for key, value in config_updates.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                    elif hasattr(self.config.detection, key):
                        setattr(self.config.detection, key, value)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update ANPR config: {e}")
            return False