"""
Enhanced ANPR Service with Unified Database Integration
Extends the existing ANPR service with comprehensive database logging and alerting
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
from datetime import datetime

# Add ANPR module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'anpr'))

try:
    from anpr_system import ANPRDetectionSystem
    from plate_recognition import EnhancedPlateRecognitionEngine
    # Import ANPRConfig from the original ANPR module
    anpr_path = os.path.join(os.path.dirname(__file__), '..', '..', 'anpr')
    if anpr_path not in sys.path:
        sys.path.append(anpr_path)
    from config import ANPRConfig as OriginalANPRConfig
    ANPRConfig = OriginalANPRConfig
except ImportError as e:
    logging.warning(f"Could not import ANPR modules: {e}")
    ANPRDetectionSystem = None
    ANPRConfig = None
    EnhancedPlateRecognitionEngine = None

from ..models.detection_models import ANPRDetection, ANPRResponse, BoundingBox
from .database_service import get_database_service

logger = logging.getLogger(__name__)

class EnhancedANPRService:
    """
    Enhanced ANPR Service with unified database integration.
    Provides all original functionality plus comprehensive logging and alerting.
    """
    
    def __init__(self, enable_database: bool = True, db_path: str = "crime_detection.db"):
        self.enabled = True
        self.anpr_system = None
        self.config = None
        self.last_error = None
        self.enable_database = enable_database
        
        # Database service integration
        self.db_service = None
        if enable_database:
            try:
                self.db_service = get_database_service(db_path)
                logger.info("Database service integration enabled")
            except Exception as e:
                logger.error(f"Failed to initialize database service: {e}")
                self.enable_database = False
        
        # Initialize ANPR components
        self._initialize_anpr_system()
        
        # Performance tracking
        self.detection_count = 0
        self.total_processing_time = 0.0
        self.last_detection_time = None
        
    def _initialize_anpr_system(self):
        """Initialize the ANPR detection system."""
        try:
            if ANPRConfig and EnhancedPlateRecognitionEngine:
                self.config = ANPRConfig()
                self.plate_engine = EnhancedPlateRecognitionEngine()
                logger.info("Enhanced ANPR Service initialized successfully")
                
                # Log system startup
                if self.db_service:
                    self.db_service._log_system_event(
                        'INFO', 'ANPR service started', 
                        module_name='ANPR'
                    )
            else:
                self.enabled = False
                logger.warning("Enhanced ANPR Service disabled - missing dependencies")
        except Exception as e:
            self.enabled = False
            self.last_error = str(e)
            logger.error(f"Failed to initialize Enhanced ANPR Service: {e}")
            
            # Log system error
            if self.db_service:
                self.db_service._log_system_event(
                    'ERROR', f'ANPR service initialization failed: {str(e)}',
                    module_name='ANPR'
                )
    
    async def detect_plates(self, image: np.ndarray, min_confidence: float = 0.5, 
                           location: str = "Camera_1", save_image: bool = True) -> ANPRResponse:
        """
        Enhanced plate detection with database logging and alerting.
        """
        start_time = time.time()
        
        if not self.enabled:
            return ANPRResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=0,
                error="Enhanced ANPR Service is disabled"
            )
        
        try:
            # Use the plate recognition engine directly
            detections_raw = self.plate_engine.detect_plates_in_frame(image)
            
            detections = []
            event_ids = []
            
            for detection in detections_raw:
                if detection['confidence'] >= min_confidence:
                    plate_number = detection['plate_number']
                    
                    # Check if plate is red-listed
                    is_red_listed, alert_reason = self._check_red_listed(plate_number)
                    
                    # Create bounding box
                    bbox = BoundingBox(
                        x1=int(detection['bbox'][0]),
                        y1=int(detection['bbox'][1]),
                        x2=int(detection['bbox'][2]),
                        y2=int(detection['bbox'][3])
                    )
                    
                    # Create ANPR detection object
                    anpr_detection = ANPRDetection(
                        bbox=bbox,
                        confidence=detection['confidence'],
                        class_name="license_plate",
                        plate_number=plate_number,
                        is_red_listed=is_red_listed,
                        alert_reason=alert_reason if is_red_listed else None
                    )
                    detections.append(anpr_detection)
                    
                    # Enhanced database logging
                    if self.db_service:
                        event_id = self._log_detection_to_database(
                            detection, plate_number, is_red_listed, alert_reason,
                            location, image, save_image
                        )
                        if event_id:
                            event_ids.append(event_id)
            
            processing_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(processing_time)
            
            # Log successful processing
            if self.db_service:
                self.db_service._log_system_event(
                    'INFO', f'ANPR detection completed: {len(detections)} plates found',
                    module_name='ANPR',
                    processing_time=processing_time,
                    additional_context={
                        'plate_count': len(detections),
                        'red_listed_count': sum(1 for d in detections if d.is_red_listed),
                        'location': location,
                        'min_confidence': min_confidence
                    }
                )
            
            response = ANPRResponse(
                success=True,
                detections=detections,
                total_detections=len(detections),
                processing_time=processing_time
            )
            
            # Add event IDs to response metadata if database is enabled
            if event_ids:
                response.event_ids = event_ids
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Enhanced ANPR detection failed: {e}"
            logger.error(error_msg)
            
            # Log error to database
            if self.db_service:
                self.db_service._log_system_event(
                    'ERROR', error_msg,
                    module_name='ANPR',
                    processing_time=processing_time,
                    additional_context={
                        'location': location,
                        'min_confidence': min_confidence,
                        'error_details': str(e)
                    }
                )
            
            return ANPRResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=processing_time,
                error=error_msg
            )
    
    def _log_detection_to_database(self, detection: Dict[str, Any], plate_number: str,
                                  is_red_listed: bool, alert_reason: Optional[str],
                                  location: str, image: np.ndarray, save_image: bool) -> Optional[int]:
        """Log detection to unified database."""
        try:
            # Save image if requested
            image_path = None
            if save_image:
                image_path = self._save_detection_image(image, plate_number, is_red_listed)
            
            # Prepare detection data for database
            detection_data = {
                'confidence': detection['confidence'],
                'location': location,
                'image_path': image_path,
                'processing_time': 0,  # Will be updated by the calling method
                
                # ANPR-specific data
                'plate_number': plate_number,
                'is_red_listed': is_red_listed,
                'alert_reason': alert_reason,
                'plate_confidence': detection['confidence'],
                'ocr_text_raw': detection.get('raw_text', plate_number),
                'region': self._detect_region(plate_number),
                'country_code': self._detect_country_code(plate_number),
                
                # Bounding box
                'bounding_box': {
                    'x1': int(detection['bbox'][0]),
                    'y1': int(detection['bbox'][1]),
                    'x2': int(detection['bbox'][2]),
                    'y2': int(detection['bbox'][3])
                },
                
                # Additional metadata
                'metadata': {
                    'detection_method': 'enhanced_plate_recognition',
                    'model_version': getattr(self.plate_engine, 'version', '1.0'),
                    'preprocessing_applied': True,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Log to database
            event_id = self.db_service.log_detection_event('ANPR', detection_data)
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log detection to database: {e}")
            return None
    
    def _save_detection_image(self, image: np.ndarray, plate_number: str, is_red_listed: bool) -> Optional[str]:
        """Save detection image to appropriate directory."""
        try:
            # Determine save directory
            if is_red_listed:
                save_dir = "red_alert_plates"
            else:
                save_dir = "detected_plates"
            
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_plate = "".join(c for c in plate_number if c.isalnum())
            filename = f"{timestamp}_{clean_plate}.jpg"
            filepath = os.path.join(save_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, image)
            logger.info(f"Detection image saved: {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save detection image: {e}")
            return None
    
    def _detect_region(self, plate_number: str) -> str:
        """Detect region from plate number pattern."""
        # Simplified region detection - can be enhanced with actual logic
        if self.config and hasattr(self.config.detection, 'validation_region'):
            return self.config.detection.validation_region
        return "generic"
    
    def _detect_country_code(self, plate_number: str) -> str:
        """Detect country code from plate number pattern."""
        # Simplified country detection - can be enhanced with actual logic
        return "XX"  # Unknown
    
    def _check_red_listed(self, plate_number: str) -> Tuple[bool, Optional[str]]:
        """Check if a plate number is red-listed."""
        if self.db_service:
            return self.db_service.is_red_listed('VEHICLE', plate_number)
        
        # Fallback to original method if database service is not available
        if hasattr(self, 'anpr_system') and self.anpr_system:
            return plate_number in getattr(self.anpr_system, 'red_listed_vehicles', {}), \
                   getattr(self.anpr_system, 'red_listed_vehicles', {}).get(plate_number)
        
        return False, None
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance tracking metrics."""
        self.detection_count += 1
        self.total_processing_time += processing_time
        self.last_detection_time = datetime.now()
    
    async def detect_from_base64(self, image_data: str, min_confidence: float = 0.5,
                               location: str = "Camera_1", save_image: bool = True) -> ANPRResponse:
        """Enhanced detection from base64 encoded image."""
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
            
            return await self.detect_plates(image, min_confidence, location, save_image)
            
        except Exception as e:
            error_msg = f"Failed to decode base64 image: {e}"
            logger.error(error_msg)
            
            # Log error to database
            if self.db_service:
                self.db_service._log_system_event(
                    'ERROR', error_msg,
                    module_name='ANPR',
                    additional_context={'location': location}
                )
            
            return ANPRResponse(
                success=False,
                detections=[],
                total_detections=0,
                processing_time=0,
                error=error_msg
            )
    
    # Enhanced Red-Listed Vehicle Management
    def add_red_listed_vehicle(self, plate_number: str, reason: str = "Suspicious Activity",
                              severity: str = "MEDIUM") -> bool:
        """Add a vehicle to the red-listed vehicles using database service."""
        try:
            if self.db_service:
                success = self.db_service.add_red_listed_item('VEHICLE', plate_number, reason, severity)
                if success:
                    logger.info(f"Added red-listed vehicle: {plate_number}")
                return success
            
            # Fallback to original method
            if self.anpr_system:
                return self.anpr_system.add_red_vehicle(plate_number, reason)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add red-listed vehicle: {e}")
            return False
    
    def remove_red_listed_vehicle(self, plate_number: str) -> bool:
        """Remove a vehicle from the red-listed vehicles."""
        try:
            if self.db_service:
                success = self.db_service.remove_red_listed_item('VEHICLE', plate_number)
                if success:
                    logger.info(f"Removed red-listed vehicle: {plate_number}")
                return success
            
            # Fallback to original method
            if self.anpr_system:
                return self.anpr_system.remove_red_vehicle(plate_number)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove red-listed vehicle: {e}")
            return False
    
    def get_red_listed_vehicles(self) -> List[Dict[str, Any]]:
        """Get all red-listed vehicles."""
        try:
            if self.db_service:
                return self.db_service.get_red_listed_items('VEHICLE')
            
            # Fallback to original method
            if self.anpr_system:
                vehicles = self.anpr_system.list_red_vehicles()
                return [
                    {
                        'identifier': plate,
                        'reason': reason,
                        'type': 'VEHICLE'
                    }
                    for plate, reason in vehicles
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get red-listed vehicles: {e}")
            return []
    
    # Enhanced Analytics and Reporting
    def get_detection_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        try:
            stats = {
                'service_stats': {
                    'detection_count': self.detection_count,
                    'average_processing_time': (
                        self.total_processing_time / self.detection_count 
                        if self.detection_count > 0 else 0
                    ),
                    'last_detection_time': (
                        self.last_detection_time.isoformat() 
                        if self.last_detection_time else None
                    ),
                    'service_enabled': self.enabled,
                    'last_error': self.last_error
                }
            }
            
            # Add database statistics if available
            if self.db_service:
                db_summary = self.db_service.get_detection_summary(days)
                anpr_stats = db_summary.get('detection_counts', {}).get('ANPR', 0)
                
                stats['database_stats'] = {
                    'total_anpr_detections': anpr_stats,
                    'summary_period_days': days,
                    'last_updated': db_summary.get('last_updated')
                }
                
                # Get recent ANPR alerts
                recent_alerts = self.db_service.get_recent_alerts(hours=24)
                anpr_alerts = [alert for alert in recent_alerts if alert['type'] == 'ANPR']
                
                stats['alert_stats'] = {
                    'recent_alerts_24h': len(anpr_alerts),
                    'red_listed_detections': len([
                        alert for alert in anpr_alerts 
                        if alert.get('is_red_listed', False)
                    ])
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get detection statistics: {e}")
            return {'error': str(e)}
    
    def get_recent_detections(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent ANPR detections from database."""
        try:
            if self.db_service:
                search_params = {
                    'event_type': 'ANPR',
                    'start_date': datetime.now() - timedelta(hours=hours),
                    'limit': 100
                }
                events = self.db_service.search_events(search_params)
                return events
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get recent detections: {e}")
            return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the ANPR service."""
        health_status = {
            "enabled": self.enabled,
            "last_error": self.last_error,
            "config_loaded": self.config is not None,
            "engine_loaded": hasattr(self, 'plate_engine') and self.plate_engine is not None,
            "database_integration": self.enable_database and self.db_service is not None,
            "performance_metrics": {
                "detection_count": self.detection_count,
                "average_processing_time": (
                    self.total_processing_time / self.detection_count 
                    if self.detection_count > 0 else 0
                ),
                "last_detection_time": (
                    self.last_detection_time.isoformat() 
                    if self.last_detection_time else None
                )
            }
        }
        
        # Add database health if available
        if self.db_service:
            try:
                db_health = self.db_service.get_system_health()
                health_status["database_health"] = db_health
            except Exception as e:
                health_status["database_health"] = {"error": str(e)}
        
        return health_status
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """Update ANPR configuration with enhanced logging."""
        try:
            if self.config:
                # Update configuration based on provided parameters
                updated_keys = []
                for key, value in config_updates.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        updated_keys.append(key)
                    elif hasattr(self.config.detection, key):
                        setattr(self.config.detection, key, value)
                        updated_keys.append(f"detection.{key}")
                
                # Log configuration changes
                if self.db_service and updated_keys:
                    self.db_service._log_system_event(
                        'INFO', f'ANPR configuration updated: {", ".join(updated_keys)}',
                        module_name='ANPR',
                        additional_context={'updated_config': config_updates}
                    )
                
                logger.info(f"ANPR configuration updated: {updated_keys}")
                return True
            
            return False
            
        except Exception as e:
            error_msg = f"Failed to update ANPR config: {e}"
            logger.error(error_msg)
            
            # Log error
            if self.db_service:
                self.db_service._log_system_event(
                    'ERROR', error_msg,
                    module_name='ANPR',
                    additional_context={'attempted_updates': config_updates}
                )
            
            return False
    
    def shutdown(self):
        """Shutdown the enhanced ANPR service."""
        logger.info("Shutting down Enhanced ANPR Service...")
        
        try:
            # Log shutdown
            if self.db_service:
                final_stats = {
                    'total_detections': self.detection_count,
                    'total_processing_time': self.total_processing_time,
                    'average_processing_time': (
                        self.total_processing_time / self.detection_count 
                        if self.detection_count > 0 else 0
                    )
                }
                
                self.db_service._log_system_event(
                    'INFO', 'ANPR service shutting down',
                    module_name='ANPR',
                    additional_context=final_stats
                )
            
            # Reset state
            self.enabled = False
            self.detection_count = 0
            self.total_processing_time = 0.0
            
            logger.info("Enhanced ANPR Service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Enhanced ANPR Service shutdown: {e}")

# Backward compatibility wrapper
class ANPRService(EnhancedANPRService):
    """Backward compatibility wrapper for existing code."""
    
    def __init__(self, enable_database: bool = True):
        super().__init__(enable_database=enable_database)
        logger.info("Using Enhanced ANPR Service with backward compatibility")