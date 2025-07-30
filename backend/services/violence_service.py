"""
Violence Detection Service Wrapper
Wraps the existing violence detection system for API integration
"""
import sys
import os
import cv2
import numpy as np
import asyncio
import time
import logging
from typing import Dict, Optional, Any
import base64
import tensorflow as tf

# Add violence detection module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'violence detection cdps'))

try:
    from flask_violence_detector import load_bensam02_model, detect_violence, preprocess_frame, enhance_frame
    VIOLENCE_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import violence detection modules: {e}")
    VIOLENCE_MODULES_AVAILABLE = False

from models.detection_models import ViolenceDetection, ViolenceResponse

logger = logging.getLogger(__name__)

class ViolenceService:
    def __init__(self):
        self.enabled = True
        self.model = None
        self.last_error = None
        
        try:
            if VIOLENCE_MODULES_AVAILABLE:
                # Load the violence detection model
                model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'violence detection cdps', 'violence', 'bensam02_model.h5')
                if os.path.exists(model_path):
                    try:
                        # Try loading with custom objects to handle compatibility issues
                        self.model = tf.keras.models.load_model(model_path, compile=False)
                        logger.info("Violence Detection Service initialized successfully")
                    except Exception as model_error:
                        try:
                            # Fallback: Try with custom objects None
                            import tensorflow.keras.utils as utils
                            self.model = tf.keras.models.load_model(model_path, custom_objects=None, compile=False)
                            logger.info("Violence Detection Service initialized successfully (with fallback)")
                        except Exception as fallback_error:
                            self.enabled = False
                            self.last_error = f"Model loading failed: {str(model_error)}, Fallback: {str(fallback_error)}"
                            logger.warning(f"Violence Detection Service disabled - {self.last_error}")
                else:
                    self.enabled = False
                    self.last_error = f"Model not found at: {model_path}"
                    logger.warning(f"Violence Detection Service disabled - {self.last_error}")
            else:
                self.enabled = False
                logger.warning("Violence Detection Service disabled - missing dependencies")
        except Exception as e:
            self.enabled = False
            self.last_error = str(e)
            logger.error(f"Failed to initialize Violence Detection Service: {e}")
    
    async def detect_violence(self, image: np.ndarray, confidence_threshold: float = 0.5) -> ViolenceResponse:
        """
        Detect violence in an image
        """
        start_time = time.time()
        
        if not self.enabled:
            return ViolenceResponse(
                success=False,
                detection=ViolenceDetection(is_violence=False, confidence=0.0),
                processing_time=0,
                error="Violence Detection Service is disabled"
            )
        
        try:
            # Preprocess the frame
            processed_frame = self._preprocess_frame(image)
            
            # Make prediction
            prediction = self.model.predict(processed_frame, verbose=0)
            
            # Get violence probability (assuming binary classification)
            violence_prob = prediction[0][1] if len(prediction[0]) > 1 else prediction[0][0]
            
            # Determine if violence is detected
            is_violence = float(violence_prob) > confidence_threshold
            
            detection = ViolenceDetection(
                is_violence=is_violence,
                confidence=float(violence_prob)
            )
            
            processing_time = time.time() - start_time
            
            return ViolenceResponse(
                success=True,
                detection=detection,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Violence detection failed: {e}")
            return ViolenceResponse(
                success=False,
                detection=ViolenceDetection(is_violence=False, confidence=0.0),
                processing_time=processing_time,
                error=str(e)
            )
    
    async def detect_from_base64(self, image_data: str, confidence_threshold: float = 0.5) -> ViolenceResponse:
        """
        Detect violence from base64 encoded image
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return ViolenceResponse(
                    success=False,
                    detection=ViolenceDetection(is_violence=False, confidence=0.0),
                    processing_time=0,
                    error="Invalid image data"
                )
            
            return await self.detect_violence(image, confidence_threshold)
            
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return ViolenceResponse(
                success=False,
                detection=ViolenceDetection(is_violence=False, confidence=0.0),
                processing_time=0,
                error=f"Image decoding failed: {str(e)}"
            )
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for Bensam02 model (128x128 input)
        """
        # Resize to 128x128 (Bensam02 model input size)
        frame = cv2.resize(frame, (128, 128))
        # Convert to RGB (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        frame = frame.astype('float32') / 255.0
        # Add batch dimension
        frame = np.expand_dims(frame, axis=0)
        return frame
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame for better detection
        """
        try:
            # Increase contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Increase sharpness
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Frame enhancement failed, using original: {e}")
            return frame
    
    async def detect_violence_video(self, video_data: bytes, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect violence in a video (batch processing)
        """
        try:
            # Create temporary file for video processing
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_data)
                tmp_path = tmp_file.name
            
            # Process video
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return {"success": False, "error": "Cannot open video file"}
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            violence_frames = 0
            confidence_values = []
            violence_timestamps = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 10th frame for efficiency
                if frame_count % 10 == 0:
                    result = await self.detect_violence(frame, confidence_threshold)
                    if result.success:
                        confidence_values.append(result.detection.confidence)
                        if result.detection.is_violence:
                            violence_frames += 1
                            violence_timestamps.append(float(frame_count / fps))
            
            cap.release()
            os.unlink(tmp_path)  # Clean up temp file
            
            return {
                "success": True,
                "total_frames_processed": frame_count,
                "violence_frames": violence_frames,
                "violence_percentage": (violence_frames / (frame_count // 10)) * 100 if frame_count > 0 else 0,
                "avg_confidence": np.mean(confidence_values) if confidence_values else 0,
                "violence_timestamps": violence_timestamps,
                "duration": float(total_frames / fps) if fps > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Video violence detection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the violence detection service
        """
        return {
            "enabled": self.enabled,
            "last_error": self.last_error,
            "model_loaded": self.model is not None,
            "modules_available": VIOLENCE_MODULES_AVAILABLE
        }
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        Update violence detection configuration
        """
        try:
            # Violence detection doesn't have complex configuration
            # This is a placeholder for future configuration options
            logger.info(f"Violence detection config update requested: {config_updates}")
            return True
        except Exception as e:
            logger.error(f"Failed to update violence config: {e}")
            return False