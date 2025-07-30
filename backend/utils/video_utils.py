"""
Video processing utilities for the unified backend
"""
import cv2
import numpy as np
import tempfile
import os
from typing import Generator, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Video processing utility class
    """
    
    def __init__(self, video_source: str):
        self.video_source = video_source
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.duration = 0
        
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def open(self) -> bool:
        """Open video source"""
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                logger.error(f"Could not open video source: {self.video_source}")
                return False
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0
            
            logger.info(f"Video opened: {self.width}x{self.height}, {self.fps} FPS, {self.duration:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to open video: {e}")
            return False
    
    def close(self):
        """Close video source"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video"""
        if not self.cap:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            return ret, frame
        except Exception as e:
            logger.error(f"Failed to read frame: {e}")
            return False, None
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """Get frame at specific timestamp (seconds)"""
        if not self.cap:
            return None
        
        try:
            frame_number = int(timestamp * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            return frame if ret else None
        except Exception as e:
            logger.error(f"Failed to get frame at time {timestamp}: {e}")
            return None
    
    def get_frame_at_position(self, frame_number: int) -> Optional[np.ndarray]:
        """Get frame at specific frame number"""
        if not self.cap:
            return None
        
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            return frame if ret else None
        except Exception as e:
            logger.error(f"Failed to get frame at position {frame_number}: {e}")
            return None
    
    def frames_generator(self, skip_frames: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator that yields frames with frame numbers"""
        if not self.cap:
            return
        
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_num % skip_frames == 0:
                yield frame_num, frame
            
            frame_num += 1
    
    def get_video_info(self) -> Dict[str, Any]:
        """Get video information"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration': self.duration,
            'source': self.video_source
        }

def create_temp_video_file(video_data: bytes, suffix: str = '.mp4') -> str:
    """
    Create temporary video file from bytes data
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(video_data)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        logger.error(f"Failed to create temp video file: {e}")
        raise

def cleanup_temp_file(file_path: str):
    """
    Clean up temporary file
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

def extract_frames(video_path: str, output_dir: str, interval: float = 1.0) -> list:
    """
    Extract frames from video at specified intervals
    """
    extracted_frames = []
    
    try:
        with VideoProcessor(video_path) as processor:
            if not processor.cap:
                return extracted_frames
            
            os.makedirs(output_dir, exist_ok=True)
            
            frame_interval = int(processor.fps * interval)
            frame_num = 0
            
            for current_frame, frame in processor.frames_generator():
                if current_frame % frame_interval == 0:
                    timestamp = current_frame / processor.fps
                    filename = f"frame_{current_frame:06d}_{timestamp:.2f}s.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    success = cv2.imwrite(filepath, frame)
                    if success:
                        extracted_frames.append({
                            'frame_number': current_frame,
                            'timestamp': timestamp,
                            'filepath': filepath
                        })
                
                frame_num += 1
        
        logger.info(f"Extracted {len(extracted_frames)} frames from {video_path}")
        return extracted_frames
        
    except Exception as e:
        logger.error(f"Failed to extract frames: {e}")
        return extracted_frames

def resize_video_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize video frame maintaining aspect ratio
    """
    try:
        h, w = frame.shape[:2]
        
        # Calculate scaling factor
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create canvas with target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate centering offsets
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        # Place resized frame on canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    except Exception as e:
        logger.error(f"Failed to resize video frame: {e}")
        return frame

def validate_video_format(file_path: str) -> bool:
    """
    Validate if video file format is supported
    """
    try:
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in supported_formats
    except Exception:
        return False

def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Get comprehensive video metadata
    """
    metadata = {}
    
    try:
        with VideoProcessor(video_path) as processor:
            if processor.cap:
                # Basic properties
                metadata.update(processor.get_video_info())
                
                # Additional properties
                metadata.update({
                    'codec': int(processor.cap.get(cv2.CAP_PROP_FOURCC)),
                    'backend': processor.cap.getBackendName(),
                    'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
                })
                
    except Exception as e:
        logger.error(f"Failed to get video metadata: {e}")
        metadata['error'] = str(e)
    
    return metadata

class CameraCapture:
    """
    Camera capture utility for live feeds
    """
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_opened = False
    
    def open(self) -> bool:
        """Open camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
            
            # Test camera
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Could not read from camera")
                self.cap.release()
                return False
            
            self.is_opened = True
            logger.info(f"Camera {self.camera_index} opened successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open camera: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera"""
        if not self.is_opened or not self.cap:
            return False, None
        
        try:
            return self.cap.read()
        except Exception as e:
            logger.error(f"Failed to read from camera: {e}")
            return False, None
    
    def close(self):
        """Close camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.is_opened = False
            logger.info(f"Camera {self.camera_index} closed")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()