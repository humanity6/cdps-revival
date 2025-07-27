import cv2
import numpy as np
import time
import logging
from datetime import datetime
from typing import List, Tuple, Dict
from config import Config

class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.detection_times = []
        
    def update_frame_time(self, frame_time: float):
        """Update frame processing time"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def update_detection_time(self, detection_time: float):
        """Update face detection time"""
        self.detection_times.append(detection_time)
        if len(self.detection_times) > self.window_size:
            self.detection_times.pop(0)
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        return len(self.frame_times) / sum(self.frame_times)
    
    def get_avg_detection_time(self) -> float:
        """Get average detection time"""
        if not self.detection_times:
            return 0.0
        return sum(self.detection_times) / len(self.detection_times)

class FrameProcessor:
    @staticmethod
    def resize_frame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
        """Resize frame for faster processing"""
        if scale == 1.0:
            return frame
        
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def convert_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB for face_recognition library"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def scale_coordinates(coordinates: List[Tuple], scale: float) -> List[Tuple]:
        """Scale face coordinates back to original frame size"""
        if scale == 1.0:
            return coordinates
        
        scaled_coords = []
        for coord in coordinates:
            if len(coord) == 4:  # (top, right, bottom, left)
                top, right, bottom, left = coord
                scaled_coords.append((
                    int(top / scale),
                    int(right / scale),
                    int(bottom / scale),
                    int(left / scale)
                ))
            else:
                scaled_coords.append(coord)
        
        return scaled_coords

class AlertManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_alerts = {}  # Track last alert time for each person
        self.alert_cooldown = 5.0  # Seconds between alerts for same person
        
        # Setup logging
        if Config.ALERT_LOG:
            log_handler = logging.FileHandler(Config.LOG_FILE)
            log_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(log_handler)
    
    def trigger_alert(self, name: str, category: str, confidence: float, 
                     frame_location: Tuple = None) -> bool:
        """Trigger alert for detected person"""
        current_time = time.time()
        
        # Check cooldown
        if name in self.last_alerts:
            time_since_last = current_time - self.last_alerts[name]
            if time_since_last < self.alert_cooldown:
                return False
        
        # Update last alert time
        self.last_alerts[name] = current_time
        
        # Log alert
        alert_msg = f"ALERT: {category.upper()} detected - {name} (confidence: {confidence:.2f})"
        if frame_location:
            alert_msg += f" at position {frame_location}"
        
        self.logger.warning(alert_msg)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {alert_msg}")
        
        # TODO: Add sound alert if needed
        # if Config.ALERT_SOUND:
        #     self.play_alert_sound()
        
        return True
    
    def cleanup_old_alerts(self, max_age: float = 300.0):
        """Remove old alert timestamps"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.last_alerts.items()
            if current_time - timestamp > max_age
        ]
        
        for key in expired_keys:
            del self.last_alerts[key]

class FrameDrawer:
    @staticmethod
    def draw_face_box(frame: np.ndarray, face_location: Tuple, 
                     name: str, category: str, confidence: float) -> np.ndarray:
        """Draw bounding box and label on face"""
        top, right, bottom, left = face_location
        
        # Choose color based on category
        if category == 'criminal':
            color = (0, 0, 255)  # Red
        elif category == 'restricted':
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green for unknown
        
        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Prepare label text
        if category in ['criminal', 'restricted']:
            label = f"{category.upper()}: {name}"
        else:
            label = "Unknown"
        
        if Config.SHOW_CONFIDENCE and confidence > 0:
            label += f" ({confidence:.1%})"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
        cv2.rectangle(frame, (left, bottom - 35), 
                     (left + label_size[0], bottom), color, cv2.FILLED)
        
        # Draw label text
        cv2.putText(frame, label, (left + 6, bottom - 6),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter on frame"""
        if not Config.SHOW_FPS:
            return frame
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    @staticmethod
    def draw_status(frame: np.ndarray, status_text: str) -> np.ndarray:
        """Draw status information"""
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

class VideoCapture:
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.is_video_file = isinstance(source, str)
        
    def initialize(self) -> bool:
        """Initialize video capture"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.is_video_file:
                # Set webcam properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.VIDEO_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.VIDEO_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
            
            return self.cap.isOpened()
            
        except Exception as e:
            logging.error(f"Error initializing video capture: {str(e)}")
            return False
    
    def read_frame(self) -> Tuple[bool, np.ndarray]:
        """Read next frame"""
        if self.cap is None:
            return False, None
        
        return self.cap.read()
    
    def release(self):
        """Release video capture"""
        if self.cap is not None:
            self.cap.release()
    
    def get_total_frames(self) -> int:
        """Get total frames (for video files)"""
        if self.is_video_file and self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1
    
    def get_fps(self) -> float:
        """Get video FPS"""
        if self.cap is not None:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0