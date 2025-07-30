"""
Performance Monitor for Face Detection System
Tracks system performance and provides optimization suggestions
"""

import time
import psutil
import cv2
import numpy as np
from collections import deque
from typing import Dict, List
import logging

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = time.time()
        
    def start_frame_timer(self):
        """Start timing a frame"""
        self.frame_start_time = time.time()
        
    def end_frame_timer(self):
        """End timing a frame"""
        if hasattr(self, 'frame_start_time'):
            frame_time = time.time() - self.frame_start_time
            self.frame_times.append(frame_time)
            self.total_frames += 1
    
    def start_detection_timer(self):
        """Start timing face detection"""
        self.detection_start_time = time.time()
        
    def end_detection_timer(self):
        """End timing face detection"""
        if hasattr(self, 'detection_start_time'):
            detection_time = time.time() - self.detection_start_time
            self.detection_times.append(detection_time)
            self.total_detections += 1
    
    def update_system_metrics(self):
        """Update CPU and memory usage"""
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(psutil.virtual_memory().percent)
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_average_detection_time(self) -> float:
        """Get average detection time in ms"""
        if not self.detection_times:
            return 0.0
        
        return (sum(self.detection_times) / len(self.detection_times)) * 1000
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        runtime = time.time() - self.start_time
        
        stats = {
            'runtime_seconds': runtime,
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'avg_fps': self.total_frames / runtime if runtime > 0 else 0,
            'current_fps': self.get_fps(),
            'avg_detection_time_ms': self.get_average_detection_time(),
            'cpu_usage_percent': list(self.cpu_usage)[-1] if self.cpu_usage else 0,
            'memory_usage_percent': list(self.memory_usage)[-1] if self.memory_usage else 0,
            'avg_cpu_usage': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            'avg_memory_usage': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        }
        
        return stats
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        stats = self.get_system_stats()
        
        report = f"""
=== Performance Report ===
Runtime: {stats['runtime_seconds']:.1f} seconds
Total Frames Processed: {stats['total_frames']}
Total Face Detections: {stats['total_detections']}

Frame Rate:
  Average FPS: {stats['avg_fps']:.1f}
  Current FPS: {stats['current_fps']:.1f}

Detection Performance:
  Average Detection Time: {stats['avg_detection_time_ms']:.1f} ms
  Detections per second: {stats['total_detections'] / stats['runtime_seconds']:.1f}

System Resources:
  Current CPU Usage: {stats['cpu_usage_percent']:.1f}%
  Current Memory Usage: {stats['memory_usage_percent']:.1f}%
  Average CPU Usage: {stats['avg_cpu_usage']:.1f}%
  Average Memory Usage: {stats['avg_memory_usage']:.1f}%
"""
        
        # Add optimization suggestions
        suggestions = self.get_optimization_suggestions(stats)
        if suggestions:
            report += "\n=== Optimization Suggestions ===\n"
            for suggestion in suggestions:
                report += f"• {suggestion}\n"
        
        return report
    
    def get_optimization_suggestions(self, stats: Dict) -> List[str]:
        """Generate optimization suggestions based on performance"""
        suggestions = []
        
        # FPS suggestions
        if stats['current_fps'] < 15:
            suggestions.append("Low FPS detected. Try increasing FRAME_SKIP or DETECTION_FREQUENCY in config.py")
            suggestions.append("Consider reducing DETECTION_SCALE_FACTOR for faster processing")
            suggestions.append("Switch from 'cnn' to 'hog' face detection model for better speed")
        
        # Detection time suggestions
        if stats['avg_detection_time_ms'] > 100:
            suggestions.append("High detection time. Consider reducing input resolution")
            suggestions.append("Process fewer frames by increasing DETECTION_FREQUENCY")
        
        # CPU suggestions
        if stats['avg_cpu_usage'] > 80:
            suggestions.append("High CPU usage detected. Reduce processing frequency or resolution")
            suggestions.append("Consider using GPU acceleration if available")
        
        # Memory suggestions
        if stats['avg_memory_usage'] > 80:
            suggestions.append("High memory usage detected. Reduce frame buffer sizes")
            suggestions.append("Clear detection history periodically")
        
        return suggestions
    
    def draw_performance_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance metrics on frame"""
        stats = self.get_system_stats()
        
        # Prepare text
        overlay_text = [
            f"FPS: {stats['current_fps']:.1f}",
            f"Detection: {stats['avg_detection_time_ms']:.1f}ms",
            f"CPU: {stats['cpu_usage_percent']:.1f}%",
            f"Memory: {stats['memory_usage_percent']:.1f}%"
        ]
        
        # Draw background
        overlay_height = len(overlay_text) * 25 + 10
        cv2.rectangle(frame, (10, 50), (250, 50 + overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 50), (250, 50 + overlay_height), (255, 255, 255), 1)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)  # Green
        
        for i, text in enumerate(overlay_text):
            y_pos = 70 + (i * 25)
            cv2.putText(frame, text, (15, y_pos), font, font_scale, color, 1)
        
        return frame
    
    def log_performance_warning(self):
        """Log performance warnings"""
        stats = self.get_system_stats()
        
        if stats['current_fps'] < 10:
            logging.warning(f"Low FPS detected: {stats['current_fps']:.1f}")
        
        if stats['avg_detection_time_ms'] > 150:
            logging.warning(f"High detection time: {stats['avg_detection_time_ms']:.1f}ms")
        
        if stats['cpu_usage_percent'] > 90:
            logging.warning(f"High CPU usage: {stats['cpu_usage_percent']:.1f}%")
        
        if stats['memory_usage_percent'] > 90:
            logging.warning(f"High memory usage: {stats['memory_usage_percent']:.1f}%")


class OptimizedFaceDetector:
    """
    Optimized version of face detection with performance monitoring
    """
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.frame_buffer = deque(maxlen=5)  # Keep last 5 frames for temporal consistency
        self.last_detection_result = None
        self.detection_skip_counter = 0
        
    def detect_faces_optimized(self, frame: np.ndarray, known_encodings: List, known_names: List, 
                             known_categories: List, config) -> tuple:
        """
        Optimized face detection with temporal consistency and smart skipping
        """
        self.monitor.start_detection_timer()
        
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Skip detection if we have recent results and not much motion
        if (self.last_detection_result is not None and 
            self.detection_skip_counter < config.DETECTION_FREQUENCY and
            len(self.frame_buffer) > 1):
            
            # Check for significant motion
            if not self._has_significant_motion():
                self.detection_skip_counter += 1
                self.monitor.end_detection_timer()
                return self.last_detection_result
        
        # Reset skip counter
        self.detection_skip_counter = 0
        
        # Perform detection
        import face_recognition
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=config.DETECTION_SCALE_FACTOR, fy=config.DETECTION_SCALE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame, model=config.FACE_DETECTION_MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Process results (same as original)
        face_names = []
        face_categories = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=config.RECOGNITION_TOLERANCE)
            name = "Unknown"
            category = config.PERSON_CATEGORIES['UNKNOWN']
            
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index] and face_distances[best_match_index] < config.MAX_FACE_DISTANCE:
                    name = known_names[best_match_index]
                    category = known_categories[best_match_index]
            
            face_names.append(name)
            face_categories.append(category)
        
        # Scale back face locations
        scaled_face_locations = []
        scale_factor = 1 / config.DETECTION_SCALE_FACTOR
        for (top, right, bottom, left) in face_locations:
            scaled_face_locations.append((
                int(top * scale_factor),
                int(right * scale_factor),
                int(bottom * scale_factor),
                int(left * scale_factor)
            ))
        
        result = (scaled_face_locations, face_names, face_categories)
        self.last_detection_result = result
        
        self.monitor.end_detection_timer()
        return result
    
    def _has_significant_motion(self) -> bool:
        """Check if there's significant motion between recent frames"""
        if len(self.frame_buffer) < 2:
            return True
        
        # Compare last two frames
        frame1 = cv2.cvtColor(self.frame_buffer[-2], cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(self.frame_buffer[-1], cv2.COLOR_BGR2GRAY)
        
        # Resize for faster computation
        frame1 = cv2.resize(frame1, (160, 120))
        frame2 = cv2.resize(frame2, (160, 120))
        
        # Calculate frame difference
        diff = cv2.absdiff(frame1, frame2)
        mean_diff = np.mean(diff)
        
        # Threshold for significant motion
        return mean_diff > 10
