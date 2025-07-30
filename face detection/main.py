"""
Main application with performance monitoring and optimizations
Enhanced version of the face detection system with better performance
"""

import cv2
import face_recognition
import numpy as np
import os
import time
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import argparse

import config
from performance_monitor import PerformanceMonitor, OptimizedFaceDetector

class EnhancedFaceDetectionSystem:
    def __init__(self, enable_performance_monitor: bool = True):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_categories = []
        self.last_alert_time = {}
        self.frame_count = 0
        self.detection_count = 0
        
        # Performance monitoring
        self.performance_enabled = enable_performance_monitor
        if self.performance_enabled:
            self.monitor = PerformanceMonitor()
            self.optimized_detector = OptimizedFaceDetector(self.monitor)
        
        # Setup logging
        self.setup_logging()
        
        # Create necessary directories
        self.create_directories()
        
        # Load known faces
        self.load_known_faces()
        
        logging.info("Enhanced Face Detection System initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        log_filename = f"{config.LOGS_DIR}/face_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            config.KNOWN_FACES_DIR,
            config.RESTRICTED_FACES_DIR,
            config.CRIMINAL_FACES_DIR,
            config.LOGS_DIR,
            config.TEMP_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_known_faces(self):
        """Load and encode known faces from directories"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_categories = []
        
        # Load restricted faces
        self._load_faces_from_directory(
            config.RESTRICTED_FACES_DIR, 
            config.PERSON_CATEGORIES['RESTRICTED']
        )
        
        # Load criminal faces
        self._load_faces_from_directory(
            config.CRIMINAL_FACES_DIR, 
            config.PERSON_CATEGORIES['CRIMINAL']
        )
        
        logging.info(f"Loaded {len(self.known_face_encodings)} known faces")
    
    def _load_faces_from_directory(self, directory: str, category: str):
        """Load faces from a specific directory"""
        if not os.path.exists(directory):
            return
        
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(directory, filename)
                try:
                    # Load image
                    image = face_recognition.load_image_file(image_path)
                    
                    # Get face encodings
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        # Use the first face found
                        encoding = encodings[0]
                        name = os.path.splitext(filename)[0]
                        
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(name)
                        self.known_face_categories.append(category)
                        
                        logging.info(f"Loaded {category} face: {name}")
                    else:
                        logging.warning(f"No face found in {image_path}")
                        
                except Exception as e:
                    logging.error(f"Error loading {image_path}: {str(e)}")
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> Tuple[List, List, List]:
        """Detect and recognize faces in a frame with performance monitoring"""
        if self.performance_enabled:
            return self.optimized_detector.detect_faces_optimized(
                frame, 
                self.known_face_encodings,
                self.known_face_names,
                self.known_face_categories,
                config
            )
        else:
            # Original detection method
            return self._basic_face_detection(frame)
    
    def _basic_face_detection(self, frame: np.ndarray) -> Tuple[List, List, List]:
        """Basic face detection without optimization"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=config.DETECTION_SCALE_FACTOR, fy=config.DETECTION_SCALE_FACTOR)
        
        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame, model=config.FACE_DETECTION_MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        face_categories = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding, 
                tolerance=config.RECOGNITION_TOLERANCE
            )
            
            name = "Unknown"
            category = config.PERSON_CATEGORIES['UNKNOWN']
            
            # Calculate face distances
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index] and face_distances[best_match_index] < config.MAX_FACE_DISTANCE:
                    name = self.known_face_names[best_match_index]
                    category = self.known_face_categories[best_match_index]
            
            face_names.append(name)
            face_categories.append(category)
        
        # Scale back up face locations
        scaled_face_locations = []
        scale_factor = 1 / config.DETECTION_SCALE_FACTOR
        for (top, right, bottom, left) in face_locations:
            scaled_face_locations.append((
                int(top * scale_factor),
                int(right * scale_factor),
                int(bottom * scale_factor),
                int(left * scale_factor)
            ))
        
        return scaled_face_locations, face_names, face_categories
    
    def draw_detection_results(self, frame: np.ndarray, face_locations: List, face_names: List, face_categories: List) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for (top, right, bottom, left), name, category in zip(face_locations, face_names, face_categories):
            # Choose color based on category
            color = config.COLORS.get(category, config.COLORS['unknown'])
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw label text
            font = cv2.FONT_HERSHEY_DUPLEX
            label = f"{name} ({category.upper()})"
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def handle_alerts(self, face_names: List, face_categories: List):
        """Handle alerts for detected restricted/criminal faces"""
        current_time = time.time()
        
        for name, category in zip(face_names, face_categories):
            if category in [config.PERSON_CATEGORIES['RESTRICTED'], config.PERSON_CATEGORIES['CRIMINAL']]:
                # Check if enough time has passed since last alert
                if (name not in self.last_alert_time or 
                    current_time - self.last_alert_time[name] > config.ALERT_COOLDOWN):
                    
                    self.trigger_alert(name, category)
                    self.last_alert_time[name] = current_time
    
    def trigger_alert(self, name: str, category: str):
        """Trigger alert for detected person"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_message = f"ALERT: {category.upper()} person detected - {name} at {timestamp}"
        
        logging.warning(alert_message)
        print(f"\n🚨 {alert_message}")
        
        # Here you can add additional alert mechanisms:
        # - Send email notifications
        # - Send to telegram bot
        # - Save alert to database
        # - Trigger alarm sound
    
    def process_webcam(self, show_performance: bool = True):
        """Process real-time webcam feed with performance monitoring"""
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        if not cap.isOpened():
            logging.error("Cannot open camera")
            return
        
        logging.info("Starting webcam processing. Press 'q' to quit, 'p' for performance report, 'r' to reload faces.")
        
        while True:
            if self.performance_enabled:
                self.monitor.start_frame_timer()
                self.monitor.update_system_metrics()
            
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera")
                break
            
            self.frame_count += 1
            
            # Process every nth frame for performance
            if self.frame_count % config.FRAME_SKIP == 0:
                # Detect faces every nth processed frame
                if self.detection_count % config.DETECTION_FREQUENCY == 0:
                    face_locations, face_names, face_categories = self.detect_faces_in_frame(frame)
                    
                    # Handle alerts
                    self.handle_alerts(face_names, face_categories)
                    
                    # Store results for drawing
                    self.current_detections = (face_locations, face_names, face_categories)
                
                self.detection_count += 1
                
                # Draw detection results if available
                if hasattr(self, 'current_detections'):
                    frame = self.draw_detection_results(frame, *self.current_detections)
            
            # Add status text
            status_text = f"Frames: {self.frame_count} | Known Faces: {len(self.known_face_encodings)}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add performance overlay if enabled
            if self.performance_enabled and show_performance:
                frame = self.monitor.draw_performance_overlay(frame)
                
                # Log performance warnings
                if self.frame_count % 100 == 0:  # Check every 100 frames
                    self.monitor.log_performance_warning()
            
            if self.performance_enabled:
                self.monitor.end_frame_timer()
            
            # Display frame
            cv2.imshow('Enhanced Crime Detection System - Face Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p') and self.performance_enabled:
                print(self.monitor.get_performance_report())
            elif key == ord('r'):
                print("Reloading known faces...")
                self.load_known_faces()
                print("Faces reloaded!")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.performance_enabled:
            print("\n" + self.monitor.get_performance_report())
        
        logging.info("Webcam processing stopped")
    
    def process_video(self, video_path: str, show_performance: bool = True):
        """Process video file with performance monitoring"""
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logging.info(f"Processing video: {video_path}")
        logging.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        while True:
            if self.performance_enabled:
                self.monitor.start_frame_timer()
                self.monitor.update_system_metrics()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Process every nth frame for performance
            if self.frame_count % config.FRAME_SKIP == 0:
                if self.detection_count % config.DETECTION_FREQUENCY == 0:
                    face_locations, face_names, face_categories = self.detect_faces_in_frame(frame)
                    
                    # Handle alerts
                    self.handle_alerts(face_names, face_categories)
                    
                    # Store results for drawing
                    self.current_detections = (face_locations, face_names, face_categories)
                
                self.detection_count += 1
                
                # Draw detection results if available
                if hasattr(self, 'current_detections'):
                    frame = self.draw_detection_results(frame, *self.current_detections)
            
            # Add status text
            progress = (self.frame_count / total_frames) * 100 if total_frames > 0 else 0
            status_text = f"Frame: {self.frame_count}/{total_frames} ({progress:.1f}%) | Known Faces: {len(self.known_face_encodings)}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add performance overlay if enabled
            if self.performance_enabled and show_performance:
                frame = self.monitor.draw_performance_overlay(frame)
            
            if self.performance_enabled:
                self.monitor.end_frame_timer()
            
            # Display frame
            cv2.imshow('Enhanced Crime Detection System - Video Processing', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to pause
                cv2.waitKey(0)
            elif key == ord('p') and self.performance_enabled:
                print(self.monitor.get_performance_report())
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.performance_enabled:
            print("\n" + self.monitor.get_performance_report())
        
        logging.info("Video processing completed")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Crime Detection System - Face Recognition')
    parser.add_argument('--mode', choices=['webcam', 'video'], default='webcam', 
                       help='Processing mode: webcam or video')
    parser.add_argument('--video', type=str, help='Path to video file (required for video mode)')
    parser.add_argument('--no-performance', action='store_true', 
                       help='Disable performance monitoring for maximum speed')
    parser.add_argument('--no-display-performance', action='store_true',
                       help='Disable performance overlay display')
    
    args = parser.parse_args()
    
    if args.mode == 'video' and not args.video:
        print("Error: Video file path required for video mode")
        return
    
    # Initialize system
    enable_performance = not args.no_performance
    detector = EnhancedFaceDetectionSystem(enable_performance_monitor=enable_performance)
    
    # Show performance overlay
    show_performance = enable_performance and not args.no_display_performance
    
    print("Enhanced Crime Detection System - Face Recognition")
    print("Controls:")
    print("  'q' - Quit")
    if enable_performance:
        print("  'p' - Show performance report")
    print("  'r' - Reload known faces")
    if args.mode == 'video':
        print("  'space' - Pause/Resume")
    print()
    
    # Start processing
    if args.mode == 'webcam':
        detector.process_webcam(show_performance=show_performance)
    elif args.mode == 'video':
        detector.process_video(args.video, show_performance=show_performance)


if __name__ == "__main__":
    main()
