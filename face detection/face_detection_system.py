"""
Face Detection and Recognition System
Real-time face detection for identifying restricted/criminal suspects
"""

import cv2
import face_recognition
import numpy as np
import os
import time
import math
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import config

class FaceDetectionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_categories = []
        self.last_alert_time = {}
        self.frame_count = 0
        self.detection_count = 0
        
        # Setup logging
        self.setup_logging()
        
        # Create necessary directories
        self.create_directories()
        
        # Load known faces
        self.load_known_faces()
        
        logging.info("Face Detection System initialized")
    
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
        """Load faces from a specific directory with enhanced validation"""
        if not os.path.exists(directory):
            logging.warning(f"Directory does not exist: {directory}")
            return
        
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        loaded_count = 0
        failed_count = 0
        
        for filename in os.listdir(directory):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(directory, filename)
                try:
                    # Load and validate image
                    image = face_recognition.load_image_file(image_path)
                    
                    # Check image dimensions
                    if image.shape[0] < config.MIN_FACE_SIZE or image.shape[1] < config.MIN_FACE_SIZE:
                        logging.warning(f"Image too small: {image_path} ({image.shape[0]}x{image.shape[1]})")
                        failed_count += 1
                        continue
                    
                    # Get face encodings with enhanced settings
                    encodings = face_recognition.face_encodings(
                        image, 
                        num_jitters=config.NUM_JITTERS,
                        model='large' if hasattr(config, 'USE_LARGE_MODEL') and config.USE_LARGE_MODEL else 'small'
                    )
                    
                    if encodings:
                        # Use the first face found (or best quality face if multiple)
                        if len(encodings) > 1:
                            logging.info(f"Multiple faces found in {filename}, using first detection")
                        
                        encoding = encodings[0]
                        name = os.path.splitext(filename)[0]
                        
                        # Validate encoding quality
                        if self._validate_face_encoding(encoding, image_path):
                            self.known_face_encodings.append(encoding)
                            self.known_face_names.append(name)
                            self.known_face_categories.append(category)
                            loaded_count += 1
                            logging.info(f"Loaded {category} face: {name}")
                        else:
                            failed_count += 1
                            logging.warning(f"Poor quality encoding rejected: {image_path}")
                    else:
                        failed_count += 1
                        logging.warning(f"No face found in {image_path}")
                        
                except Exception as e:
                    failed_count += 1
                    logging.error(f"Error loading {image_path}: {str(e)}")
        
        logging.info(f"Directory {directory}: {loaded_count} faces loaded, {failed_count} failed")

    def _validate_face_encoding(self, encoding: np.ndarray, image_path: str) -> bool:
        """Validate face encoding quality"""
        try:
            # Check if encoding is valid (not all zeros or NaN)
            if np.all(encoding == 0) or np.any(np.isnan(encoding)):
                return False
            
            # Check encoding variance (too low variance might indicate poor quality)
            if np.var(encoding) < 0.001:
                logging.warning(f"Low variance encoding detected: {image_path}")
                return False
            
            return True
        except Exception as e:
            logging.error(f"Error validating encoding: {e}")
            return False
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> Tuple[List, List, List]:
        """Detect and recognize faces in a frame with enhanced accuracy"""
        try:
            # Enhance image quality for better detection
            enhanced_frame = self._enhance_frame_quality(frame)
            
            # Resize frame for faster processing
            small_frame = cv2.resize(enhanced_frame, (0, 0), fx=config.DETECTION_SCALE_FACTOR, fy=config.DETECTION_SCALE_FACTOR)
            
            # Convert BGR to RGB
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations with multiple detection methods for better accuracy
            face_locations = self._detect_faces_multi_method(rgb_small_frame)
            
            if not face_locations:
                return [], [], []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
            
            face_names = []
            face_categories = []
            face_confidences = []
            
            for i, face_encoding in enumerate(face_encodings):
                name, category, confidence = self._identify_face(face_encoding)
                face_names.append(name)
                face_categories.append(category)
                face_confidences.append(confidence)
            
            # Store confidence scores for drawing
            self.last_confidences = face_confidences
            
        except Exception as e:
            logging.error(f"Error in face detection: {str(e)}")
            return [], [], []
        
        # Scale back up face locations with improved precision
        scaled_face_locations = []
        scale_factor = 1 / config.DETECTION_SCALE_FACTOR
        for (top, right, bottom, left) in face_locations:
            # Add some padding for better bounding boxes
            padding = int(5 * scale_factor)
            scaled_face_locations.append((
                max(0, int(top * scale_factor) - padding),
                min(frame.shape[1], int(right * scale_factor) + padding),
                min(frame.shape[0], int(bottom * scale_factor) + padding),
                max(0, int(left * scale_factor) - padding)
            ))
        
        return scaled_face_locations, face_names, face_categories

    def _enhance_frame_quality(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame quality for better face detection"""
        try:
            # Apply histogram equalization to improve contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Convert back to BGR
            enhanced_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            
            # Blend with original for natural look
            enhanced_frame = cv2.addWeighted(frame, 0.7, enhanced_frame, 0.3, 0)
            
            return enhanced_frame
        except Exception as e:
            logging.warning(f"Frame enhancement failed: {e}")
            return frame

    def _detect_faces_multi_method(self, rgb_frame: np.ndarray) -> List:
        """Use multiple detection methods for better accuracy"""
        face_locations = []
        
        try:
            # Primary method: HOG (fast and reliable)
            locations_hog = face_recognition.face_locations(rgb_frame, model='hog', number_of_times_to_upsample=1)
            face_locations.extend(locations_hog)
            
            # If no faces found with HOG, try CNN method (more accurate but slower)
            if not face_locations and hasattr(config, 'USE_CNN_FALLBACK') and config.USE_CNN_FALLBACK:
                locations_cnn = face_recognition.face_locations(rgb_frame, model='cnn', number_of_times_to_upsample=0)
                face_locations.extend(locations_cnn)
            
            # Remove duplicate detections
            face_locations = self._remove_duplicate_faces(face_locations)
            
        except Exception as e:
            logging.error(f"Multi-method detection failed: {e}")
            # Fallback to basic detection
            try:
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            except Exception as fallback_e:
                logging.error(f"Fallback detection failed: {fallback_e}")
                face_locations = []
        
        return face_locations

    def _remove_duplicate_faces(self, face_locations: List) -> List:
        """Remove duplicate face detections based on overlap"""
        if len(face_locations) <= 1:
            return face_locations
        
        unique_faces = []
        
        for face in face_locations:
            top, right, bottom, left = face
            is_duplicate = False
            
            for existing_face in unique_faces:
                existing_top, existing_right, existing_bottom, existing_left = existing_face
                
                # Calculate overlap
                overlap_area = max(0, min(right, existing_right) - max(left, existing_left)) * \
                              max(0, min(bottom, existing_bottom) - max(top, existing_top))
                
                face_area = (right - left) * (bottom - top)
                existing_area = (existing_right - existing_left) * (existing_bottom - existing_top)
                
                if overlap_area > 0.3 * min(face_area, existing_area):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces

    def _identify_face(self, face_encoding: np.ndarray) -> Tuple[str, str, float]:
        """Identify a face with confidence scoring"""
        if len(self.known_face_encodings) == 0:
            return "Unknown", config.PERSON_CATEGORIES['UNKNOWN'], 0.0
        
        try:
            # Calculate face distances
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) == 0:
                return "Unknown", config.PERSON_CATEGORIES['UNKNOWN'], 0.0
            
            # Find best match
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            # Calculate confidence (inverse of distance)
            confidence = max(0.0, 1.0 - best_distance)
            
            # Check if the match is good enough
            if best_distance < config.MAX_FACE_DISTANCE:
                name = self.known_face_names[best_match_index]
                category = self.known_face_categories[best_match_index]
                return name, category, confidence
            else:
                return "Unknown", config.PERSON_CATEGORIES['UNKNOWN'], confidence
                
        except Exception as e:
            logging.error(f"Face identification failed: {e}")
            return "Unknown", config.PERSON_CATEGORIES['UNKNOWN'], 0.0
    
    def draw_detection_results(self, frame: np.ndarray, face_locations: List, face_names: List, face_categories: List) -> np.ndarray:
        """Draw enhanced bounding boxes and labels on frame"""
        try:
            confidences = getattr(self, 'last_confidences', [1.0] * len(face_locations))
            
            for i, ((top, right, bottom, left), name, category) in enumerate(zip(face_locations, face_names, face_categories)):
                confidence = confidences[i] if i < len(confidences) else 1.0
                
                # Choose color and thickness based on category
                color = config.COLORS.get(category, config.COLORS['unknown'])
                thickness = 3 if category in ['restricted', 'criminal'] else 2
                
                # Draw main rectangle around face
                cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
                
                # Draw corner markers for better visibility
                corner_length = 15
                corner_thickness = 3
                
                # Top-left corner
                cv2.line(frame, (left, top), (left + corner_length, top), color, corner_thickness)
                cv2.line(frame, (left, top), (left, top + corner_length), color, corner_thickness)
                
                # Top-right corner
                cv2.line(frame, (right, top), (right - corner_length, top), color, corner_thickness)
                cv2.line(frame, (right, top), (right, top + corner_length), color, corner_thickness)
                
                # Bottom-left corner
                cv2.line(frame, (left, bottom), (left + corner_length, bottom), color, corner_thickness)
                cv2.line(frame, (left, bottom), (left, bottom - corner_length), color, corner_thickness)
                
                # Bottom-right corner
                cv2.line(frame, (right, bottom), (right - corner_length, bottom), color, corner_thickness)
                cv2.line(frame, (right, bottom), (right, bottom - corner_length), color, corner_thickness)
                
                # Calculate label dimensions
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                label_thickness = 1
                
                # Create multi-line label
                label_lines = [
                    f"{name}",
                    f"{category.upper()}",
                    f"Conf: {confidence:.2f}"
                ]
                
                # Calculate text sizes
                text_sizes = [cv2.getTextSize(line, font, font_scale, label_thickness)[0] for line in label_lines]
                max_text_width = max(size[0] for size in text_sizes)
                line_height = max(size[1] for size in text_sizes) + 5
                
                # Draw label background with rounded corners effect
                label_height = len(label_lines) * line_height + 10
                label_top = bottom + 5
                
                # Main background rectangle
                cv2.rectangle(frame, (left, label_top), (left + max_text_width + 12, label_top + label_height), color, cv2.FILLED)
                
                # Semi-transparent overlay for better readability
                overlay = frame.copy()
                cv2.rectangle(overlay, (left, label_top), (left + max_text_width + 12, label_top + label_height), (0, 0, 0), cv2.FILLED)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.rectangle(frame, (left, label_top), (left + max_text_width + 12, label_top + label_height), color, cv2.FILLED)
                
                # Draw text lines
                for j, line in enumerate(label_lines):
                    text_color = (255, 255, 255) if category != 'unknown' else (0, 0, 0)
                    y_position = label_top + 15 + j * line_height
                    cv2.putText(frame, line, (left + 6, y_position), font, font_scale, text_color, label_thickness)
                
                # Add alert indicator for restricted/criminal persons
                if category in ['restricted', 'criminal']:
                    # Draw pulsing alert circle
                    alert_radius = 8 + int(3 * abs(np.sin(time.time() * 3)))  # Pulsing effect
                    cv2.circle(frame, (right - 15, top + 15), alert_radius, (0, 0, 255), -1)
                    cv2.circle(frame, (right - 15, top + 15), alert_radius + 2, (255, 255, 255), 2)
                    cv2.putText(frame, "!", (right - 19, top + 20), font, 0.8, (255, 255, 255), 2)
            
        except Exception as e:
            logging.error(f"Error drawing detection results: {e}")
        
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
    
    def process_webcam(self):
        """Process real-time webcam feed with enhanced features"""
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        # Configure camera settings
        if config.AUTO_CAMERA_SETTINGS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for real-time processing
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure for consistent lighting
        
        if not cap.isOpened():
            logging.error("Cannot open camera")
            return
        
        # Initialize performance tracking
        fps_counter = 0
        fps_start_time = time.time()
        last_detection_time = 0
        
        # Initialize face tracking
        if config.ENABLE_FACE_TRACKING:
            self.tracked_faces = []
        
        logging.info("Starting webcam processing. Press 'q' to quit, 'r' to reload faces, 'p' to pause.")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        logging.error("Failed to read frame from camera")
                        break
                    
                    self.frame_count += 1
                    fps_counter += 1
                    
                    # Calculate and display FPS
                    current_time = time.time()
                    if current_time - fps_start_time >= 1.0:
                        fps = fps_counter / (current_time - fps_start_time)
                        fps_start_time = current_time
                        fps_counter = 0
                    else:
                        fps = 0
                    
                    # Process every nth frame for performance
                    if self.frame_count % config.FRAME_SKIP == 0:
                        # Detect faces every nth processed frame
                        if (self.detection_count % config.DETECTION_FREQUENCY == 0 or 
                            current_time - last_detection_time > 0.5):  # Force detection every 0.5 seconds
                            
                            face_locations, face_names, face_categories = self.detect_faces_in_frame(frame)
                            
                            # Handle alerts
                            if face_locations:
                                self.handle_alerts(face_names, face_categories)
                            
                            # Store results for drawing
                            self.current_detections = (face_locations, face_names, face_categories)
                            last_detection_time = current_time
                        
                        self.detection_count += 1
                        
                        # Draw detection results if available
                        if hasattr(self, 'current_detections'):
                            frame = self.draw_detection_results(frame, *self.current_detections)
                    
                    # Add enhanced status information
                    self._draw_status_info(frame, fps)
                    
                    # Display frame
                    cv2.imshow(config.WINDOW_TITLE, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    logging.info("Reloading known faces...")
                    self.reload_known_faces()
                elif key == ord('p'):
                    paused = not paused
                    logging.info(f"{'Paused' if paused else 'Resumed'} processing")
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{config.TEMP_DIR}/frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logging.info(f"Frame saved: {filename}")
                    
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt")
        except Exception as e:
            logging.error(f"Error in webcam processing: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Webcam processing stopped")

    def _draw_status_info(self, frame: np.ndarray, fps: float):
        """Draw status information on frame"""
        try:
            # Status background
            status_height = 80
            cv2.rectangle(frame, (0, 0), (frame.shape[1], status_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], status_height), (50, 50, 50), 2)
            
            # Status text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 255, 0)
            
            status_lines = [
                f"FPS: {fps:.1f} | Frames: {self.frame_count} | Known Faces: {len(self.known_face_encodings)}",
                f"Detections: {self.detection_count} | Scale: {config.DETECTION_SCALE_FACTOR} | Model: {config.FACE_DETECTION_MODEL.upper()}",
                f"Press: Q=Quit, R=Reload Faces, P=Pause, S=Save Frame"
            ]
            
            for i, line in enumerate(status_lines):
                y_pos = 20 + i * 20
                cv2.putText(frame, line, (10, y_pos), font, font_scale, color, 1)
                
        except Exception as e:
            logging.warning(f"Error drawing status info: {e}")
    
    def process_video(self, video_path: str):
        """Process video file"""
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            return
        
        logging.info(f"Processing video: {video_path}")
        
        while True:
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
            status_text = f"Frame: {self.frame_count} | Known Faces: {len(self.known_face_encodings)}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Crime Detection System - Video Processing', frame)
            
            # Break on 'q' key press or space to pause
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to pause
                cv2.waitKey(0)
        
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Video processing completed")
    
    def add_known_face(self, image_path: str, name: str, category: str):
        """Add a new known face to the system"""
        if category not in config.PERSON_CATEGORIES.values():
            logging.error(f"Invalid category: {category}")
            return False
        
        try:
            # Load and encode the image
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                logging.error(f"No face found in {image_path}")
                return False
            
            # Add to known faces
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(name)
            self.known_face_categories.append(category)
            
            logging.info(f"Added {category} face: {name}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding face {name}: {str(e)}")
            return False
    
    def reload_known_faces(self):
        """Reload all known faces from directories"""
        logging.info("Reloading known faces...")
        self.load_known_faces()
        self.last_alert_time.clear()  # Clear alert history


if __name__ == "__main__":
    # Initialize the face detection system
    detector = FaceDetectionSystem()
    
    # Start processing
    print("Crime Detection System - Face Recognition")
    print("1. Press 'w' for webcam")
    print("2. Press 'v' for video file")
    print("3. Press 'q' to quit")
    
    choice = input("Enter your choice: ").lower()
    
    if choice == 'w':
        detector.process_webcam()
    elif choice == 'v':
        video_path = input("Enter video file path: ")
        detector.process_video(video_path)
    else:
        print("Exiting...")
