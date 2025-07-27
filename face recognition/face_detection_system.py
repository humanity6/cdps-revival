import cv2
import face_recognition
import threading
import queue
import time
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging

from config import Config
from face_database import FaceDatabase
from utils import (
    PerformanceMonitor, FrameProcessor, AlertManager, 
    FrameDrawer, VideoCapture
)

class FaceDetectionSystem:
    def __init__(self):
        # Initialize components
        self.face_db = FaceDatabase()
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        self.frame_processor = FrameProcessor()
        self.frame_drawer = FrameDrawer()
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=Config.QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=Config.QUEUE_SIZE)
        self.running = False
        
        # Processing variables
        self.frame_count = 0
        self.skip_frames = Config.SKIP_FRAMES
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load face database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize and load face database"""
        stats = self.face_db.get_database_stats()
        if stats['total_faces'] == 0:
            self.logger.info("No faces found in database. Loading from directories...")
            restricted, criminal = self.face_db.load_all_known_faces()
            self.face_db.save_encodings()
            self.logger.info(f"Loaded {restricted} restricted and {criminal} criminal faces")
        else:
            self.logger.info(f"Database loaded: {stats}")
    
    def _capture_frames(self, video_capture: VideoCapture):
        """Thread function to capture frames"""
        while self.running:
            ret, frame = video_capture.read_frame()
            
            if not ret:
                self.logger.warning("Failed to read frame")
                break
            
            try:
                # Add frame to queue (non-blocking)
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Skip frame if queue is full
                continue
    
    def _process_frames(self):
        """Thread function to process frames for face detection"""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Skip frames for performance
                self.frame_count += 1
                if self.frame_count % (self.skip_frames + 1) != 0:
                    # Put frame back for display without processing
                    self.result_queue.put({
                        'frame': frame,
                        'faces': [],
                        'processed': False
                    }, block=False)
                    continue
                
                start_time = time.time()
                
                # Resize frame for faster processing
                small_frame = self.frame_processor.resize_frame(
                    frame, Config.FRAME_SCALE
                )
                
                # Convert BGR to RGB
                rgb_frame = self.frame_processor.convert_bgr_to_rgb(small_frame)
                
                # Find faces
                face_locations = face_recognition.face_locations(
                    rgb_frame, model=Config.FACE_MODEL
                )
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(
                    rgb_frame, face_locations
                )
                
                # Scale face locations back to original size
                scaled_locations = self.frame_processor.scale_coordinates(
                    face_locations, Config.FRAME_SCALE
                )
                
                # Recognize faces
                recognitions = self.face_db.recognize_faces(face_encodings)
                
                # Process results
                faces = []
                for i, (location, recognition) in enumerate(zip(scaled_locations, recognitions)):
                    face_data = {
                        'location': location,
                        'name': recognition['name'],
                        'category': recognition['category'],
                        'confidence': recognition['confidence']
                    }
                    faces.append(face_data)
                    
                    # Trigger alerts for restricted/criminal faces
                    if recognition['category'] in ['restricted', 'criminal']:
                        if recognition['confidence'] >= Config.DETECTION_CONFIDENCE:
                            self.alert_manager.trigger_alert(
                                recognition['name'],
                                recognition['category'],
                                recognition['confidence'],
                                location
                            )
                
                # Calculate processing time
                detection_time = time.time() - start_time
                self.performance_monitor.update_detection_time(detection_time)
                
                # Put result in queue
                self.result_queue.put({
                    'frame': frame,
                    'faces': faces,
                    'processed': True,
                    'detection_time': detection_time
                }, block=False)
                
            except queue.Empty:
                continue
            except queue.Full:
                continue
            except Exception as e:
                self.logger.error(f"Error in frame processing: {str(e)}")
    
    def _display_frames(self):
        """Thread function to display processed frames"""
        while self.running:
            try:
                # Get result from queue
                result = self.result_queue.get(timeout=1.0)
                
                frame = result['frame'].copy()
                
                # Draw faces if processed
                if result['processed'] and Config.DRAW_FACE_BOXES:
                    for face in result['faces']:
                        frame = self.frame_drawer.draw_face_box(
                            frame,
                            face['location'],
                            face['name'],
                            face['category'],
                            face['confidence']
                        )
                
                # Draw FPS
                current_fps = self.performance_monitor.get_fps()
                frame = self.frame_drawer.draw_fps(frame, current_fps)
                
                # Draw status
                status = f"Faces in DB: {len(self.face_db.known_encodings)}"
                if result['processed']:
                    status += f" | Detected: {len(result['faces'])}"
                frame = self.frame_drawer.draw_status(frame, status)
                
                # Update frame time for FPS calculation
                frame_time = time.time()
                self.performance_monitor.update_frame_time(0.033)  # Approximate
                
                # Display frame
                cv2.imshow('Crime Detection System', frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in frame display: {str(e)}")
    
    def run_webcam(self, camera_index: int = 0):
        """Run face detection on webcam feed"""
        self.logger.info(f"Starting webcam detection (camera {camera_index})")
        
        # Initialize video capture
        video_capture = VideoCapture(camera_index)
        if not video_capture.initialize():
            self.logger.error("Failed to initialize webcam")
            return False
        
        return self._run_detection(video_capture)
    
    def run_video_file(self, video_path: str):
        """Run face detection on video file"""
        self.logger.info(f"Starting video file detection: {video_path}")
        
        # Initialize video capture
        video_capture = VideoCapture(video_path)
        if not video_capture.initialize():
            self.logger.error(f"Failed to open video file: {video_path}")
            return False
        
        return self._run_detection(video_capture)
    
    def _run_detection(self, video_capture: VideoCapture) -> bool:
        """Main detection loop"""
        try:
            self.running = True
            
            # Start threads
            capture_thread = threading.Thread(
                target=self._capture_frames, 
                args=(video_capture,)
            )
            process_thread = threading.Thread(target=self._process_frames)
            display_thread = threading.Thread(target=self._display_frames)
            
            capture_thread.start()
            process_thread.start()
            display_thread.start()
            
            # Wait for threads to complete
            capture_thread.join()
            process_thread.join()
            display_thread.join()
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user")
            return True
        except Exception as e:
            self.logger.error(f"Error in detection: {str(e)}")
            return False
        finally:
            self.running = False
            video_capture.release()
            cv2.destroyAllWindows()
    
    def stop(self):
        """Stop the detection system"""
        self.running = False
    
    def add_face_to_database(self, image_path: str, name: str, category: str) -> bool:
        """Add a new face to the database"""
        success = self.face_db.add_face_from_image(image_path, name, category)
        if success:
            self.face_db.save_encodings()
        return success
    
    def rebuild_database(self) -> bool:
        """Rebuild the face database"""
        return self.face_db.rebuild_database()
    
    def get_system_stats(self) -> Dict:
        """Get system performance and database statistics"""
        db_stats = self.face_db.get_database_stats()
        
        return {
            'database': db_stats,
            'performance': {
                'current_fps': self.performance_monitor.get_fps(),
                'avg_detection_time': self.performance_monitor.get_avg_detection_time(),
                'frame_count': self.frame_count
            }
        }

def main():
    """Main function for testing the system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Face Detection System')
    parser.add_argument('--source', type=str, default='webcam',
                       help='Video source: "webcam" or path to video file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index for webcam mode')
    parser.add_argument('--rebuild-db', action='store_true',
                       help='Rebuild face database from image directories')
    
    args = parser.parse_args()
    
    # Initialize system
    detection_system = FaceDetectionSystem()
    
    # Rebuild database if requested
    if args.rebuild_db:
        print("Rebuilding face database...")
        detection_system.rebuild_database()
        print("Database rebuilt successfully")
    
    # Display system stats
    stats = detection_system.get_system_stats()
    print(f"System initialized with {stats['database']['total_faces']} known faces")
    print(f"({stats['database']['restricted_faces']} restricted, "
          f"{stats['database']['criminal_faces']} criminal)")
    
    try:
        if args.source == 'webcam':
            print("Starting webcam detection. Press 'q' to quit.")
            detection_system.run_webcam(args.camera)
        else:
            print(f"Starting video file detection: {args.source}")
            detection_system.run_video_file(args.source)
    except KeyboardInterrupt:
        print("\nStopping detection system...")
        detection_system.stop()

if __name__ == "__main__":
    main()