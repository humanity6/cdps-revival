import cv2
import os
import time
import threading
from datetime import datetime, timedelta
import numpy as np
from config import ANPRConfig
from plate_recognition import PlateRecognitionEngine
from telegram_bot import TelegramAlertBot, run_async
from database import FaceDatabase  # We'll use the same database class
import logging
from collections import deque
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anpr_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ANPRDetectionSystem:
    def __init__(self):
        self.config = ANPRConfig()
        self.plate_engine = None
        self.telegram_bot = None
        self.database = FaceDatabase()  # Reusing the same database class
        self.camera = None
        self.running = False
        self.last_alert_times = {}  # Track last alert time for each red-listed plate
        self.frame_skip_counter = 0  # For frame skipping optimization
        self.performance_stats = {
            'frames_processed': 0,
            'plates_detected': 0,
            'red_alerts_sent': 0,
            'processing_times': deque(maxlen=30),
            'avg_fps': 0
        }
        
        # Create directories
        self.create_directories()
        
        # Initialize components
        self.initialize_components()
    
    def create_directories(self):
        """Create necessary directories."""
        os.makedirs('detected_plates', exist_ok=True)
        os.makedirs('red_alert_plates', exist_ok=True)
    
    def initialize_components(self):
        """Initialize all system components."""
        try:
            logger.info("🚀 Initializing ANPR Detection System...")
            
            # Initialize plate recognition engine
            logger.info("🔍 Initializing Plate Recognition Engine...")
            self.plate_engine = PlateRecognitionEngine()
            
            # Initialize Telegram bot
            logger.info("📱 Initializing Telegram Bot...")
            self.telegram_bot = TelegramAlertBot()
            
            logger.info("✅ All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def initialize_camera(self, camera_index=None):
        """Initialize camera capture with optimized settings."""
        try:
            camera_index = camera_index if camera_index is not None else self.config.camera.index
            logger.info(f"📹 Initializing camera {camera_index}...")
            
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise Exception(f"Could not open camera {camera_index}")
            
            # Set camera properties for optimal performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time processing
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Could not read from camera")
            
            logger.info("✅ Camera initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing camera: {e}")
            return False
    
    def save_plate_image(self, frame, bbox, plate_number, is_red_listed=False):
        """Save plate image for review."""
        try:
            # Extract plate region
            plate_img = self.plate_engine.extract_plate_image(frame, bbox)
            
            if plate_img.size == 0:
                return None
            
            # Generate filename
            timestamp = datetime.now()
            safe_plate = plate_number.replace(' ', '_').replace('-', '_')
            
            if is_red_listed:
                directory = 'red_alert_plates'
                filename = f"RED_ALERT_{safe_plate}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            else:
                directory = 'detected_plates'
                filename = f"{safe_plate}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            
            filepath = os.path.join(directory, filename)
            
            # Save image
            cv2.imwrite(filepath, plate_img)
            logger.info(f"💾 Saved plate image: {filename}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error saving plate image: {e}")
            return None
    
    def should_send_alert(self, plate_number):
        """Check if enough time has passed since last alert for this plate."""
        current_time = datetime.now()
        
        if plate_number not in self.last_alert_times:
            self.last_alert_times[plate_number] = current_time
            return True
        
        time_diff = current_time - self.last_alert_times[plate_number]
        if time_diff.total_seconds() >= self.config.alert.interval:
            self.last_alert_times[plate_number] = current_time
            return True
        
        return False
    
    def process_frame_optimized(self, frame):
        """Optimized frame processing for plate detection."""
        start_time = time.time()
        timestamp = datetime.now()
        
        # Skip frames for performance if configured
        self.frame_skip_counter += 1
        every_n_frames = getattr(getattr(self.config, 'performance', None), 'every_n_frames', 1)
        if self.frame_skip_counter % every_n_frames != 0:
            return frame, []
        
        # Detect plates
        detections = self.plate_engine.detect_plates_in_frame(frame)
        
        # Update performance stats
        self.performance_stats['frames_processed'] += 1
        self.performance_stats['plates_detected'] += len(detections)
        
        # Draw plate boxes
        annotated_frame = self.plate_engine.draw_plate_boxes(frame.copy(), detections)
        
        # Process each detected plate
        for detection in detections:
            self.process_plate_detection(detection, frame, timestamp)
        
        # Record processing time
        processing_time = time.time() - start_time
        self.performance_stats['processing_times'].append(processing_time)
        
        return annotated_frame, detections
    
    def process_plate_detection(self, detection, frame, timestamp):
        """Process a single plate detection."""
        try:
            plate_number = detection['plate_number']
            confidence = detection['confidence']
            bbox = detection['bbox']
            
            # Normalize plate number
            normalized_plate = self.plate_engine.normalize_plate_number(plate_number)
            
            # Check if plate is red-listed
            is_red_listed, alert_reason = self.database.is_plate_red_listed(normalized_plate)
            
            # Save plate image
            image_path = self.save_plate_image(frame, bbox, normalized_plate, is_red_listed)
            
            # Record detection in database
            detection_id = self.database.add_plate_detection(
                normalized_plate, is_red_listed, confidence, image_path
            )
            
            if is_red_listed:
                logger.warning(f"🚨 RED ALERT: Detected red-listed vehicle {normalized_plate} - {alert_reason}")
                
                # Send alert if enough time has passed
                if self.should_send_alert(normalized_plate):
                    self.send_red_alert_async(frame, normalized_plate, alert_reason, timestamp, detection_id)
                    self.performance_stats['red_alerts_sent'] += 1
            else:
                logger.info(f"✅ Normal vehicle detected: {normalized_plate} (conf: {confidence:.3f})")
                
        except Exception as e:
            logger.error(f"❌ Error processing plate detection: {e}")
    
    def send_red_alert_async(self, frame, plate_number, alert_reason, timestamp, detection_id):
        """Send red alert asynchronously."""
        try:
            # Create a thread to send the alert
            alert_thread = threading.Thread(
                target=self._send_red_alert_thread,
                args=(frame.copy(), plate_number, alert_reason, timestamp, detection_id)
            )
            alert_thread.daemon = True
            alert_thread.start()
            
        except Exception as e:
            logger.error(f"❌ Error starting alert thread: {e}")
    
    def _send_red_alert_thread(self, frame, plate_number, alert_reason, timestamp, detection_id):
        """Thread function to send red alerts."""
        try:
            # Send Telegram alert
            success = run_async(
                self.telegram_bot.send_red_plate_alert(
                    frame, plate_number, alert_reason, timestamp
                )
            )
            
            if success:
                # Mark alert as sent in database
                self.database.mark_plate_alert_sent(detection_id)
                logger.info(f"✅ Red alert sent successfully for plate {plate_number}")
            else:
                logger.error(f"❌ Failed to send red alert for plate {plate_number}")
                
        except Exception as e:
            logger.error(f"❌ Error in alert thread: {e}")
    
    def run_detection(self, show_video=True):
        """Run the main detection loop."""
        if not self.initialize_camera():
            logger.error("❌ Failed to initialize camera")
            return
        
        self.running = True
        logger.info("🎯 Starting ANPR detection...")
        
        # Send startup message
        try:
            run_async(self.telegram_bot.send_startup_message())
        except Exception as e:
            logger.warning(f"⚠️ Could not send startup message: {e}")
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("❌ Failed to read frame from camera")
                    break
                
                # Process frame
                annotated_frame, detections = self.process_frame_optimized(frame)
                
                # Add system info to frame
                info_frame = self.add_system_info_to_frame(annotated_frame, detections)
                
                # Show video if requested
                if show_video:
                    cv2.imshow('ANPR Detection System', info_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("🛑 Quit key pressed")
                        break
                    elif key == ord('s'):
                        # Send system status
                        self.send_system_status()
                
                # Update performance stats
                self.update_performance_stats()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in detection loop: {e}")
        finally:
            self.stop_detection()
    
    def add_system_info_to_frame(self, frame, detections):
        """Add system information overlay to the frame."""
        info_frame = frame.copy()
        
        # System info
        fps = self.performance_stats['avg_fps']
        plates_count = len(detections)
        red_alerts = self.performance_stats['red_alerts_sent']
        
        # Add text overlay
        cv2.putText(info_frame, f"ANPR System - FPS: {fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_frame, f"Plates: {plates_count} | Red Alerts: {red_alerts}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(info_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return info_frame
    
    def update_performance_stats(self):
        """Update performance statistics."""
        if self.performance_stats['processing_times']:
            avg_time = sum(self.performance_stats['processing_times']) / len(self.performance_stats['processing_times'])
            self.performance_stats['avg_fps'] = 1.0 / avg_time if avg_time > 0 else 0
    
    def send_system_status(self):
        """Send system status via Telegram."""
        try:
            stats = self.database.get_plate_detection_stats()
            stats.update(self.performance_stats)
            
            run_async(self.telegram_bot.send_plate_detection_status(stats))
            logger.info("📊 System status sent")
            
        except Exception as e:
            logger.error(f"❌ Error sending system status: {e}")
    
    def stop_detection(self):
        """Stop the detection system."""
        logger.info("🛑 Stopping ANPR detection system...")
        self.running = False
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        # Send shutdown message
        try:
            run_async(self.telegram_bot.send_shutdown_message())
        except Exception as e:
            logger.warning(f"⚠️ Could not send shutdown message: {e}")
        
        logger.info("✅ ANPR detection system stopped")
    
    def add_red_vehicle(self, plate_number, alert_reason="Suspicious Activity"):
        """Add a vehicle to the red alert list."""
        try:
            vehicle_id = self.database.add_red_alerted_vehicle(plate_number, alert_reason)
            if vehicle_id:
                logger.info(f"✅ Added red-listed vehicle: {plate_number}")
                return True
            else:
                logger.info(f"📝 Updated existing red-listed vehicle: {plate_number}")
                return True
        except Exception as e:
            logger.error(f"❌ Error adding red-listed vehicle: {e}")
            return False
    
    def remove_red_vehicle(self, plate_number):
        """Remove a vehicle from the red alert list."""
        try:
            self.database.remove_red_alerted_vehicle(plate_number)
            logger.info(f"✅ Removed red-listed vehicle: {plate_number}")
            return True
        except Exception as e:
            logger.error(f"❌ Error removing red-listed vehicle: {e}")
            return False
    
    def list_red_vehicles(self):
        """List all red-alerted vehicles."""
        try:
            vehicles = self.database.get_all_red_alerted_vehicles()
            return vehicles
        except Exception as e:
            logger.error(f"❌ Error listing red-listed vehicles: {e}")
            return []

def main():
    """Main function to run the ANPR detection system."""
    try:
        # Create and run the ANPR system
        anpr_system = ANPRDetectionSystem()
        
        # Add some sample red-listed vehicles (for demonstration)
        print("Adding sample red-listed vehicles...")
        anpr_system.add_red_vehicle("ABC123", "Stolen Vehicle")
        anpr_system.add_red_vehicle("XYZ789", "Wanted for Investigation")
        anpr_system.add_red_vehicle("DEF456", "Unpaid Fines")
        
        print("Red-listed vehicles:")
        for vehicle in anpr_system.list_red_vehicles():
            print(f"  - {vehicle[0]}: {vehicle[1]}")
        
        print("\nStarting ANPR detection system...")
        print("Press 'q' to quit, 's' to send system status")
        
        # Run detection
        anpr_system.run_detection(show_video=True)
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    except KeyboardInterrupt:
        logger.info("🛑 System interrupted by user")

if __name__ == "__main__":
    main()