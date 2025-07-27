import os

class Config:
    # Performance Settings
    FRAME_SCALE = 0.5  # Scale frames to improve performance
    SKIP_FRAMES = 2    # Process every Nth frame
    DETECTION_CONFIDENCE = 0.6  # Minimum confidence for face matches
    
    # Threading Settings
    MAX_WORKERS = 4    # Maximum worker threads
    QUEUE_SIZE = 30    # Frame queue size
    
    # Video Settings
    WEBCAM_INDEX = 0   # Default webcam index
    VIDEO_WIDTH = 640  # Video capture width
    VIDEO_HEIGHT = 480 # Video capture height
    TARGET_FPS = 30    # Target frames per second
    
    # Face Detection Settings
    FACE_MODEL = 'hog'  # 'hog' for speed, 'cnn' for accuracy
    FACE_TOLERANCE = 0.6  # Lower = more strict matching
    
    # Directory Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'known_faces')
    RESTRICTED_FACES_DIR = os.path.join(KNOWN_FACES_DIR, 'restricted')
    CRIMINAL_FACES_DIR = os.path.join(KNOWN_FACES_DIR, 'criminals')
    TEST_VIDEOS_DIR = os.path.join(BASE_DIR, 'test_videos')
    
    # Alert Settings
    ALERT_SOUND = True
    ALERT_LOG = True
    LOG_FILE = os.path.join(BASE_DIR, 'detections.log')
    
    # Display Settings
    SHOW_FPS = True
    SHOW_CONFIDENCE = True
    DRAW_FACE_BOXES = True
    
    # Database Settings
    FACE_ENCODINGS_FILE = os.path.join(BASE_DIR, 'face_encodings.pkl')
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.KNOWN_FACES_DIR,
            cls.RESTRICTED_FACES_DIR,
            cls.CRIMINAL_FACES_DIR,
            cls.TEST_VIDEOS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)