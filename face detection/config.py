"""
Face Recognition System Configuration
Enhanced settings for better performance and accuracy
"""

# Detection settings
DETECTION_SCALE_FACTOR = 0.4  # Scale down input for faster processing (reduced for better quality)
FACE_DETECTION_MODEL = 'hog'  # 'hog' for speed, 'cnn' for accuracy
RECOGNITION_TOLERANCE = 0.5   # Lower = more strict matching (reduced for better accuracy)
MAX_FACE_DISTANCE = 0.5      # Maximum distance for face matching (reduced for better accuracy)
USE_CNN_FALLBACK = True       # Use CNN method if HOG fails to find faces
MIN_FACE_SIZE = 50           # Minimum face size in pixels to consider

# Performance settings
FRAME_SKIP = 1               # Process every nth frame (reduced for better detection)
RESIZE_WIDTH = 640           # Resize frame width for processing
DETECTION_FREQUENCY = 3      # Detect faces every nth processed frame (reduced for more frequent detection)
MAX_DETECTION_THREADS = 2    # Maximum number of detection threads

# Alert settings
ALERT_COOLDOWN = 15          # Seconds between alerts for same person (reduced for more responsive alerts)
CONFIDENCE_THRESHOLD = 0.6   # Minimum confidence for detection (increased for better accuracy)
HIGH_PRIORITY_THRESHOLD = 0.8 # Threshold for high-priority alerts

# Enhanced detection settings
ENABLE_FACE_ENHANCEMENT = True  # Enable image enhancement for better detection
ENABLE_MULTI_SCALE = True      # Enable multi-scale detection
NUM_JITTERS = 1               # Number of times to re-sample face for encoding
UPSAMPLING_TIMES = 1          # Times to upsample image for detection

# File paths
KNOWN_FACES_DIR = "known_faces"
RESTRICTED_FACES_DIR = "restricted_faces"
CRIMINAL_FACES_DIR = "criminal_faces"
LOGS_DIR = "logs"
TEMP_DIR = "temp"

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
AUTO_CAMERA_SETTINGS = True   # Automatically adjust camera settings

# Enhanced colors for bounding boxes (BGR format)
COLORS = {
    'unknown': (0, 255, 255),      # Yellow
    'restricted': (0, 69, 255),    # Orange-Red  
    'criminal': (0, 0, 255),       # Bright Red
    'safe': (0, 255, 0),           # Green
    'high_priority': (0, 0, 139),  # Dark Red for high-priority threats
}

# Detection categories
PERSON_CATEGORIES = {
    'UNKNOWN': 'unknown',
    'RESTRICTED': 'restricted', 
    'CRIMINAL': 'criminal',
    'SAFE': 'safe',
    'HIGH_PRIORITY': 'high_priority'
}

# UI Settings
SHOW_FPS = True              # Show FPS counter
SHOW_DETECTION_COUNT = True  # Show detection statistics
SHOW_CONFIDENCE = True       # Show confidence scores
WINDOW_TITLE = "Enhanced Crime Detection System"

# Performance monitoring
ENABLE_PERFORMANCE_MONITORING = True
PERFORMANCE_LOG_INTERVAL = 30  # Log performance stats every N seconds

# Advanced features
ENABLE_FACE_TRACKING = True   # Enable face tracking between frames
TRACKING_MAX_DISTANCE = 100   # Maximum distance for face tracking
ENABLE_MOTION_DETECTION = False  # Enable motion-based optimization
