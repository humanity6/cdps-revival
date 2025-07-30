"""
ANPR System Configuration
This module contains all configuration settings for the Enhanced ANPR System.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class CameraConfig:
    """Camera configuration settings."""
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1

@dataclass
class DetectionConfig:
    """Detection configuration settings."""
    min_confidence: float = 0.5
    min_plate_width: int = 50
    min_plate_height: int = 20
    max_plate_width: int = 400
    max_plate_height: int = 200
    
    # Image preprocessing
    enable_clahe: bool = True
    enable_sharpening: bool = True
    enable_noise_reduction: bool = True
    
    # Validation settings
    min_plate_length: int = 3
    max_plate_length: int = 12
    validation_region: str = 'generic'  # 'indian', 'us', 'uk', 'generic'

@dataclass
class AlertConfig:
    """Alert configuration settings."""
    interval: int = 60  # seconds between alerts for the same plate

@dataclass
class StorageConfig:
    """File storage configuration settings."""
    detected_plates_dir: str = "detected_plates"
    red_alert_plates_dir: str = "red_alert_plates"
    uploads_dir: str = "web_frontend/uploads"
    
    # File management
    max_file_age_days: int = 30
    auto_cleanup: bool = True
    save_annotated_images: bool = True
    image_format: str = "jpg"
    image_quality: int = 95

@dataclass
class TelegramConfig:
    """Telegram bot configuration settings."""
    enabled: bool = False
    bot_token: str = "7560595961:AAG5sAVZv4QaMVdjqx0KFJQqTVZVHtuvQ5E"
    chat_id: str = "6639956728"
    send_images: bool = True
    send_notifications: bool = True

@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""
    max_detection_queue_size: int = 10
    processing_threads: int = 2
    enable_gpu: bool = False
    batch_processing: bool = False
    
    # Monitoring
    enable_performance_monitoring: bool = True
    stats_update_interval: int = 10  # seconds
    max_stats_samples: int = 1000

@dataclass
class WebConfig:
    """Web interface configuration settings."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True
    threaded: bool = True
    upload_folder: str = "uploads"
    max_file_size_mb: int = 16
    allowed_extensions: List[str] = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']

class ANPRConfig:
    """Main ANPR system configuration."""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration with default values."""
        # Core configuration objects
        self.camera = CameraConfig()
        self.detection = DetectionConfig()
        self.alert = AlertConfig()
        self.storage = StorageConfig()
        self.telegram = TelegramConfig()
        self.performance = PerformanceConfig()
        self.web = WebConfig()
        
        # System settings
        self.system_name = "Enhanced ANPR System"
        self.version = "2.0.0"
        self.enable_logging = True
        self.log_level = "INFO"
        self.log_file = "anpr_system.log"
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.storage.detected_plates_dir,
            self.storage.red_alert_plates_dir,
            self.storage.uploads_dir
        ]
        
        for directory in directories:
            if directory and directory != '.':
                os.makedirs(directory, exist_ok=True)
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration objects
            for section, values in config_data.items():
                if hasattr(self, section) and isinstance(getattr(self, section), object):
                    for key, value in values.items():
                        if hasattr(getattr(self, section), key):
                            setattr(getattr(self, section), key, value)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file."""
        try:
            import json
            from dataclasses import asdict
            
            config_data = {
                'camera': asdict(self.camera),
                'detection': asdict(self.detection),
                'alert': asdict(self.alert),
                'storage': asdict(self.storage),
                'telegram': asdict(self.telegram),
                'performance': asdict(self.performance),
                'web': asdict(self.web)
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving config file {config_file}: {e}")
    
    def get_summary(self) -> Dict:
        """Get configuration summary."""
        return {
            'system_name': self.system_name,
            'version': self.version,
            'camera_enabled': True,
            'detection_min_confidence': self.detection.min_confidence,
            'alert_interval': self.alert.interval,
            'telegram_enabled': self.telegram.enabled,
            'web_interface_enabled': True,
            'storage_directories': {
                'detected_plates': self.storage.detected_plates_dir,
                'red_alerts': self.storage.red_alert_plates_dir
            }
        }

# Default configuration instance
default_config = ANPRConfig()

# Environment variable overrides
def load_env_config():
    """Load configuration from environment variables."""
    config = ANPRConfig()
    
    # Camera settings
    if os.getenv('ANPR_CAMERA_INDEX'):
        config.camera.index = int(os.getenv('ANPR_CAMERA_INDEX'))
    if os.getenv('ANPR_CAMERA_WIDTH'):
        config.camera.width = int(os.getenv('ANPR_CAMERA_WIDTH'))
    if os.getenv('ANPR_CAMERA_HEIGHT'):
        config.camera.height = int(os.getenv('ANPR_CAMERA_HEIGHT'))
    
    # Detection settings
    if os.getenv('ANPR_MIN_CONFIDENCE'):
        config.detection.min_confidence = float(os.getenv('ANPR_MIN_CONFIDENCE'))
    if os.getenv('ANPR_VALIDATION_REGION'):
        config.detection.validation_region = os.getenv('ANPR_VALIDATION_REGION')
    
    # Alert settings
    if os.getenv('ANPR_ALERT_INTERVAL'):
        config.alert.interval = int(os.getenv('ANPR_ALERT_INTERVAL'))
    
    # Telegram settings
    if os.getenv('ANPR_TELEGRAM_TOKEN'):
        config.telegram.bot_token = os.getenv('ANPR_TELEGRAM_TOKEN')
        config.telegram.enabled = True
    if os.getenv('ANPR_TELEGRAM_CHAT_ID'):
        config.telegram.chat_id = os.getenv('ANPR_TELEGRAM_CHAT_ID')
    
    # Web settings
    if os.getenv('ANPR_WEB_HOST'):
        config.web.host = os.getenv('ANPR_WEB_HOST')
    if os.getenv('ANPR_WEB_PORT'):
        config.web.port = int(os.getenv('ANPR_WEB_PORT'))
    if os.getenv('ANPR_WEB_DEBUG'):
        config.web.debug = os.getenv('ANPR_WEB_DEBUG').lower() == 'true'
    
    return config

# Configuration validation
def validate_config(config: ANPRConfig) -> Tuple[bool, List[str]]:
    """Validate configuration settings."""
    errors = []
    
    # Validate camera settings
    if config.camera.index < 0:
        errors.append("Camera index must be non-negative")
    if config.camera.width <= 0 or config.camera.height <= 0:
        errors.append("Camera dimensions must be positive")
    
    # Validate detection settings
    if not (0.0 <= config.detection.min_confidence <= 1.0):
        errors.append("Detection confidence must be between 0.0 and 1.0")
    
    # Validate alert settings
    if config.alert.interval < 0:
        errors.append("Alert interval must be non-negative")
    
    # Validate telegram settings
    if config.telegram.enabled and not config.telegram.bot_token:
        errors.append("Telegram bot token is required when Telegram is enabled")
    
    # Validate web settings
    if not (1 <= config.web.port <= 65535):
        errors.append("Web port must be between 1 and 65535")
    if config.web.max_file_size_mb <= 0:
        errors.append("Max file size must be positive")
    
    return len(errors) == 0, errors