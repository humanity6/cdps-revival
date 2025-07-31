"""
Global Configuration Management for Unified Backend API
"""
import os
import sys
from typing import Dict, Any, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from enum import Enum

# Import ANPRConfig from the ANPR module
try:
    # Add ANPR module to path
    anpr_path = os.path.join(os.path.dirname(__file__), '..', 'anpr')
    abs_anpr_path = os.path.abspath(anpr_path)
    if abs_anpr_path not in sys.path:
        sys.path.insert(0, abs_anpr_path)
    
    # Import using importlib to avoid circular import
    import importlib.util
    config_path = os.path.join(abs_anpr_path, "config.py")
    spec = importlib.util.spec_from_file_location("anpr_config", config_path)
    anpr_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anpr_config_module)
    ANPRConfig = anpr_config_module.ANPRConfig
    ANPR_CONFIG_AVAILABLE = True
except Exception as e:
    ANPRConfig = None
    ANPR_CONFIG_AVAILABLE = False

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class BackendConfig(BaseSettings):
    # API Settings
    api_title: str = "Real-Time Crime Detection API"
    api_version: str = "1.0.0"
    api_description: str = "Unified backend for crime detection modules"
    
    # Server Settings
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    debug: bool = Field(False, env="API_DEBUG")
    reload: bool = Field(False, env="API_RELOAD")
    
    # CORS Settings
    cors_origins: list = Field(default=["*"])
    cors_methods: list = Field(default=["*"])
    cors_headers: list = Field(default=["*"])
    
    # Logging
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    log_file: str = Field("backend.log", env="LOG_FILE")
    
    # File Upload Settings
    max_file_size: int = Field(16 * 1024 * 1024, env="MAX_FILE_SIZE")  # 16MB
    upload_dir: str = Field("uploads", env="UPLOAD_DIR")
    temp_dir: str = Field("temp", env="TEMP_DIR")
    
    # WebSocket Settings
    websocket_ping_interval: int = Field(30, env="WS_PING_INTERVAL")
    websocket_ping_timeout: int = Field(10, env="WS_PING_TIMEOUT")
    max_websocket_connections: int = Field(100, env="MAX_WS_CONNECTIONS")
    
    # Performance Settings
    max_workers: int = Field(4, env="MAX_WORKERS")
    request_timeout: int = Field(300, env="REQUEST_TIMEOUT")  # 5 minutes
    
    # Live Feed Settings
    default_camera_index: int = Field(0, env="DEFAULT_CAMERA_INDEX")
    default_camera_width: int = Field(640, env="DEFAULT_CAMERA_WIDTH")
    default_camera_height: int = Field(480, env="DEFAULT_CAMERA_HEIGHT")
    default_camera_fps: int = Field(30, env="DEFAULT_CAMERA_FPS")
    
    # Module Enable/Disable
    enable_anpr: bool = Field(True, env="ENABLE_ANPR")
    enable_face: bool = Field(True, env="ENABLE_FACE")
    enable_violence: bool = Field(True, env="ENABLE_VIOLENCE")
    enable_weapon: bool = Field(True, env="ENABLE_WEAPON")
    
    # Security (placeholder for future use)
    secret_key: str = Field("your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }

class ModuleConfig:
    """
    Configuration manager for individual detection modules
    """
    
    def __init__(self):
        self.configs = {
            'anpr': {
                'enabled': True,
                'min_confidence': 0.5,
                'alert_interval': 60,
                'camera_index': 0,
                'save_detections': True
            },
            'face': {
                'enabled': True,
                'recognition_tolerance': 0.5,
                'alert_cooldown': 15,
                'detection_scale_factor': 0.4,
                'save_detections': True
            },
            'violence': {
                'enabled': True,
                'confidence_threshold': 0.5,
                'model_path': 'violence/bensam02_model.h5',
                'save_detections': True
            },
            'weapon': {
                'enabled': True,
                'confidence_threshold': 0.5,
                'model_path': 'models/best.pt',
                'save_detections': True
            }
        }
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get configuration for a specific module"""
        return self.configs.get(module_name, {})
    
    def update_module_config(self, module_name: str, updates: Dict[str, Any]) -> bool:
        """Update configuration for a specific module"""
        if module_name in self.configs:
            self.configs[module_name].update(updates)
            return True
        return False
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all module configurations"""
        return self.configs.copy()
    
    def reset_module_config(self, module_name: str) -> bool:
        """Reset module configuration to defaults"""
        defaults = {
            'anpr': {
                'enabled': True,
                'min_confidence': 0.5,
                'alert_interval': 60,
                'camera_index': 0,
                'save_detections': True
            },
            'face': {
                'enabled': True,
                'recognition_tolerance': 0.5,
                'alert_cooldown': 15,
                'detection_scale_factor': 0.4,
                'save_detections': True
            },
            'violence': {
                'enabled': True,
                'confidence_threshold': 0.5,
                'model_path': 'violence/bensam02_model.h5',
                'save_detections': True
            },
            'weapon': {
                'enabled': True,
                'confidence_threshold': 0.5,
                'model_path': 'models/best.pt',
                'save_detections': True
            }
        }
        
        if module_name in defaults:
            self.configs[module_name] = defaults[module_name].copy()
            return True
        return False

# Global instances
backend_config = BackendConfig()
module_config = ModuleConfig()

# Utility functions
def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        backend_config.upload_dir,
        backend_config.temp_dir,
        "logs",
        "static",
        "detected_plates",
        "red_alert_plates",
        "face_detections"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_module_enabled_status() -> Dict[str, bool]:
    """Get enabled status for all modules"""
    return {
        'anpr': backend_config.enable_anpr and module_config.get_module_config('anpr').get('enabled', True),
        'face': backend_config.enable_face and module_config.get_module_config('face').get('enabled', True),
        'violence': backend_config.enable_violence and module_config.get_module_config('violence').get('enabled', True),
        'weapon': backend_config.enable_weapon and module_config.get_module_config('weapon').get('enabled', True)
    }

def update_module_status(module_name: str, enabled: bool) -> bool:
    """Update the enabled status of a module"""
    return module_config.update_module_config(module_name, {'enabled': enabled})

# Environment-based configuration loading
def load_config_from_env():
    """Load additional configuration from environment variables"""
    # This function can be extended to load more complex configurations
    # from environment variables or configuration files
    pass

# Initialize directories on import
ensure_directories()