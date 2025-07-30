# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive Real-Time Crime Detection and Prevention System that implements multiple computer vision and machine learning modules for security applications. The system includes:

- **ANPR (Automatic Number Plate Recognition)** - License plate detection and red-listed vehicle tracking
- **Face Detection** - Facial recognition for unauthorized access and criminal detection
- **Violence Detection** - Real-time violence detection using deep learning models
- **Weapon Detection** - YOLO-based weapon detection system

## Architecture

The system uses a **modular architecture** where each detection component operates independently:

- Each module has its own Flask web interface for testing
- Shared configuration systems with dataclass-based configs
- In-memory tracking for red-listed items (no database dependencies)
- Telegram bot integration for real-time alerts
- Performance monitoring and optimization features

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# Windows:
myenv\Scripts\activate
# Linux/Mac:
source myenv/bin/activate

# Install dependencies for each module
pip install -r "face detection/requirements.txt"
pip install -r "violence detection cdps/requirements.txt" 
pip install -r "weapon/requirements.txt"
```

### Running Individual Modules

**ANPR System:**
```bash
cd anpr
python anpr_system.py
```

**Face Detection System:**
```bash
cd "face detection"
python main.py --mode webcam  # For webcam
python main.py --mode video --video path/to/video.mp4  # For video file
```

**Violence Detection:**
```bash
cd "violence detection cdps"
python flask_violence_detector.py
# Access web interface at http://localhost:5000
```

**Weapon Detection:**
```bash
cd weapon
python app.py
# Access web interface at http://localhost:5000
```

### Testing Commands
```bash
# Face detection system testing
cd "face detection"
python test_system.py

# Face detection setup
python setup_system.py
```

## Key Configuration Files

- `anpr/config.py` - Comprehensive ANPR system configuration with dataclasses
- `face detection/config.py` - Face recognition settings and performance tuning  
- `weapon/config.yaml` - YOLO model configuration for weapon detection

## Module-Specific Architecture

### ANPR System (anpr/)
- **Main**: `anpr_system.py` - Core detection system with camera integration
- **Config**: `config.py` - Dataclass-based configuration system
- **Recognition**: `plate_recognition.py` - Enhanced plate recognition engine
- **Alerts**: `telegram_bot.py` - Telegram integration for alerts
- **Database**: `database.py` - Database operations (currently deprecated in favor of in-memory tracking)

### Face Detection System (face detection/)
- **Main**: `main.py` - Enhanced face detection with performance monitoring
- **Utils**: `face_utils.py` - Face processing utilities
- **Performance**: `performance_monitor.py` - System performance tracking
- **Config**: `config.py` - Detection parameters and thresholds

### Violence Detection (violence detection cdps/)
- **Flask App**: `flask_violence_detector.py` - Web interface for violence detection
- **Model**: Uses Bensam02 MobileNetV2 model (violence/bensam02_model.h5)

### Weapon Detection (weapon/)
- **Flask App**: `app.py` - Web interface for weapon detection  
- **Model**: YOLO-based detection using ultralytics (models/best.pt)

## Important Development Notes

### Performance Considerations
- ANPR system uses frame skipping and optimized processing for real-time performance
- Face detection has configurable detection frequency and frame scaling
- All systems support performance monitoring and statistics

### Alert Systems
- Telegram bot integration across modules for real-time notifications
- Configurable alert intervals to prevent spam
- Alert cooldown mechanisms for repeated detections

### File Management
- Detected images saved to module-specific directories
- Automatic directory creation on startup
- Configurable image quality and formats

### Threading and Async
- ANPR system uses threading for non-blocking alert sending
- Face detection supports multi-threaded processing
- Async operations for Telegram communications

### Model Management
- Models are loaded at application startup
- Error handling for missing models
- Support for different model formats (H5, PT files)

## Configuration Management

Most modules use either:
1. **Dataclass-based configs** (ANPR) - Structured, type-safe configuration
2. **Python config files** (Face Detection) - Simple variable-based configuration  
3. **YAML configs** (Weapon Detection) - External configuration files

Environment variables can override default settings in ANPR system using `ANPR_*` prefixes.

## Testing and Validation

- Face detection includes dedicated test and setup scripts
- Web interfaces provide immediate testing capabilities
- Performance monitoring helps identify bottlenecks
- Confidence thresholds configurable per module

## Security Considerations

- No hardcoded credentials (uses environment variables for Telegram tokens)
- Input validation for uploaded files
- Configurable file size limits
- Temporary file cleanup after processing