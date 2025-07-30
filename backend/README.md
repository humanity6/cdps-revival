# Real-Time Crime Detection Backend API

A unified FastAPI backend that integrates all crime detection modules (ANPR, Face Recognition, Violence Detection, and Weapon Detection) under a single API with real-time WebSocket support.

## Features

- **Multi-Module Detection**: ANPR, Face Recognition, Violence Detection, Weapon Detection
- **RESTful API**: Individual endpoint testing for each detection module
- **Real-Time WebSocket**: Live camera feed with toggleable detection modules
- **Multi-Detection**: Run multiple detection modules on a single image
- **Configuration Management**: Runtime configuration updates for all modules
- **Health Monitoring**: Service health checks and status monitoring
- **File Upload Support**: Process images and videos via file upload or base64

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Server

```bash
# Using the startup script (recommended)
python run.py

# Or directly with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Check dependencies and models only
python run.py --check-only
```

### 3. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **System Info**: http://localhost:8000/info

## API Endpoints

### Detection Endpoints

- `POST /api/detect/anpr` - License plate detection
- `POST /api/detect/face` - Face recognition
- `POST /api/detect/violence` - Violence detection
- `POST /api/detect/weapon` - Weapon detection
- `POST /api/detect/multi` - Multi-module detection

### Live Feed Endpoints

- `WS /api/live/ws/{client_id}` - WebSocket for real-time feed
- `POST /api/live/start` - Start live camera feed
- `POST /api/live/stop` - Stop live camera feed
- `GET /api/live/status` - Get live feed status
- `POST /api/live/camera/config` - Update camera configuration

### Settings Endpoints

- `GET /api/settings/` - Get all system settings
- `GET /api/settings/{module}` - Get module-specific settings
- `PUT /api/settings/{module}` - Update module settings
- `POST /api/settings/{module}/toggle` - Enable/disable module
- `GET /api/settings/health` - Get all modules health status

## WebSocket Usage

### Connect to Live Feed

```javascript
const ws = new WebSocket('ws://localhost:8000/api/live/ws/client123');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'frame') {
        // Display frame data (base64 image)
        displayFrame(data.frame_data);
        updateStats(data.fps, data.detections);
    } else if (data.type === 'status') {
        // Handle status updates
        updateStatus(data);
    }
};

// Start live feed
ws.send(JSON.stringify({type: 'start_feed'}));

// Toggle detection modules
ws.send(JSON.stringify({
    type: 'toggle_module',
    module: 'weapon',
    enabled: true
}));
```

### WebSocket Message Types

#### Outgoing (Client → Server)
- `start_feed` - Start camera feed
- `stop_feed` - Stop camera feed
- `toggle_module` - Enable/disable detection module
- `update_camera_config` - Update camera settings
- `ping` - Ping server

#### Incoming (Server → Client)
- `frame` - Real-time frame with detections
- `status` - System status update
- `feed_started`/`feed_stopped` - Feed state changes
- `error` - Error messages
- `pong` - Ping response

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Module Control
ENABLE_ANPR=true
ENABLE_FACE=true
ENABLE_VIOLENCE=true
ENABLE_WEAPON=true

# Camera Settings
DEFAULT_CAMERA_INDEX=0
DEFAULT_CAMERA_WIDTH=640
DEFAULT_CAMERA_HEIGHT=480
DEFAULT_CAMERA_FPS=30

# File Upload
MAX_FILE_SIZE=16777216  # 16MB
```

### Runtime Configuration

Update module settings via API:

```bash
# Update ANPR confidence threshold
curl -X PUT "http://localhost:8000/api/settings/anpr" \
  -H "Content-Type: application/json" \
  -d '{"min_confidence": 0.7}'

# Toggle weapon detection
curl -X POST "http://localhost:8000/api/settings/weapon/toggle?enabled=false"
```

## Usage Examples

### Image Detection

```python
import requests
import base64

# Read image file
with open('test_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# ANPR Detection
response = requests.post(
    'http://localhost:8000/api/detect/anpr',
    json={'image_data': image_data, 'min_confidence': 0.6}
)

print(response.json())
```

### File Upload

```python
import requests

# Upload image file
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/detect/face',
        files={'file': f}
    )

print(response.json())
```

### Multi-Detection

```python
import requests
import base64

with open('test_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post(
    'http://localhost:8000/api/detect/multi',
    json={
        'image_data': image_data,
        'enabled_modules': ['anpr', 'face', 'weapon'],
        'anpr_config': {'min_confidence': 0.6},
        'face_config': {'recognition_tolerance': 0.5},
        'weapon_config': {'confidence_threshold': 0.7}
    }
)

print(response.json())
```

## Health Monitoring

```bash
# Overall system health
curl http://localhost:8000/health

# Module-specific health
curl http://localhost:8000/api/settings/health/anpr

# All modules health
curl http://localhost:8000/api/settings/health
```

## Development

### Project Structure

```
backend/
├── main.py                 # FastAPI application
├── run.py                  # Startup script
├── requirements.txt        # Dependencies
├── config.py              # Configuration management
├── models/                # Pydantic models
├── services/              # Detection service wrappers
├── routers/               # API route handlers
├── utils/                 # Utilities and WebSocket manager
└── static/               # Static files
```

### Adding New Detection Modules

1. Create service wrapper in `services/`
2. Add Pydantic models in `models/`
3. Add API routes in `routers/detection.py`
4. Update configuration in `config.py`
5. Register service in `main.py`

### Custom Commands

```bash
# Development server with auto-reload
python run.py --reload --debug

# Custom host and port
python run.py --host 127.0.0.1 --port 9000

# Check dependencies only
python run.py --check-only

# Set log level
python run.py --log-level DEBUG
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure all detection modules are in the correct paths
2. **Model Loading Errors**: Check if model files exist in expected locations
3. **Camera Access Errors**: Verify camera permissions and availability
4. **WebSocket Connection Issues**: Check firewall and CORS settings

### Logging

Logs are written to `backend.log` and console. Adjust log level via:
- Environment variable: `LOG_LEVEL=DEBUG`
- Command line: `python run.py --log-level DEBUG`

### Performance Tips

- Use appropriate confidence thresholds for your use case
- Skip violence detection in live feeds for better performance
- Process every nth frame for real-time applications
- Use GPU acceleration where available

## Integration with Frontend

This backend is designed to work seamlessly with web frontends. Key integration points:

- RESTful API for individual testing panels
- WebSocket for live feed display
- Configuration API for settings panels
- Health monitoring for status displays
- File upload support for drag-and-drop interfaces

For frontend development, use the OpenAPI schema available at `/docs` or `/openapi.json`.