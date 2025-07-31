# Real-Time Crime Detection and Prevention System

A comprehensive, modular system for real-time crime detection using computer vision and machine learning. The system includes multiple detection modules (ANPR, Face Recognition, Violence Detection, Weapon Detection) with a unified backend API and modern React frontend.

## 🏗️ System Architecture

```
Real-Time Crime Detection System
├── backend/           # FastAPI Backend (Python)
├── frontend/          # React Frontend (TypeScript)
├── anpr/             # License Plate Recognition Module
├── face detection/   # Face Recognition Module  
├── violence detection cdps/  # Violence Detection Module
├── weapon/           # Weapon Detection Module
└── database/         # Unified Database System
```

## ✨ Features

### 🎯 Detection Modules
- **ANPR (Automatic Number Plate Recognition)**: Real-time license plate detection with red-list checking
- **Face Recognition**: Face detection and person identification with unknown face alerts
- **Violence Detection**: AI-powered violence incident detection using MobileNetV2
- **Weapon Detection**: YOLO-based weapon detection for firearms, knives, and dangerous objects

### 🖥️ User Interface
- **Real-time Dashboard**: System status, analytics, and detection trends
- **Live Video Feeds**: WebSocket-based streaming with multi-module overlay
- **Detection Management**: Upload images, webcam capture, and historical browsing
- **Alert System**: Real-time notifications with Telegram integration
- **Advanced Search**: Filter and export detection data
- **Settings Panel**: Configure detection thresholds and parameters

### 🔧 Technical Features
- **Unified FastAPI Backend** with modular service architecture
- **WebSocket Support** for real-time video streaming and alerts
- **SQLite Database** with comprehensive event logging and analytics
- **Telegram Bot Integration** for instant security alerts
- **RESTful API** with OpenAPI documentation
- **Modern React Frontend** with Material-UI design
- **Docker Support** for easy deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with pip
- Node.js 18+ with npm
- Webcam (optional, for live detection)

### Option 1: Automated Setup (Windows)
```bash
# Run the automated startup script
start_system.bat
```

### Option 2: Manual Setup

**Backend Setup:**
```bash
cd backend
pip install -r requirements.txt
python run.py
```

**Frontend Setup:**
```bash
cd frontend  
npm install
npm run dev
```

### Access Points
- **Frontend Application**: http://localhost:5173
- **Backend API**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs
- **WebSocket Endpoint**: ws://127.0.0.1:8000/api/live/ws/{client_id}

## 📊 System Components

### Backend Services (`backend/`)
- **FastAPI Application** (`main.py`) - Main API server
- **Detection Services** (`services/`) - Modular detection engines
- **Database Layer** (`database/`) - Unified data management
- **WebSocket Manager** (`utils/`) - Real-time communication
- **API Routers** (`routers/`) - Endpoint organization

### Frontend Application (`frontend/`)
- **React Components** (`src/components/`) - Reusable UI elements
- **Page Components** (`src/pages/`) - Main application views
- **API Services** (`src/services/`) - Backend communication
- **WebSocket Integration** (`src/hooks/`) - Real-time features
- **Type Definitions** (`src/types/`) - TypeScript interfaces

### Detection Modules
- **ANPR Module** (`anpr/`) - License plate recognition
- **Face Module** (`face detection/`) - Face detection and recognition
- **Violence Module** (`violence detection cdps/`) - Violence incident detection
- **Weapon Module** (`weapon/`) - Weapon detection system

## 🔌 API Endpoints

### Detection APIs
- `POST /api/detect/image` - Multi-module image detection
- `POST /api/detect/face` - Face-specific detection
- `POST /api/detect/weapon` - Weapon detection
- `POST /api/detect/violence` - Violence detection
- `POST /api/detect/anpr` - License plate recognition

### Live Streaming
- `POST /api/live/start` - Start camera stream
- `POST /api/live/stop` - Stop camera stream
- `GET /api/live/status` - Camera status
- `WebSocket /api/live/ws/{client_id}` - Live video feed

### Data Management
- `GET /api/detections/recent` - Recent detections
- `GET /api/detections/search` - Search detections
- `GET /api/analytics` - System analytics
- `GET /api/alerts/recent` - Recent alerts

### Configuration
- `GET /api/settings` - Get system settings
- `POST /api/settings/{module}` - Update module settings
- `GET /api/system/status` - System health status

## 🛠️ Configuration

### Backend Configuration (`backend/config.py`)
```python
# Detection modules
ENABLE_ANPR = True
ENABLE_FACE = True  
ENABLE_VIOLENCE = True
ENABLE_WEAPON = True

# API settings
API_HOST = "127.0.0.1"
API_PORT = 8000

# Database
DATABASE_URL = "sqlite:///crime_detection.db"
```

### Frontend Configuration (`frontend/src/services/api.ts`)
```typescript
const API_BASE_URL = 'http://127.0.0.1:8000';
```

### Module Settings
Each detection module has configurable parameters:
- **Detection thresholds** (0.1 - 1.0)
- **Processing intervals** 
- **Alert sensitivity levels**
- **Output directories**

## 📱 User Interface Guide

### Dashboard
- Real-time system metrics and detection analytics
- Interactive charts showing detection trends
- System health monitoring
- Recent detection timeline

### Live Feeds
- Real-time video streaming with detection overlay
- Multi-module detection toggle controls
- Camera settings (resolution, FPS)
- Manual capture and detection

### Detection Modules
Each module provides:
- Image upload interface
- Webcam capture functionality
- Detection results with confidence scores
- Historical detection browsing
- Export capabilities

### Alerts & Search
- Real-time alert notifications
- Alert history and management
- Advanced search with filters
- Data export (CSV, JSON)

## 🔧 Development

### Adding New Detection Modules
1. Create module directory with detection logic
2. Add service class in `backend/services/`
3. Create API endpoints in `backend/routers/`
4. Add frontend components in `frontend/src/pages/`
5. Update navigation and routing

### Database Schema
The system uses a unified SQLite database with tables for:
- `events` - All detection events
- `detections` - Detection results
- `alerts` - System alerts
- `red_listed_items` - Blocked items (license plates, faces)
- `system_logs` - Application logs

### WebSocket Events
- `detection` - New detection result
- `alert` - Security alert
- `video_frame` - Live video data
- `system_status` - Health updates

## 🐳 Docker Deployment

### Backend Container
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "run.py"]
```

### Frontend Container  
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 5173
CMD ["npm", "run", "preview"]
```

## 📊 Performance & Monitoring

### System Requirements
- **Minimum**: 4GB RAM, 2-core CPU
- **Recommended**: 8GB RAM, 4-core CPU, GPU support
- **Storage**: 2GB for system, additional space for detection logs

### Performance Monitoring
- Detection processing times
- API response times  
- WebSocket connection health
- Database query performance
- Memory and CPU usage

### Scaling Considerations
- Horizontal scaling with load balancers
- Database clustering for high-volume deployments
- CDN integration for static assets
- Microservice architecture for large installations

## 🔒 Security Features

### Authentication & Authorization
- API key authentication
- Role-based access control
- Session management
- CORS configuration

### Data Security
- Encrypted database connections
- Secure file upload validation
- Input sanitization
- Rate limiting

### Privacy Protection
- Configurable data retention policies
- Personal data anonymization
- Audit trail logging
- GDPR compliance features

## 🧪 Testing

### Backend Testing
```bash
cd backend
python -m pytest tests/
```

### Frontend Testing
```bash
cd frontend
npm run test
```

### Integration Testing
- End-to-end API testing
- WebSocket connection testing
- Detection accuracy validation
- Performance benchmarking

## 📚 Documentation

- **API Documentation**: Available at `/docs` when backend is running
- **Frontend Documentation**: See `frontend/README.md`
- **Module Documentation**: Each detection module includes specific docs
- **Database Schema**: See `backend/database/schema.py`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is part of a final year project for educational purposes.

## 🆘 Troubleshooting

### Common Issues

**Backend won't start:**
- Check Python dependencies: `pip install -r requirements.txt`
- Verify model files exist in correct locations
- Check port 8000 availability

**Frontend build errors:**
- Update Node.js to 18+
- Clear node_modules: `rm -rf node_modules && npm install`
- Check TypeScript configuration

**WebSocket connection failed:**
- Ensure backend is running on correct port
- Check firewall settings
- Verify CORS configuration

**Detection not working:**
- Check model file paths and permissions
- Verify camera access permissions
- Monitor backend logs for errors

### Debug Mode
Enable detailed logging:
```bash
# Backend
python run.py --debug --log-level DEBUG

# Frontend  
npm run dev -- --mode development
```

### Support
For issues and questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Enable debug logging
4. Check system requirements