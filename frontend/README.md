# Real-Time Crime Detection Frontend

A comprehensive React-based web application for monitoring and interacting with the Real-Time Crime Detection and Prevention System.

## Features

### Core Modules
- **Live Feeds Panel**: Real-time video streaming with multi-module detection overlay
- **Face Recognition**: Upload images or use webcam for face detection and recognition
- **Weapon Detection**: Detect various weapons using YOLO-based models
- **Violence Detection**: Real-time violence incident detection and alerting
- **ANPR**: Automatic Number Plate Recognition with red-list checking

### Analytics & Monitoring
- **Dashboard**: Real-time system status and detection analytics
- **Smart Search**: Advanced filtering and search of past detections
- **Alert System**: Real-time notifications and alert management
- **Settings**: Configure detection thresholds and module parameters

### Technical Features
- Real-time WebSocket communication for live video streaming
- RESTful API integration with the backend system
- Responsive Material-UI design
- Interactive charts and visualizations
- Image upload and webcam capture capabilities

## Quick Start

### Prerequisites
- Node.js 18+ and npm
- Backend system running on `http://localhost:8000`

### Installation
```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:5173`

### Build for Production
```bash
npm run build
npm run preview
```

## Technology Stack
- **React 18** with TypeScript
- **Material-UI v5** for UI components
- **React Router** for navigation
- **Axios** for HTTP requests
- **Socket.io** for WebSocket communication
- **Chart.js** for data visualization
- **Vite** for build tooling

## API Integration

The frontend communicates with the backend through:

### REST API Endpoints
- `/api/detect/*` - Detection services
- `/api/live/*` - Camera and streaming control
- `/api/settings/*` - System configuration
- `/api/alerts/*` - Alert management
- `/api/analytics/*` - Analytics data
- `/api/system/*` - System status

### WebSocket Events
- `detection` - Real-time detection results
- `alert` - Live security alerts
- `video_frame` - Live video stream data
- `system_status` - System health updates

## Configuration

### Backend Connection
Update the API base URL in `src/services/api.ts`:
```typescript
const API_BASE_URL = 'http://localhost:8000';
```

### WebSocket Connection
Update the WebSocket URL in `src/services/websocket.ts`:
```typescript
this.socket = io('http://localhost:8000');
```
