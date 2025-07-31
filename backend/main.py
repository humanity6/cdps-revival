"""
Main FastAPI application for Unified Crime Detection Backend
"""
import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Configuration
from config import backend_config, ensure_directories

# Services
from services import ANPRService, FaceService, ViolenceService, WeaponService

# Routers
from routers import (
    detection_router, 
    live_feed_router, 
    settings_router,
    system_router,
    analytics_router,
    detections_router,
    alerts_router
)
from routers.detection import set_detection_services
from routers.live_feed import set_websocket_manager
from routers.settings import set_detection_services as set_settings_services

# Utilities
from utils.websocket_manager import WebSocketManager
from utils.socketio_manager import socketio_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, backend_config.log_level.value),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(backend_config.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global services
detection_services: Dict[str, Any] = {}
websocket_manager: WebSocketManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    """
    # Startup
    logger.info("🚀 Starting Real-Time Crime Detection Backend API")
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Initialize detection services
        await initialize_services()
        
        # Initialize WebSocket manager
        initialize_websocket_manager()
        
        logger.info("✅ All services initialized successfully")
        
        yield  # Application is running
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Crime Detection Backend API")
        await cleanup_services()

async def initialize_services():
    """
    Initialize all detection services
    """
    global detection_services
    
    logger.info("🔧 Initializing detection services...")
    
    # Initialize ANPR Service
    if backend_config.enable_anpr:
        try:
            anpr_service = ANPRService()
            detection_services['anpr'] = anpr_service
            logger.info("✅ ANPR Service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ANPR Service: {e}")
    
    # Initialize Face Recognition Service
    if backend_config.enable_face:
        try:
            face_service = FaceService()
            detection_services['face'] = face_service
            logger.info("✅ Face Recognition Service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Face Recognition Service: {e}")
    
    # Initialize Violence Detection Service
    if backend_config.enable_violence:
        try:
            violence_service = ViolenceService()
            detection_services['violence'] = violence_service
            logger.info("✅ Violence Detection Service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Violence Detection Service: {e}")
    
    # Initialize Weapon Detection Service
    if backend_config.enable_weapon:
        try:
            weapon_service = WeaponService()
            detection_services['weapon'] = weapon_service
            logger.info("✅ Weapon Detection Service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Weapon Detection Service: {e}")
    
    # Set services in routers
    set_detection_services(detection_services)
    set_settings_services(detection_services)
    
    # Set services in Socket.IO manager
    socketio_manager.set_detection_services(detection_services)
    
    logger.info(f"📊 Initialized {len(detection_services)} detection services")

def initialize_websocket_manager():
    """
    Initialize WebSocket manager for live feeds
    """
    global websocket_manager
    
    try:
        websocket_manager = WebSocketManager(detection_services)
        set_websocket_manager(websocket_manager)
        logger.info("✅ WebSocket Manager initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize WebSocket Manager: {e}")

async def cleanup_services():
    """
    Cleanup services on shutdown
    """
    global websocket_manager
    
    # Cleanup WebSocket manager
    if websocket_manager:
        try:
            await websocket_manager.cleanup()
            logger.info("✅ WebSocket Manager cleaned up")
        except Exception as e:
            logger.error(f"❌ Error cleaning up WebSocket Manager: {e}")
    
    # Cleanup detection services
    for service_name, service in detection_services.items():
        try:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
            logger.info(f"✅ {service_name} service cleaned up")
        except Exception as e:
            logger.error(f"❌ Error cleaning up {service_name} service: {e}")

# Create FastAPI application
app = FastAPI(
    title=backend_config.api_title,
    version=backend_config.api_version,
    description=backend_config.api_description,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=backend_config.cors_origins,
    allow_credentials=True,
    allow_methods=backend_config.cors_methods,
    allow_headers=backend_config.cors_headers,
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Mount Socket.io app
app.mount("/socket.io", socketio_manager.get_asgi_app())

# Include routers
app.include_router(detection_router)
app.include_router(live_feed_router)  
app.include_router(settings_router)
app.include_router(system_router, prefix="/api/system", tags=["system"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])
app.include_router(detections_router, prefix="/api/detections", tags=["detections"])
app.include_router(alerts_router, prefix="/api/alerts", tags=["alerts"])

# Root endpoints
@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Real-Time Crime Detection Backend API",
        "version": backend_config.api_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "services": list(detection_services.keys()),
        "websocket_endpoint": "/api/live/ws/{client_id}"
    }

@app.get("/health")
async def health_check():
    """
    Overall system health check
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - app.extra.get("start_time", time.time()),
            "services": {},
            "websocket": {
                "enabled": websocket_manager is not None,
                "connected_clients": websocket_manager.connection_manager.get_client_count() if websocket_manager else 0
            }
        }
        
        # Check each service
        for service_name, service in detection_services.items():
            try:
                if hasattr(service, 'get_health_status'):
                    service_health = service.get_health_status()
                    health_status["services"][service_name] = service_health
                else:
                    health_status["services"][service_name] = {"status": "unknown"}
            except Exception as e:
                health_status["services"][service_name] = {"status": "error", "error": str(e)}
                health_status["status"] = "degraded"
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@app.get("/info")
async def system_info():
    """
    System information endpoint
    """
    try:
        return {
            "api": {
                "title": backend_config.api_title,
                "version": backend_config.api_version,
                "description": backend_config.api_description
            },
            "config": {
                "host": backend_config.host,
                "port": backend_config.port,
                "debug": backend_config.debug,
                "log_level": backend_config.log_level.value
            },
            "services": {
                "available": list(detection_services.keys()),
                "enabled": {
                    "anpr": backend_config.enable_anpr,
                    "face": backend_config.enable_face,
                    "violence": backend_config.enable_violence,
                    "weapon": backend_config.enable_weapon
                }
            },
            "features": {
                "file_upload": True,
                "base64_processing": True,
                "multi_detection": True,
                "live_feed": websocket_manager is not None,
                "websocket": True
            },
            "limits": {
                "max_file_size": f"{backend_config.max_file_size // (1024 * 1024)}MB",
                "max_websocket_connections": backend_config.max_websocket_connections
            }
        }
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Store start time for uptime calculation
app.extra = {"start_time": time.time()}

# Custom exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    logger.info("🌟 Starting Crime Detection Backend API Server")
    
    try:
        uvicorn.run(
            "main:app",
            host=backend_config.host,
            port=backend_config.port,
            reload=backend_config.reload,
            log_level=backend_config.log_level.value.lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Server error: {e}")
        sys.exit(1)