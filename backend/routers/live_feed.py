"""
Live Feed API routes for real-time detection streaming
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import JSONResponse
import json
import logging
import uuid
from typing import Dict, Any

from models.detection_models import LiveFeedConfig
from models.response_models import StandardResponse
from utils.websocket_manager import WebSocketManager
from config import get_module_enabled_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/live", tags=["live_feed"])

# Global WebSocket manager (will be injected)
websocket_manager: WebSocketManager = None

def get_websocket_manager():
    """Dependency to get WebSocket manager"""
    return websocket_manager

def set_websocket_manager(manager: WebSocketManager):
    """Set WebSocket manager"""
    global websocket_manager
    websocket_manager = manager

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    client_id: str,
    manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    WebSocket endpoint for real-time live feed
    """
    if not manager:
        await websocket.close(code=1003, reason="WebSocket manager not available")
        return
    
    try:
        # Connect client
        await manager.connect_client(websocket, client_id)
        
        # Listen for messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle message
                await manager.handle_client_message(client_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client {client_id} disconnected normally")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from client {client_id}: {e}")
                await manager.send_error_message(client_id, "Invalid JSON format")
            except Exception as e:
                logger.error(f"Error handling message from client {client_id}: {e}")
                await manager.send_error_message(client_id, str(e))
                
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        # Clean up
        manager.disconnect_client(client_id)

@router.post("/start")
async def start_live_feed(manager: WebSocketManager = Depends(get_websocket_manager)):
    """
    Start the live camera feed
    """
    try:
        if not manager:
            raise HTTPException(status_code=503, detail="WebSocket manager not available")
        
        await manager.start_live_feed()
        
        return StandardResponse(
            success=True,
            message="Live feed started successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to start live feed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_live_feed(manager: WebSocketManager = Depends(get_websocket_manager)):
    """
    Stop the live camera feed
    """
    try:
        if not manager:
            raise HTTPException(status_code=503, detail="WebSocket manager not available")
        
        await manager.stop_live_feed()
        
        return StandardResponse(
            success=True,
            message="Live feed stopped successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to stop live feed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_live_feed_status(manager: WebSocketManager = Depends(get_websocket_manager)):
    """
    Get current live feed status
    """
    try:
        if not manager:
            raise HTTPException(status_code=503, detail="WebSocket manager not available")
        
        status = {
            'is_streaming': manager.is_streaming,
            'connected_clients': manager.connection_manager.get_client_count(),
            'camera_config': manager.camera_config,
            'current_fps': manager.current_fps,
            'enabled_modules': get_module_enabled_status()
        }
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Failed to get live feed status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/camera/config")
async def update_camera_config(
    config: LiveFeedConfig,
    manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    Update camera configuration
    """
    try:
        if not manager:
            raise HTTPException(status_code=503, detail="WebSocket manager not available")
        
        camera_config = {
            'index': config.camera_index,
            'width': config.width,
            'height': config.height,
            'fps': config.fps
        }
        
        await manager.update_camera_config(camera_config)
        
        return StandardResponse(
            success=True,
            message="Camera configuration updated successfully",
            data=camera_config
        )
        
    except Exception as e:
        logger.error(f"Failed to update camera config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/camera/config")
async def get_camera_config(manager: WebSocketManager = Depends(get_websocket_manager)):
    """
    Get current camera configuration
    """
    try:
        if not manager:
            raise HTTPException(status_code=503, detail="WebSocket manager not available")
        
        return JSONResponse(content=manager.camera_config)
        
    except Exception as e:
        logger.error(f"Failed to get camera config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/module/toggle")
async def toggle_module_globally(
    module: str,
    enabled: bool,
    manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    Toggle detection module globally for all clients
    """
    try:
        if not manager:
            raise HTTPException(status_code=503, detail="WebSocket manager not available")
        
        # Update module status in config
        from config import update_module_status
        success = update_module_status(module, enabled)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Unknown module: {module}")
        
        # Broadcast update to all clients
        await manager.connection_manager.broadcast({
            'type': 'global_module_toggled',
            'module': module,
            'enabled': enabled,
            'enabled_modules': get_module_enabled_status()
        })
        
        return StandardResponse(
            success=True,
            message=f"Module {module} {'enabled' if enabled else 'disabled'} globally",
            data={'module': module, 'enabled': enabled}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle module {module}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clients")
async def get_connected_clients(manager: WebSocketManager = Depends(get_websocket_manager)):
    """
    Get information about connected WebSocket clients
    """
    try:
        if not manager:
            raise HTTPException(status_code=503, detail="WebSocket manager not available")
        
        client_info = []
        for client_id, data in manager.connection_manager.connection_data.items():
            client_info.append({
                'client_id': client_id,
                'connected_at': data.get('connected_at'),
                'enabled_modules': data.get('enabled_modules', []),
                'last_ping': data.get('last_ping')
            })
        
        return JSONResponse(content={
            'total_clients': len(client_info),
            'clients': client_info
        })
        
    except Exception as e:
        logger.error(f"Failed to get client info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clients/{client_id}")
async def disconnect_client(
    client_id: str,
    manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    Disconnect a specific WebSocket client
    """
    try:
        if not manager:
            raise HTTPException(status_code=503, detail="WebSocket manager not available")
        
        if client_id not in manager.connection_manager.active_connections:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Send disconnect message and close connection
        await manager.connection_manager.send_personal_message({
            'type': 'disconnect',
            'reason': 'Disconnected by server'
        }, client_id)
        
        # Remove client
        manager.disconnect_client(client_id)
        
        return StandardResponse(
            success=True,
            message=f"Client {client_id} disconnected successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disconnect client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))