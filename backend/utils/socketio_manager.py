"""
Socket.io WebSocket Manager
Provides Socket.io integration for real-time communication
"""
import socketio
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
import json
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class SocketIOManager:
    """Socket.io WebSocket manager for real-time communication"""
    
    def __init__(self):
        self.sio = socketio.AsyncServer(
            cors_allowed_origins="*",  # Allow all origins for development  
            logger=False,
            engineio_logger=False,
            async_mode='asgi',
            ping_timeout=60,  # Increase ping timeout
            ping_interval=25  # Send ping every 25 seconds
        )
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.detection_services: Dict[str, Any] = {}
        self.camera_capture = None
        self.is_streaming = False
        self.stream_task = None
        self.setup_events()
    
    def get_asgi_app(self):
        """Get the ASGI app for Socket.io"""
        return socketio.ASGIApp(self.sio, other_asgi_app=None)
    
    def set_detection_services(self, services: Dict[str, Any]):
        """Set detection services for the Socket.IO manager"""
        self.detection_services = services
        logger.info(f"Socket.IO manager configured with services: {list(services.keys())}")
    
    def setup_events(self):
        """Set up Socket.io event handlers"""
        
        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection"""
            logger.info(f"Socket.io client connected: {sid}")
            self.connected_clients[sid] = {
                "connected_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
            
            # Send connection confirmation
            await self.sio.emit('connection', {
                'status': 'connected',
                'client_id': sid,
                'timestamp': datetime.now().isoformat()
            }, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            logger.info(f"Socket.io client disconnected: {sid}")
            if sid in self.connected_clients:
                del self.connected_clients[sid]
        
        @self.sio.event
        async def start_stream(sid, data=None):
            """Handle start stream request"""
            logger.info(f"Client {sid} requested to start stream")
            try:
                if not self.is_streaming:
                    await self._start_camera_stream()
                
                await self.sio.emit('stream_status', {
                    'status': 'started',
                    'message': 'Video stream started',
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
            except Exception as e:
                logger.error(f"Failed to start stream for client {sid}: {e}")
                await self.sio.emit('error', {
                    'message': f'Failed to start stream: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
        
        @self.sio.event
        async def stop_stream(sid, data=None):
            """Handle stop stream request"""
            logger.info(f"Client {sid} requested to stop stream")
            try:
                if self.is_streaming and len(self.connected_clients) <= 1:
                    await self._stop_camera_stream()
                
                await self.sio.emit('stream_status', {
                    'status': 'stopped', 
                    'message': 'Video stream stopped',
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
            except Exception as e:
                logger.error(f"Failed to stop stream for client {sid}: {e}")
                await self.sio.emit('error', {
                    'message': f'Failed to stop stream: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
        
        @self.sio.event
        async def update_stream_settings(sid, data):
            """Handle stream settings update"""
            logger.info(f"Client {sid} updated stream settings: {data}")
            try:
                # Update stream settings if provided
                if data and isinstance(data, dict):
                    # Here you could update camera settings, detection parameters, etc.
                    pass
                
                await self.sio.emit('settings_updated', {
                    'status': 'success',
                    'settings': data,
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
            except Exception as e:
                logger.error(f"Failed to update settings for client {sid}: {e}")
                await self.sio.emit('error', {
                    'message': f'Failed to update settings: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
        
        @self.sio.event
        async def ping(sid, data=None):
            """Handle ping from client"""
            await self.sio.emit('pong', {
                'timestamp': datetime.now().isoformat()
            }, room=sid)
    
    async def broadcast_detection(self, detection: Dict[str, Any]):
        """Broadcast detection to all connected clients"""
        await self.sio.emit('detection', detection)
        logger.debug(f"Broadcasted detection: {detection.get('type', 'unknown')}")
    
    async def broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast alert to all connected clients"""
        await self.sio.emit('alert', alert)
        logger.info(f"Broadcasted alert: {alert.get('type', 'unknown')}")
    
    async def broadcast_video_frame(self, frame_data: str):
        """Broadcast video frame to all connected clients"""
        await self.sio.emit('video_frame', frame_data)
    
    async def broadcast_system_status(self, status: Dict[str, Any]):
        """Broadcast system status to all connected clients"""
        await self.sio.emit('system_status', status)
        logger.debug("Broadcasted system status update")
    
    async def send_to_client(self, client_id: str, event: str, data: Dict[str, Any]):
        """Send data to specific client"""
        if client_id in self.connected_clients:
            await self.sio.emit(event, data, room=client_id)
            logger.debug(f"Sent {event} to client {client_id}")
        else:
            logger.warning(f"Client {client_id} not found")
    
    def get_connected_clients_info(self) -> Dict[str, Any]:
        """Get information about connected clients"""
        return {
            "total_clients": len(self.connected_clients),
            "clients": self.connected_clients
        }
    
    async def _start_camera_stream(self):
        """Start camera streaming"""
        if self.is_streaming:
            return
        
        try:
            # Import here to avoid circular imports
            from utils.video_utils import CameraCapture
            
            self.camera_capture = CameraCapture(
                camera_index=0,
                width=640,
                height=480,
                fps=30
            )
            
            if not self.camera_capture.open():
                raise Exception("Failed to open camera")
            
            self.is_streaming = True
            self.stream_task = asyncio.create_task(self._stream_loop())
            logger.info("Camera stream started")
            
        except Exception as e:
            logger.error(f"Failed to start camera stream: {e}")
            raise
    
    async def _stop_camera_stream(self):
        """Stop camera streaming"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
            self.stream_task = None
        
        if self.camera_capture:
            self.camera_capture.close()
            self.camera_capture = None
        
        logger.info("Camera stream stopped")
    
    async def _stream_loop(self):
        """Main streaming loop"""
        try:
            from utils.image_utils import image_to_base64
            
            while self.is_streaming and self.camera_capture:
                ret, frame = self.camera_capture.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                # Convert frame to base64
                frame_data = image_to_base64(frame)
                if frame_data:
                    await self.broadcast_video_frame(frame_data)
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)  # 10 FPS
                
        except asyncio.CancelledError:
            logger.info("Stream loop cancelled")
        except Exception as e:
            logger.error(f"Error in stream loop: {e}")
        finally:
            if self.is_streaming:
                await self._stop_camera_stream()

# Global Socket.io manager instance
socketio_manager = SocketIOManager()