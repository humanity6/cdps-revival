"""
Socket.io WebSocket Manager
Provides Socket.io integration for real-time communication
"""
import socketio
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SocketIOManager:
    """Socket.io WebSocket manager for real-time communication"""
    
    def __init__(self):
        self.sio = socketio.AsyncServer(
            cors_allowed_origins="*",  # Allow all origins for development
            logger=False,
            engineio_logger=False
        )
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.setup_events()
    
    def get_asgi_app(self):
        """Get the ASGI app for Socket.io"""
        return socketio.ASGIApp(self.sio)
    
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
            # TODO: Start camera stream for this client
            await self.sio.emit('stream_status', {
                'status': 'started',
                'message': 'Video stream started',
                'timestamp': datetime.now().isoformat()
            }, room=sid)
        
        @self.sio.event
        async def stop_stream(sid, data=None):
            """Handle stop stream request"""
            logger.info(f"Client {sid} requested to stop stream")
            # TODO: Stop camera stream for this client
            await self.sio.emit('stream_status', {
                'status': 'stopped', 
                'message': 'Video stream stopped',
                'timestamp': datetime.now().isoformat()
            }, room=sid)
        
        @self.sio.event
        async def update_stream_settings(sid, data):
            """Handle stream settings update"""
            logger.info(f"Client {sid} updated stream settings: {data}")
            # TODO: Update stream settings
            await self.sio.emit('settings_updated', {
                'status': 'success',
                'settings': data,
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

# Global Socket.io manager instance
socketio_manager = SocketIOManager()