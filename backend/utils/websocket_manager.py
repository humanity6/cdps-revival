"""
WebSocket Manager for Real-time Live Feeds
Handles WebSocket connections and real-time detection streaming
"""
import asyncio
import json
import time
import logging
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import cv2
import numpy as np
from datetime import datetime

from utils.image_utils import image_to_base64, draw_bounding_boxes
from utils.video_utils import CameraCapture
from config import backend_config, get_module_enabled_status

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_data: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_data[client_id] = {
            'connected_at': time.time(),
            'enabled_modules': ['anpr', 'face', 'weapon', 'violence'],
            'last_ping': time.time()
        }
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_data:
            del self.connection_data[client_id]
        logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                    return True
                else:
                    self.disconnect(client_id)
                    return False
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)
                return False
        return False
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                else:
                    disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Failed to broadcast to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def get_client_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def update_client_modules(self, client_id: str, enabled_modules: List[str]):
        """Update enabled modules for a client"""
        if client_id in self.connection_data:
            self.connection_data[client_id]['enabled_modules'] = enabled_modules
    
    def get_client_modules(self, client_id: str) -> List[str]:
        """Get enabled modules for a client"""
        return self.connection_data.get(client_id, {}).get('enabled_modules', [])

class WebSocketManager:
    """Main WebSocket manager for live feeds"""
    
    def __init__(self, detection_services: Dict[str, Any]):
        self.connection_manager = ConnectionManager()
        self.detection_services = detection_services
        self.camera_capture = None
        self.is_streaming = False
        self.stream_task = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Live feed configuration
        self.camera_config = {
            'index': backend_config.default_camera_index,
            'width': backend_config.default_camera_width,
            'height': backend_config.default_camera_height,
            'fps': backend_config.default_camera_fps
        }
    
    async def connect_client(self, websocket: WebSocket, client_id: str):
        """Connect a new WebSocket client"""
        await self.connection_manager.connect(websocket, client_id)
        
        # Send initial status
        await self.send_status_update(client_id)
    
    def disconnect_client(self, client_id: str):
        """Disconnect a WebSocket client"""
        self.connection_manager.disconnect(client_id)
        
        # Stop streaming if no clients
        if self.connection_manager.get_client_count() == 0:
            asyncio.create_task(self.stop_live_feed())
    
    async def handle_client_message(self, client_id: str, message: dict):
        """Handle incoming message from WebSocket client"""
        try:
            message_type = message.get('type')
            
            if message_type == 'start_feed':
                await self.start_live_feed()
            elif message_type == 'stop_feed':
                await self.stop_live_feed()
            elif message_type == 'toggle_module':
                module_name = message.get('module')
                enabled = message.get('enabled', True)
                await self.toggle_detection_module(client_id, module_name, enabled)
            elif message_type == 'update_camera_config':
                config = message.get('config', {})
                await self.update_camera_config(config)
            elif message_type == 'ping':
                await self.handle_ping(client_id)
            else:
                logger.warning(f"Unknown message type from {client_id}: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self.send_error_message(client_id, str(e))
    
    async def start_live_feed(self):
        """Start the live camera feed"""
        if self.is_streaming:
            return
        
        try:
            # Initialize camera
            self.camera_capture = CameraCapture(
                camera_index=self.camera_config['index'],
                width=self.camera_config['width'],
                height=self.camera_config['height'],
                fps=self.camera_config['fps']
            )
            
            if not self.camera_capture.open():
                await self.broadcast_error("Failed to open camera")
                return
            
            self.is_streaming = True
            self.stream_task = asyncio.create_task(self._stream_loop())
            
            await self.connection_manager.broadcast({
                'type': 'feed_started',
                'timestamp': time.time(),
                'camera_config': self.camera_config
            })
            
            logger.info("Live feed started")
            
        except Exception as e:
            logger.error(f"Failed to start live feed: {e}")
            await self.broadcast_error(f"Failed to start live feed: {str(e)}")
    
    async def stop_live_feed(self):
        """Stop the live camera feed"""
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
        
        await self.connection_manager.broadcast({
            'type': 'feed_stopped',
            'timestamp': time.time()
        })
        
        logger.info("Live feed stopped")
    
    async def _stream_loop(self):
        """Main streaming loop"""
        try:
            while self.is_streaming and self.camera_capture:
                ret, frame = self.camera_capture.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                # Update FPS counter
                self._update_fps()
                
                # Process frame with detections for each connected client
                await self._process_and_send_frame(frame)
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            logger.info("Stream loop cancelled")
        except Exception as e:
            logger.error(f"Error in stream loop: {e}")
            await self.broadcast_error(f"Stream error: {str(e)}")
        finally:
            if self.is_streaming:
                await self.stop_live_feed()
    
    async def _process_and_send_frame(self, frame: np.ndarray):
        """Process frame and send to connected clients"""
        if not self.connection_manager.active_connections:
            return
        
        # Get system-wide module status
        system_modules = get_module_enabled_status()
        
        # Process detections for all enabled modules
        all_detections = {}
        
        # ANPR Detection
        if system_modules.get('anpr') and 'anpr' in self.detection_services:
            try:
                anpr_result = await self.detection_services['anpr'].detect_plates(frame)
                if anpr_result.success:
                    all_detections['anpr'] = anpr_result.detections
            except Exception as e:
                logger.error(f"ANPR detection error: {e}")
        
        # Face Detection
        if system_modules.get('face') and 'face' in self.detection_services:
            try:
                face_result = await self.detection_services['face'].detect_faces(frame)
                if face_result.success:
                    all_detections['face'] = face_result.detections
            except Exception as e:
                logger.error(f"Face detection error: {e}")
        
        # Weapon Detection
        if system_modules.get('weapon') and 'weapon' in self.detection_services:
            try:
                weapon_result = await self.detection_services['weapon'].detect_weapons(frame)
                if weapon_result.success:
                    all_detections['weapon'] = weapon_result.detections
            except Exception as e:
                logger.error(f"Weapon detection error: {e}")
        
        # Violence Detection (skip for live feed due to performance)
        # Violence detection is typically too slow for real-time processing
        
        # Send frame to each client with their enabled modules
        for client_id in list(self.connection_manager.active_connections.keys()):
            await self._send_frame_to_client(client_id, frame, all_detections)
    
    async def _send_frame_to_client(self, client_id: str, frame: np.ndarray, all_detections: Dict[str, List]):
        """Send processed frame to specific client"""
        try:
            # Get client's enabled modules
            client_modules = self.connection_manager.get_client_modules(client_id)
            
            # Filter detections based on client's enabled modules
            client_detections = []
            for module in client_modules:
                if module in all_detections:
                    client_detections.extend(all_detections[module])
            
            # Draw bounding boxes on frame
            annotated_frame = draw_bounding_boxes(frame, client_detections)
            
            # Convert frame to base64
            frame_data = image_to_base64(annotated_frame)
            if not frame_data:
                return
            
            # Prepare frame message
            frame_message = {
                'type': 'frame',
                'timestamp': time.time(),
                'frame_data': frame_data,
                'fps': self.current_fps,
                'detections': {
                    module: len([d for d in client_detections if getattr(d, 'class_name', '').startswith(module)])
                    for module in client_modules
                },
                'total_detections': len(client_detections)
            }
            
            # Send to client
            await self.connection_manager.send_personal_message(frame_message, client_id)
            
        except Exception as e:
            logger.error(f"Error sending frame to client {client_id}: {e}")
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    async def toggle_detection_module(self, client_id: str, module_name: str, enabled: bool):
        """Toggle detection module for specific client"""
        client_modules = self.connection_manager.get_client_modules(client_id)
        
        if enabled and module_name not in client_modules:
            client_modules.append(module_name)
        elif not enabled and module_name in client_modules:
            client_modules.remove(module_name)
        
        self.connection_manager.update_client_modules(client_id, client_modules)
        
        await self.connection_manager.send_personal_message({
            'type': 'module_toggled',
            'module': module_name,
            'enabled': enabled,
            'enabled_modules': client_modules
        }, client_id)
    
    async def update_camera_config(self, config: Dict[str, Any]):
        """Update camera configuration"""
        # Stop current feed
        was_streaming = self.is_streaming
        if was_streaming:
            await self.stop_live_feed()
        
        # Update configuration
        self.camera_config.update(config)
        
        # Restart feed if it was running
        if was_streaming and self.connection_manager.get_client_count() > 0:
            await self.start_live_feed()
        
        # Notify clients
        await self.connection_manager.broadcast({
            'type': 'camera_config_updated',
            'config': self.camera_config
        })
    
    async def send_status_update(self, client_id: str):
        """Send status update to client"""
        status = {
            'type': 'status',
            'is_streaming': self.is_streaming,
            'camera_config': self.camera_config,
            'enabled_modules': self.connection_manager.get_client_modules(client_id),
            'system_modules': get_module_enabled_status(),
            'connected_clients': self.connection_manager.get_client_count(),
            'current_fps': self.current_fps
        }
        
        await self.connection_manager.send_personal_message(status, client_id)
    
    async def handle_ping(self, client_id: str):
        """Handle ping from client"""
        if client_id in self.connection_manager.connection_data:
            self.connection_manager.connection_data[client_id]['last_ping'] = time.time()
        
        await self.connection_manager.send_personal_message({
            'type': 'pong',
            'timestamp': time.time()
        }, client_id)
    
    async def send_error_message(self, client_id: str, error: str):
        """Send error message to specific client"""
        await self.connection_manager.send_personal_message({
            'type': 'error',
            'message': error,
            'timestamp': time.time()
        }, client_id)
    
    async def broadcast_error(self, error: str):
        """Broadcast error to all clients"""
        await self.connection_manager.broadcast({
            'type': 'error',
            'message': error,
            'timestamp': time.time()
        })
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.stop_live_feed()
        
        # Disconnect all clients
        for client_id in list(self.connection_manager.active_connections.keys()):
            self.disconnect_client(client_id)