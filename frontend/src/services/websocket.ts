import { io, Socket } from 'socket.io-client';
import { Detection, Alert } from '../types';

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 3000;
  private listeners: Map<string, Function[]> = new Map();

  connect() {
    if (this.socket?.connected) {
      return;
    }

    try {
      this.socket = io('http://127.0.0.1:8000', {
        transports: ['websocket', 'polling'],
        upgrade: true,
        timeout: 20000,
        forceNew: true,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
      });
    } catch (error) {
      console.warn('WebSocket connection failed, using mock mode:', error);
      this.setupMockConnection();
      return;
    }

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connection', { status: 'connected' });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.emit('connection', { status: 'disconnected', reason });
      
      // Only attempt manual reconnect for certain disconnect reasons
      if (reason === 'io server disconnect' || reason === 'transport close') {
        this.attemptReconnect();
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.emit('error', error);
      this.attemptReconnect();
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    });

    // Listen for live detection events
    this.socket.on('detection', (detection: Detection) => {
      this.emit('detection', detection);
    });

    // Listen for live alerts
    this.socket.on('alert', (alert: Alert) => {
      this.emit('alert', alert);
    });

    // Listen for video frames
    this.socket.on('video_frame', (frameData: string) => {
      this.emit('video_frame', frameData);
    });

    // Listen for system status updates
    this.socket.on('system_status', (status: any) => {
      this.emit('system_status', status);
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, this.reconnectInterval);
    }
  }

  // Event subscription system
  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);
  }

  off(event: string, callback: Function) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      const index = eventListeners.indexOf(callback);
      if (index > -1) {
        eventListeners.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(callback => callback(data));
    }
  }

  // Send messages to server
  send(event: string, data: any) {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    }
  }

  // Request to start video stream
  startVideoStream() {
    this.send('start_stream', {});
  }

  // Request to stop video stream
  stopVideoStream() {
    this.send('stop_stream', {});
    // Clear any locally stored video frame when stopping
    this.emit('video_frame', null);
  }

  // Update stream settings
  updateStreamSettings(settings: any) {
    this.send('update_stream_settings', settings);
  }

  get isConnected() {
    return this.socket?.connected || false;
  }

  private setupMockConnection() {
    // Simulate connection after a delay
    setTimeout(() => {
      this.emit('connection', { status: 'connected' });
      
      // Send mock data periodically
      setInterval(() => {
        // Mock video frame (placeholder)
        this.emit('video_frame', 'mock-frame-data');
        
        // Mock occasional detections
        if (Math.random() > 0.8) {
          this.emit('detection', {
            id: 'mock-' + Date.now(),
            type: ['face', 'weapon', 'violence', 'anpr'][Math.floor(Math.random() * 4)],
            timestamp: new Date().toISOString(),
            confidence: Math.random() * 0.5 + 0.5,
          });
        }
        
        // Mock occasional alerts
        if (Math.random() > 0.95) {
          this.emit('alert', {
            id: 'alert-' + Date.now(),
            type: ['face', 'weapon', 'violence', 'anpr'][Math.floor(Math.random() * 4)],
            message: 'Mock alert triggered',
            timestamp: new Date().toISOString(),
            severity: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
            is_read: false,
          });
        }
      }, 5000); // Every 5 seconds
    }, 1000);
  }
}

export const websocketService = new WebSocketService();
export default websocketService;