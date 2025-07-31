import { useEffect, useCallback, useState } from 'react';
import websocketService from '../services/websocket';
import { Detection, Alert } from '../types';

export const useWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [latestDetection, setLatestDetection] = useState<Detection | null>(null);
  const [latestAlert, setLatestAlert] = useState<Alert | null>(null);
  const [videoFrame, setVideoFrame] = useState<string | null>(null);

  useEffect(() => {
    // Connection status handler
    const handleConnection = (data: { status: string }) => {
      setIsConnected(data.status === 'connected');
    };

    // Detection handler
    const handleDetection = (detection: Detection) => {
      setLatestDetection(detection);
    };

    // Alert handler
    const handleAlert = (alert: Alert) => {
      setLatestAlert(alert);
    };

    // Video frame handler
    const handleVideoFrame = (frameData: string) => {
      setVideoFrame(frameData);
    };

    // Subscribe to events
    websocketService.on('connection', handleConnection);
    websocketService.on('detection', handleDetection);
    websocketService.on('alert', handleAlert);
    websocketService.on('video_frame', handleVideoFrame);

    // Connect to WebSocket
    websocketService.connect();

    // Cleanup on unmount
    return () => {
      websocketService.off('connection', handleConnection);
      websocketService.off('detection', handleDetection);
      websocketService.off('alert', handleAlert);
      websocketService.off('video_frame', handleVideoFrame);
    };
  }, []);

  const startVideoStream = useCallback(() => {
    websocketService.startVideoStream();
  }, []);

  const stopVideoStream = useCallback(() => {
    websocketService.stopVideoStream();
  }, []);

  const updateStreamSettings = useCallback((settings: any) => {
    websocketService.updateStreamSettings(settings);
  }, []);

  return {
    isConnected,
    latestDetection,
    latestAlert,
    videoFrame,
    startVideoStream,
    stopVideoStream,
    updateStreamSettings,
  };
};