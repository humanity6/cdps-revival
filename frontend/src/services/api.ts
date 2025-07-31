import axios from 'axios';
import { Detection, Alert, ModuleSettings, SystemStatus, AnalyticsData } from '../types';

const API_BASE_URL = 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Detection API
export const detectionsApi = {
  // Upload image for detection
  uploadImage: async (file: File, modules: string[]) => {
    const formData = new FormData();
    formData.append('file', file);
    
    // Use the multi-detection endpoint that exists
    const response = await api.post('/api/detect/multi', {
      image_data: await fileToBase64(file),
      enabled_modules: modules,
    });
    return response.data;
  },

  // Get recent detections
  getRecentDetections: async (limit: number = 50): Promise<Detection[]> => {
    // Return mock data until backend endpoint is available
    const detectionTypes = ['face', 'weapon', 'violence', 'anpr'];
    const mockDetections: Detection[] = [];
    
    for (let i = 0; i < limit; i++) {
      const type = detectionTypes[Math.floor(Math.random() * detectionTypes.length)] as any;
      mockDetections.push({
        id: `mock-${i}`,
        type,
        timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString(),
        confidence: Math.random() * 0.5 + 0.5, // 0.5 to 1.0
        location: {
          x: Math.floor(Math.random() * 400),
          y: Math.floor(Math.random() * 300),
          width: Math.floor(Math.random() * 100) + 50,
          height: Math.floor(Math.random() * 100) + 50,
        },
        metadata: { mock: true },
      });
    }
    
    return mockDetections.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  },

  // Search detections
  searchDetections: async (params: {
    type?: string;
    start_time?: string;
    end_time?: string;
    min_confidence?: number;
    limit?: number;
  }): Promise<Detection[]> => {
    // Use the recent detections mock for now, filtered by type if specified
    const recentDetections = await detectionsApi.getRecentDetections(params.limit || 50);
    
    if (params.type) {
      return recentDetections.filter(d => d.type === params.type);
    }
    
    return recentDetections;
  },
};

// Helper function to convert file to base64
async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.split(',')[1]); // Remove data:image/jpeg;base64, prefix
    };
    reader.onerror = error => reject(error);
  });
}

// Live streaming API
export const liveApi = {
  // Start camera
  startCamera: async () => {
    const response = await api.post('/api/live/start');
    return response.data;
  },

  // Stop camera
  stopCamera: async () => {
    const response = await api.post('/api/live/stop');
    return response.data;
  },

  // Get camera status
  getCameraStatus: async () => {
    const response = await api.get('/api/live/status');
    return response.data;
  },

  // Update camera settings
  updateCameraSettings: async (settings: any) => {
    const response = await api.post('/api/live/settings', settings);
    return response.data;
  },
};

// Settings API
export const settingsApi = {
  // Get all module settings
  getSettings: async (): Promise<ModuleSettings> => {
    const response = await api.get('/api/settings');
    return response.data;
  },

  // Update module settings
  updateSettings: async (module: string, settings: any) => {
    const response = await api.post(`/api/settings/${module}`, settings);
    return response.data;
  },

  // Reset settings to defaults
  resetSettings: async (module: string) => {
    const response = await api.post(`/api/settings/${module}/reset`);
    return response.data;
  },
};

// Alerts API
export const alertsApi = {
  // Get recent alerts
  getRecentAlerts: async (limit: number = 20): Promise<Alert[]> => {
    // Return mock data until backend endpoint is available
    const alertTypes = ['face', 'weapon', 'violence', 'anpr'];
    const severityLevels = ['low', 'medium', 'high'];
    const messages = {
      face: ['Unknown person detected', 'Unauthorized access attempt'],
      weapon: ['Weapon detected in area', 'Security threat identified'],
      violence: ['Violence incident detected', 'Aggressive behavior observed'],
      anpr: ['Red-listed vehicle detected', 'Unauthorized vehicle'],
    };
    
    const mockAlerts: Alert[] = [];
    
    for (let i = 0; i < limit; i++) {
      const type = alertTypes[Math.floor(Math.random() * alertTypes.length)] as any;
      const severity = severityLevels[Math.floor(Math.random() * severityLevels.length)] as any;
      const typeMessages = messages[type] || ['Alert triggered'];
      
      mockAlerts.push({
        id: `alert-${i}`,
        type,
        message: typeMessages[Math.floor(Math.random() * typeMessages.length)],
        timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString(),
        severity,
        is_read: Math.random() > 0.3, // 70% chance of being read
      });
    }
    
    return mockAlerts.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  },

  // Mark alert as read
  markAlertAsRead: async (alertId: string) => {
    // Mock implementation
    return {
      success: true,
      message: 'Alert marked as read',
      alertId,
    };
  },

  // Clear all alerts
  clearAllAlerts: async () => {
    // Mock implementation
    return {
      success: true,
      message: 'All alerts cleared',
    };
  },
};

// Analytics API
export const analyticsApi = {
  // Get analytics data
  getAnalytics: async (timeRange: string = '24h'): Promise<AnalyticsData> => {
    // Return mock data until backend endpoints are available
    return {
      total_detections: Math.floor(Math.random() * 100) + 50,
      detections_by_type: {
        face: Math.floor(Math.random() * 30) + 10,
        weapon: Math.floor(Math.random() * 5),
        violence: Math.floor(Math.random() * 3),
        anpr: Math.floor(Math.random() * 40) + 15,
      },
      detections_by_hour: Array.from({ length: 24 }, (_, i) => ({
        hour: `${String(i).padStart(2, '0')}:00`,
        count: Math.floor(Math.random() * 10),
      })),
      confidence_distribution: [
        { range: '0-20%', count: Math.floor(Math.random() * 5) },
        { range: '20-40%', count: Math.floor(Math.random() * 8) },
        { range: '40-60%', count: Math.floor(Math.random() * 15) + 5 },
        { range: '60-80%', count: Math.floor(Math.random() * 25) + 10 },
        { range: '80-100%', count: Math.floor(Math.random() * 35) + 15 },
      ],
    };
  },

  // Get detection trends
  getDetectionTrends: async (type?: string) => {
    // Return mock data
    return {
      trends: Array.from({ length: 7 }, (_, i) => ({
        date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        count: Math.floor(Math.random() * 50) + 10,
      })).reverse(),
    };
  },
};

// System API
export const systemApi = {
  // Get system status
  getStatus: async (): Promise<SystemStatus> => {
    // Use the health endpoint which actually exists
    const response = await api.get('/health');
    
    // Transform the response to match our SystemStatus interface
    const healthData = response.data;
    return {
      is_healthy: healthData.status === 'healthy',
      camera_connected: true, // Assume true for now
      modules_active: {
        face: healthData.services?.face?.enabled || false,
        weapon: healthData.services?.weapon?.enabled || false,
        violence: healthData.services?.violence?.enabled || false,
        anpr: healthData.services?.anpr?.enabled || false,
      },
      last_heartbeat: new Date().toISOString(),
    };
  },

  // Get system info
  getInfo: async () => {
    const response = await api.get('/info');
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;