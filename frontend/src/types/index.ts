export interface Detection {
  id: string;
  type: 'face' | 'weapon' | 'violence' | 'anpr';
  timestamp: string;
  confidence: number;
  location?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  metadata?: Record<string, any>;
}

export interface FaceDetection extends Detection {
  type: 'face';
  person_name?: string;
  is_unknown: boolean;
  face_encoding?: number[];
}

export interface WeaponDetection extends Detection {
  type: 'weapon';
  weapon_type: string;
  severity: 'low' | 'medium' | 'high';
}

export interface ViolenceDetection extends Detection {
  type: 'violence';
  violence_type: string;
  severity: 'low' | 'medium' | 'high';
}

export interface ANPRDetection extends Detection {
  type: 'anpr';
  plate_number: string;
  is_red_listed: boolean;
}

export interface Alert {
  id: string;
  type: 'face' | 'weapon' | 'violence' | 'anpr';
  message: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high';
  is_read: boolean;
}

export interface CameraSettings {
  resolution: string;
  fps: number;
  enabled: boolean;
}

export interface ModuleSettings {
  face: {
    detection_threshold: number;
    recognition_threshold: number;
    max_faces_per_frame: number;
  };
  weapon: {
    detection_threshold: number;
    nms_threshold: number;
  };
  violence: {
    detection_threshold: number;
    frame_buffer_size: number;
  };
  anpr: {
    detection_threshold: number;
    ocr_threshold: number;
  };
}

export interface SystemStatus {
  is_healthy: boolean;
  camera_connected: boolean;
  modules_active: {
    face: boolean;
    weapon: boolean;
    violence: boolean;
    anpr: boolean;
  };
  last_heartbeat: string;
}

export interface AnalyticsData {
  total_detections: number;
  detections_by_type: Record<string, number>;
  detections_by_hour: Array<{
    hour: string;
    count: number;
  }>;
  confidence_distribution: Array<{
    range: string;
    count: number;
  }>;
}