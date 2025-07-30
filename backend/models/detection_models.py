"""
Pydantic models for detection requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class DetectionType(str, Enum):
    ANPR = "anpr"
    FACE = "face"
    VIOLENCE = "violence"
    WEAPON = "weapon"

class ImageFormat(str, Enum):
    JPG = "jpg"
    PNG = "png"
    JPEG = "jpeg"

# Base Models
class BoundingBox(BaseModel):
    x1: int = Field(..., description="Left coordinate")
    y1: int = Field(..., description="Top coordinate") 
    x2: int = Field(..., description="Right coordinate")
    y2: int = Field(..., description="Bottom coordinate")

class Detection(BaseModel):
    bbox: BoundingBox
    confidence: float = Field(..., ge=0, le=1)
    class_name: str

# ANPR Models
class ANPRDetection(Detection):
    plate_number: str
    is_red_listed: bool = False
    alert_reason: Optional[str] = None

class ANPRRequest(BaseModel):
    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = None
    min_confidence: float = Field(0.5, ge=0, le=1)

class ANPRResponse(BaseModel):
    success: bool
    detections: List[ANPRDetection]
    total_detections: int
    processing_time: float
    error: Optional[str] = None

# Face Recognition Models  
class FaceDetection(Detection):
    person_name: str
    category: str  # unknown, restricted, criminal, safe
    alert_triggered: bool = False

class FaceRequest(BaseModel):
    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = None
    recognition_tolerance: float = Field(0.5, ge=0, le=1)

class FaceResponse(BaseModel):
    success: bool
    detections: List[FaceDetection] 
    total_detections: int
    processing_time: float
    error: Optional[str] = None

# Violence Detection Models
class ViolenceDetection(BaseModel):
    is_violence: bool
    confidence: float = Field(..., ge=0, le=1)
    frame_number: Optional[int] = None
    timestamp: Optional[float] = None

class ViolenceRequest(BaseModel):
    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    video_data: Optional[str] = Field(None, description="Base64 encoded video")
    image_url: Optional[str] = None
    confidence_threshold: float = Field(0.5, ge=0, le=1)

class ViolenceResponse(BaseModel):
    success: bool
    detection: ViolenceDetection
    processing_time: float
    error: Optional[str] = None

# Weapon Detection Models
class WeaponDetection(Detection):
    weapon_type: str

class WeaponRequest(BaseModel):
    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = None
    confidence_threshold: float = Field(0.5, ge=0, le=1)

class WeaponResponse(BaseModel):
    success: bool
    detections: List[WeaponDetection]
    total_detections: int
    processing_time: float
    error: Optional[str] = None

# Multi-Detection Models
class MultiDetectionRequest(BaseModel):
    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = None
    enabled_modules: List[DetectionType] = Field(default_factory=lambda: [DetectionType.ANPR, DetectionType.FACE, DetectionType.WEAPON])
    anpr_config: Optional[Dict[str, Any]] = None
    face_config: Optional[Dict[str, Any]] = None
    weapon_config: Optional[Dict[str, Any]] = None

class MultiDetectionResponse(BaseModel):
    success: bool
    anpr_results: Optional[ANPRResponse] = None
    face_results: Optional[FaceResponse] = None
    weapon_results: Optional[WeaponResponse] = None
    processing_time: float
    error: Optional[str] = None

# Live Feed Models
class LiveFeedConfig(BaseModel):
    camera_index: int = Field(0, ge=0)
    width: int = Field(640, gt=0)
    height: int = Field(480, gt=0)
    fps: int = Field(30, gt=0)
    enabled_modules: List[DetectionType] = Field(default_factory=lambda: [DetectionType.ANPR, DetectionType.FACE, DetectionType.WEAPON])

class LiveFeedFrame(BaseModel):
    frame_data: str = Field(..., description="Base64 encoded frame")
    timestamp: float
    detections: Dict[str, Any] = Field(default_factory=dict)
    fps: float

# Settings Models
class ModuleSettings(BaseModel):
    enabled: bool = True
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    additional_params: Dict[str, Any] = Field(default_factory=dict)

class SystemSettings(BaseModel):
    anpr: ModuleSettings = Field(default_factory=ModuleSettings)
    face: ModuleSettings = Field(default_factory=ModuleSettings)
    violence: ModuleSettings = Field(default_factory=ModuleSettings)
    weapon: ModuleSettings = Field(default_factory=ModuleSettings)
    live_feed: LiveFeedConfig = Field(default_factory=LiveFeedConfig)

# Health Check Models
class ModuleHealth(BaseModel):
    name: str
    status: str  # healthy, error, disabled
    last_check: float
    error_message: Optional[str] = None

class SystemHealth(BaseModel):
    overall_status: str
    modules: List[ModuleHealth]
    uptime: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None