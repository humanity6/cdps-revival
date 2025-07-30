"""
Detection API routes for individual module testing
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any, Optional
import asyncio

from models.detection_models import (
    ANPRRequest, ANPRResponse,
    FaceRequest, FaceResponse,
    ViolenceRequest, ViolenceResponse,
    WeaponRequest, WeaponResponse,
    MultiDetectionRequest, MultiDetectionResponse
)
from models.response_models import StandardResponse, ErrorResponse
from utils.image_utils import base64_to_image, image_to_base64, draw_bounding_boxes
from config import get_module_enabled_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detect", tags=["detection"])

# Global variable to hold detection services (will be injected)
detection_services: Dict[str, Any] = {}

def get_detection_services():
    """Dependency to get detection services"""
    return detection_services

def set_detection_services(services: Dict[str, Any]):
    """Set detection services"""
    global detection_services
    detection_services = services

@router.post("/anpr", response_model=ANPRResponse)
async def detect_anpr(
    request: ANPRRequest = None,
    file: UploadFile = File(None),
    services: Dict[str, Any] = Depends(get_detection_services)
):
    """
    Detect license plates in an image
    """
    try:
        # Check if ANPR service is enabled
        if not get_module_enabled_status().get('anpr', False):
            raise HTTPException(status_code=503, detail="ANPR service is disabled")
        
        if 'anpr' not in services:
            raise HTTPException(status_code=503, detail="ANPR service not available")
        
        anpr_service = services['anpr']
        
        # Handle file upload
        if file:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            image_data = await file.read()
            import base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            result = await anpr_service.detect_from_base64(
                image_b64, 
                min_confidence=0.5
            )
        
        # Handle base64 data
        elif request and request.image_data:
            result = await anpr_service.detect_from_base64(
                request.image_data,
                min_confidence=request.min_confidence
            )
        
        # Handle image URL (placeholder - would need to fetch image)
        elif request and request.image_url:
            raise HTTPException(status_code=400, detail="Image URL processing not implemented yet")
        
        else:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ANPR detection error: {e}")
        return ANPRResponse(
            success=False,
            detections=[],
            total_detections=0,
            processing_time=0,
            error=str(e)
        )

@router.post("/face", response_model=FaceResponse)
async def detect_faces(
    request: FaceRequest = None,
    file: UploadFile = File(None),
    services: Dict[str, Any] = Depends(get_detection_services)
):
    """
    Detect and recognize faces in an image
    """
    try:
        # Check if Face service is enabled
        if not get_module_enabled_status().get('face', False):
            raise HTTPException(status_code=503, detail="Face recognition service is disabled")
        
        if 'face' not in services:
            raise HTTPException(status_code=503, detail="Face recognition service not available")
        
        face_service = services['face']
        
        # Handle file upload
        if file:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            image_data = await file.read()
            import base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            result = await face_service.detect_from_base64(
                image_b64,
                recognition_tolerance=0.5
            )
        
        # Handle base64 data
        elif request and request.image_data:
            result = await face_service.detect_from_base64(
                request.image_data,
                recognition_tolerance=request.recognition_tolerance
            )
        
        # Handle image URL
        elif request and request.image_url:
            raise HTTPException(status_code=400, detail="Image URL processing not implemented yet")
        
        else:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return FaceResponse(
            success=False,
            detections=[],
            total_detections=0,
            processing_time=0,
            error=str(e)
        )

@router.post("/violence", response_model=ViolenceResponse)
async def detect_violence(
    request: ViolenceRequest = None,
    file: UploadFile = File(None),
    services: Dict[str, Any] = Depends(get_detection_services)
):
    """
    Detect violence in an image
    """
    try:
        # Check if Violence service is enabled
        if not get_module_enabled_status().get('violence', False):
            raise HTTPException(status_code=503, detail="Violence detection service is disabled")
        
        if 'violence' not in services:
            raise HTTPException(status_code=503, detail="Violence detection service not available")
        
        violence_service = services['violence']
        
        # Handle file upload
        if file:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            image_data = await file.read()
            import base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            result = await violence_service.detect_from_base64(
                image_b64,
                confidence_threshold=0.5
            )
        
        # Handle base64 data
        elif request and request.image_data:
            result = await violence_service.detect_from_base64(
                request.image_data,
                confidence_threshold=request.confidence_threshold
            )
        
        # Handle video data (for batch processing)
        elif request and request.video_data:
            import base64
            video_bytes = base64.b64decode(request.video_data)
            result_dict = await violence_service.detect_violence_video(
                video_bytes,
                confidence_threshold=request.confidence_threshold
            )
            
            # Convert to ViolenceResponse format
            if result_dict.get('success'):
                from models.detection_models import ViolenceDetection
                detection = ViolenceDetection(
                    is_violence=result_dict.get('violence_frames', 0) > 0,
                    confidence=result_dict.get('avg_confidence', 0.0)
                )
                result = ViolenceResponse(
                    success=True,
                    detection=detection,
                    processing_time=0
                )
            else:
                result = ViolenceResponse(
                    success=False,
                    detection=ViolenceDetection(is_violence=False, confidence=0.0),
                    processing_time=0,
                    error=result_dict.get('error', 'Unknown error')
                )
        
        # Handle image URL
        elif request and request.image_url:
            raise HTTPException(status_code=400, detail="Image URL processing not implemented yet")
        
        else:
            raise HTTPException(status_code=400, detail="No image or video data provided")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Violence detection error: {e}")
        from models.detection_models import ViolenceDetection
        return ViolenceResponse(
            success=False,
            detection=ViolenceDetection(is_violence=False, confidence=0.0),
            processing_time=0,
            error=str(e)
        )

@router.post("/weapon", response_model=WeaponResponse)
async def detect_weapons(
    request: WeaponRequest = None,
    file: UploadFile = File(None),
    services: Dict[str, Any] = Depends(get_detection_services)
):
    """
    Detect weapons in an image
    """
    try:
        # Check if Weapon service is enabled
        if not get_module_enabled_status().get('weapon', False):
            raise HTTPException(status_code=503, detail="Weapon detection service is disabled")
        
        if 'weapon' not in services:
            raise HTTPException(status_code=503, detail="Weapon detection service not available")
        
        weapon_service = services['weapon']
        
        # Handle file upload
        if file:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            image_data = await file.read()
            import base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            result = await weapon_service.detect_from_base64(
                image_b64,
                confidence_threshold=0.5
            )
        
        # Handle base64 data
        elif request and request.image_data:
            result = await weapon_service.detect_from_base64(
                request.image_data,
                confidence_threshold=request.confidence_threshold
            )
        
        # Handle image URL
        elif request and request.image_url:
            raise HTTPException(status_code=400, detail="Image URL processing not implemented yet")
        
        else:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Weapon detection error: {e}")
        return WeaponResponse(
            success=False,
            detections=[],
            total_detections=0,
            processing_time=0,
            error=str(e)
        )

@router.post("/multi", response_model=MultiDetectionResponse)
async def multi_detection(
    request: MultiDetectionRequest,
    services: Dict[str, Any] = Depends(get_detection_services)
):
    """
    Run multiple detection modules on a single image
    """
    import time
    start_time = time.time()
    
    try:
        if not request.image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Convert base64 to image once
        image = base64_to_image(request.image_data)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Initialize results
        results = MultiDetectionResponse(
            success=True,
            processing_time=0
        )
        
        # Run enabled detections concurrently
        detection_tasks = []
        enabled_status = get_module_enabled_status()
        
        # ANPR Detection
        if 'anpr' in request.enabled_modules and enabled_status.get('anpr') and 'anpr' in services:
            detection_tasks.append(('anpr', services['anpr'].detect_plates(image)))
        
        # Face Detection
        if 'face' in request.enabled_modules and enabled_status.get('face') and 'face' in services:
            tolerance = request.face_config.get('recognition_tolerance', 0.5) if request.face_config else 0.5
            detection_tasks.append(('face', services['face'].detect_faces(image, tolerance)))
        
        # Weapon Detection
        if 'weapon' in request.enabled_modules and enabled_status.get('weapon') and 'weapon' in services:
            confidence = request.weapon_config.get('confidence_threshold', 0.5) if request.weapon_config else 0.5
            detection_tasks.append(('weapon', services['weapon'].detect_weapons(image, confidence)))
        
        # Violence Detection (if requested)
        if 'violence' in request.enabled_modules and enabled_status.get('violence') and 'violence' in services:
            confidence = request.violence_config.get('confidence_threshold', 0.5) if request.violence_config else 0.5
            detection_tasks.append(('violence', services['violence'].detect_violence(image, confidence)))
        
        # Run detections concurrently
        if detection_tasks:
            detection_results = await asyncio.gather(
                *[task[1] for task in detection_tasks],
                return_exceptions=True
            )
            
            # Process results
            for i, (module_name, _) in enumerate(detection_tasks):
                result = detection_results[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Error in {module_name} detection: {result}")
                    continue
                
                if module_name == 'anpr':
                    results.anpr_results = result
                elif module_name == 'face':
                    results.face_results = result
                elif module_name == 'weapon':
                    results.weapon_results = result
                elif module_name == 'violence':
                    results.violence_results = result
        
        results.processing_time = time.time() - start_time
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-detection error: {e}")
        return MultiDetectionResponse(
            success=False,
            processing_time=time.time() - start_time,
            error=str(e)
        )

@router.get("/health")
async def health_check(services: Dict[str, Any] = Depends(get_detection_services)):
    """
    Health check for all detection services
    """
    try:
        health_status = {
            'overall_status': 'healthy',
            'services': {},
            'enabled_modules': get_module_enabled_status()
        }
        
        for service_name, service in services.items():
            try:
                if hasattr(service, 'get_health_status'):
                    service_health = service.get_health_status()
                    health_status['services'][service_name] = service_health
                else:
                    health_status['services'][service_name] = {
                        'enabled': True,
                        'status': 'unknown'
                    }
            except Exception as e:
                health_status['services'][service_name] = {
                    'enabled': False,
                    'error': str(e)
                }
                health_status['overall_status'] = 'degraded'
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                'overall_status': 'error',
                'error': str(e)
            }
        )