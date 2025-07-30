"""
Settings API routes for configuration management
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

from models.detection_models import SystemSettings, ModuleSettings
from models.response_models import StandardResponse
from config import module_config, get_module_enabled_status, update_module_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Global detection services reference
detection_services: Dict[str, Any] = {}

def get_detection_services():
    """Dependency to get detection services"""
    return detection_services

def set_detection_services(services: Dict[str, Any]):
    """Set detection services"""
    global detection_services
    detection_services = services

@router.get("/")
async def get_all_settings():
    """
    Get all system and module settings
    """
    try:
        settings = {
            'modules': module_config.get_all_configs(),
            'enabled_status': get_module_enabled_status(),
            'system': {
                'api_version': '1.0.0',
                'max_file_size': '16MB',
                'supported_formats': ['jpg', 'jpeg', 'png', 'mp4', 'avi']
            }
        }
        
        return JSONResponse(content=settings)
        
    except Exception as e:
        logger.error(f"Failed to get settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{module_name}")
async def get_module_settings(module_name: str):
    """
    Get settings for a specific module
    """
    try:
        if module_name not in ['anpr', 'face', 'violence', 'weapon']:
            raise HTTPException(status_code=404, detail=f"Unknown module: {module_name}")
        
        config = module_config.get_module_config(module_name)
        if not config:
            raise HTTPException(status_code=404, detail=f"No configuration found for module: {module_name}")
        
        return JSONResponse(content=config)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get {module_name} settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{module_name}")
async def update_module_settings(
    module_name: str, 
    settings: Dict[str, Any],
    services: Dict[str, Any] = Depends(get_detection_services)
):
    """
    Update settings for a specific module
    """
    try:
        if module_name not in ['anpr', 'face', 'violence', 'weapon']:
            raise HTTPException(status_code=404, detail=f"Unknown module: {module_name}")
        
        # Update local configuration
        success = module_config.update_module_config(module_name, settings)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update module configuration")
        
        # Update service configuration if available
        if module_name in services:
            service = services[module_name]
            if hasattr(service, 'update_config'):
                try:
                    service.update_config(settings)
                except Exception as e:
                    logger.warning(f"Failed to update {module_name} service config: {e}")
        
        # Get updated configuration
        updated_config = module_config.get_module_config(module_name)
        
        return StandardResponse(
            success=True,
            message=f"{module_name} settings updated successfully",
            data=updated_config
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update {module_name} settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{module_name}/reset")
async def reset_module_settings(module_name: str):
    """
    Reset module settings to defaults
    """
    try:
        if module_name not in ['anpr', 'face', 'violence', 'weapon']:
            raise HTTPException(status_code=404, detail=f"Unknown module: {module_name}")
        
        success = module_config.reset_module_config(module_name)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to reset module configuration")
        
        # Get reset configuration
        reset_config = module_config.get_module_config(module_name)
        
        return StandardResponse(
            success=True,
            message=f"{module_name} settings reset to defaults",
            data=reset_config
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset {module_name} settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{module_name}/toggle")
async def toggle_module(module_name: str, enabled: bool):
    """
    Enable or disable a detection module
    """
    try:
        if module_name not in ['anpr', 'face', 'violence', 'weapon']:
            raise HTTPException(status_code=404, detail=f"Unknown module: {module_name}")
        
        success = update_module_status(module_name, enabled)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update module status")
        
        return StandardResponse(
            success=True,
            message=f"{module_name} module {'enabled' if enabled else 'disabled'}",
            data={'module': module_name, 'enabled': enabled}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle {module_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/modules")
async def get_module_status():
    """
    Get enabled/disabled status of all modules
    """
    try:
        status = get_module_enabled_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Failed to get module status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/{module_name}")
async def get_module_health(
    module_name: str,
    services: Dict[str, Any] = Depends(get_detection_services)
):
    """
    Get health status of a specific module
    """
    try:
        if module_name not in ['anpr', 'face', 'violence', 'weapon']:
            raise HTTPException(status_code=404, detail=f"Unknown module: {module_name}")
        
        if module_name not in services:
            return JSONResponse(content={
                'module': module_name,
                'status': 'not_available',
                'enabled': False,
                'error': 'Service not loaded'
            })
        
        service = services[module_name]
        
        if hasattr(service, 'get_health_status'):
            health = service.get_health_status()
            health['module'] = module_name
            return JSONResponse(content=health)
        else:
            return JSONResponse(content={
                'module': module_name,
                'status': 'unknown',
                'enabled': True,
                'message': 'Health check not implemented for this module'
            })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get {module_name} health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_all_modules_health(services: Dict[str, Any] = Depends(get_detection_services)):
    """
    Get health status of all modules
    """
    try:
        health_status = {}
        
        for module_name in ['anpr', 'face', 'violence', 'weapon']:
            if module_name in services:
                service = services[module_name]
                if hasattr(service, 'get_health_status'):
                    health_status[module_name] = service.get_health_status()
                else:
                    health_status[module_name] = {
                        'status': 'unknown',
                        'enabled': True
                    }
            else:
                health_status[module_name] = {
                    'status': 'not_available',
                    'enabled': False,
                    'error': 'Service not loaded'
                }
        
        # Add overall system health
        overall_status = 'healthy'
        if any(status.get('status') == 'not_available' for status in health_status.values()):
            overall_status = 'degraded'
        if any(status.get('enabled') == False and status.get('status') == 'error' for status in health_status.values()):
            overall_status = 'error'
        
        return JSONResponse(content={
            'overall_status': overall_status,
            'modules': health_status,
            'enabled_modules': get_module_enabled_status()
        })
        
    except Exception as e:
        logger.error(f"Failed to get all modules health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anpr/red-listed")
async def get_red_listed_vehicles(services: Dict[str, Any] = Depends(get_detection_services)):
    """
    Get list of red-listed vehicles (ANPR specific)
    """
    try:
        if 'anpr' not in services:
            raise HTTPException(status_code=503, detail="ANPR service not available")
        
        anpr_service = services['anpr']
        if hasattr(anpr_service, 'get_red_listed_vehicles'):
            vehicles = anpr_service.get_red_listed_vehicles()
            return JSONResponse(content={
                'vehicles': [{'plate': plate, 'reason': reason} for plate, reason in vehicles],
                'count': len(vehicles)
            })
        else:
            return JSONResponse(content={'vehicles': [], 'count': 0})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get red-listed vehicles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/anpr/red-listed")
async def add_red_listed_vehicle(
    plate_number: str,
    reason: str = "Suspicious Activity",
    services: Dict[str, Any] = Depends(get_detection_services)
):
    """
    Add a vehicle to the red-listed vehicles (ANPR specific)
    """
    try:
        if 'anpr' not in services:
            raise HTTPException(status_code=503, detail="ANPR service not available")
        
        anpr_service = services['anpr']
        if hasattr(anpr_service, 'add_red_listed_vehicle'):
            success = anpr_service.add_red_listed_vehicle(plate_number, reason)
            if success:
                return StandardResponse(
                    success=True,
                    message=f"Vehicle {plate_number} added to red list",
                    data={'plate': plate_number, 'reason': reason}
                )
            else:
                raise HTTPException(status_code=400, detail="Failed to add vehicle to red list")
        else:
            raise HTTPException(status_code=501, detail="Red list management not implemented")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add red-listed vehicle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/anpr/red-listed/{plate_number}")
async def remove_red_listed_vehicle(
    plate_number: str,
    services: Dict[str, Any] = Depends(get_detection_services)
):
    """
    Remove a vehicle from the red-listed vehicles (ANPR specific)
    """
    try:
        if 'anpr' not in services:
            raise HTTPException(status_code=503, detail="ANPR service not available")
        
        anpr_service = services['anpr']
        if hasattr(anpr_service, 'remove_red_listed_vehicle'):
            success = anpr_service.remove_red_listed_vehicle(plate_number)
            if success:
                return StandardResponse(
                    success=True,
                    message=f"Vehicle {plate_number} removed from red list"
                )
            else:
                raise HTTPException(status_code=404, detail="Vehicle not found in red list")
        else:
            raise HTTPException(status_code=501, detail="Red list management not implemented")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove red-listed vehicle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/face/known-faces")
async def get_known_faces_info(services: Dict[str, Any] = Depends(get_detection_services)):
    """
    Get information about known faces (Face Recognition specific)
    """
    try:
        if 'face' not in services:
            raise HTTPException(status_code=503, detail="Face recognition service not available")
        
        face_service = services['face']
        if hasattr(face_service, 'get_known_faces_count'):
            counts = face_service.get_known_faces_count()
            return JSONResponse(content=counts)
        else:
            return JSONResponse(content={'total': 0, 'restricted': 0, 'criminal': 0, 'unknown': 0})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get known faces info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/face/reload")
async def reload_known_faces(services: Dict[str, Any] = Depends(get_detection_services)):
    """
    Reload known faces from directories (Face Recognition specific)
    """
    try:
        if 'face' not in services:
            raise HTTPException(status_code=503, detail="Face recognition service not available")
        
        face_service = services['face']
        if hasattr(face_service, 'reload_known_faces'):
            success = face_service.reload_known_faces()
            if success:
                return StandardResponse(
                    success=True,
                    message="Known faces reloaded successfully"
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to reload known faces")
        else:
            raise HTTPException(status_code=501, detail="Face reload not implemented")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reload known faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))