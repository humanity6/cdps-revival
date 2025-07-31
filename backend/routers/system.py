"""
System Status and Health API Router
Provides system information and health monitoring
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import psutil
import os
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "is_healthy": True,
            "timestamp": datetime.now().isoformat(),
            "camera_connected": True,  # TODO: Check actual camera status
            "modules_active": {
                "face": True,
                "weapon": True, 
                "violence": True,
                "anpr": True
            },
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3)
            },
            "last_heartbeat": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "is_healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "camera_connected": False,
            "modules_active": {
                "face": False,
                "weapon": False,
                "violence": False, 
                "anpr": False
            },
            "last_heartbeat": datetime.now().isoformat()
        }

@router.get("/info")
async def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    try:
        return {
            "name": "Real-Time Crime Detection System",
            "version": "1.0.0",
            "python_version": os.sys.version,
            "platform": os.name,
            "uptime": "Running",  # TODO: Calculate actual uptime
            "modules": ["anpr", "face", "violence", "weapon"],
            "api_version": "1.0.0",
            "documentation": "/docs"
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))