"""
Detections History API Router
Provides detection history and search functionality
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import random
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

def generate_sample_detection(detection_type: str) -> Dict[str, Any]:
    """Generate a sample detection for testing"""
    base_detection = {
        "id": str(uuid.uuid4()),
        "type": detection_type,
        "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 1440))).isoformat(),
        "confidence": round(random.uniform(0.5, 0.99), 3),
        "location": {
            "x": random.randint(50, 500),
            "y": random.randint(50, 400),
            "width": random.randint(50, 200),
            "height": random.randint(50, 200)
        },
        "metadata": {}
    }
    
    # Add type-specific data
    if detection_type == "face":
        base_detection.update({
            "person_name": random.choice(["John Doe", "Jane Smith", "Unknown Person", "Mike Johnson", "Sarah Wilson"]),
            "is_unknown": random.choice([True, False]),
            "face_encoding": [random.uniform(-1, 1) for _ in range(128)]
        })
    elif detection_type == "weapon":
        base_detection.update({
            "weapon_type": random.choice(["pistol", "knife", "rifle", "suspicious_object"]),
            "severity": random.choice(["low", "medium", "high"])
        })
    elif detection_type == "violence":
        base_detection.update({
            "violence_type": random.choice(["fighting", "assault", "aggressive_behavior"]),
            "severity": random.choice(["low", "medium", "high"])
        })
    elif detection_type == "anpr":
        base_detection.update({
            "plate_number": f"{random.choice(['ABC', 'XYZ', 'DEF'])}{random.randint(100, 999)}",
            "is_red_listed": random.choice([True, False])
        })
    
    return base_detection

@router.get("/recent")
async def get_recent_detections(limit: int = Query(50, ge=1, le=200)) -> List[Dict[str, Any]]:
    """Get recent detections"""
    try:
        # Generate sample detections
        # TODO: Replace with actual database queries
        
        detections = []
        detection_types = ["face", "weapon", "violence", "anpr"]
        
        for _ in range(limit):
            detection_type = random.choice(detection_types)
            detection = generate_sample_detection(detection_type)
            detections.append(detection)
        
        # Sort by timestamp (most recent first)
        detections.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return detections
        
    except Exception as e:
        logger.error(f"Error getting recent detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_detections(
    type: Optional[str] = Query(None, description="Detection type filter"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset")
) -> List[Dict[str, Any]]:
    """Search detections with filters"""
    try:
        # Generate sample filtered results
        # TODO: Replace with actual database queries
        
        detections = []
        
        # Determine types to generate
        if type:
            detection_types = [type] if type in ["face", "weapon", "violence", "anpr"] else ["face"]
        else:
            detection_types = ["face", "weapon", "violence", "anpr"]
        
        # Generate sample data
        total_results = limit + offset
        for i in range(total_results):
            detection_type = random.choice(detection_types)
            detection = generate_sample_detection(detection_type)
            
            # Apply confidence filter
            if min_confidence and detection["confidence"] < min_confidence:
                continue
                
            # Apply time filters (basic simulation)
            if start_time:
                detection_time = datetime.fromisoformat(detection["timestamp"].replace('Z', '+00:00'))
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                if detection_time < start_dt:
                    continue
            
            if end_time:
                detection_time = datetime.fromisoformat(detection["timestamp"].replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                if detection_time > end_dt:
                    continue
            
            detections.append(detection)
        
        # Apply pagination
        paginated_results = detections[offset:offset + limit]
        
        # Sort by timestamp (most recent first)
        paginated_results.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return paginated_results
        
    except Exception as e:
        logger.error(f"Error searching detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/count")
async def get_detection_count(
    type: Optional[str] = Query(None, description="Detection type filter"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)")
) -> Dict[str, Any]:
    """Get total count of detections matching criteria"""
    try:
        # Generate sample count
        # TODO: Replace with actual database queries
        
        if type:
            count = random.randint(10, 100)
        else:
            count = random.randint(50, 500)
        
        return {
            "total_count": count,
            "filters": {
                "type": type,
                "start_time": start_time,
                "end_time": end_time
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting detection count: {e}")
        raise HTTPException(status_code=500, detail=str(e))