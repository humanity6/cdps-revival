"""
Unified Database Package for Real-Time Crime Detection System
"""

from .schema import UnifiedDatabaseSchema, initialize_database, EventType, SeverityLevel, EventStatus, AlertStatus
from .models import DatabaseManager, Event, Detection, AnprEvent, FaceEvent, ViolenceEvent, WeaponEvent, Alert, RedListedItem, SystemLog, DetectionStatistics, QueryHelpers
from .operations import DatabaseOperations
from .config import DatabaseConfig, DatabaseConnectionManager, get_connection_manager, initialize_database_connection, shutdown_database_connection

__version__ = "1.0.0"
__all__ = [
    # Schema components
    "UnifiedDatabaseSchema",
    "initialize_database",
    "EventType", 
    "SeverityLevel",
    "EventStatus",
    "AlertStatus",
    
    # ORM Models
    "DatabaseManager",
    "Event",
    "Detection", 
    "AnprEvent",
    "FaceEvent",
    "ViolenceEvent", 
    "WeaponEvent",
    "Alert",
    "RedListedItem",
    "SystemLog",
    "DetectionStatistics",
    "QueryHelpers",
    
    # Operations
    "DatabaseOperations",
    
    # Configuration and Connection Management
    "DatabaseConfig",
    "DatabaseConnectionManager", 
    "get_connection_manager",
    "initialize_database_connection",
    "shutdown_database_connection"
]