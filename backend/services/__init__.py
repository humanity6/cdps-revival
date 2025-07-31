# Services package - import services individually to avoid circular imports

def get_anpr_service():
    from .anpr_service import ANPRService
    return ANPRService

def get_face_service():
    from .face_service import FaceService
    return FaceService

def get_violence_service():
    from .violence_service import ViolenceService
    return ViolenceService

def get_weapon_service():
    from .weapon_service import WeaponService
    return WeaponService

def get_alert_service():
    from .alert_service import TelegramAlertService
    return TelegramAlertService

def get_analytics_service():
    from .analytics_service import AnalyticsService
    return AnalyticsService

def get_database_service():
    from .database_service import DatabaseService
    return DatabaseService

# For backward compatibility, provide the services as module-level imports
# but only import them when actually needed
try:
    from .anpr_service import ANPRService
    from .face_service import FaceService  
    from .violence_service import ViolenceService
    from .weapon_service import WeaponService
except ImportError as e:
    import logging
    logging.warning(f"Some services could not be imported: {e}")
    # Create dummy classes to prevent import errors
    class ANPRService: pass
    class FaceService: pass
    class ViolenceService: pass
    class WeaponService: pass

__all__ = [
    'ANPRService',
    'FaceService', 
    'ViolenceService',
    'WeaponService',
    'get_anpr_service',
    'get_face_service',
    'get_violence_service', 
    'get_weapon_service',
    'get_alert_service',
    'get_analytics_service',
    'get_database_service'
]