"""
Database Operations Service for Unified Crime Detection System
Provides comprehensive CRUD operations for all detection modules
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func, and_, or_, desc, asc
import json

from .models import (
    DatabaseManager, Event, Detection, AnprEvent, FaceEvent, 
    ViolenceEvent, WeaponEvent, Alert, RedListedItem, 
    SystemLog, DetectionStatistics, QueryHelpers
)
from .schema import EventType, SeverityLevel, EventStatus, AlertStatus

logger = logging.getLogger(__name__)

class DatabaseOperations:
    """
    Comprehensive database operations service for crime detection system.
    Provides CRUD operations, analytics queries, and utility functions.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.query_helpers = QueryHelpers()
    
    # Event Operations
    def create_event(self, event_data: Dict[str, Any]) -> Optional[int]:
        """Create a new event record."""
        session = self.db_manager.get_session()
        try:
            event = Event(
                event_type=event_data['event_type'],
                confidence_score=event_data.get('confidence_score'),
                location=event_data.get('location', 'Camera_1'),
                image_path=event_data.get('image_path'),
                processing_time=event_data.get('processing_time'),
                severity_level=event_data.get('severity_level', 'MEDIUM'),
                alert_triggered=event_data.get('alert_triggered', False)
            )
            
            # Set metadata if provided
            if 'metadata' in event_data:
                event.set_metadata(event_data['metadata'])
            
            session.add(event)
            session.commit()
            event_id = event.id
            
            logger.info(f"Created event with ID: {event_id}, type: {event.event_type}")
            return event_id
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create event: {e}")
            return None
        finally:
            self.db_manager.close_session(session)
    
    def get_event(self, event_id: int) -> Optional[Event]:
        """Get an event by ID."""
        session = self.db_manager.get_session()
        try:
            event = session.query(Event).filter(Event.id == event_id).first()
            return event
        finally:
            self.db_manager.close_session(session)
    
    def update_event_status(self, event_id: int, status: str) -> bool:
        """Update event status."""
        session = self.db_manager.get_session()
        try:
            event = session.query(Event).filter(Event.id == event_id).first()
            if event:
                event.status = status
                event.updated_at = datetime.now()
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to update event status: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    # Detection Operations
    def add_detection(self, event_id: int, detection_data: Dict[str, Any]) -> Optional[int]:
        """Add detection details to an event."""
        session = self.db_manager.get_session()
        try:
            detection = Detection(
                event_id=event_id,
                detection_type=detection_data['detection_type'],
                confidence=detection_data.get('confidence'),
                class_name=detection_data.get('class_name')
            )
            
            # Set bounding box if provided
            if 'bounding_box' in detection_data:
                bbox = detection_data['bounding_box']
                detection.set_bounding_box(bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'])
            
            # Set additional data if provided
            if 'additional_data' in detection_data:
                detection.set_additional_data(detection_data['additional_data'])
            
            session.add(detection)
            session.commit()
            return detection.id
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to add detection: {e}")
            return None
        finally:
            self.db_manager.close_session(session)
    
    # ANPR Operations
    def create_anpr_event(self, event_id: int, anpr_data: Dict[str, Any]) -> bool:
        """Create ANPR-specific event details."""
        session = self.db_manager.get_session()
        try:
            anpr_event = AnprEvent(
                event_id=event_id,
                plate_number=anpr_data['plate_number'].upper(),
                is_red_listed=anpr_data.get('is_red_listed', False),
                alert_reason=anpr_data.get('alert_reason'),
                vehicle_make_model=anpr_data.get('vehicle_make_model'),
                country_code=anpr_data.get('country_code'),
                region=anpr_data.get('region'),
                plate_confidence=anpr_data.get('plate_confidence'),
                ocr_text_raw=anpr_data.get('ocr_text_raw')
            )
            
            session.add(anpr_event)
            session.commit()
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create ANPR event: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def get_anpr_detections_by_plate(self, plate_number: str, days: int = 30) -> List[Event]:
        """Get ANPR detections for a specific plate number."""
        session = self.db_manager.get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            events = session.query(Event).join(AnprEvent).filter(
                and_(
                    AnprEvent.plate_number == plate_number.upper(),
                    Event.timestamp >= cutoff_date
                )
            ).order_by(desc(Event.timestamp)).all()
            
            return events
        finally:
            self.db_manager.close_session(session)
    
    # Face Recognition Operations
    def create_face_event(self, event_id: int, face_data: Dict[str, Any]) -> bool:
        """Create face recognition-specific event details."""
        session = self.db_manager.get_session()
        try:
            face_event = FaceEvent(
                event_id=event_id,
                person_name=face_data.get('person_name'),
                person_category=face_data.get('person_category', 'unknown'),
                recognition_confidence=face_data.get('recognition_confidence'),
                face_encoding=face_data.get('face_encoding'),
                age_estimate=face_data.get('age_estimate'),
                gender_estimate=face_data.get('gender_estimate'),
                emotion_detected=face_data.get('emotion_detected'),
                face_quality_score=face_data.get('face_quality_score'),
                is_masked=face_data.get('is_masked', False)
            )
            
            session.add(face_event)
            session.commit()
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create face event: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def get_unknown_face_detections(self, hours: int = 24) -> List[Event]:
        """Get recent unknown face detections."""
        session = self.db_manager.get_session()
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            events = session.query(Event).join(FaceEvent).filter(
                and_(
                    FaceEvent.person_category == 'unknown',
                    Event.timestamp >= cutoff_time
                )
            ).order_by(desc(Event.timestamp)).all()
            
            return events
        finally:
            self.db_manager.close_session(session)
    
    # Violence Detection Operations
    def create_violence_event(self, event_id: int, violence_data: Dict[str, Any]) -> bool:
        """Create violence detection-specific event details."""
        session = self.db_manager.get_session()
        try:
            violence_event = ViolenceEvent(
                event_id=event_id,
                is_violence=violence_data['is_violence'],
                violence_type=violence_data.get('violence_type'),
                violence_intensity=violence_data.get('violence_intensity'),
                duration_seconds=violence_data.get('duration_seconds'),
                people_count=violence_data.get('people_count'),
                movement_intensity=violence_data.get('movement_intensity')
            )
            
            # Set frame analysis if provided
            if 'frame_analysis' in violence_data:
                violence_event.set_frame_analysis(violence_data['frame_analysis'])
            
            session.add(violence_event)
            session.commit()
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create violence event: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def get_violence_detections(self, hours: int = 24) -> List[Event]:
        """Get recent violence detections."""
        session = self.db_manager.get_session()
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            events = session.query(Event).join(ViolenceEvent).filter(
                and_(
                    ViolenceEvent.is_violence == True,
                    Event.timestamp >= cutoff_time
                )
            ).order_by(desc(Event.timestamp)).all()
            
            return events
        finally:
            self.db_manager.close_session(session)
    
    # Weapon Detection Operations
    def create_weapon_event(self, event_id: int, weapon_data: Dict[str, Any]) -> bool:
        """Create weapon detection-specific event details."""
        session = self.db_manager.get_session()
        try:
            weapon_event = WeaponEvent(
                event_id=event_id,
                weapon_type=weapon_data['weapon_type'],
                weapon_category=weapon_data.get('weapon_category'),
                threat_level=weapon_data.get('threat_level', 'MEDIUM'),
                weapon_size_estimate=weapon_data.get('weapon_size_estimate'),
                is_concealed=weapon_data.get('is_concealed', False),
                person_holding_weapon=weapon_data.get('person_holding_weapon', False),
                weapon_condition=weapon_data.get('weapon_condition'),
                multiple_weapons=weapon_data.get('multiple_weapons', False),
                weapon_count=weapon_data.get('weapon_count', 1)
            )
            
            session.add(weapon_event)
            session.commit()
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create weapon event: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def get_weapon_detections(self, hours: int = 24, threat_level: str = None) -> List[Event]:
        """Get recent weapon detections."""
        session = self.db_manager.get_session()
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = session.query(Event).join(WeaponEvent).filter(
                Event.timestamp >= cutoff_time
            )
            
            if threat_level:
                query = query.filter(WeaponEvent.threat_level == threat_level)
            
            events = query.order_by(desc(Event.timestamp)).all()
            return events
        finally:
            self.db_manager.close_session(session)
    
    # Alert Operations
    def create_alert(self, alert_data: Dict[str, Any]) -> Optional[int]:
        """Create a new alert record."""
        session = self.db_manager.get_session()
        try:
            alert = Alert(
                event_id=alert_data['event_id'],
                alert_type=alert_data['alert_type'],
                recipient=alert_data['recipient'],
                message_content=alert_data.get('message_content'),
                message_template=alert_data.get('message_template'),
                priority=alert_data.get('priority', 5),
                expires_at=alert_data.get('expires_at')
            )
            
            session.add(alert)
            session.commit()
            return alert.id
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create alert: {e}")
            return None
        finally:
            self.db_manager.close_session(session)
    
    def update_alert_status(self, alert_id: int, status: str, error_message: str = None, 
                           response_data: Dict[str, Any] = None) -> bool:
        """Update alert status and response data."""
        session = self.db_manager.get_session()
        try:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                alert.status = status
                if status == 'SENT':
                    alert.sent_at = datetime.now()
                    alert.delivery_confirmed = True
                elif status == 'FAILED':
                    alert.error_message = error_message
                    alert.retry_count += 1
                
                if response_data:
                    alert.set_response_data(response_data)
                
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to update alert status: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def get_pending_alerts(self, alert_type: str = None, limit: int = 100) -> List[Alert]:
        """Get pending alerts for processing."""
        session = self.db_manager.get_session()
        try:
            query = session.query(Alert).filter(
                Alert.status.in_(['PENDING', 'RETRY'])
            )
            
            if alert_type:
                query = query.filter(Alert.alert_type == alert_type)
            
            # Order by priority (ascending) and created_at
            alerts = query.order_by(asc(Alert.priority), asc(Alert.created_at)).limit(limit).all()
            return alerts
        finally:
            self.db_manager.close_session(session)
    
    # Red-Listed Items Operations
    def add_red_listed_item(self, item_data: Dict[str, Any]) -> Optional[int]:
        """Add an item to the red-listed items."""
        session = self.db_manager.get_session()
        try:
            red_item = RedListedItem(
                item_type=item_data['item_type'],
                identifier=item_data['identifier'].upper() if item_data['item_type'] == 'VEHICLE' else item_data['identifier'],
                reason=item_data['reason'],
                severity_level=item_data.get('severity_level', 'MEDIUM'),
                added_by=item_data.get('added_by', 'system'),
                notes=item_data.get('notes'),
                expiry_date=item_data.get('expiry_date')
            )
            
            session.add(red_item)
            session.commit()
            return red_item.id
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to add red-listed item: {e}")
            return None
        finally:
            self.db_manager.close_session(session)
    
    def is_red_listed(self, item_type: str, identifier: str) -> Tuple[bool, Optional[str]]:
        """Check if an item is red-listed."""
        session = self.db_manager.get_session()
        try:
            clean_identifier = identifier.upper() if item_type == 'VEHICLE' else identifier
            
            red_item = session.query(RedListedItem).filter(
                and_(
                    RedListedItem.item_type == item_type,
                    RedListedItem.identifier == clean_identifier,
                    RedListedItem.is_active == True
                )
            ).first()
            
            if red_item:
                # Update last detected and count
                red_item.last_detected = datetime.now()
                red_item.detection_count += 1
                session.commit()
                return True, red_item.reason
            
            return False, None
        except SQLAlchemyError as e:
            logger.error(f"Failed to check red-listed status: {e}")
            return False, None
        finally:
            self.db_manager.close_session(session)
    
    def remove_red_listed_item(self, item_type: str, identifier: str) -> bool:
        """Remove an item from red-listed items."""
        session = self.db_manager.get_session()
        try:
            clean_identifier = identifier.upper() if item_type == 'VEHICLE' else identifier
            
            red_item = session.query(RedListedItem).filter(
                and_(
                    RedListedItem.item_type == item_type,
                    RedListedItem.identifier == clean_identifier
                )
            ).first()
            
            if red_item:
                red_item.is_active = False
                red_item.updated_at = datetime.now()
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to remove red-listed item: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def get_red_listed_items(self, item_type: str = None, active_only: bool = True) -> List[RedListedItem]:
        """Get red-listed items."""
        session = self.db_manager.get_session()
        try:
            query = session.query(RedListedItem)
            
            if active_only:
                query = query.filter(RedListedItem.is_active == True)
            
            if item_type:
                query = query.filter(RedListedItem.item_type == item_type)
            
            items = query.order_by(desc(RedListedItem.updated_at)).all()
            return items
        finally:
            self.db_manager.close_session(session)
    
    # System Logging Operations
    def add_system_log(self, log_data: Dict[str, Any]) -> bool:
        """Add a system log entry."""
        session = self.db_manager.get_session()
        try:
            log_entry = SystemLog(
                module_name=log_data['module_name'],
                log_level=log_data['log_level'],
                message=log_data['message'],
                function_name=log_data.get('function_name'),
                line_number=log_data.get('line_number'),
                event_id=log_data.get('event_id'),
                processing_time=log_data.get('processing_time'),
                memory_usage=log_data.get('memory_usage')
            )
            
            if 'additional_context' in log_data:
                log_entry.set_additional_context(log_data['additional_context'])
            
            session.add(log_entry)
            session.commit()
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to add system log: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    # Analytics and Reporting Operations
    def get_detection_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get detection summary for dashboard."""
        session = self.db_manager.get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Total detections by type
            detection_counts = session.query(
                Event.event_type,
                func.count(Event.id).label('count')
            ).filter(
                Event.timestamp >= cutoff_date
            ).group_by(Event.event_type).all()
            
            # Alert detections
            alert_count = session.query(func.count(Event.id)).filter(
                and_(
                    Event.timestamp >= cutoff_date,
                    Event.alert_triggered == True
                )
            ).scalar()
            
            # Recent critical events
            critical_events = session.query(func.count(Event.id)).filter(
                and_(
                    Event.timestamp >= cutoff_date,
                    Event.severity_level == 'CRITICAL'
                )
            ).scalar()
            
            return {
                'detection_counts': {dc.event_type: dc.count for dc in detection_counts},
                'total_alerts': alert_count,
                'critical_events': critical_events,
                'summary_period_days': days,
                'last_updated': datetime.now()
            }
        finally:
            self.db_manager.close_session(session)
    
    def get_hourly_detection_stats(self, date: datetime = None) -> List[Dict[str, Any]]:
        """Get hourly detection statistics for a specific date."""
        session = self.db_manager.get_session()
        try:
            target_date = date or datetime.now().date()
            
            stats = session.query(DetectionStatistics).filter(
                func.date(DetectionStatistics.stat_date) == target_date
            ).order_by(DetectionStatistics.stat_hour).all()
            
            return [
                {
                    'hour': stat.stat_hour,
                    'module': stat.module_name,
                    'location': stat.location,
                    'detections': stat.total_detections,
                    'alerts': stat.alert_detections,
                    'avg_confidence': stat.average_confidence,
                    'avg_processing_time': stat.processing_time_avg
                }
                for stat in stats
            ]
        finally:
            self.db_manager.close_session(session)
    
    def search_events(self, search_params: Dict[str, Any]) -> List[Event]:
        """Advanced search for events with multiple filters."""
        session = self.db_manager.get_session()
        try:
            query = session.query(Event)
            
            # Filter by event type
            if 'event_type' in search_params:
                query = query.filter(Event.event_type == search_params['event_type'])
            
            # Filter by date range
            if 'start_date' in search_params:
                query = query.filter(Event.timestamp >= search_params['start_date'])
            if 'end_date' in search_params:
                query = query.filter(Event.timestamp <= search_params['end_date'])
            
            # Filter by location
            if 'location' in search_params:
                query = query.filter(Event.location == search_params['location'])
            
            # Filter by alert status
            if 'alert_triggered' in search_params:
                query = query.filter(Event.alert_triggered == search_params['alert_triggered'])
            
            # Filter by severity level
            if 'severity_level' in search_params:
                query = query.filter(Event.severity_level == search_params['severity_level'])
            
            # Filter by confidence range
            if 'min_confidence' in search_params:
                query = query.filter(Event.confidence_score >= search_params['min_confidence'])
            if 'max_confidence' in search_params:
                query = query.filter(Event.confidence_score <= search_params['max_confidence'])
            
            # Module-specific filters
            if search_params.get('event_type') == 'ANPR' and 'plate_number' in search_params:
                query = query.join(AnprEvent).filter(
                    AnprEvent.plate_number.contains(search_params['plate_number'].upper())
                )
            
            if search_params.get('event_type') == 'FACE' and 'person_name' in search_params:
                query = query.join(FaceEvent).filter(
                    FaceEvent.person_name.contains(search_params['person_name'])
                )
            
            if search_params.get('event_type') == 'WEAPON' and 'weapon_type' in search_params:
                query = query.join(WeaponEvent).filter(
                    WeaponEvent.weapon_type.contains(search_params['weapon_type'])
                )
            
            # Order by timestamp (most recent first)
            events = query.order_by(desc(Event.timestamp)).limit(
                search_params.get('limit', 100)
            ).all()
            
            return events
        finally:
            self.db_manager.close_session(session)
    
    def cleanup_old_data(self, retention_days: int = 90) -> Dict[str, int]:
        """Clean up old data based on retention policy."""
        session = self.db_manager.get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            cleanup_stats = {}
            
            # Clean up old events (this will cascade to related tables)
            old_events = session.query(Event).filter(Event.timestamp < cutoff_date)
            cleanup_stats['events_deleted'] = old_events.count()
            old_events.delete(synchronize_session=False)
            
            # Clean up old system logs
            old_logs = session.query(SystemLog).filter(SystemLog.timestamp < cutoff_date)
            cleanup_stats['logs_deleted'] = old_logs.count()
            old_logs.delete(synchronize_session=False)
            
            # Clean up old statistics (keep longer retention for stats)
            stats_cutoff = datetime.now() - timedelta(days=retention_days * 2)
            old_stats = session.query(DetectionStatistics).filter(DetectionStatistics.stat_date < stats_cutoff)
            cleanup_stats['stats_deleted'] = old_stats.count()
            old_stats.delete(synchronize_session=False)
            
            session.commit()
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to cleanup old data: {e}")
            return {}
        finally:
            self.db_manager.close_session(session)