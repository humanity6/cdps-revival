"""
SQLAlchemy ORM Models for Unified Crime Detection Database
Provides object-relational mapping for all detection modules
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, 
    ForeignKey, UniqueConstraint, CheckConstraint, LargeBinary
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import enum

Base = declarative_base()

class EventTypeEnum(enum.Enum):
    ANPR = "ANPR"
    FACE = "FACE"
    VIOLENCE = "VIOLENCE" 
    WEAPON = "WEAPON"

class SeverityLevelEnum(enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class EventStatusEnum(enum.Enum):
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    ARCHIVED = "ARCHIVED"

class AlertStatusEnum(enum.Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    RETRY = "RETRY"

class Event(Base):
    """Master events table - stores all detection events."""
    
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=func.current_timestamp())
    confidence_score = Column(Float)
    status = Column(String(20), default='PENDING')
    location = Column(String(100), default='Camera_1')
    metadata = Column(Text)  # JSON field
    alert_triggered = Column(Boolean, default=False)
    severity_level = Column(String(20), default='MEDIUM')
    image_path = Column(String(500))
    processing_time = Column(Float)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    detections = relationship("Detection", back_populates="event", cascade="all, delete-orphan")
    anpr_event = relationship("AnprEvent", back_populates="event", uselist=False, cascade="all, delete-orphan")
    face_event = relationship("FaceEvent", back_populates="event", uselist=False, cascade="all, delete-orphan")
    violence_event = relationship("ViolenceEvent", back_populates="event", uselist=False, cascade="all, delete-orphan")
    weapon_event = relationship("WeaponEvent", back_populates="event", uselist=False, cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="event", cascade="all, delete-orphan")
    system_logs = relationship("SystemLog", back_populates="event")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("event_type IN ('ANPR', 'FACE', 'VIOLENCE', 'WEAPON')", name='ck_event_type'),
        CheckConstraint("status IN ('PENDING', 'PROCESSED', 'ARCHIVED')", name='ck_event_status'),
        CheckConstraint("severity_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')", name='ck_severity_level'),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name='ck_confidence_range'),
    )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Parse metadata JSON field."""
        if self.metadata:
            try:
                return json.loads(self.metadata)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_metadata(self, data: Dict[str, Any]):
        """Set metadata JSON field."""
        self.metadata = json.dumps(data)
    
    def __repr__(self):
        return f"<Event(id={self.id}, type={self.event_type}, timestamp={self.timestamp})>"

class Detection(Base):
    """Detection details table - stores bounding boxes and detection specifics."""
    
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False)
    detection_type = Column(String(50), nullable=False)
    bounding_box = Column(Text)  # JSON format
    confidence = Column(Float)
    class_name = Column(String(100))
    additional_data = Column(Text)  # JSON field
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    event = relationship("Event", back_populates="detections")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name='ck_detection_confidence'),
    )
    
    def get_bounding_box(self) -> Dict[str, int]:
        """Parse bounding box JSON."""
        if self.bounding_box:
            try:
                return json.loads(self.bounding_box)
            except json.JSONDecodeError:
                return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
    
    def set_bounding_box(self, x1: int, y1: int, x2: int, y2: int):
        """Set bounding box coordinates."""
        self.bounding_box = json.dumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    
    def get_additional_data(self) -> Dict[str, Any]:
        """Parse additional data JSON."""
        if self.additional_data:
            try:
                return json.loads(self.additional_data)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_additional_data(self, data: Dict[str, Any]):
        """Set additional data JSON."""
        self.additional_data = json.dumps(data)
    
    def __repr__(self):
        return f"<Detection(id={self.id}, type={self.detection_type}, event_id={self.event_id})>"

class AnprEvent(Base):
    """ANPR-specific event details."""
    
    __tablename__ = 'anpr_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False, unique=True)
    plate_number = Column(String(20), nullable=False)
    is_red_listed = Column(Boolean, default=False)
    alert_reason = Column(String(500))
    vehicle_make_model = Column(String(100))
    country_code = Column(String(10))
    region = Column(String(50))
    plate_confidence = Column(Float)
    ocr_text_raw = Column(String(50))
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    event = relationship("Event", back_populates="anpr_event")
    
    def __repr__(self):
        return f"<AnprEvent(id={self.id}, plate={self.plate_number}, red_listed={self.is_red_listed})>"

class FaceEvent(Base):
    """Face recognition-specific event details."""
    
    __tablename__ = 'face_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False, unique=True)
    person_name = Column(String(100))
    person_category = Column(String(20))
    recognition_confidence = Column(Float)
    face_encoding = Column(LargeBinary)  # Face embedding
    age_estimate = Column(Integer)
    gender_estimate = Column(String(20))
    emotion_detected = Column(String(50))
    face_quality_score = Column(Float)
    is_masked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    event = relationship("Event", back_populates="face_event")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("person_category IN ('unknown', 'restricted', 'criminal', 'safe', 'authorized')", 
                       name='ck_person_category'),
    )
    
    def __repr__(self):
        return f"<FaceEvent(id={self.id}, person={self.person_name}, category={self.person_category})>"

class ViolenceEvent(Base):
    """Violence detection-specific event details."""
    
    __tablename__ = 'violence_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False, unique=True)
    is_violence = Column(Boolean, nullable=False)
    violence_type = Column(String(50))
    violence_intensity = Column(String(20))
    frame_analysis = Column(Text)  # JSON field
    duration_seconds = Column(Float)
    people_count = Column(Integer)
    movement_intensity = Column(Float)
    audio_analysis = Column(Text)  # JSON field
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    event = relationship("Event", back_populates="violence_event")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("violence_intensity IN ('LOW', 'MEDIUM', 'HIGH')", name='ck_violence_intensity'),
    )
    
    def get_frame_analysis(self) -> Dict[str, Any]:
        """Parse frame analysis JSON."""
        if self.frame_analysis:
            try:
                return json.loads(self.frame_analysis)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_frame_analysis(self, data: Dict[str, Any]):
        """Set frame analysis JSON."""
        self.frame_analysis = json.dumps(data)
    
    def __repr__(self):
        return f"<ViolenceEvent(id={self.id}, is_violence={self.is_violence}, type={self.violence_type})>"

class WeaponEvent(Base):
    """Weapon detection-specific event details."""
    
    __tablename__ = 'weapon_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False, unique=True)
    weapon_type = Column(String(50), nullable=False)
    weapon_category = Column(String(50))
    threat_level = Column(String(20))
    weapon_size_estimate = Column(String(20))
    is_concealed = Column(Boolean, default=False)
    person_holding_weapon = Column(Boolean, default=False)
    weapon_condition = Column(String(50))
    multiple_weapons = Column(Boolean, default=False)
    weapon_count = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    event = relationship("Event", back_populates="weapon_event")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("threat_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')", name='ck_threat_level'),
    )
    
    def __repr__(self):
        return f"<WeaponEvent(id={self.id}, weapon={self.weapon_type}, threat={self.threat_level})>"

class Alert(Base):
    """Alert tracking table."""
    
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False)
    alert_type = Column(String(20))
    recipient = Column(String(200), nullable=False)
    message_content = Column(Text)
    message_template = Column(String(100))
    status = Column(String(20), default='PENDING')
    sent_at = Column(DateTime)
    delivery_confirmed = Column(Boolean, default=False)
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    response_data = Column(Text)  # JSON field
    priority = Column(Integer, default=5)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    event = relationship("Event", back_populates="alerts")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("alert_type IN ('telegram', 'email', 'sms', 'webhook', 'system')", name='ck_alert_type'),
        CheckConstraint("status IN ('PENDING', 'SENT', 'FAILED', 'RETRY')", name='ck_alert_status'),
        CheckConstraint("priority >= 1 AND priority <= 10", name='ck_alert_priority'),
    )
    
    def get_response_data(self) -> Dict[str, Any]:
        """Parse response data JSON."""
        if self.response_data:
            try:
                return json.loads(self.response_data)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_response_data(self, data: Dict[str, Any]):
        """Set response data JSON."""
        self.response_data = json.dumps(data)
    
    def __repr__(self):
        return f"<Alert(id={self.id}, type={self.alert_type}, status={self.status})>"

class RedListedItem(Base):
    """Red-listed items tracking table."""
    
    __tablename__ = 'red_listed_items'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    item_type = Column(String(20))
    identifier = Column(String(100), nullable=False)
    reason = Column(String(500), nullable=False)
    severity_level = Column(String(20), default='MEDIUM')
    added_by = Column(String(100), default='system')
    notes = Column(Text)
    expiry_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    last_detected = Column(DateTime)
    detection_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Constraints
    __table_args__ = (
        CheckConstraint("item_type IN ('VEHICLE', 'PERSON', 'OBJECT')", name='ck_item_type'),
        CheckConstraint("severity_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')", name='ck_red_list_severity'),
        UniqueConstraint('item_type', 'identifier', name='uq_red_listed_item'),
    )
    
    def __repr__(self):
        return f"<RedListedItem(id={self.id}, type={self.item_type}, identifier={self.identifier})>"

class SystemLog(Base):
    """System logs table."""
    
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    module_name = Column(String(50), nullable=False)
    log_level = Column(String(20))
    message = Column(Text, nullable=False)
    function_name = Column(String(100))
    line_number = Column(Integer)
    additional_context = Column(Text)  # JSON field
    event_id = Column(Integer, ForeignKey('events.id'))
    processing_time = Column(Float)
    memory_usage = Column(Float)
    timestamp = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    event = relationship("Event", back_populates="system_logs")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')", name='ck_log_level'),
    )
    
    def get_additional_context(self) -> Dict[str, Any]:
        """Parse additional context JSON."""
        if self.additional_context:
            try:
                return json.loads(self.additional_context)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_additional_context(self, data: Dict[str, Any]):
        """Set additional context JSON."""
        self.additional_context = json.dumps(data)
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, module={self.module_name}, level={self.log_level})>"

class DetectionStatistics(Base):
    """Detection statistics table for analytics."""
    
    __tablename__ = 'detection_statistics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stat_date = Column(DateTime, nullable=False)
    stat_hour = Column(Integer)
    module_name = Column(String(50), nullable=False)
    location = Column(String(100), default='Camera_1')
    total_detections = Column(Integer, default=0)
    alert_detections = Column(Integer, default=0)
    average_confidence = Column(Float)
    min_confidence = Column(Float)
    max_confidence = Column(Float)
    processing_time_avg = Column(Float)
    processing_time_max = Column(Float)
    false_positive_count = Column(Integer, default=0)
    true_positive_count = Column(Integer, default=0)
    system_uptime_seconds = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Constraints
    __table_args__ = (
        CheckConstraint("stat_hour >= 0 AND stat_hour <= 23", name='ck_hour_range'),
        UniqueConstraint('stat_date', 'stat_hour', 'module_name', 'location', name='uq_detection_stats'),
    )
    
    def __repr__(self):
        return f"<DetectionStatistics(date={self.stat_date}, module={self.module_name})>"

# Database session and engine management
class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: str = "sqlite:///crime_detection.db"):
        from sqlalchemy import create_engine
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close a database session."""
        session.close()
    
    def get_table_names(self) -> List[str]:
        """Get all table names in the database."""
        return list(Base.metadata.tables.keys())

# Utility functions for common queries
class QueryHelpers:
    """Helper functions for common database queries."""
    
    @staticmethod
    def get_recent_events(session, hours: int = 24, event_type: Optional[str] = None) -> List[Event]:
        """Get recent events within specified hours."""
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = session.query(Event).filter(Event.timestamp >= cutoff_time)
        if event_type:
            query = query.filter(Event.event_type == event_type)
        
        return query.order_by(Event.timestamp.desc()).all()
    
    @staticmethod
    def get_alert_events(session, limit: int = 100) -> List[Event]:
        """Get events that triggered alerts."""
        return session.query(Event).filter(
            Event.alert_triggered == True
        ).order_by(Event.timestamp.desc()).limit(limit).all()
    
    @staticmethod
    def get_red_listed_detections(session, days: int = 30) -> List[Event]:
        """Get red-listed detections from the past N days."""
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(days=days)
        
        return session.query(Event).filter(
            Event.timestamp >= cutoff_time,
            Event.alert_triggered == True
        ).order_by(Event.timestamp.desc()).all()
    
    @staticmethod
    def get_detection_counts_by_type(session, days: int = 7) -> Dict[str, int]:
        """Get detection counts grouped by event type."""
        from datetime import datetime, timedelta
        from sqlalchemy import func
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        results = session.query(
            Event.event_type,
            func.count(Event.id).label('count')
        ).filter(
            Event.timestamp >= cutoff_time
        ).group_by(Event.event_type).all()
        
        return {result.event_type: result.count for result in results}
    
    @staticmethod
    def get_system_health_summary(session) -> Dict[str, Any]:
        """Get system health summary from logs."""
        from datetime import datetime, timedelta
        from sqlalchemy import func
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Get error counts by module
        error_counts = session.query(
            SystemLog.module_name,
            func.count(SystemLog.id).label('error_count')
        ).filter(
            SystemLog.timestamp >= cutoff_time,
            SystemLog.log_level == 'ERROR'
        ).group_by(SystemLog.module_name).all()
        
        # Get total log counts
        total_logs = session.query(func.count(SystemLog.id)).filter(
            SystemLog.timestamp >= cutoff_time
        ).scalar()
        
        return {
            'total_logs_24h': total_logs,
            'error_counts_by_module': {ec.module_name: ec.error_count for ec in error_counts},
            'last_updated': datetime.now()
        }