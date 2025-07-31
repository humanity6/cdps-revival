"""
Unified Database Schema for Real-Time Crime Detection System
Supports ANPR, Face Recognition, Violence Detection, and Weapon Detection modules
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class EventType(Enum):
    ANPR = "ANPR"
    FACE = "FACE"
    VIOLENCE = "VIOLENCE"
    WEAPON = "WEAPON"

class SeverityLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class EventStatus(Enum):
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    ARCHIVED = "ARCHIVED"

class AlertStatus(Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    RETRY = "RETRY"

class UnifiedDatabaseSchema:
    """
    Unified database schema manager for the crime detection system.
    Handles all four detection modules with optimized structure for analytics and search.
    """
    
    def __init__(self, db_path: str = "crime_detection.db"):
        self.db_path = db_path
        self.schema_version = "1.0.0"
        
    def create_database(self) -> bool:
        """Create the complete database schema with all tables and indexes."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enable foreign key support
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Create all tables
            self._create_events_table(cursor)
            self._create_detections_table(cursor)
            self._create_anpr_events_table(cursor)
            self._create_face_events_table(cursor)
            self._create_violence_events_table(cursor)
            self._create_weapon_events_table(cursor)
            self._create_alerts_table(cursor)
            self._create_red_listed_items_table(cursor)
            self._create_system_logs_table(cursor)
            self._create_detection_statistics_table(cursor)
            self._create_search_index_table(cursor)
            
            # Create indexes for performance
            self._create_indexes(cursor)
            
            # Create views for analytics
            self._create_analytics_views(cursor)
            
            # Insert schema version
            self._insert_schema_info(cursor)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database schema created successfully at {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database schema: {e}")
            return False
    
    def _create_events_table(self, cursor: sqlite3.Cursor):
        """Create the master events table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL CHECK(event_type IN ('ANPR', 'FACE', 'VIOLENCE', 'WEAPON')),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
                status TEXT DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'PROCESSED', 'ARCHIVED')),
                location TEXT DEFAULT 'Camera_1',
                event_metadata TEXT, -- JSON field for module-specific data
                alert_triggered BOOLEAN DEFAULT FALSE,
                severity_level TEXT DEFAULT 'MEDIUM' CHECK(severity_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                image_path TEXT,
                processing_time REAL, -- Time taken to process in seconds
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_detections_table(self, cursor: sqlite3.Cursor):
        """Create the detection details table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                detection_type TEXT NOT NULL, -- specific type: license_plate, face, violence, weapon_type
                bounding_box TEXT, -- JSON format: {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
                confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
                class_name TEXT,
                additional_data TEXT, -- JSON for type-specific fields
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE
            )
        """)
    
    def _create_anpr_events_table(self, cursor: sqlite3.Cursor):
        """Create ANPR-specific events table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anpr_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL UNIQUE,
                plate_number TEXT NOT NULL,
                is_red_listed BOOLEAN DEFAULT FALSE,
                alert_reason TEXT,
                vehicle_make_model TEXT,
                country_code TEXT,
                region TEXT,
                plate_confidence REAL,
                ocr_text_raw TEXT, -- Raw OCR output for debugging
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE
            )
        """)
    
    def _create_face_events_table(self, cursor: sqlite3.Cursor):
        """Create face recognition-specific events table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL UNIQUE,
                person_name TEXT,
                person_category TEXT CHECK(person_category IN ('unknown', 'restricted', 'criminal', 'safe', 'authorized')),
                recognition_confidence REAL,
                face_encoding BLOB, -- Face embedding for recognition
                age_estimate INTEGER,
                gender_estimate TEXT,
                emotion_detected TEXT,
                face_quality_score REAL,
                is_masked BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE
            )
        """)
    
    def _create_violence_events_table(self, cursor: sqlite3.Cursor):
        """Create violence detection-specific events table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violence_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL UNIQUE,
                is_violence BOOLEAN NOT NULL,
                violence_type TEXT, -- fight, weapon_usage, aggressive_behavior, etc.
                violence_intensity TEXT CHECK(violence_intensity IN ('LOW', 'MEDIUM', 'HIGH')),
                frame_analysis TEXT, -- JSON for multi-frame analysis
                duration_seconds REAL,
                people_count INTEGER,
                movement_intensity REAL,
                audio_analysis TEXT, -- JSON for audio features if available
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE
            )
        """)
    
    def _create_weapon_events_table(self, cursor: sqlite3.Cursor):
        """Create weapon detection-specific events table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weapon_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL UNIQUE,
                weapon_type TEXT NOT NULL,
                weapon_category TEXT, -- firearm, knife, blunt_object, etc.
                threat_level TEXT CHECK(threat_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                weapon_size_estimate TEXT, -- small, medium, large
                is_concealed BOOLEAN DEFAULT FALSE,
                person_holding_weapon BOOLEAN DEFAULT FALSE,
                weapon_condition TEXT, -- drawn, holstered, brandished
                multiple_weapons BOOLEAN DEFAULT FALSE,
                weapon_count INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE
            )
        """)
    
    def _create_alerts_table(self, cursor: sqlite3.Cursor):
        """Create alerts tracking table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                alert_type TEXT CHECK(alert_type IN ('telegram', 'email', 'sms', 'webhook', 'system')),
                recipient TEXT NOT NULL, -- chat_id, email, phone, url
                message_content TEXT,
                message_template TEXT,
                status TEXT DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'SENT', 'FAILED', 'RETRY')),
                sent_at DATETIME,
                delivery_confirmed BOOLEAN DEFAULT FALSE,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                response_data TEXT, -- JSON response from alert service
                priority INTEGER DEFAULT 5, -- 1-10 scale
                expires_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE
            )
        """)
    
    def _create_red_listed_items_table(self, cursor: sqlite3.Cursor):
        """Create red-listed items tracking table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS red_listed_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_type TEXT CHECK(item_type IN ('VEHICLE', 'PERSON', 'OBJECT')),
                identifier TEXT NOT NULL, -- plate_number, person_name, object_description
                reason TEXT NOT NULL,
                severity_level TEXT DEFAULT 'MEDIUM' CHECK(severity_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                added_by TEXT DEFAULT 'system',
                notes TEXT,
                expiry_date DATETIME,
                is_active BOOLEAN DEFAULT TRUE,
                last_detected DATETIME,
                detection_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(item_type, identifier)
            )
        """)
    
    def _create_system_logs_table(self, cursor: sqlite3.Cursor):
        """Create system logs table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_name TEXT NOT NULL,
                log_level TEXT CHECK(log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
                message TEXT NOT NULL,
                function_name TEXT,
                line_number INTEGER,
                additional_context TEXT, -- JSON field
                event_id INTEGER, -- Link to related event if applicable
                processing_time REAL,
                memory_usage REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE SET NULL
            )
        """)
    
    def _create_detection_statistics_table(self, cursor: sqlite3.Cursor):
        """Create detection statistics table for dashboard analytics."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_date DATE NOT NULL,
                stat_hour INTEGER CHECK(stat_hour >= 0 AND stat_hour <= 23),
                module_name TEXT NOT NULL,
                location TEXT DEFAULT 'Camera_1',
                total_detections INTEGER DEFAULT 0,
                alert_detections INTEGER DEFAULT 0,
                average_confidence REAL,
                min_confidence REAL,
                max_confidence REAL,
                processing_time_avg REAL,
                processing_time_max REAL,
                false_positive_count INTEGER DEFAULT 0,
                true_positive_count INTEGER DEFAULT 0,
                system_uptime_seconds INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(stat_date, stat_hour, module_name, location)
            )
        """)
    
    def _create_search_index_table(self, cursor: sqlite3.Cursor):
        """Create search index table for full-text search capabilities."""
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                event_id UNINDEXED,
                event_type,
                content,
                tags,
                location,
                timestamp UNINDEXED
            )
        """)
    
    def _create_indexes(self, cursor: sqlite3.Cursor):
        """Create database indexes for optimal query performance."""
        indexes = [
            # Events table indexes
            "CREATE INDEX IF NOT EXISTS idx_events_type_timestamp ON events(event_type, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)",
            "CREATE INDEX IF NOT EXISTS idx_events_alert_triggered ON events(alert_triggered)",
            "CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity_level)",
            "CREATE INDEX IF NOT EXISTS idx_events_location ON events(location)",
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)",
            
            # Detections table indexes
            "CREATE INDEX IF NOT EXISTS idx_detections_event_id ON detections(event_id)",
            "CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(detection_type)",
            "CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence)",
            
            # ANPR events indexes
            "CREATE INDEX IF NOT EXISTS idx_anpr_plate_number ON anpr_events(plate_number)",
            "CREATE INDEX IF NOT EXISTS idx_anpr_red_listed ON anpr_events(is_red_listed)",
            "CREATE INDEX IF NOT EXISTS idx_anpr_event_id ON anpr_events(event_id)",
            
            # Face events indexes
            "CREATE INDEX IF NOT EXISTS idx_face_person_name ON face_events(person_name)",
            "CREATE INDEX IF NOT EXISTS idx_face_category ON face_events(person_category)",
            "CREATE INDEX IF NOT EXISTS idx_face_event_id ON face_events(event_id)",
            
            # Violence events indexes
            "CREATE INDEX IF NOT EXISTS idx_violence_is_violence ON violence_events(is_violence)",
            "CREATE INDEX IF NOT EXISTS idx_violence_type ON violence_events(violence_type)",
            "CREATE INDEX IF NOT EXISTS idx_violence_event_id ON violence_events(event_id)",
            
            # Weapon events indexes
            "CREATE INDEX IF NOT EXISTS idx_weapon_type ON weapon_events(weapon_type)",
            "CREATE INDEX IF NOT EXISTS idx_weapon_threat_level ON weapon_events(threat_level)",
            "CREATE INDEX IF NOT EXISTS idx_weapon_event_id ON weapon_events(event_id)",
            
            # Alerts table indexes
            "CREATE INDEX IF NOT EXISTS idx_alerts_event_id ON alerts(event_id)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_sent_at ON alerts(sent_at)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority)",
            
            # Red-listed items indexes
            "CREATE INDEX IF NOT EXISTS idx_red_listed_identifier ON red_listed_items(identifier)",
            "CREATE INDEX IF NOT EXISTS idx_red_listed_type_active ON red_listed_items(item_type, is_active)",
            "CREATE INDEX IF NOT EXISTS idx_red_listed_active ON red_listed_items(is_active)",
            
            # System logs indexes
            "CREATE INDEX IF NOT EXISTS idx_logs_module_timestamp ON system_logs(module_name, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(log_level)",
            "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs(timestamp DESC)",
            
            # Statistics indexes
            "CREATE INDEX IF NOT EXISTS idx_stats_date_module ON detection_statistics(stat_date, module_name)",
            "CREATE INDEX IF NOT EXISTS idx_stats_location ON detection_statistics(location)",
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
    
    def _create_analytics_views(self, cursor: sqlite3.Cursor):
        """Create database views for analytics and reporting."""
        
        # Daily summary view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS daily_detection_summary AS
            SELECT 
                DATE(timestamp) as detection_date,
                event_type,
                location,
                COUNT(*) as total_detections,
                COUNT(CASE WHEN alert_triggered = 1 THEN 1 END) as alert_detections,
                AVG(confidence_score) as avg_confidence,
                AVG(processing_time) as avg_processing_time
            FROM events
            WHERE timestamp >= DATE('now', '-30 days')
            GROUP BY DATE(timestamp), event_type, location
            ORDER BY detection_date DESC
        """)
        
        # Red-listed detection view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS red_listed_detections AS
            SELECT 
                e.id as event_id,
                e.timestamp,
                e.event_type,
                e.location,
                CASE 
                    WHEN e.event_type = 'ANPR' THEN a.plate_number 
                    WHEN e.event_type = 'FACE' THEN f.person_name
                    ELSE 'N/A'
                END as identifier,
                CASE 
                    WHEN e.event_type = 'ANPR' THEN a.alert_reason
                    WHEN e.event_type = 'FACE' THEN f.person_category
                    ELSE 'N/A'
                END as alert_reason,
                e.confidence_score,
                e.image_path
            FROM events e
            LEFT JOIN anpr_events a ON e.id = a.event_id AND e.event_type = 'ANPR' AND a.is_red_listed = 1
            LEFT JOIN face_events f ON e.id = f.event_id AND e.event_type = 'FACE' AND f.person_category IN ('restricted', 'criminal')
            WHERE e.alert_triggered = 1
            ORDER BY e.timestamp DESC
        """)
        
        # System health view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS system_health_summary AS
            SELECT 
                module_name,
                COUNT(*) as total_logs,
                COUNT(CASE WHEN log_level = 'ERROR' THEN 1 END) as error_count,
                COUNT(CASE WHEN log_level = 'WARNING' THEN 1 END) as warning_count,
                AVG(processing_time) as avg_processing_time,
                MAX(timestamp) as last_activity
            FROM system_logs
            WHERE timestamp >= DATETIME('now', '-24 hours')
            GROUP BY module_name
        """)
    
    def _insert_schema_info(self, cursor: sqlite3.Cursor):
        """Insert schema version information."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                version TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        
        cursor.execute("""
            INSERT OR REPLACE INTO schema_info (version, description)
            VALUES (?, ?)
        """, (self.schema_version, "Unified Crime Detection Database Schema"))
    
    def get_schema_version(self) -> Optional[str]:
        """Get the current schema version."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM schema_info ORDER BY created_at DESC LIMIT 1")
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to get schema version: {e}")
            return None
    
    def validate_schema(self) -> Tuple[bool, List[str]]:
        """Validate that all required tables and indexes exist."""
        errors = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check required tables
            required_tables = [
                'events', 'detections', 'anpr_events', 'face_events', 
                'violence_events', 'weapon_events', 'alerts', 
                'red_listed_items', 'system_logs', 'detection_statistics'
            ]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            for table in required_tables:
                if table not in existing_tables:
                    errors.append(f"Missing required table: {table}")
            
            conn.close()
            
        except Exception as e:
            errors.append(f"Schema validation failed: {e}")
        
        return len(errors) == 0, errors

def initialize_database(db_path: str = "crime_detection.db") -> bool:
    """Initialize the unified crime detection database."""
    schema_manager = UnifiedDatabaseSchema(db_path)
    return schema_manager.create_database()

if __name__ == "__main__":
    # Initialize database when run directly
    success = initialize_database()
    if success:
        print("Database schema created successfully!")
    else:
        print("Failed to create database schema!")