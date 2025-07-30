import sqlite3
import numpy as np
import json
from datetime import datetime
from config import ANPRConfig

class FaceDatabase:
    def __init__(self, db_path=None):
        self.config = ANPRConfig()
        self.db_path = db_path or (self.config.database.db_path if hasattr(self.config.database, 'db_path') else 'known_faces.db')
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create known faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create detection history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT,
                is_known BOOLEAN,
                confidence REAL,
                image_path TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alert_sent BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create alert history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER,
                alert_type TEXT,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (detection_id) REFERENCES detection_history (id)
            )
        ''')
        
        # Create red alerted vehicles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS red_alerted_vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL UNIQUE,
                alert_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # Create plate detection history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plate_detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                is_red_listed BOOLEAN DEFAULT FALSE,
                confidence REAL,
                image_path TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alert_sent BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_known_face(self, person_name, embedding, image_path=None):
        """Add a known face embedding to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy array to binary
        embedding_blob = embedding.tobytes()
        
        cursor.execute('''
            INSERT INTO known_faces (person_name, embedding, image_path)
            VALUES (?, ?, ?)
        ''', (person_name, embedding_blob, image_path))
        
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return face_id
    
    def get_all_known_faces(self):
        """Retrieve all known face embeddings from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, person_name, embedding, image_path 
            FROM known_faces
        ''')
        
        faces = []
        for row in cursor.fetchall():
            face_id, name, embedding_blob, image_path = row
            # Convert binary back to numpy array
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            faces.append({
                'id': face_id,
                'name': name,
                'embedding': embedding,
                'image_path': image_path
            })
        
        conn.close()
        return faces
    
    def add_detection(self, person_name, is_known, confidence, image_path=None):
        """Record a face detection event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detection_history (person_name, is_known, confidence, image_path)
            VALUES (?, ?, ?, ?)
        ''', (person_name, is_known, confidence, image_path))
        
        detection_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return detection_id
    
    def add_alert(self, detection_id, alert_type):
        """Record an alert being sent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alert_history (detection_id, alert_type)
            VALUES (?, ?)
        ''', (detection_id, alert_type))
        
        conn.commit()
        conn.close()
    
    def get_recent_unknown_detections(self, minutes=5):
        """Get recent unknown person detections within specified minutes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM detection_history 
            WHERE is_known = FALSE 
            AND detected_at > datetime('now', '-{} minutes')
            ORDER BY detected_at DESC
        '''.format(minutes))
        
        detections = cursor.fetchall()
        conn.close()
        
        return detections
    
    def update_known_face(self, face_id, person_name=None, embedding=None):
        """Update a known face record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if person_name:
            cursor.execute('''
                UPDATE known_faces 
                SET person_name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (person_name, face_id))
        
        if embedding is not None:
            embedding_blob = embedding.tobytes()
            cursor.execute('''
                UPDATE known_faces 
                SET embedding = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (embedding_blob, face_id))
        
        conn.commit()
        conn.close()
    
    def delete_known_face(self, face_id):
        """Delete a known face from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM known_faces WHERE id = ?', (face_id,))
        
        conn.commit()
        conn.close()
    
    def get_detection_stats(self):
        """Get detection statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total detections
        cursor.execute('SELECT COUNT(*) FROM detection_history')
        total_detections = cursor.fetchone()[0]
        
        # Known vs unknown
        cursor.execute('SELECT is_known, COUNT(*) FROM detection_history GROUP BY is_known')
        stats = dict(cursor.fetchall())
        
        # Recent alerts
        cursor.execute('''
            SELECT COUNT(*) FROM alert_history 
            WHERE sent_at > datetime('now', '-24 hours')
        ''')
        recent_alerts = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_detections': total_detections,
            'known_detections': stats.get(1, 0),
            'unknown_detections': stats.get(0, 0),
            'recent_alerts': recent_alerts
        }
    
    # Plate-related database methods
    def add_red_alerted_vehicle(self, plate_number, alert_reason="Unknown"):
        """Add a vehicle to the red alert list."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO red_alerted_vehicles (plate_number, alert_reason)
                VALUES (?, ?)
            ''', (plate_number.upper(), alert_reason))
            
            vehicle_id = cursor.lastrowid
            conn.commit()
            return vehicle_id
            
        except sqlite3.IntegrityError:
            # Plate already exists, update it
            cursor.execute('''
                UPDATE red_alerted_vehicles 
                SET alert_reason = ?, updated_at = CURRENT_TIMESTAMP, is_active = TRUE
                WHERE plate_number = ?
            ''', (alert_reason, plate_number.upper()))
            conn.commit()
            return None
            
        finally:
            conn.close()
    
    def remove_red_alerted_vehicle(self, plate_number):
        """Remove a vehicle from the red alert list."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE red_alerted_vehicles 
            SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
            WHERE plate_number = ?
        ''', (plate_number.upper(),))
        
        conn.commit()
        conn.close()
    
    def is_plate_red_listed(self, plate_number):
        """Check if a plate number is in the red alert list. Logs the check and strips/uppercases input."""
        clean_plate = plate_number.strip().upper() if plate_number else ''
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Checking red list for plate: '{clean_plate}' (original: '{plate_number}')")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, alert_reason FROM red_alerted_vehicles 
            WHERE plate_number = ? AND is_active = TRUE
        ''', (clean_plate,))
        result = cursor.fetchone()
        conn.close()
        if result:
            logger.info(f"Plate '{clean_plate}' IS red-listed. Reason: {result[1]}")
            return True, result[1]  # Return True and the alert reason
        logger.info(f"Plate '{clean_plate}' is NOT red-listed.")
        return False, None
    
    def get_all_red_alerted_vehicles(self):
        """Get all red-alerted vehicles."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT plate_number, alert_reason, created_at, updated_at 
            FROM red_alerted_vehicles 
            WHERE is_active = TRUE
            ORDER BY updated_at DESC
        ''')
        
        vehicles = cursor.fetchall()
        conn.close()
        
        return vehicles
    
    def add_plate_detection(self, plate_number, is_red_listed, confidence, image_path=None):
        """Record a plate detection event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO plate_detection_history (plate_number, is_red_listed, confidence, image_path)
            VALUES (?, ?, ?, ?)
        ''', (plate_number.upper(), is_red_listed, confidence, image_path))
        
        detection_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return detection_id
    
    def mark_plate_alert_sent(self, detection_id):
        """Mark that an alert has been sent for a plate detection."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE plate_detection_history 
            SET alert_sent = TRUE 
            WHERE id = ?
        ''', (detection_id,))
        
        conn.commit()
        conn.close()
    
    def get_plate_detection_stats(self):
        """Get plate detection statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total plate detections
        cursor.execute('SELECT COUNT(*) FROM plate_detection_history')
        total_detections = cursor.fetchone()[0]
        
        # Red-listed vs normal
        cursor.execute('SELECT is_red_listed, COUNT(*) FROM plate_detection_history GROUP BY is_red_listed')
        stats = dict(cursor.fetchall())
        
        # Recent red-listed detections
        cursor.execute('''
            SELECT COUNT(*) FROM plate_detection_history 
            WHERE is_red_listed = TRUE AND detected_at > datetime('now', '-24 hours')
        ''')
        recent_red_alerts = cursor.fetchone()[0]
        
        # Total red-alerted vehicles
        cursor.execute('SELECT COUNT(*) FROM red_alerted_vehicles WHERE is_active = TRUE')
        total_red_vehicles = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_plate_detections': total_detections,
            'normal_detections': stats.get(0, 0),
            'red_listed_detections': stats.get(1, 0),
            'recent_red_alerts': recent_red_alerts,
            'total_red_vehicles': total_red_vehicles
        }