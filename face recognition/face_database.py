import os
import pickle
import face_recognition
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from config import Config

class FaceDatabase:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.known_categories = []  # 'restricted' or 'criminal'
        self.encoding_cache = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        Config.ensure_directories()
        
        # Load existing encodings
        self.load_encodings()
    
    def add_face_from_image(self, image_path: str, name: str, category: str) -> bool:
        """Add a face from an image file to the database"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) == 0:
                self.logger.warning(f"No face found in {image_path}")
                return False
            
            if len(face_encodings) > 1:
                self.logger.warning(f"Multiple faces found in {image_path}, using first one")
            
            # Add to database
            encoding = face_encodings[0]
            self.known_encodings.append(encoding)
            self.known_names.append(name)
            self.known_categories.append(category)
            
            self.logger.info(f"Added {category} face: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding face from {image_path}: {str(e)}")
            return False
    
    def load_faces_from_directory(self, directory: str, category: str) -> int:
        """Load all faces from a directory"""
        if not os.path.exists(directory):
            self.logger.warning(f"Directory {directory} does not exist")
            return 0
        
        count = 0
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for filename in os.listdir(directory):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(directory, filename)
                name = os.path.splitext(filename)[0]
                
                if self.add_face_from_image(image_path, name, category):
                    count += 1
        
        self.logger.info(f"Loaded {count} {category} faces from {directory}")
        return count
    
    def load_all_known_faces(self) -> Tuple[int, int]:
        """Load all restricted and criminal faces"""
        restricted_count = self.load_faces_from_directory(
            Config.RESTRICTED_FACES_DIR, 'restricted'
        )
        criminal_count = self.load_faces_from_directory(
            Config.CRIMINAL_FACES_DIR, 'criminal'
        )
        
        return restricted_count, criminal_count
    
    def save_encodings(self) -> bool:
        """Save encodings to file for faster loading"""
        try:
            data = {
                'encodings': self.known_encodings,
                'names': self.known_names,
                'categories': self.known_categories
            }
            
            with open(Config.FACE_ENCODINGS_FILE, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"Saved {len(self.known_encodings)} encodings to cache")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving encodings: {str(e)}")
            return False
    
    def load_encodings(self) -> bool:
        """Load encodings from cache file"""
        try:
            if os.path.exists(Config.FACE_ENCODINGS_FILE):
                with open(Config.FACE_ENCODINGS_FILE, 'rb') as f:
                    data = pickle.load(f)
                
                self.known_encodings = data['encodings']
                self.known_names = data['names']
                self.known_categories = data['categories']
                
                self.logger.info(f"Loaded {len(self.known_encodings)} encodings from cache")
                return True
            else:
                # Load from directories if no cache exists
                self.load_all_known_faces()
                self.save_encodings()
                return True
                
        except Exception as e:
            self.logger.error(f"Error loading encodings: {str(e)}")
            # Fallback to loading from directories
            self.load_all_known_faces()
            return False
    
    def recognize_faces(self, face_encodings: List[np.ndarray]) -> List[Dict]:
        """Recognize faces from encodings"""
        results = []
        
        for face_encoding in face_encodings:
            name = "Unknown"
            category = "unknown"
            confidence = 0.0
            
            if len(self.known_encodings) > 0:
                # Calculate distances to all known faces
                face_distances = face_recognition.face_distance(
                    self.known_encodings, face_encoding
                )
                
                # Find best match
                best_match_index = np.argmin(face_distances)
                
                # Check if match is within tolerance
                if face_distances[best_match_index] <= Config.FACE_TOLERANCE:
                    name = self.known_names[best_match_index]
                    category = self.known_categories[best_match_index]
                    # Convert distance to confidence (0-1)
                    confidence = 1 - face_distances[best_match_index]
            
            results.append({
                'name': name,
                'category': category,
                'confidence': confidence
            })
        
        return results
    
    def rebuild_database(self) -> bool:
        """Rebuild the entire face database from image directories"""
        try:
            # Clear current data
            self.known_encodings.clear()
            self.known_names.clear()
            self.known_categories.clear()
            
            # Reload from directories
            self.load_all_known_faces()
            
            # Save to cache
            self.save_encodings()
            
            self.logger.info("Database rebuilt successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error rebuilding database: {str(e)}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the database"""
        restricted_count = sum(1 for cat in self.known_categories if cat == 'restricted')
        criminal_count = sum(1 for cat in self.known_categories if cat == 'criminal')
        
        return {
            'total_faces': len(self.known_encodings),
            'restricted_faces': restricted_count,
            'criminal_faces': criminal_count
        }