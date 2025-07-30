"""
Utility functions for the Face Detection System
"""

import cv2
import os
import numpy as np
from typing import List, Tuple
import face_recognition
import config

class FaceUtils:
    @staticmethod
    def extract_faces_from_image(image_path: str, output_dir: str, person_name: str) -> List[str]:
        """
        Extract individual faces from an image and save them with enhanced processing
        Returns list of saved face image paths
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load and enhance image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return []
            
            # Enhance image quality
            enhanced_image = FaceUtils.enhance_image_quality(image)
            rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            
            # Find face locations with multiple methods
            face_locations_hog = face_recognition.face_locations(rgb_image, model='hog', number_of_times_to_upsample=1)
            face_locations = face_locations_hog
            
            # Try CNN if no faces found with HOG
            if not face_locations:
                try:
                    face_locations = face_recognition.face_locations(rgb_image, model='cnn')
                    print(f"Using CNN detection for {image_path}")
                except Exception as e:
                    print(f"CNN detection failed: {e}")
            
            if not face_locations:
                print(f"No faces found in {image_path}")
                return []
            
            saved_faces = []
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Calculate face dimensions
                face_width = right - left
                face_height = bottom - top
                
                # Skip very small faces
                if face_width < config.MIN_FACE_SIZE or face_height < config.MIN_FACE_SIZE:
                    print(f"Skipping small face in {image_path}: {face_width}x{face_height}")
                    continue
                
                # Extract face region with adaptive padding
                padding = max(20, int(min(face_width, face_height) * 0.3))
                face_top = max(0, top - padding)
                face_bottom = min(image.shape[0], bottom + padding)
                face_left = max(0, left - padding)
                face_right = min(image.shape[1], right + padding)
                
                face_image = enhanced_image[face_top:face_bottom, face_left:face_right]
                
                # Ensure minimum size
                if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                    face_image = cv2.resize(face_image, (128, 128), interpolation=cv2.INTER_CUBIC)
                
                # Further enhance the face image
                face_image = FaceUtils.enhance_face_image(face_image)
                
                # Save face image with quality assessment
                quality_score = FaceUtils.assess_face_quality(face_image)
                face_filename = f"{person_name}_face_{i+1}_q{quality_score:.2f}.jpg"
                face_path = os.path.join(output_dir, face_filename)
                
                # Save with high quality
                cv2.imwrite(face_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                saved_faces.append(face_path)
                print(f"Saved face: {face_path} (Quality: {quality_score:.2f})")
            
            return saved_faces
            
        except Exception as e:
            print(f"Error extracting faces from {image_path}: {e}")
            return []
    
    @staticmethod
    def resize_image(image_path: str, max_width: int = 800, max_height: int = 600) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    @staticmethod
    def enhance_image_quality(image: np.ndarray) -> np.ndarray:
        """Apply comprehensive image enhancement for better face recognition"""
        try:
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Apply bilateral filter for noise reduction while preserving edges
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Slight sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)
            
            return enhanced
            
        except Exception as e:
            print(f"Image enhancement failed: {e}")
            return image

    @staticmethod
    def enhance_face_image(face_image: np.ndarray) -> np.ndarray:
        """Specific enhancement for face images"""
        try:
            # Histogram equalization
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            
            # Convert back to color
            enhanced_face = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            
            # Blend with original
            enhanced_face = cv2.addWeighted(face_image, 0.7, enhanced_face, 0.3, 0)
            
            # Gaussian blur for slight smoothing
            enhanced_face = cv2.GaussianBlur(enhanced_face, (3, 3), 0)
            
            return enhanced_face
            
        except Exception as e:
            print(f"Face enhancement failed: {e}")
            return face_image

    @staticmethod
    def assess_face_quality(face_image: np.ndarray) -> float:
        """Assess the quality of a face image for recognition"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = gray.std()
            
            # Normalize and combine metrics
            sharpness_score = min(1.0, laplacian_var / 1000.0)  # Normalize to 0-1
            brightness_score = 1.0 - abs(brightness - 128) / 128.0  # Prefer mid-range brightness
            contrast_score = min(1.0, contrast / 128.0)  # Normalize to 0-1
            
            # Weighted quality score
            quality_score = (sharpness_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
            
            return quality_score
            
        except Exception as e:
            print(f"Quality assessment failed: {e}")
            return 0.5  # Default medium quality
    
    @staticmethod
    def batch_process_images(input_dir: str, output_dir: str, category: str):
        """
        Process all images in a directory and extract faces
        """
        if not os.path.exists(input_dir):
            print(f"Input directory not found: {input_dir}")
            return
        
        # Create category-specific output directory
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        processed_count = 0
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(input_dir, filename)
                person_name = os.path.splitext(filename)[0]
                
                print(f"Processing: {filename}")
                
                # Extract faces
                faces = FaceUtils.extract_faces_from_image(image_path, category_dir, person_name)
                
                if faces:
                    processed_count += len(faces)
                    print(f"Extracted {len(faces)} faces from {filename}")
                else:
                    print(f"No faces found in {filename}")
        
        print(f"Total faces processed: {processed_count}")
    
    @staticmethod
    def verify_face_quality(image_path: str) -> Tuple[bool, str]:
        """
        Verify if an image has good quality for face recognition
        Returns (is_good_quality, message)
        """
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return False, "No face detected in image"
            
            if len(face_locations) > 1:
                return False, f"Multiple faces detected ({len(face_locations)}). Use single face images."
            
            # Check face size
            top, right, bottom, left = face_locations[0]
            face_width = right - left
            face_height = bottom - top
            
            if face_width < 100 or face_height < 100:
                return False, f"Face too small ({face_width}x{face_height}). Minimum 100x100 pixels recommended."
            
            # Try to encode the face
            face_encodings = face_recognition.face_encodings(image, face_locations)
            if not face_encodings:
                return False, "Failed to encode face. Image quality may be poor."
            
            return True, "Face quality is good"
            
        except Exception as e:
            return False, f"Error processing image: {str(e)}"


def setup_face_database():
    """Interactive setup for face database"""
    print("=== Face Database Setup ===")
    print("This utility helps you organize known faces for the detection system.")
    print()
    
    while True:
        print("Options:")
        print("1. Extract faces from image")
        print("2. Batch process directory")
        print("3. Verify face quality")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            person_name = input("Enter person name: ").strip()
            category = input("Enter category (restricted/criminal): ").strip().lower()
            
            if category == 'restricted':
                output_dir = config.RESTRICTED_FACES_DIR
            elif category == 'criminal':
                output_dir = config.CRIMINAL_FACES_DIR
            else:
                print("Invalid category. Use 'restricted' or 'criminal'")
                continue
            
            faces = FaceUtils.extract_faces_from_image(image_path, output_dir, person_name)
            print(f"Extracted {len(faces)} faces")
        
        elif choice == '2':
            input_dir = input("Enter input directory path: ").strip()
            category = input("Enter category (restricted/criminal): ").strip().lower()
            
            if category == 'restricted':
                output_dir = config.RESTRICTED_FACES_DIR
            elif category == 'criminal':
                output_dir = config.CRIMINAL_FACES_DIR
            else:
                print("Invalid category. Use 'restricted' or 'criminal'")
                continue
            
            FaceUtils.batch_process_images(input_dir, ".", category)
        
        elif choice == '3':
            image_path = input("Enter image path: ").strip()
            is_good, message = FaceUtils.verify_face_quality(image_path)
            print(f"Quality check: {message}")
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice")
        
        print()


if __name__ == "__main__":
    setup_face_database()
