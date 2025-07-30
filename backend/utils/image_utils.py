"""
Image processing utilities for the unified backend
"""
import cv2
import numpy as np
import base64
import io
from PIL import Image
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Convert base64 string to OpenCV image
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None

def image_to_base64(image: np.ndarray, format: str = 'jpg') -> Optional[str]:
    """
    Convert OpenCV image to base64 string
    """
    try:
        # Encode image
        success, encoded_image = cv2.imencode(f'.{format}', image)
        if not success:
            return None
        
        # Convert to base64
        base64_string = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        return f"data:image/{format};base64,{base64_string}"
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        return None

def resize_image(image: np.ndarray, width: int, height: int, maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image with optional aspect ratio maintenance
    """
    try:
        if maintain_aspect:
            # Calculate aspect ratio
            h, w = image.shape[:2]
            aspect = w / h
            
            # Calculate new dimensions
            if aspect > width / height:
                new_width = width
                new_height = int(width / aspect)
            else:
                new_height = height
                new_width = int(height * aspect)
            
            # Resize image
            resized = cv2.resize(image, (new_width, new_height))
            
            # Create canvas with target size
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Calculate centering offsets
            y_offset = (height - new_height) // 2
            x_offset = (width - new_width) // 2
            
            # Place resized image on canvas
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return canvas
        else:
            return cv2.resize(image, (width, height))
    except Exception as e:
        logger.error(f"Failed to resize image: {e}")
        return image

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better detection
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    except Exception as e:
        logger.error(f"Failed to enhance image: {e}")
        return image

def crop_image(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Crop image to specified bounding box
    """
    try:
        h, w = image.shape[:2]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Ensure valid bounding box
        if x2 <= x1 or y2 <= y1:
            return image
        
        return image[y1:y2, x1:x2]
    except Exception as e:
        logger.error(f"Failed to crop image: {e}")
        return image

def draw_bounding_boxes(image: np.ndarray, detections: list, colors: dict = None) -> np.ndarray:
    """
    Draw bounding boxes on image with labels
    """
    try:
        annotated = image.copy()
        
        default_colors = {
            'face': (0, 255, 0),
            'weapon': (0, 0, 255),
            'license_plate': (255, 0, 0),
            'unknown': (255, 255, 0)
        }
        
        if colors:
            default_colors.update(colors)
        
        for detection in detections:
            # Extract bounding box
            if hasattr(detection, 'bbox'):
                bbox = detection.bbox
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            else:
                continue
            
            # Get color for detection type
            color = default_colors.get(detection.class_name, default_colors['unknown'])
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_parts = [detection.class_name]
            if hasattr(detection, 'confidence'):
                label_parts.append(f"{detection.confidence:.2f}")
            if hasattr(detection, 'plate_number'):
                label_parts.append(detection.plate_number)
            if hasattr(detection, 'person_name'):
                label_parts.append(detection.person_name)
            if hasattr(detection, 'weapon_type'):
                label_parts.append(detection.weapon_type)
            
            label = " | ".join(label_parts)
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    except Exception as e:
        logger.error(f"Failed to draw bounding boxes: {e}")
        return image

def validate_image(image: np.ndarray) -> bool:
    """
    Validate if image is valid for processing
    """
    try:
        if image is None:
            return False
        
        if len(image.shape) < 2:
            return False
        
        if image.shape[0] < 10 or image.shape[1] < 10:
            return False
        
        return True
    except Exception:
        return False

def convert_color_space(image: np.ndarray, conversion: str) -> np.ndarray:
    """
    Convert image color space
    """
    try:
        conversion_map = {
            'BGR2RGB': cv2.COLOR_BGR2RGB,
            'RGB2BGR': cv2.COLOR_RGB2BGR,
            'BGR2GRAY': cv2.COLOR_BGR2GRAY,
            'RGB2GRAY': cv2.COLOR_RGB2GRAY,
            'GRAY2BGR': cv2.COLOR_GRAY2BGR,
            'GRAY2RGB': cv2.COLOR_GRAY2RGB,
            'BGR2HSV': cv2.COLOR_BGR2HSV,
            'HSV2BGR': cv2.COLOR_HSV2BGR
        }
        
        if conversion in conversion_map:
            return cv2.cvtColor(image, conversion_map[conversion])
        else:
            logger.warning(f"Unknown color conversion: {conversion}")
            return image
    except Exception as e:
        logger.error(f"Failed to convert color space: {e}")
        return image

def get_image_info(image: np.ndarray) -> dict:
    """
    Get information about an image
    """
    try:
        if image is None:
            return {}
        
        shape = image.shape
        info = {
            'height': shape[0],
            'width': shape[1],
            'channels': shape[2] if len(shape) > 2 else 1,
            'dtype': str(image.dtype),
            'size_bytes': image.nbytes
        }
        
        return info
    except Exception as e:
        logger.error(f"Failed to get image info: {e}")
        return {}