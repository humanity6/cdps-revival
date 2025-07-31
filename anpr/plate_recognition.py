# Standard libs
import cv2
import numpy as np
import logging
import re
import asyncio
from typing import List, Dict, Tuple, Optional
import math

# Third-party
from fastanpr import FastANPR
import pytesseract

# Optional dependencies with fallbacks
try:
    from scipy import ndimage
    from skimage import morphology, measure, segmentation
    from skimage.filters import threshold_otsu, gaussian
    from skimage.transform import rotate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Local
from config import ANPRConfig

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedPlateRecognitionEngine:
    def __init__(self, min_confidence: float = 0.6):
        """Initialize the enhanced number plate recognition engine."""
        try:
            self.config = ANPRConfig()
            # Initialize FastANPR with optimized configuration
            self.anpr = FastANPR()
            logger.info("✅ FastANPR initialized successfully with default configuration")
        except Exception as e:
            logger.debug(f"Basic initialization failed: {e}, trying with GPU configuration")
            try:
                use_gpu = bool(getattr(self.config.performance, "enable_gpu", False))
                self.anpr = FastANPR(device="cuda" if use_gpu else "cpu")
                logger.info("✅ FastANPR initialized successfully with device configuration")
            except Exception as e:
                logger.error(f"❌ Failed to initialize FastANPR: {e}")
                raise

        # Detection parameters
        self.min_confidence = max(0.0, min_confidence)
        self.debug_mode = getattr(self.config, "debug_mode", False)

        # Enhanced plate validation patterns (supports multiple international formats)
        patterns = [
            # Indian formats
            r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',       # New Indian format
            r'^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$',       # Old Indian format
            
            # Pakistani formats (enhanced)
            r'^[A-Z]{3}\d{3,4}$',                   # Pakistani standard: LEA486, MNA1234
            r'^[A-Z]{2,4}\d{2,4}$',                 # Pakistani variant: MNA486, ABCD1234
            r'^[A-Z]{2}\d{2}[A-Z]{1}\d{3}$',       # Pakistani old format
            
            # International formats
            r'^[A-Z]{1,3}\d{1,4}[A-Z]{0,2}$',      # Generic alpha-numeric
            r'^\d{1,3}[A-Z]{1,3}\d{1,4}$',         # Numeric-alpha-numeric
            r'^[A-Z]{2,3}\d{2,4}$',                 # European style
            r'^[A-Z0-9]{4,10}$',                    # Generic fallback
            
            # Two-line patterns (enhanced for Pakistani)
            r'^[A-Z]{2,4}\n\d{2,4}$',              # Two-line format: MNA\n486
            r'^[A-Z]{2,4}\s*\d{3,5}$',             # Spaced format: MNA 486
        ]
        self.plate_patterns = [re.compile(p) for p in patterns]

        # Character correction mappings for common OCR errors
        self.char_corrections = {
            'O': '0', 'Q': '0', 'D': '0',  # Round shapes to zero
            'I': '1', 'L': '1', '|': '1',  # Vertical lines to one
            'Z': '2', 'S': '5', 'G': '6',  # Similar shapes
            'B': '8', 'T': '7', 'A': '4'   # Shape similarities
        }

        logger.info("✅ Enhanced number plate recognition engine initialized successfully")
    
    def detect_plates_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect number plates in a video frame using enhanced processing.
        """
        try:
            # Preprocess frame for better detection
            enhanced_frame = self._enhance_frame_for_detection(frame)
            
            # FastANPR expects RGB images
            rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

            # Run FastANPR detection
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # If we're in an event loop, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.anpr.run([rgb_frame]))
                    detections_nested = future.result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                detections_nested = asyncio.run(self.anpr.run([rgb_frame]))
            detections_raw = detections_nested[0] if detections_nested else []

            results = []
            for plate in detections_raw:
                try:
                    raw_text = getattr(plate, 'rec_text', '') or ''
                    plate_text = raw_text.strip().upper()
                    confidence = float(getattr(plate, 'rec_conf', 0.0) or 0.0)
                    bbox = getattr(plate, 'det_box', [0, 0, 0, 0])

                    # Enhanced multi-strategy plate processing
                    enhanced_results = self._process_detected_plate(frame, bbox, plate_text, confidence)
                    
                    for result in enhanced_results:
                        if result['confidence'] >= self.min_confidence and self._is_valid_plate(result['plate_number']):
                            results.append(result)

                except Exception as inner_e:
                    logger.debug(f"Skipping a detection due to parse error: {inner_e}")
                    continue
            
            # Remove duplicates and sort by confidence
            results = self._remove_duplicate_detections(results)
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"❌ Error detecting plates in frame: {e}")
            return []
    
    def _enhance_frame_for_detection(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced frame preprocessing for better plate detection."""
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Sharpen the image
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(blurred, -1, kernel)
            
            # Convert back to BGR
            enhanced_frame = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            
            return enhanced_frame
        except Exception as e:
            logger.debug(f"Frame enhancement error: {e}")
            return frame
    
    def _process_detected_plate(self, frame: np.ndarray, bbox: List[int], raw_text: str, confidence: float) -> List[Dict]:
        """
        Enhanced plate processing with multiple strategies for better accuracy.
        """
        results = []
        
        try:
            # Strategy 1: Use original FastANPR result
            if raw_text and self._is_valid_plate(raw_text):
                results.append({
                    'plate_number': self._normalize_plate_text(raw_text),
                    'confidence': confidence,
                    'bbox': bbox,
                    'method': 'fastanpr_direct'
                })
            
            # Strategy 2: Enhanced OCR on extracted plate region
            plate_img = self._extract_and_enhance_plate(frame, bbox)
            if plate_img.size > 0:
                # Try multiple OCR approaches
                ocr_results = self._multi_strategy_ocr(plate_img)
                
                for method, text in ocr_results.items():
                    if text and self._is_valid_plate(text):
                        # Calculate confidence based on method and text quality
                        method_confidence = self._calculate_ocr_confidence(text, method, confidence)
                        results.append({
                            'plate_number': self._normalize_plate_text(text),
                            'confidence': method_confidence,
                            'bbox': bbox,
                            'method': method
                        })
            
            # Strategy 3: Two-line specific processing
            two_line_results = self._process_two_line_plate(frame, bbox, raw_text)
            results.extend(two_line_results)
            
            # Strategy 4: Pakistani year-detection fallback
            # If we got a result like "LEB06", try to find the real registration number
            for result in results[:]:  # Copy to avoid modification during iteration
                plate_text = result['plate_number']
                if self._looks_like_year_mistake(plate_text):
                    corrected = self._try_fix_year_mistake(frame, bbox, plate_text)
                    if corrected:
                        results.append({
                            'plate_number': corrected,
                            'confidence': result['confidence'] * 0.9,  # Slightly lower confidence
                            'bbox': bbox,
                            'method': 'year_mistake_correction'
                        })
            
            return results
            
        except Exception as e:
            logger.debug(f"Plate processing error: {e}")
            return []
    
    def _extract_and_enhance_plate(self, frame: np.ndarray, bbox: List[int], padding: int = 15) -> np.ndarray:
        """
        Extract and enhance plate region with advanced preprocessing.
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract region
            plate_img = frame[y1:y2, x1:x2].copy()
            
            if plate_img.size == 0:
                return np.array([])
            
            # Enhance the extracted plate
            enhanced_plate = self._enhance_plate_image(plate_img)
            return enhanced_plate
            
        except Exception as e:
            logger.debug(f"Plate extraction error: {e}")
            return np.array([])
    
    def _enhance_plate_image(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Advanced plate image enhancement for better OCR.
        """
        try:
            # Resize if too small
            h, w = plate_img.shape[:2]
            if min(h, w) < 50:
                scale = max(2.0, 100 / min(h, w))
                plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img.copy()
            
            # Deskew the image
            deskewed = self._deskew_image(gray)
            
            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(deskewed)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            enhanced = clahe.apply(denoised)
            
            # Morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.debug(f"Plate enhancement error: {e}")
            return plate_img
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Correct skew in plate images for better OCR.
        """
        try:
            # Apply edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
            
            if lines is not None and len(lines) > 0:
                # Calculate the most common angle
                angles = []
                for rho, theta in lines[:min(10, len(lines))]:
                    angle = theta * 180 / np.pi - 90
                    angles.append(angle)
                
                # Use median angle to avoid outliers
                if angles:
                    median_angle = np.median(angles)
                    
                    # Only correct if angle is significant
                    if abs(median_angle) > 1:
                        # Rotate image
                        center = tuple(np.array(image.shape[1::-1]) / 2)
                        rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        corrected = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
                        return corrected
            
            return image
            
        except Exception as e:
            logger.debug(f"Deskew error: {e}")
            return image
    
    def _multi_strategy_ocr(self, plate_img: np.ndarray) -> Dict[str, str]:
        """
        Apply multiple OCR strategies for better text extraction.
        """
        results = {}
        
        try:
            # Strategy 1: Standard OCR with preprocessing
            results['standard_ocr'] = self._ocr_with_preprocessing(plate_img, 'standard')
            
            # Strategy 2: OCR optimized for single line
            results['single_line_ocr'] = self._ocr_with_preprocessing(plate_img, 'single_line')
            
            # Strategy 3: OCR optimized for two lines
            results['two_line_ocr'] = self._ocr_with_preprocessing(plate_img, 'two_line')
            
            # Strategy 4: Character-level segmentation OCR
            results['segmented_ocr'] = self._segmented_character_ocr(plate_img)
            
            # Strategy 5: Inverted image OCR (for dark plates)
            inverted = cv2.bitwise_not(plate_img)
            results['inverted_ocr'] = self._ocr_with_preprocessing(inverted, 'standard')
            
        except Exception as e:
            logger.debug(f"Multi-strategy OCR error: {e}")
        
        return results
    
    def _ocr_with_preprocessing(self, plate_img: np.ndarray, strategy: str) -> str:
        """
        OCR with different preprocessing strategies.
        """
        try:
            # Choose PSM based on strategy
            psm_map = {
                'standard': 7,      # Single text line
                'single_line': 8,   # Single word
                'two_line': 6,      # Uniform block of text
            }
            psm = psm_map.get(strategy, 7)
            
            # Apply strategy-specific preprocessing
            if strategy == 'two_line':
                # For two-line plates, apply different thresholding
                thresh = cv2.adaptiveThreshold(
                    plate_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            else:
                # Standard thresholding
                _, thresh = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR configuration
            config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            # Extract text
            text = pytesseract.image_to_string(thresh, config=config)
            
            # Clean and normalize text
            cleaned_text = re.sub(r'[^A-Z0-9\n]', '', text.upper())
            
            return cleaned_text
            
        except Exception as e:
            logger.debug(f"OCR preprocessing error ({strategy}): {e}")
            return ""
    
    def _segmented_character_ocr(self, plate_img: np.ndarray) -> str:
        """
        Character-level segmentation and OCR for difficult plates.
        """
        try:
            # Apply morphological operations to separate characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            processed = cv2.morphologyEx(plate_img, cv2.MORPH_CLOSE, kernel)
            
            # Find contours (potential characters)
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort contours by position
            valid_contours = []
            h, w = plate_img.shape
            min_area = (h * w) * 0.01  # Minimum 1% of image area
            max_area = (h * w) * 0.3   # Maximum 30% of image area
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    x, y, cw, ch = cv2.boundingRect(contour)
                    # Check aspect ratio (characters should be taller than wide typically)
                    if 0.2 < ch/cw < 5:
                        valid_contours.append((x, y, cw, ch, contour))
            
            # Sort by x-coordinate (left to right)
            valid_contours.sort(key=lambda x: x[0])
            
            # OCR each character
            characters = []
            for x, y, cw, ch, contour in valid_contours:
                # Extract character region with padding
                padding = 2
                char_region = plate_img[max(0, y-padding):min(h, y+ch+padding), 
                                      max(0, x-padding):min(w, x+cw+padding)]
                
                if char_region.size > 0:
                    # Resize character for better OCR
                    char_region = cv2.resize(char_region, (50, 70), interpolation=cv2.INTER_CUBIC)
                    
                    # OCR single character
                    config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    char_text = pytesseract.image_to_string(char_region, config=config).strip()
                    
                    if char_text and len(char_text) == 1:
                        characters.append(char_text.upper())
            
            return ''.join(characters)
            
        except Exception as e:
            logger.debug(f"Segmented OCR error: {e}")
            return ""
    
    def _process_two_line_plate(self, frame: np.ndarray, bbox: List[int], raw_text: str) -> List[Dict]:
        """
        Enhanced two-line plate processing for various formats.
        """
        results = []
        
        try:
            # Extract plate with extra vertical padding for two-line detection
            x1, y1, x2, y2 = map(int, bbox)
            h_box = y2 - y1
            
            # Extend bounding box vertically
            y1_ext = max(0, y1 - int(h_box * 0.2))
            y2_ext = min(frame.shape[0], y2 + int(h_box * 0.5))
            
            extended_plate = frame[y1_ext:y2_ext, x1:x2].copy()
            
            if extended_plate.size == 0:
                return results
            
            # Enhance the extended plate image
            enhanced_plate = self._enhance_plate_image(extended_plate)
            
            # Try to detect two-line structure
            two_line_text = self._detect_two_line_structure(enhanced_plate)
            
            if two_line_text:
                # Validate and format two-line result
                formatted_text = self._format_two_line_plate(two_line_text)
                if formatted_text and self._is_valid_plate(formatted_text):
                    confidence = 0.8  # High confidence for successful two-line detection
                    results.append({
                        'plate_number': formatted_text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'method': 'two_line_detection'
                    })
            
            # Fallback: Try splitting the original detection into two parts
            if raw_text and len(raw_text) >= 6:
                fallback_results = self._try_two_line_fallback(raw_text, bbox)
                results.extend(fallback_results)
            
        except Exception as e:
            logger.debug(f"Two-line processing error: {e}")
        
        return results
    
    def _detect_two_line_structure(self, plate_img: np.ndarray) -> Optional[str]:
        """
        Detect and extract text from two-line plate structure.
        Enhanced for Pakistani plates with city code + year on top, number on bottom.
        """
        try:
            h, w = plate_img.shape[:2]
            logger.debug(f"Processing two-line plate image: {h}x{w}")
            
            # Enhanced region separation for better OCR
            # More aggressive separation to ensure clean text regions
            top_end = int(h * 0.4)   # Top 40% for city code + year
            bottom_start = int(h * 0.6)  # Bottom 40% for registration number
            
            # Extract regions with better boundaries
            top_region = plate_img[0:top_end, :]
            bottom_region = plate_img[bottom_start:, :]
            
            # Apply additional preprocessing for each region
            top_enhanced = self._enhance_text_region(top_region)
            bottom_enhanced = self._enhance_text_region(bottom_region)
            
            # OCR each region with multiple strategies
            results = []
            
            # Try different OCR configurations
            configs = [
                ('--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 'psm7'),
                ('--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 'psm8'),
                ('--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 'psm13')
            ]
            
            for config, config_name in configs:
                try:
                    # OCR top region (city code + year)
                    top_text = pytesseract.image_to_string(top_enhanced, config=config).strip().upper()
                    # OCR bottom region (registration number)
                    bottom_text = pytesseract.image_to_string(bottom_enhanced, config=config).strip().upper()
                    
                    logger.debug(f"OCR {config_name} - Top: '{top_text}', Bottom: '{bottom_text}'")
                    
                    # Clean the text but preserve spaces in top text for parsing
                    top_clean = re.sub(r'[^A-Z0-9\s]', '', top_text).strip()
                    bottom_clean = re.sub(r'[^A-Z0-9]', '', bottom_text).strip()
                    
                    # Skip if either region is empty
                    if not top_clean or not bottom_clean:
                        continue
                    
                    # Process as Pakistani two-line plate
                    processed_result = self._process_pakistani_two_line(top_clean, bottom_clean)
                    if processed_result:
                        results.append((processed_result, config_name))
                        logger.debug(f"Valid result from {config_name}: {processed_result}")
                        
                except Exception as e:
                    logger.debug(f"OCR config {config_name} failed: {e}")
                    continue
            
            # Return the best result prioritizing longer registration numbers
            if results:
                def score_result(result_tuple):
                    result, config = result_tuple
                    lines = result.split('\n')
                    if len(lines) == 2:
                        city_code, reg_number = lines
                        # Prefer longer registration numbers (more likely to be correct)
                        return len(reg_number) * 10 + len(city_code)
                    return 0
                
                best_result = max(results, key=score_result)[0]
                logger.debug(f"Selected best result: {best_result}")
                return best_result
            
            return None
            
        except Exception as e:
            logger.debug(f"Two-line structure detection error: {e}")
            return None
    
    def _looks_like_year_mistake(self, plate_text: str) -> bool:
        """
        Check if a plate looks like we accidentally picked the year instead of registration.
        Example: "LEB06" looks like year mistake, "LEB5700" does not.
        """
        try:
            clean_plate = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
            
            # Pakistani pattern: 2-4 letters + 2-4 digits
            match = re.match(r'^([A-Z]{2,4})(\d{2,4})$', clean_plate)
            if match:
                city_code, number = match.groups()
                # Likely year mistake if:
                # 1. Number is 2 digits (like 06, 17, 18, 19, 20, 21, etc.)
                # 2. Number looks like a year (00-30 range typical for Pakistani plates)
                if len(number) == 2 and 0 <= int(number) <= 30:
                    logger.debug(f"Detected possible year mistake: {plate_text} (number: {number})")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Year mistake detection error: {e}")
            return False
    
    def _try_fix_year_mistake(self, frame: np.ndarray, bbox: List[int], wrong_plate: str) -> Optional[str]:
        """
        Try to fix a year mistake by re-OCRing with focus on bottom region.
        """
        try:
            # Extract city code from wrong plate
            clean_wrong = re.sub(r'[^A-Z0-9]', '', wrong_plate.upper())
            match = re.match(r'^([A-Z]{2,4})\d{2,4}$', clean_wrong)
            if not match:
                return None
                
            city_code = match.group(1)
            
            # Re-extract plate with focus on bottom region
            plate_img = self._extract_and_enhance_plate(frame, bbox)
            if plate_img.size == 0:
                return None
            
            h, w = plate_img.shape[:2]
            # Focus heavily on bottom region where registration number should be
            bottom_region = plate_img[int(h * 0.65):, :]  # Bottom 35%
            
            if bottom_region.size == 0:
                return None
            
            # Enhance specifically for numbers
            enhanced_bottom = self._enhance_text_region(bottom_region)
            
            # OCR with focus on digits
            config = '--psm 8 -c tessedit_char_whitelist=0123456789'
            bottom_text = pytesseract.image_to_string(enhanced_bottom, config=config).strip()
            bottom_clean = re.sub(r'[^0-9]', '', bottom_text)
            
            logger.debug(f"Year mistake correction: extracted '{bottom_clean}' from bottom region")
            
            # Validate the extracted number
            if bottom_clean and len(bottom_clean) >= 3 and bottom_clean.isdigit():
                corrected = f"{city_code}{bottom_clean}"
                logger.debug(f"Corrected {wrong_plate} to {corrected}")
                return corrected
            
            return None
            
        except Exception as e:
            logger.debug(f"Year mistake correction error: {e}")
            return None

    def _enhance_text_region(self, region_img: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing specifically for text regions.
        """
        try:
            if region_img.size == 0:
                return region_img
                
            # Resize small regions for better OCR
            h, w = region_img.shape[:2]
            if h < 30 or w < 60:
                scale_h = max(2.0, 40 / h)
                scale_w = max(2.0, 100 / w)
                scale = min(scale_h, scale_w, 4.0)  # Cap at 4x to avoid excessive scaling
                region_img = cv2.resize(region_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale if needed
            if len(region_img.shape) == 3:
                gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = region_img.copy()
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
            enhanced = clahe.apply(filtered)
            
            # Apply adaptive thresholding for better text separation
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Light morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.debug(f"Text region enhancement error: {e}")
            return region_img
    
    def _process_pakistani_two_line(self, top_text: str, bottom_text: str) -> Optional[str]:
        """
        Process Pakistani two-line plates: City Code + Year / Registration Number
        Example: "LEB 06" / "5700" -> "LEB5700" (NOT "LEB06")
        """
        try:
            if not top_text or not bottom_text:
                return None
                
            logger.debug(f"Processing Pakistani two-line: top='{top_text}', bottom='{bottom_text}'")
            
            # Clean bottom text (should be the registration number)
            bottom_clean = re.sub(r'[^0-9]', '', bottom_text.strip())
            
            # Process top text to extract city code (ignore year)
            top_parts = top_text.strip().split()
            city_code = None
            
            # Try to extract city code from the first part of top line
            if len(top_parts) >= 1:
                potential_city = re.sub(r'[^A-Z]', '', top_parts[0])
                if len(potential_city) >= 2 and potential_city.isalpha():
                    city_code = potential_city
                    logger.debug(f"Extracted city code from first part: '{city_code}'")
            
            # If we couldn't extract city code from first part, try the whole top line
            if not city_code:
                all_letters = re.sub(r'[^A-Z]', '', top_text)
                if len(all_letters) >= 2:
                    # Take first 2-4 letters as city code
                    city_code = all_letters[:4] if len(all_letters) <= 4 else all_letters[:3]
                    logger.debug(f"Extracted city code from all letters: '{city_code}'")
            
            # Enhanced validation for Pakistani plates
            if city_code and bottom_clean:
                # Pakistani city codes are typically 2-4 letters
                if (2 <= len(city_code) <= 4 and city_code.isalpha() and
                    2 <= len(bottom_clean) <= 4 and bottom_clean.isdigit()):
                    
                    # Strong preference for bottom line registration number over top line year
                    # Bottom line should always be the registration number in Pakistani plates
                    if len(bottom_clean) >= 2:  # Accept any 2+ digit number from bottom line
                        logger.debug(f"Pakistani plate detected: {city_code} + {bottom_clean} (bottom line priority)")
                        return f"{city_code}\n{bottom_clean}"
                    else:
                        logger.debug(f"Rejected: bottom line too short: {bottom_clean}")
                        
            return None
            
        except Exception as e:
            logger.debug(f"Pakistani two-line processing error: {e}")
            return None
    
    def _format_two_line_plate(self, two_line_text: str) -> str:
        """
        Format two-line plate text into standard format.
        """
        try:
            lines = two_line_text.split('\n')
            if len(lines) != 2:
                return ""
            
            top_line, bottom_line = lines[0].strip(), lines[1].strip()
            
            # Apply character corrections
            top_line = self._apply_character_corrections(top_line, prefer_letters=True)
            bottom_line = self._apply_character_corrections(bottom_line, prefer_letters=False)
            
            # Combine lines
            combined = f"{top_line}{bottom_line}"
            
            return combined
            
        except Exception as e:
            logger.debug(f"Two-line formatting error: {e}")
            return ""
    
    def _apply_character_corrections(self, text: str, prefer_letters: bool = False) -> str:
        """
        Apply character corrections based on context.
        """
        corrected = ""
        for char in text:
            if prefer_letters:
                # For letter-expected positions, convert digits that look like letters
                if char in ['0', '6', '8']:
                    corrected += {'0': 'O', '6': 'G', '8': 'B'}.get(char, char)
                else:
                    corrected += char
            else:
                # For digit-expected positions, convert letters that look like digits
                corrected += self.char_corrections.get(char, char)
        
        return corrected
    
    def _try_two_line_fallback(self, raw_text: str, bbox: List[int]) -> List[Dict]:
        """
        Fallback method for two-line plates when structure detection fails.
        """
        results = []
        
        try:
            clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
            
            # Try common splitting patterns
            split_patterns = [
                (3, None),  # First 3 characters, rest
                (2, 4),     # First 2, next 4
                (4, None),  # First 4, rest
            ]
            
            for first_n, second_n in split_patterns:
                if len(clean_text) >= first_n + 3:  # Minimum total length
                    first_part = clean_text[:first_n]
                    second_part = clean_text[first_n:first_n + second_n] if second_n else clean_text[first_n:]
                    
                    # Validate parts
                    if (any(c.isalpha() for c in first_part) and 
                        any(c.isdigit() for c in second_part) and
                        len(second_part) >= 3):
                        
                        combined = f"{first_part}{second_part}"
                        if self._is_valid_plate(combined):
                            results.append({
                                'plate_number': combined,
                                'confidence': 0.6,  # Lower confidence for fallback
                                'bbox': bbox,
                                'method': 'two_line_fallback'
                            })
        
        except Exception as e:
            logger.debug(f"Two-line fallback error: {e}")
        
        return results
    
    def _calculate_ocr_confidence(self, text: str, method: str, base_confidence: float) -> float:
        """
        Calculate confidence score for OCR results.
        """
        try:
            # Base confidence from detection
            confidence = base_confidence * 0.7  # Start with 70% of original
            
            # Method-based adjustments
            method_multipliers = {
                'fastanpr_direct': 1.0,
                'standard_ocr': 0.9,
                'single_line_ocr': 0.85,
                'two_line_ocr': 0.9,
                'segmented_ocr': 0.8,
                'inverted_ocr': 0.75,
                'two_line_detection': 0.95,
                'two_line_fallback': 0.6
            }
            
            confidence *= method_multipliers.get(method, 0.7)
            
            # Text quality adjustments
            if text:
                # Length bonus/penalty
                if 6 <= len(text) <= 10:
                    confidence *= 1.1
                elif len(text) < 4:
                    confidence *= 0.7
                
                # Character type balance
                letters = sum(1 for c in text if c.isalpha())
                digits = sum(1 for c in text if c.isdigit())
                if letters > 0 and digits > 0:
                    confidence *= 1.05
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.debug(f"Confidence calculation error: {e}")
            return base_confidence * 0.5
    
    def _remove_duplicate_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Remove duplicate detections based on similarity and bbox overlap.
        Enhanced to handle cases like LEB06 vs AFLEB5700 from same plate.
        """
        if not detections:
            return detections
        
        unique_detections = []
        
        for detection in detections:
            is_duplicate = False
            
            for existing in unique_detections:
                # Check if plates are similar OR if they're from overlapping regions
                if (self._are_plates_similar(detection['plate_number'], existing['plate_number']) or
                    self._are_bboxes_overlapping(detection['bbox'], existing['bbox'])):
                    
                    # Choose the better detection based on multiple criteria
                    if self._is_better_detection(detection, existing):
                        unique_detections.remove(existing)
                        unique_detections.append(detection)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(detection)
        
        return unique_detections
    
    def _are_bboxes_overlapping(self, bbox1: List[int], bbox2: List[int], threshold: float = 0.7) -> bool:
        """
        Check if two bounding boxes significantly overlap (likely same plate).
        """
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)
            
            if x_right < x_left or y_bottom < y_top:
                return False
            
            # Calculate intersection and union areas
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area
            
            # IoU (Intersection over Union)
            iou = intersection_area / union_area if union_area > 0 else 0
            
            return iou >= threshold
            
        except Exception as e:
            logger.debug(f"Bbox overlap check error: {e}")
            return False
    
    def _is_better_detection(self, detection1: Dict, detection2: Dict) -> bool:
        """
        Determine which detection is better based on multiple criteria.
        """
        try:
            # Criteria for determining better detection
            score1 = 0
            score2 = 0
            
            # 1. Confidence score (40% weight)
            conf1 = detection1.get('confidence', 0)
            conf2 = detection2.get('confidence', 0)
            if conf1 > conf2:
                score1 += 4
            elif conf2 > conf1:
                score2 += 4
                
            # 2. Plate length (30% weight) - prefer longer, more complete plates
            plate1 = detection1.get('plate_number', '')
            plate2 = detection2.get('plate_number', '')
            clean1 = re.sub(r'[^A-Z0-9]', '', plate1)
            clean2 = re.sub(r'[^A-Z0-9]', '', plate2)
            
            if len(clean1) > len(clean2):
                score1 += 3
            elif len(clean2) > len(clean1):
                score2 += 3
                
            # 3. Method quality (20% weight)
            method1 = detection1.get('method', '')
            method2 = detection2.get('method', '')
            
            method_priority = {
                'two_line_detection': 5,
                'pakistani_two_line': 5,
                'fastanpr_direct': 4,
                'standard_ocr': 3,
                'single_line_ocr': 2,
                'segmented_ocr': 3,
                'inverted_ocr': 2,
                'two_line_fallback': 1
            }
            
            priority1 = method_priority.get(method1, 1)
            priority2 = method_priority.get(method2, 1)
            
            if priority1 > priority2:
                score1 += 2
            elif priority2 > priority1:
                score2 += 2
                
            # 4. Avoid obvious junk (10% weight)
            # Penalize plates with obvious OCR artifacts
            if self._has_ocr_artifacts(plate1):
                score1 -= 1
            if self._has_ocr_artifacts(plate2):
                score2 -= 1
                
            return score1 > score2
            
        except Exception as e:
            logger.debug(f"Detection comparison error: {e}")
            return detection1.get('confidence', 0) > detection2.get('confidence', 0)
    
    def _has_ocr_artifacts(self, plate_text: str) -> bool:
        """
        Check if plate text has obvious OCR artifacts.
        """
        try:
            clean_plate = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
            
            # Check for common OCR artifacts
            artifacts = [
                len(clean_plate) > 10,  # Too long
                len(clean_plate) < 4,   # Too short
                clean_plate.startswith('AF'),  # Common OCR prefix error
                clean_plate.startswith('0'),   # Plates don't start with 0
                clean_plate.endswith('0' * 3), # Too many trailing zeros
                bool(re.search(r'[0-9]{6,}', clean_plate)),  # Too many consecutive digits
                bool(re.search(r'[A-Z]{6,}', clean_plate)),  # Too many consecutive letters
            ]
            
            return any(artifacts)
            
        except Exception as e:
            logger.debug(f"OCR artifact check error: {e}")
            return False
    
    def _are_plates_similar(self, plate1: str, plate2: str, threshold: float = 0.8) -> bool:
        """
        Check if two plate numbers are similar (likely the same plate).
        """
        try:
            # Normalize both plates
            p1 = re.sub(r'[^A-Z0-9]', '', plate1.upper())
            p2 = re.sub(r'[^A-Z0-9]', '', plate2.upper())
            
            if p1 == p2:
                return True
            
            # Check similarity using Levenshtein-like approach
            if len(p1) == 0 or len(p2) == 0:
                return False
            
            # Simple similarity check
            matches = sum(1 for a, b in zip(p1, p2) if a == b)
            similarity = matches / max(len(p1), len(p2))
            
            return similarity >= threshold
            
        except Exception as e:
            logger.debug(f"Plate similarity check error: {e}")
            return False
    
    def _is_valid_plate(self, plate_text: str) -> bool:
        """
        Enhanced validation for license plates.
        """
        try:
            if not plate_text:
                return False
            
            # Clean the plate text
            clean_plate = re.sub(r'[\s\-\n]', '', plate_text.upper())
            
            # Length check
            if not (3 <= len(clean_plate) <= 12):
                return False
            
            # Must contain both letters and numbers
            has_letters = any(c.isalpha() for c in clean_plate)
            has_digits = any(c.isdigit() for c in clean_plate)
            
            if not (has_letters and has_digits):
                return False
            
            # Pattern matching
            for pattern in self.plate_patterns:
                if pattern.match(clean_plate) or pattern.match(plate_text.upper()):
                    return True
            
            # Fallback validation for reasonable alphanumeric mix
            if len(clean_plate) >= 5:
                letter_count = sum(1 for c in clean_plate if c.isalpha())
                digit_count = sum(1 for c in clean_plate if c.isdigit())
                
                # At least 2 letters and 2 digits
                if letter_count >= 2 and digit_count >= 2:
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Plate validation error: {e}")
            return False
    
    def _normalize_plate_text(self, plate_text: str) -> str:
        """
        Robust normalization for Pakistani plates:
        - Multi-line: first line (alphabets) + last line (digits, 1-4), ignore all middle lines (issuance year).
        - Single-line: only accept 2-3 letters + 1, 3, or 4 digits. Never accept 2-3 letters + 2 digits.
        """
        import re
        try:
            # Split into lines and clean
            lines = [l.strip().replace(' ', '') for l in plate_text.splitlines() if l.strip()]
            if len(lines) >= 2:
                first = lines[0]
                last = lines[-1]
                # Only accept if first is 2-3 letters, last is 1-4 digits
                if re.fullmatch(r"[A-Z]{2,3}", first, re.IGNORECASE) and re.fullmatch(r"\d{1,4}", last):
                    return f"{first.upper()}{last}"
                else:
                    return ''  # Only accept if last line is all digits
            # Fallback: single-line logic
            normalized = re.sub(r'[\n\r\s-]+', '', plate_text.upper())
            match = re.match(r'^([A-Z]{2,3})(\d{1}|\d{3,4})$', normalized)
            if match:
                city, number = match.groups()
                return f"{city}{number}"
            return ''  # Never accept 2-3 letters + 2 digits
        except Exception as e:
            logger.debug(f"Plate normalization error: {e}")
            return ''

    def draw_plate_boxes(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw enhanced bounding boxes around detected plates.
        """
        annotated_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            plate_number = detection['plate_number']
            confidence = detection['confidence']
            method = detection.get('method', 'unknown')
            
            # Extract coordinates
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Color coding based on confidence
                if confidence >= 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence >= 0.6:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 165, 255)  # Orange for low confidence
                
                # Draw bounding box with thickness based on confidence
                thickness = max(2, int(confidence * 4))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Prepare label with method info
                label = f"{plate_number} ({confidence:.2f})"
                if self.debug_mode:
                    label += f" [{method}]"
                
                # Calculate label dimensions
                font_scale = 0.5
                font_thickness = 2
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                
                # Draw background for text
                label_y = y1 - 10 if y1 > 30 else y2 + 25
                cv2.rectangle(annotated_frame, 
                            (x1, label_y - label_size[1] - 5),
                            (x1 + label_size[0] + 5, label_y + 5),
                            color, -1)
                
                # Add text
                cv2.putText(annotated_frame, label,
                          (x1 + 2, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                          (0, 0, 0), font_thickness)
                
                # Add ranking number
                cv2.putText(annotated_frame, f"#{i+1}",
                          (x2 - 30, y1 + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                          color, 2)
        
        return annotated_frame

    def extract_plate_image(self, frame: np.ndarray, bbox: List[int], padding: int = 10) -> np.ndarray:
        """
        Extract the plate region from the frame (backward compatibility method).
        """
        return self._extract_and_enhance_plate(frame, bbox, padding)

    def normalize_plate_number(self, plate_number: str) -> str:
        """
        Normalize plate number for consistent database storage and comparison (backward compatibility).
        """
        return self._normalize_plate_text(plate_number)

    def get_detection_statistics(self, detections: List[Dict]) -> Dict:
        """
        Get statistics about the detections for analysis and debugging.
        """
        try:
            if not detections:
                return {
                    'total_detections': 0,
                    'avg_confidence': 0.0,
                    'max_confidence': 0.0,
                    'min_confidence': 0.0,
                    'methods_used': [],
                    'plate_lengths': [],
                    'unique_plates': 0
                }

            confidences = [d['confidence'] for d in detections]
            methods = [d.get('method', 'unknown') for d in detections]
            plates = [d['plate_number'] for d in detections]
            lengths = [len(re.sub(r'[^A-Z0-9]', '', plate)) for plate in plates]
            unique_plates = len(set(plates))

            return {
                'total_detections': len(detections),
                'avg_confidence': sum(confidences) / len(confidences),
                'max_confidence': max(confidences),
                'min_confidence': min(confidences),
                'methods_used': list(set(methods)),
                'plate_lengths': lengths,
                'unique_plates': unique_plates,
                'method_counts': {method: methods.count(method) for method in set(methods)}
            }
        except Exception as e:
            logger.debug(f"Statistics calculation error: {e}")
            return {'error': str(e)}

    def validate_plate_format(self, plate_text: str, country_code: str = None) -> Dict:
        """
        Validate plate format against specific country standards.
        """
        try:
            result = {
                'is_valid': False,
                'country': country_code,
                'format': None,
                'confidence': 0.0,
                'issues': []
            }

            if not plate_text:
                result['issues'].append('Empty plate text')
                return result

            clean_plate = re.sub(r'[\s\-\n]', '', plate_text.upper())

            # Country-specific validation
            if country_code:
                if country_code.upper() == 'IN':  # India
                    patterns = [
                        (r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', 'New Indian Format'),
                        (r'^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$', 'Old Indian Format'),
                    ]
                elif country_code.upper() == 'PK':  # Pakistan
                    patterns = [
                        (r'^[A-Z]{3}\d{3,4}$', 'Pakistani Standard'),
                        (r'^[A-Z]{2}\d{2}[A-Z]{1}\d{3}$', 'Pakistani Variant'),
                    ]
                else:
                    patterns = [(r'^[A-Z0-9]{4,10}$', 'Generic Format')]
            else:
                # Use all patterns
                patterns = [
                    (r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', 'New Indian Format'),
                    (r'^[A-Z]{3}\d{3,4}$', 'Pakistani Standard'),
                    (r'^[A-Z0-9]{4,10}$', 'Generic Format'),
                ]

            # Check against patterns
            for pattern, format_name in patterns:
                if re.match(pattern, clean_plate):
                    result['is_valid'] = True
                    result['format'] = format_name
                    result['confidence'] = 0.9
                    break

            # Additional checks
            if not result['is_valid']:
                # Check basic requirements
                has_letters = any(c.isalpha() for c in clean_plate)
                has_digits = any(c.isdigit() for c in clean_plate)
                
                if not has_letters:
                    result['issues'].append('Missing letters')
                if not has_digits:
                    result['issues'].append('Missing digits')
                
                if 3 <= len(clean_plate) <= 12 and has_letters and has_digits:
                    result['is_valid'] = True
                    result['format'] = 'Generic Valid'
                    result['confidence'] = 0.6
                elif len(clean_plate) < 3:
                    result['issues'].append('Too short')
                elif len(clean_plate) > 12:
                    result['issues'].append('Too long')

            return result

        except Exception as e:
            logger.debug(f"Plate format validation error: {e}")
            return {'error': str(e), 'is_valid': False}

    def batch_process_plates(self, frames: List[np.ndarray], max_workers: int = 4) -> List[List[Dict]]:
        """
        Process multiple frames in parallel for better performance.
        """
        try:
            import concurrent.futures
            
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_frame = {executor.submit(self.detect_plates_in_frame, frame): i 
                                 for i, frame in enumerate(frames)}
                
                # Collect results in order
                frame_results = [None] * len(frames)
                for future in concurrent.futures.as_completed(future_to_frame):
                    frame_idx = future_to_frame[future]
                    try:
                        detections = future.result()
                        frame_results[frame_idx] = detections
                    except Exception as e:
                        logger.debug(f"Frame {frame_idx} processing error: {e}")
                        frame_results[frame_idx] = []
                
                return frame_results
                
        except ImportError:
            logger.warning("concurrent.futures not available, processing sequentially")
            return [self.detect_plates_in_frame(frame) for frame in frames]
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [self.detect_plates_in_frame(frame) for frame in frames]

# Maintain backward compatibility
class PlateRecognitionEngine(EnhancedPlateRecognitionEngine):
    """Backward compatibility alias."""
    pass