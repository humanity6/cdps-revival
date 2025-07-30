from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import time
from datetime import datetime
import base64
import json

app = Flask(__name__)

# Global model variable
model = None

def load_bensam02_model():
    """Load the Bensam02 MobileNetV2 model."""
    global model
    try:
        model_path = "violence/bensam02_model.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("✓ Bensam02 MobileNetV2 model loaded successfully")
            return True
        else:
            print(f"✗ Model not found at: {model_path}")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def preprocess_frame(frame):
    """Preprocess frame for Bensam02 model (128x128 input)."""
    # Resize to 128x128 (Bensam02 model input size)
    frame = cv2.resize(frame, (128, 128))
    # Convert to RGB (OpenCV uses BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    frame = frame.astype('float32') / 255.0
    # Add batch dimension
    frame = np.expand_dims(frame, axis=0)
    return frame

def detect_violence(frame):
    """Detect violence in a single frame."""
    global model
    if model is None:
        return False, 0.0
    
    try:
        # Preprocess frame
        processed_frame = preprocess_frame(frame)
        
        # Make prediction
        prediction = model.predict(processed_frame, verbose=0)
        
        # Get violence probability (assuming binary classification)
        violence_prob = prediction[0][1] if len(prediction[0]) > 1 else prediction[0][0]
        
        # Determine if violence is detected (threshold: 0.5)
        is_violence = violence_prob > 0.5
        
        return is_violence, violence_prob
        
    except Exception as e:
        print(f"Error in violence detection: {e}")
        return False, 0.0

def enhance_frame(frame):
    """Enhance frame for better detection."""
    # Increase contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Increase sharpness
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def process_video_flask(video_file):
    """Process video file and return results."""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Open video file
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize variables
        violence_frames = 0
        total_processed_frames = 0
        confidence_values = []
        violence_timestamps = []
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            total_processed_frames += 1
            
            # Enhance frame
            enhanced_frame = enhance_frame(frame)
            
            # Detect violence
            is_violence, confidence = detect_violence(enhanced_frame)
            confidence_values.append(float(confidence))
            
            if is_violence:
                violence_frames += 1
                violence_timestamps.append(float(total_processed_frames / fps))
        
        # Calculate statistics
        violence_percentage = (violence_frames / total_processed_frames) * 100 if total_processed_frames > 0 else 0
        avg_confidence = np.mean(confidence_values) if confidence_values else 0
        
        # Cleanup
        cap.release()
        os.unlink(tmp_path)  # Delete temporary file
        
        return {
            'success': True,
            'total_frames': total_processed_frames,
            'violence_frames': violence_frames,
            'violence_percentage': float(violence_percentage),
            'avg_confidence': float(avg_confidence),
            'confidence_values': confidence_values,
            'violence_timestamps': violence_timestamps,
            'duration': float(total_processed_frames / fps),
            'fps': fps,
            'resolution': f"{frame_width}x{frame_height}",
            'filename': video_file.filename
        }
        
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'})
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Process the video
    results = process_video_flask(video_file)
    return jsonify(results)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Load model on startup
    if load_bensam02_model():
        print("✅ Model loaded successfully")
    else:
        print("❌ Failed to load model")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 