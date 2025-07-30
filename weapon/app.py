#!/usr/bin/env python3
"""
Flask Web UI for Weapon Detection System
Allows uploading images and videos for weapon detection testing.
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import io
import base64
import tempfile
import json
from datetime import datetime
import yaml

# Load configuration
def load_config():
    """Load configuration from config.yaml."""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {
            'model': {'path': 'models/best.pt'},
            'detection': {'confidence_threshold': 0.5}
        }

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load configuration and model
config = load_config()
model_path = config.get('model', {}).get('path', 'models/best.pt')
confidence_threshold = config.get('detection', {}).get('confidence_threshold', 0.5)

# Initialize YOLO model
try:
    model = YOLO(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def process_image(image_path):
    """Process image and return detection results."""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Run inference
        results = model(image_path)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    if confidence >= confidence_threshold:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'class': class_name,
                            'confidence': confidence
                        })
        
        return {
            'success': True,
            'detections': detections,
            'total_detections': len(detections)
        }
        
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

def process_video(video_path):
    """Process video and return detection results."""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Process video
        frame_detections = []
        frame_count = 0
        detection_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 10th frame for efficiency
            if frame_count % 10 == 0:
                results = model(frame)
                
                frame_detection = {
                    'frame': frame_count,
                    'time': frame_count / fps,
                    'detections': []
                }
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            
                            if confidence >= confidence_threshold:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                frame_detection['detections'].append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'class': class_name,
                                    'confidence': confidence
                                })
                
                if frame_detection['detections']:
                    frame_detections.append(frame_detection)
                    detection_frames += 1
        
        cap.release()
        
        return {
            'success': True,
            'total_frames': frame_count,
            'detection_frames': detection_frames,
            'duration': duration,
            'fps': fps,
            'frame_detections': frame_detections
        }
        
    except Exception as e:
        return {"error": f"Error processing video: {str(e)}"}

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and detection."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not allowed_file(file.filename, {'png', 'jpg', 'jpeg', 'gif', 'bmp'}):
        return jsonify({'error': 'Invalid file type. Please upload an image.'})
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        results = process_image(filepath)
        
        if 'error' in results:
            return jsonify(results)
        
        # Create annotated image
        if results['detections']:
            img = cv2.imread(filepath)
            for detection in results['detections']:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                label = f"{detection['class']}: {detection['confidence']:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Save annotated image
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            cv2.imwrite(result_path, img)
            
            # Convert to base64 for display
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            results['annotated_image'] = img_base64
            results['result_path'] = result_path
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and detection."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not allowed_file(file.filename, {'mp4', 'avi', 'mov', 'mkv', 'wmv'}):
        return jsonify({'error': 'Invalid file type. Please upload a video.'})
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process video
        results = process_video(filepath)
        
        if 'error' in results:
            return jsonify(results)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})

@app.route('/download_result/<filename>')
def download_result(filename):
    """Download result file."""
    try:
        return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename),
                        as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'})

if __name__ == '__main__':
    print("🚀 Starting Flask Weapon Detection Web UI")
    print("=" * 50)
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Results folder: {app.config['RESULTS_FOLDER']}")
    print(f"🤖 Model path: {model_path}")
    print(f"🎯 Confidence threshold: {confidence_threshold}")
    print("=" * 50)
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 