#!/usr/bin/env python3
"""
Test script for the Face Detection System
This script helps test the system performance and functionality
"""

import time
import sys
import os
from face_detection_system import FaceDetectionSystem
from config import Config

def test_system_initialization():
    """Test system initialization"""
    print("Testing system initialization...")
    
    try:
        detection_system = FaceDetectionSystem()
        stats = detection_system.get_system_stats()
        
        print(f"✓ System initialized successfully")
        print(f"  - Total faces in database: {stats['database']['total_faces']}")
        print(f"  - Restricted faces: {stats['database']['restricted_faces']}")
        print(f"  - Criminal faces: {stats['database']['criminal_faces']}")
        
        return True
    except Exception as e:
        print(f"✗ System initialization failed: {str(e)}")
        return False

def test_performance_config():
    """Test performance configuration"""
    print("\nTesting performance configuration...")
    
    print(f"✓ Frame scale: {Config.FRAME_SCALE} (smaller = faster)")
    print(f"✓ Skip frames: {Config.SKIP_FRAMES} (higher = faster)")
    print(f"✓ Face model: {Config.FACE_MODEL} ('hog' is faster than 'cnn')")
    print(f"✓ Target FPS: {Config.TARGET_FPS}")
    print(f"✓ Queue size: {Config.QUEUE_SIZE}")
    
    if Config.FRAME_SCALE <= 0.5 and Config.FACE_MODEL == 'hog':
        print("✓ Configuration optimized for speed")
        return True
    else:
        print("⚠ Configuration may not achieve 30+ FPS")
        return True

def test_webcam_availability():
    """Test webcam availability"""
    print("\nTesting webcam availability...")
    
    import cv2
    
    cap = cv2.VideoCapture(Config.WEBCAM_INDEX)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Webcam {Config.WEBCAM_INDEX} is available")
            print(f"  - Frame size: {frame.shape}")
            cap.release()
            return True
        else:
            print(f"✗ Cannot read from webcam {Config.WEBCAM_INDEX}")
            cap.release()
            return False
    else:
        print(f"✗ Cannot open webcam {Config.WEBCAM_INDEX}")
        return False

def test_directories():
    """Test directory structure"""
    print("\nTesting directory structure...")
    
    directories = [
        Config.KNOWN_FACES_DIR,
        Config.RESTRICTED_FACES_DIR,
        Config.CRIMINAL_FACES_DIR,
        Config.TEST_VIDEOS_DIR
    ]
    
    all_exist = True
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} (missing)")
            all_exist = False
    
    return all_exist

def benchmark_face_detection():
    """Benchmark face detection performance"""
    print("\nBenchmarking face detection performance...")
    
    try:
        import cv2
        import face_recognition
        import numpy as np
        
        # Create a test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test face detection speed
        start_time = time.time()
        iterations = 10
        
        for i in range(iterations):
            # Resize frame
            small_frame = cv2.resize(test_frame, None, fx=Config.FRAME_SCALE, fy=Config.FRAME_SCALE)
            # Convert color
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame, model=Config.FACE_MODEL)
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        estimated_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"✓ Face detection benchmark completed")
        print(f"  - Average processing time: {avg_time:.3f}s per frame")
        print(f"  - Estimated FPS (detection only): {estimated_fps:.1f}")
        
        if estimated_fps >= 30:
            print("✓ Performance target achieved!")
        else:
            print("⚠ Performance target not met (consider adjusting config)")
        
        return estimated_fps >= 30
        
    except Exception as e:
        print(f"✗ Benchmark failed: {str(e)}")
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n" + "="*50)
    print("USAGE EXAMPLES")
    print("="*50)
    
    print("\n1. Run with webcam:")
    print("   python face_detection_system.py --source webcam")
    
    print("\n2. Run with video file:")
    print("   python face_detection_system.py --source path/to/video.mp4")
    
    print("\n3. Rebuild face database:")
    print("   python face_detection_system.py --rebuild-db")
    
    print("\n4. Test with specific camera:")
    print("   python face_detection_system.py --source webcam --camera 1")
    
    print("\nControls during detection:")
    print("   - Press 'q' to quit")
    print("   - FPS and detection info shown on screen")

def main():
    """Main test function"""
    print("Face Detection System - Test Suite")
    print("="*50)
    
    tests = [
        ("System Initialization", test_system_initialization),
        ("Performance Configuration", test_performance_config),
        ("Directory Structure", test_directories),
        ("Webcam Availability", test_webcam_availability),
        ("Face Detection Benchmark", benchmark_face_detection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {str(e)}")
    
    print("\n" + "="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*50)
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
    else:
        print("⚠ Some tests failed. Check the issues above.")
    
    show_usage_examples()

if __name__ == "__main__":
    main()