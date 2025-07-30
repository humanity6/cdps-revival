"""
Test script for the Face Detection System
Run this to verify system functionality before main deployment
"""

import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    import config
    from face_detection_system import FaceDetectionSystem
    from performance_monitor import PerformanceMonitor
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)

def test_camera_access():
    """Test camera accessibility"""
    print("\n📹 Testing camera access...")
    
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera {config.CAMERA_INDEX} accessible - Frame size: {frame.shape}")
            cap.release()
            return True
        else:
            print(f"❌ Cannot read from camera {config.CAMERA_INDEX}")
    else:
        print(f"❌ Cannot open camera {config.CAMERA_INDEX}")
        print("💡 Try changing CAMERA_INDEX in config.py (usually 0, 1, or 2)")
    
    cap.release()
    return False

def test_enhanced_face_detection():
    """Test enhanced face detection functionality"""
    print("\n🔍 Testing enhanced face detection...")
    
    try:
        import face_recognition
        
        # Create a more realistic test image
        test_image = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Add some texture to make it more realistic
        noise = np.random.randint(0, 50, test_image.shape, dtype=np.uint8)
        test_image = cv2.add(test_image, noise)
        
        # Test face_recognition library with enhanced settings
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        # Test multiple detection methods
        face_locations_hog = face_recognition.face_locations(rgb_image, model='hog')
        print(f"   - HOG detection: {len(face_locations_hog)} faces found")
        
        try:
            face_locations_cnn = face_recognition.face_locations(rgb_image, model='cnn')
            print(f"   - CNN detection: {len(face_locations_cnn)} faces found")
        except Exception as e:
            print(f"   - CNN detection unavailable: {e}")
        
        print("✅ Enhanced face detection library working")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced face detection test failed: {e}")
        return False

def test_config_validation():
    """Test configuration settings"""
    print("\n⚙️ Testing configuration validation...")
    
    try:
        # Test required configuration parameters
        required_params = [
            'DETECTION_SCALE_FACTOR', 'MAX_FACE_DISTANCE', 'RECOGNITION_TOLERANCE',
            'COLORS', 'PERSON_CATEGORIES', 'CAMERA_INDEX'
        ]
        
        missing_params = []
        for param in required_params:
            if not hasattr(config, param):
                missing_params.append(param)
        
        if missing_params:
            print(f"❌ Missing configuration parameters: {missing_params}")
            return False
        
        # Validate parameter ranges
        if not (0.1 <= config.DETECTION_SCALE_FACTOR <= 1.0):
            print(f"❌ Invalid DETECTION_SCALE_FACTOR: {config.DETECTION_SCALE_FACTOR}")
            return False
        
        if not (0.1 <= config.MAX_FACE_DISTANCE <= 1.0):
            print(f"❌ Invalid MAX_FACE_DISTANCE: {config.MAX_FACE_DISTANCE}")
            return False
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def test_performance_features():
    """Test performance monitoring features"""
    print("\n📊 Testing performance features...")
    
    try:
        # Test if performance monitoring is available
        if hasattr(config, 'ENABLE_PERFORMANCE_MONITORING') and config.ENABLE_PERFORMANCE_MONITORING:
            monitor = PerformanceMonitor()
            
            # Simulate some operations
            monitor.start_frame_timer()
            time.sleep(0.01)  # Simulate frame processing
            monitor.end_frame_timer()
            
            monitor.start_detection_timer()
            time.sleep(0.005)  # Simulate detection
            monitor.end_detection_timer()
            
            monitor.update_system_metrics()
            
            stats = monitor.get_system_stats()
            print(f"   - FPS monitoring: {stats['current_fps']:.1f}")
            print(f"   - CPU usage: {stats['cpu_usage_percent']:.1f}%")
            print(f"   - Memory usage: {stats['memory_usage_percent']:.1f}%")
            
        print("✅ Performance features working")
        return True
        
    except Exception as e:
        print(f"❌ Performance features test failed: {e}")
        return False

def test_enhanced_system_initialization():
    """Test enhanced system initialization"""
    print("\n🚀 Testing enhanced system initialization...")
    
    try:
        detector = FaceDetectionSystem()
        
        # Test enhanced features
        print(f"   - Known faces loaded: {len(detector.known_face_encodings)}")
        
        # Test if enhanced methods are available
        if hasattr(detector, '_enhance_frame_quality'):
            test_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
            enhanced_frame = detector._enhance_frame_quality(test_frame)
            print("   - Frame enhancement: Available")
        
        if hasattr(detector, '_detect_faces_multi_method'):
            print("   - Multi-method detection: Available")
        
        if hasattr(detector, '_identify_face'):
            print("   - Enhanced identification: Available")
        
        print("✅ Enhanced system initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced system initialization failed: {e}")
        return False

def run_enhanced_detection_test():
    """Run enhanced detection test with real images"""
    print("\n⚡ Running enhanced detection test...")
    
    try:
        detector = FaceDetectionSystem()
        
        # Create a more realistic test frame with better contrast
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
        
        # Add some facial features simulation
        center_x, center_y = 320, 240
        cv2.circle(test_frame, (center_x, center_y), 80, (180, 160, 140), -1)  # Face
        cv2.circle(test_frame, (center_x-20, center_y-20), 8, (50, 50, 50), -1)  # Left eye
        cv2.circle(test_frame, (center_x+20, center_y-20), 8, (50, 50, 50), -1)  # Right eye
        cv2.ellipse(test_frame, (center_x, center_y+10), (15, 8), 0, 0, 180, (50, 50, 50), 2)  # Mouth
        
        # Run enhanced detection
        face_locations, face_names, face_categories = detector.detect_faces_in_frame(test_frame)
        
        print(f"   - Faces detected: {len(face_locations)}")
        print(f"   - Known faces in database: {len(detector.known_face_encodings)}")
        
        # Test enhanced drawing
        if face_locations:
            result_frame = detector.draw_detection_results(test_frame, face_locations, face_names, face_categories)
            print("   - Enhanced drawing: Successful")
        
        print("✅ Enhanced detection test completed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced detection test failed: {e}")
        return False

def test_directories():
    """Test directory structure"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        config.RESTRICTED_FACES_DIR,
        config.CRIMINAL_FACES_DIR,
        config.KNOWN_FACES_DIR,
        config.LOGS_DIR,
        config.TEMP_DIR
    ]
    
    all_good = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory} exists")
        else:
            print(f"❌ {directory} missing")
            all_good = False
    
    return all_good

def test_system_initialization():
    """Test system initialization"""
    print("\n🚀 Testing system initialization...")
    
    try:
        detector = FaceDetectionSystem()
        print("✅ Face Detection System initialized successfully")
        print(f"   - Known faces loaded: {len(detector.known_face_encodings)}")
        return True
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

def test_performance_monitor():
    """Test performance monitoring"""
    print("\n📊 Testing performance monitor...")
    
    try:
        monitor = PerformanceMonitor()
        monitor.start_frame_timer()
        monitor.end_frame_timer()
        monitor.update_system_metrics()
        
        stats = monitor.get_system_stats()
        print(f"✅ Performance monitor working - FPS: {stats['current_fps']:.1f}")
        return True
    except Exception as e:
        print(f"❌ Performance monitor test failed: {e}")
        return False

def create_sample_faces():
    """Create sample face images for testing"""
    print("\n🎭 Creating sample test faces...")
    
    try:
        # Create a simple test face image
        sample_image = np.ones((200, 200, 3), dtype=np.uint8) * 128  # Gray background
        
        # Draw a simple face-like pattern
        cv2.circle(sample_image, (100, 100), 80, (200, 180, 150), -1)  # Face circle
        cv2.circle(sample_image, (80, 80), 10, (50, 50, 50), -1)       # Left eye
        cv2.circle(sample_image, (120, 80), 10, (50, 50, 50), -1)      # Right eye
        cv2.ellipse(sample_image, (100, 110), (15, 10), 0, 0, 180, (50, 50, 50), 2)  # Mouth
        
        # Save test images
        test_restricted_path = os.path.join(config.RESTRICTED_FACES_DIR, "test_restricted_person.jpg")
        test_criminal_path = os.path.join(config.CRIMINAL_FACES_DIR, "test_criminal_person.jpg")
        
        cv2.imwrite(test_restricted_path, sample_image)
        cv2.imwrite(test_criminal_path, sample_image)
        
        print(f"✅ Sample faces created:")
        print(f"   - {test_restricted_path}")
        print(f"   - {test_criminal_path}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create sample faces: {e}")
        return False

def run_quick_detection_test():
    """Run a quick detection test"""
    print("\n⚡ Running quick detection test...")
    
    try:
        detector = FaceDetectionSystem()
        
        # Create a test frame
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        
        # Run detection
        face_locations, face_names, face_categories = detector.detect_faces_in_frame(test_frame)
        
        print(f"✅ Detection test completed")
        print(f"   - Faces detected: {len(face_locations)}")
        print(f"   - Known faces in database: {len(detector.known_face_encodings)}")
        
        return True
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 Face Detection System - Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directories),
        ("Camera Access", test_camera_access),
        ("Enhanced Face Detection", test_enhanced_face_detection),
        ("Configuration Validation", test_config_validation),
        ("Performance Features", test_performance_features),
        ("Enhanced System Initialization", test_enhanced_system_initialization),
        ("Sample Face Creation", create_sample_faces),
        ("Enhanced Detection Test", run_enhanced_detection_test),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Add face images to restricted_faces/ and criminal_faces/ directories")
        print("2. Run: python main.py --mode webcam")
        print("3. Or run: python main.py --mode video --video your_video.mp4")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before proceeding.")
        print("\nTroubleshooting:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Check camera permissions and availability")
        print("- Verify directory permissions")

if __name__ == "__main__":
    main()
