#!/usr/bin/env python3
"""
Integration test script for the Crime Detection Backend API
"""
import sys
import os
import asyncio
import base64
import logging
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_service_initialization():
    """Test if all detection services can be initialized"""
    print("Testing service initialization...")
    
    # Test ANPR Service
    try:
        from services.anpr_service import ANPRService
        anpr_service = ANPRService()
        health = anpr_service.get_health_status()
        print(f"   [OK] ANPR Service: {'Enabled' if health.get('enabled') else 'Disabled'}")
        if not health.get('enabled'):
            print(f"      [WARN] Reason: {health.get('last_error', 'Unknown')}")
    except Exception as e:
        print(f"   [ERROR] ANPR Service: Failed to initialize - {e}")
    
    # Test Face Service
    try:
        from services.face_service import FaceService
        face_service = FaceService()
        health = face_service.get_health_status()
        print(f"   [OK] Face Service: {'Enabled' if health.get('enabled') else 'Disabled'}")
        if not health.get('enabled'):
            print(f"      [WARN] Reason: {health.get('last_error', 'Unknown')}")
    except Exception as e:
        print(f"   [ERROR] Face Service: Failed to initialize - {e}")
    
    # Test Violence Service
    try:
        from services.violence_service import ViolenceService
        violence_service = ViolenceService()
        health = violence_service.get_health_status()
        print(f"   [OK] Violence Service: {'Enabled' if health.get('enabled') else 'Disabled'}")
        if not health.get('enabled'):
            print(f"      [WARN] Reason: {health.get('last_error', 'Unknown')}")
    except Exception as e:
        print(f"   [ERROR] Violence Service: Failed to initialize - {e}")
    
    # Test Weapon Service
    try:
        from services.weapon_service import WeaponService
        weapon_service = WeaponService()
        health = weapon_service.get_health_status()
        print(f"   [OK] Weapon Service: {'Enabled' if health.get('enabled') else 'Disabled'}")
        if not health.get('enabled'):
            print(f"      [WARN] Reason: {health.get('last_error', 'Unknown')}")
    except Exception as e:
        print(f"   [ERROR] Weapon Service: Failed to initialize - {e}")

def test_configuration():
    """Test configuration system"""
    print("\nTesting configuration system...")
    
    try:
        from config import backend_config, module_config, get_module_enabled_status
        
        print(f"   [OK] Backend Config: Host={backend_config.host}, Port={backend_config.port}")
        print(f"   [OK] Module Status: {get_module_enabled_status()}")
        
        # Test module configuration
        anpr_config = module_config.get_module_config('anpr')
        print(f"   [OK] ANPR Config: {anpr_config}")
        
    except Exception as e:
        print(f"   [ERROR] Configuration Test Failed: {e}")

def test_models():
    """Test Pydantic models"""
    print("\nTesting Pydantic models...")
    
    try:
        from models.detection_models import ANPRRequest, FaceRequest, WeaponRequest, ViolenceRequest
        
        # Test ANPR model
        anpr_req = ANPRRequest(image_data="test_data", min_confidence=0.6)
        print(f"   [OK] ANPR Model: confidence={anpr_req.min_confidence}")
        
        # Test Face model
        face_req = FaceRequest(image_data="test_data", recognition_tolerance=0.5)
        print(f"   [OK] Face Model: tolerance={face_req.recognition_tolerance}")
        
        # Test Weapon model
        weapon_req = WeaponRequest(image_data="test_data", confidence_threshold=0.7)
        print(f"   [OK] Weapon Model: confidence={weapon_req.confidence_threshold}")
        
        # Test Violence model
        violence_req = ViolenceRequest(image_data="test_data", confidence_threshold=0.5)
        print(f"   [OK] Violence Model: confidence={violence_req.confidence_threshold}")
        
    except Exception as e:
        print(f"   [ERROR] Models Test Failed: {e}")

def test_utilities():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        from utils.image_utils import base64_to_image, image_to_base64
        import numpy as np
        
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image.fill(128)  # Gray image
        
        # Test image to base64 conversion
        b64_data = image_to_base64(test_image)
        if b64_data:
            print("   [OK] Image to Base64: Working")
            
            # Test base64 to image conversion
            decoded_image = base64_to_image(b64_data.split(',')[1])  # Remove data URL prefix
            if decoded_image is not None:
                print("   [OK] Base64 to Image: Working")
            else:
                print("   [ERROR] Base64 to Image: Failed")
        else:
            print("   [ERROR] Image to Base64: Failed")
        
    except Exception as e:
        print(f"   [ERROR] Utilities Test Failed: {e}")

def check_dependencies():
    """Check critical dependencies"""
    print("\nChecking critical dependencies...")
    
    critical_deps = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('pydantic', 'pydantic'),
        ('pillow', 'PIL'),
    ]
    
    optional_deps = [
        ('ultralytics', 'ultralytics'),
        ('tensorflow', 'tensorflow'),
        ('torch', 'torch'),
        ('face-recognition', 'face_recognition'),
    ]
    
    print("   Critical dependencies:")
    for pkg_name, import_name in critical_deps:
        try:
            __import__(import_name.split('.')[0])
            print(f"     [OK] {pkg_name}")
        except ImportError:
            print(f"     [ERROR] {pkg_name} - MISSING")
    
    print("   Optional dependencies:")
    for pkg_name, import_name in optional_deps:
        try:
            __import__(import_name.split('.')[0])
            print(f"     [OK] {pkg_name}")
        except ImportError:
            print(f"     [WARN] {pkg_name} - Not installed (some features may not work)")

def check_model_files():
    """Check if model files exist"""
    print("\nChecking model files...")
    
    model_files = [
        ("Violence Detection", "../violence detection cdps/violence/bensam02_model.h5"),
        ("Weapon Detection", "../weapon/models/best.pt"),
    ]
    
    for model_name, rel_path in model_files:
        full_path = os.path.join(os.path.dirname(__file__), rel_path)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"   [OK] {model_name}: Found ({size_mb:.1f} MB)")
        else:
            print(f"   [WARN] {model_name}: Not found at {rel_path}")

async def main():
    """Main test function"""
    print("Crime Detection Backend Integration Test")
    print("=" * 50)
    
    # Run all tests
    check_dependencies()
    check_model_files()
    test_configuration()
    test_models()
    test_utilities()
    await test_service_initialization()
    
    print("\n" + "=" * 50)
    print("[OK] Integration test completed!")
    print("\nNext steps:")
    print("   1. Start the server: python run.py")
    print("   2. Check API docs: http://localhost:8000/docs")
    print("   3. Test endpoints with sample data")

if __name__ == "__main__":
    asyncio.run(main())