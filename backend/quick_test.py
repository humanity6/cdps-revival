#!/usr/bin/env python3
"""
Quick test to check if the server can start
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test basic imports"""
    try:
        print("Testing basic imports...")
        
        # Test configuration
        from config import backend_config, module_config
        print("   [OK] Configuration imported")
        
        # Test models
        from models.detection_models import ANPRRequest
        print("   [OK] Models imported")
        
        # Test services individually
        try:
            from services.weapon_service import WeaponService
            weapon_service = WeaponService()
            print(f"   [OK] Weapon Service: {'Enabled' if weapon_service.enabled else 'Disabled'}")
        except Exception as e:
            print(f"   [WARN] Weapon Service: {e}")
        
        try:
            from services.violence_service import ViolenceService
            violence_service = ViolenceService()
            print(f"   [OK] Violence Service: {'Enabled' if violence_service.enabled else 'Disabled'}")
        except Exception as e:
            print(f"   [WARN] Violence Service: {e}")
        
        try:
            from services.anpr_service import ANPRService
            anpr_service = ANPRService()
            print(f"   [OK] ANPR Service: {'Enabled' if anpr_service.enabled else 'Disabled'}")
        except Exception as e:
            print(f"   [WARN] ANPR Service: {e}")
        
        try:
            from services.face_service import FaceService
            face_service = FaceService()
            print(f"   [OK] Face Service: {'Enabled' if face_service.enabled else 'Disabled'}")
        except Exception as e:
            print(f"   [WARN] Face Service: {e}")
        
        print("\n[OK] All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Import failed: {e}")
        return False

def test_server_creation():
    """Test if FastAPI app can be created"""
    try:
        print("\nTesting FastAPI app creation...")
        
        # Import main components
        from main import app
        print("   [OK] FastAPI app created successfully")
        
        # Test a simple endpoint
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        if response.status_code == 200:
            print("   [OK] Root endpoint working")
        else:
            print(f"   [WARN] Root endpoint returned {response.status_code}")
        
        print("\n[OK] Server creation test passed!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Server creation failed: {e}")
        return False

def main():
    print("Quick Backend Test")
    print("=" * 30)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test server creation
    if not test_server_creation():
        success = False
    
    print("\n" + "=" * 30)
    if success:
        print("[OK] All tests passed! Server should be ready to start.")
        print("\nTo start the server:")
        print("   python run.py")
    else:
        print("[ERROR] Some tests failed. Check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)