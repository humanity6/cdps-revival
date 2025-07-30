#!/usr/bin/env python3
"""
Install missing dependencies for the backend
"""
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def main():
    print("Installing missing dependencies...")
    
    # Install pydantic-settings
    print("\n1. Installing pydantic-settings...")
    if install_package("pydantic-settings>=2.0.0"):
        print("   [OK] pydantic-settings installed")
    else:
        print("   [ERROR] Failed to install pydantic-settings")
    
    # Try to install fastanpr with version
    print("\n2. Installing fastanpr...")
    if install_package("fastanpr>=1.2.0"):
        print("   [OK] fastanpr installed")
    else:
        print("   [WARN] fastanpr installation failed, trying without version...")
        if install_package("fastanpr"):
            print("   [OK] fastanpr installed (without version)")
        else:
            print("   [ERROR] Failed to install fastanpr")
    
    print("\n3. Upgrading TensorFlow (may help with model compatibility)...")
    if install_package("tensorflow>=2.13.0,<2.16.0"):
        print("   [OK] TensorFlow upgraded")
    else:
        print("   [WARN] TensorFlow upgrade failed")
    
    print("\nInstallation completed!")
    print("Run 'python test_integration.py' to test again")

if __name__ == "__main__":
    main()