"""
Enhanced Setup Script for Face Detection System
Handles installation issues and system configuration
"""

import subprocess
import sys
import os
import platform
import urllib.request
import logging
from pathlib import Path

class SystemSetup:
    def __init__(self):
        self.system = platform.system()
        self.architecture = platform.architecture()[0]
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def check_system_requirements(self):
        """Check system requirements"""
        self.logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error("Python 3.8 or higher required")
            return False
        
        # Check OS
        if self.system not in ['Windows', 'Linux', 'Darwin']:
            self.logger.warning(f"Untested OS: {self.system}")
        
        self.logger.info(f"System: {self.system} {self.architecture}")
        self.logger.info(f"Python: {self.python_version}")
        
        return True
    
    def install_dlib_windows(self):
        """Install dlib on Windows using pre-built wheel"""
        self.logger.info("Installing dlib for Windows...")
        
        try:
            # Try to install from PyPI first
            subprocess.check_call([sys.executable, "-m", "pip", "install", "dlib"])
            self.logger.info("dlib installed successfully from PyPI")
            return True
            
        except subprocess.CalledProcessError:
            self.logger.warning("PyPI installation failed, trying alternative methods...")
            
            # Try installing from wheel
            wheel_urls = {
                "3.8": f"https://github.com/sachadee/Dlib/raw/master/dlib-19.22.0-cp38-cp38-win_amd64.whl",
                "3.9": f"https://github.com/sachadee/Dlib/raw/master/dlib-19.22.0-cp39-cp39-win_amd64.whl",
                "3.10": f"https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp310-cp310-win_amd64.whl",
                "3.11": f"https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl"
            }
            
            if self.python_version in wheel_urls:
                wheel_url = wheel_urls[self.python_version]
                try:
                    self.logger.info(f"Downloading dlib wheel for Python {self.python_version}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_url])
                    self.logger.info("dlib installed successfully from wheel")
                    return True
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Wheel installation failed: {e}")
            
            # Final fallback: conda installation
            try:
                self.logger.info("Trying conda installation...")
                subprocess.check_call(["conda", "install", "-c", "conda-forge", "dlib", "-y"])
                self.logger.info("dlib installed successfully via conda")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.error("All dlib installation methods failed")
                return False
    
    def install_dependencies(self):
        """Install all dependencies"""
        self.logger.info("Installing dependencies...")
        
        # Upgrade pip first
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        except subprocess.CalledProcessError:
            self.logger.warning("Failed to upgrade pip")
        
        # Install basic dependencies first
        basic_deps = [
            "numpy>=1.24.0,<2.0.0",
            "opencv-python>=4.8.0", 
            "Pillow>=10.0.0",
            "imutils==0.5.4",
            "psutil>=5.9.0"
        ]
        
        for dep in basic_deps:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                self.logger.info(f"Installed: {dep}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to install {dep}: {e}")
                return False
        
        # Install dlib (platform-specific)
        if self.system == "Windows":
            if not self.install_dlib_windows():
                return False
        else:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "dlib"])
            except subprocess.CalledProcessError:
                self.logger.error("dlib installation failed on non-Windows system")
                return False
        
        # Install face-recognition
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "face-recognition==1.3.0"])
            self.logger.info("face-recognition installed successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install face-recognition: {e}")
            return False
        
        # Install optional dependencies
        optional_deps = ["scikit-learn>=1.3.0", "matplotlib>=3.7.0"]
        for dep in optional_deps:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                self.logger.info(f"Installed optional: {dep}")
            except subprocess.CalledProcessError:
                self.logger.warning(f"Failed to install optional dependency: {dep}")
        
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        self.logger.info("Setting up directories...")
        
        directories = [
            "known_faces",
            "restricted_faces", 
            "criminal_faces",
            "logs",
            "temp"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
        
        return True
    
    def test_installation(self):
        """Test if everything is installed correctly"""
        self.logger.info("Testing installation...")
        
        try:
            import cv2
            self.logger.info("✅ OpenCV imported successfully")
        except ImportError as e:
            self.logger.error(f"❌ OpenCV import failed: {e}")
            return False
        
        try:
            import dlib
            self.logger.info("✅ dlib imported successfully")
        except ImportError as e:
            self.logger.error(f"❌ dlib import failed: {e}")
            return False
        
        try:
            import face_recognition
            self.logger.info("✅ face-recognition imported successfully")
        except ImportError as e:
            self.logger.error(f"❌ face-recognition import failed: {e}")
            return False
        
        try:
            import numpy as np
            self.logger.info("✅ numpy imported successfully")
        except ImportError as e:
            self.logger.error(f"❌ numpy import failed: {e}")
            return False
        
        # Test basic functionality
        try:
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            self.logger.info("✅ Basic face detection test passed")
        except Exception as e:
            self.logger.error(f"❌ Basic functionality test failed: {e}")
            return False
        
        return True
    
    def run_setup(self):
        """Run complete setup process"""
        self.logger.info("Starting Face Detection System Setup...")
        
        if not self.check_system_requirements():
            return False
        
        if not self.setup_directories():
            return False
        
        if not self.install_dependencies():
            return False
        
        if not self.test_installation():
            return False
        
        self.logger.info("✅ Setup completed successfully!")
        self.logger.info("Next steps:")
        self.logger.info("1. Add face images to restricted_faces/ and criminal_faces/ directories")
        self.logger.info("2. Run: python test_system.py")
        self.logger.info("3. Run: python main.py --mode webcam")
        
        return True

if __name__ == "__main__":
    setup = SystemSetup()
    success = setup.run_setup()
    
    if not success:
        print("\n❌ Setup failed. Please check the logs above for errors.")
        print("\nCommon solutions:")
        print("1. Install Visual Studio Build Tools for Windows")
        print("2. Try running: conda install -c conda-forge dlib")
        print("3. Download pre-built dlib wheel from: https://github.com/z-mahmud22/Dlib_Windows_Python3.x")
        sys.exit(1)
    else:
        print("\n🎉 Setup completed successfully!")
