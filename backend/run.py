#!/usr/bin/env python3
"""
Startup script for the Crime Detection Backend API
"""
import sys
import os
import argparse
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'opencv-python',
        'numpy',
        'pillow',
        'pydantic'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_deps.append(package)
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Install missing dependencies with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_models():
    """Check if required model files exist"""
    model_paths = [
        ("ANPR FastANPR", "Available through pip package"),
        ("Violence Detection", "../violence detection cdps/violence/bensam02_model.h5"),
        ("Weapon Detection", "../weapon/models/best.pt"),
        ("Face Recognition", "Available through face-recognition package")
    ]
    
    print("🔍 Checking model availability...")
    
    for model_name, path in model_paths:
        if path.startswith("Available"):
            print(f"   ✅ {model_name}: {path}")
        else:
            full_path = os.path.join(os.path.dirname(__file__), path)
            if os.path.exists(full_path):
                print(f"   ✅ {model_name}: Found at {path}")
            else:
                print(f"   ⚠️  {model_name}: Not found at {path}")

def main():
    parser = argparse.ArgumentParser(description="Crime Detection Backend API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Log level")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check dependencies and models, don't start server")
    
    args = parser.parse_args()
    
    print("🚀 Crime Detection Backend API")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check models
    check_models()
    
    if args.check_only:
        print("\n✅ Dependency and model check completed")
        return
    
    print(f"\n🌐 Starting server on {args.host}:{args.port}")
    print(f"📊 Log level: {args.log_level}")
    print(f"🔄 Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    print(f"🐛 Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print("\n📖 API Documentation:")
    print(f"   - Swagger UI: http://{args.host}:{args.port}/docs")
    print(f"   - ReDoc: http://{args.host}:{args.port}/redoc")
    print("\n🔌 WebSocket Endpoint:")
    print(f"   - Live Feed: ws://{args.host}:{args.port}/api/live/ws/{{client_id}}")
    print("\n" + "=" * 50)
    
    # Set environment variables if provided
    if args.host != "0.0.0.0":
        os.environ["API_HOST"] = args.host
    if args.port != 8000:
        os.environ["API_PORT"] = str(args.port)
    if args.debug:
        os.environ["API_DEBUG"] = "true"
    if args.reload:
        os.environ["API_RELOAD"] = "true"
    os.environ["LOG_LEVEL"] = args.log_level
    
    try:
        # Import and run the application
        import uvicorn
        from main import app
        
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()