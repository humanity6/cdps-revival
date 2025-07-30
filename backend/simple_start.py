#!/usr/bin/env python3
"""
Simple server startup test - bypasses problematic imports
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Starting simplified backend server...")
    
    # Import only what we need
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    
    # Create simple app
    app = FastAPI(
        title="Crime Detection API",
        version="1.0.0",
        description="Unified backend for crime detection modules"
    )
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {
            "message": "Crime Detection Backend API",
            "status": "running",
            "version": "1.0.0",
            "note": "This is a simplified startup. Some services may not be available."
        }
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "message": "Basic server is running"
        }
    
    print("Basic FastAPI app created successfully!")
    print("Starting server on http://0.0.0.0:8000")
    print("Press Ctrl+C to stop")
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
except KeyboardInterrupt:
    print("\nServer stopped by user")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)