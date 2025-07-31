@echo off
echo Starting Real-Time Crime Detection System
echo ==========================================
echo.

echo Starting Backend Server...
start cmd /k "cd backend && python run.py --host 127.0.0.1 --port 8000"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend Development Server...
start cmd /k "cd frontend && npm run dev"

echo.
echo System is starting up!
echo.
echo Backend API will be available at: http://127.0.0.1:8000
echo Frontend will be available at: http://localhost:5173 (or similar)
echo.
echo API Documentation: http://127.0.0.1:8000/docs
echo.
echo Press any key to exit...
pause > nul