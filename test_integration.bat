@echo off
echo Testing Real-Time Crime Detection System Integration
echo =====================================================
echo.

echo Testing Backend API endpoints...
echo.

echo 1. Testing Health Endpoint:
curl -s http://127.0.0.1:8000/health | findstr "healthy"
if %errorlevel% == 0 (
    echo [OK] Health endpoint working
) else (
    echo [ERROR] Health endpoint failed
)
echo.

echo 2. Testing System Info:
curl -s http://127.0.0.1:8000/info > nul
if %errorlevel% == 0 (
    echo [OK] Info endpoint working
) else (
    echo [ERROR] Info endpoint failed
)
echo.

echo 3. Testing Detection Health:
curl -s http://127.0.0.1:8000/api/detect/health > nul
if %errorlevel% == 0 (
    echo [OK] Detection health endpoint working
) else (
    echo [ERROR] Detection health endpoint failed
)
echo.

echo 4. Testing Live Feed Status:
curl -s http://127.0.0.1:8000/api/live/status > nul
if %errorlevel% == 0 (
    echo [OK] Live feed status endpoint working
) else (
    echo [ERROR] Live feed status endpoint failed
)
echo.

echo 5. Testing Settings:
curl -s http://127.0.0.1:8000/api/settings/ > nul
if %errorlevel% == 0 (
    echo [OK] Settings endpoint working
) else (
    echo [ERROR] Settings endpoint failed
)
echo.

echo Testing Frontend accessibility...
curl -s http://localhost:5175/ > nul
if %errorlevel% == 0 (
    echo [OK] Frontend accessible at http://localhost:5175/
) else (
    echo [ERROR] Frontend not accessible
)
echo.

echo =====================================================
echo Integration Test Summary:
echo - Backend API: http://127.0.0.1:8000
echo - Frontend App: http://localhost:5175/
echo - API Documentation: http://127.0.0.1:8000/docs
echo =====================================================
echo.
echo If all endpoints show [OK], the system is ready to use!
echo.
pause