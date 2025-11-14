@echo off
echo ==================================================
echo   AI DJ Mixing Platform - Windows Startup
echo ==================================================

echo.
echo Checking Docker status...
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not accessible.
    echo.
    echo SOLUTIONS:
    echo 1. Make sure Docker Desktop is running
    echo 2. Run this script as Administrator
    echo 3. Add yourself to docker-users group:
    echo    - Run PowerShell as Admin
    echo    - Run: net localgroup docker-users "%USERNAME%" /add
    echo    - Restart your computer
    echo.
    pause
    exit /b 1
)

echo Docker is running! ✓

echo.
echo Setting up environment...
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env >nul
    echo Created .env file ✓
) else (
    echo .env file already exists ✓
)

echo.
echo Building and starting services...
echo This may take a few minutes on first run...
docker-compose up -d

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to start services
    echo Try running this script as Administrator
    pause
    exit /b 1
)

echo.
echo ==================================================
echo   🚀 AI DJ Mixing Platform Started Successfully!
echo ==================================================
echo.
echo Services available at:
echo   Frontend:     http://localhost:3000
echo   API Docs:     http://localhost:8000/docs
echo   Task Monitor: http://localhost:5555
echo   RabbitMQ UI:  http://localhost:15672 (guest/guest)
echo.
echo Checking service health...
timeout /t 10 /nobreak >nul

echo.
echo Service Status:
docker-compose ps

echo.
echo To view logs: docker-compose logs -f [service-name]
echo To stop:      docker-compose down
echo.
echo Ready to upload some music and test the AI! 🎵
pause