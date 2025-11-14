# AI DJ Mixing Platform - Windows PowerShell Startup Script

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   AI DJ Mixing Platform - Windows Startup" -ForegroundColor Cyan  
Write-Host "==================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Checking Docker status..." -ForegroundColor Yellow

try {
    $dockerVersion = docker version --format "{{.Server.Version}}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not accessible"
    }
    Write-Host "Docker is running! ✓" -ForegroundColor Green
    Write-Host "Docker Version: $dockerVersion" -ForegroundColor Gray
} catch {
    Write-Host "ERROR: Docker is not running or not accessible." -ForegroundColor Red
    Write-Host ""
    Write-Host "SOLUTIONS:" -ForegroundColor Yellow
    Write-Host "1. Make sure Docker Desktop is running" -ForegroundColor White
    Write-Host "2. Run PowerShell as Administrator" -ForegroundColor White  
    Write-Host "3. Add yourself to docker-users group:" -ForegroundColor White
    Write-Host "   - Run: net localgroup docker-users `"$env:USERNAME`" /add" -ForegroundColor Gray
    Write-Host "   - Restart your computer" -ForegroundColor Gray
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Setting up environment..." -ForegroundColor Yellow

if (!(Test-Path ".env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Gray
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env file ✓" -ForegroundColor Green
} else {
    Write-Host ".env file already exists ✓" -ForegroundColor Green
}

Write-Host ""
Write-Host "Building and starting services..." -ForegroundColor Yellow
Write-Host "This may take a few minutes on first run..." -ForegroundColor Gray

docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to start services" -ForegroundColor Red
    Write-Host "Try running PowerShell as Administrator" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "   🚀 AI DJ Mixing Platform Started Successfully!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

Write-Host ""
Write-Host "Services available at:" -ForegroundColor Cyan
Write-Host "  Frontend:     " -NoNewline -ForegroundColor White
Write-Host "http://localhost:3000" -ForegroundColor Blue
Write-Host "  API Docs:     " -NoNewline -ForegroundColor White  
Write-Host "http://localhost:8000/docs" -ForegroundColor Blue
Write-Host "  Task Monitor: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:5555" -ForegroundColor Blue
Write-Host "  RabbitMQ UI:  " -NoNewline -ForegroundColor White
Write-Host "http://localhost:15672" -ForegroundColor Blue -NoNewline
Write-Host " (guest/guest)" -ForegroundColor Gray

Write-Host ""
Write-Host "Checking service health..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "Service Status:" -ForegroundColor Cyan
docker-compose ps

Write-Host ""
Write-Host "Useful Commands:" -ForegroundColor Cyan
Write-Host "  View logs:    docker-compose logs -f [service-name]" -ForegroundColor Gray
Write-Host "  Stop all:     docker-compose down" -ForegroundColor Gray
Write-Host "  Restart:      docker-compose restart [service-name]" -ForegroundColor Gray

Write-Host ""
Write-Host "Ready to upload some music and test the AI! 🎵" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to continue"