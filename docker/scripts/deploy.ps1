# deploy.ps1 - Script de déploiement Docker
param(
    [string]$Environment = "dev",
    [switch]$Build = $false,
    [switch]$Migrate = $false,
    [switch]$Reset = $false
)

Write-Host "🚀 Déploiement Docker - Environnement: $Environment" -ForegroundColor Green
Write-Host "=" * 60

if ($Reset) {
    Write-Host "🔄 Réinitialisation complète..." -ForegroundColor Red
    docker-compose down -v
    docker system prune -f
    $Build = $true
}

if ($Build) {
    Write-Host "🔨 Construction des images Docker..." -ForegroundColor Yellow
    docker-compose build --no-cache
}

if ($Migrate) {
    Write-Host "📊 Migration base de données..." -ForegroundColor Yellow
    docker-compose run --rm ml-pipeline python -c "print('Migration simulée - à implémenter')"
}

Write-Host "▶️ Démarrage des services..." -ForegroundColor Yellow
docker-compose up -d

Write-Host "🏥 Vérification de la santé des services..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

$services = @("timescaledb", "redis", "api")
foreach ($service in $services) {
    $status = docker-compose ps -q $service
    if ($status) {
        $health = docker inspect $status --format='{{.State.Status}}'
        if ($health -eq "running") {
            Write-Host "✅ $service: Running" -ForegroundColor Green
        } else {
            Write-Host "❌ $service: $health" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ $service: Not found" -ForegroundColor Red
    }
}

Write-Host "🎉 Déploiement terminé!" -ForegroundColor Green
Write-Host "🌐 Services disponibles:" -ForegroundColor Cyan
Write-Host "  Dashboard: http://localhost:3000" -ForegroundColor White
Write-Host "  API: http://localhost:8000" -ForegroundColor White
Write-Host "  Grafana: http://localhost:3001" -ForegroundColor White
