# startup.ps1 - Script de démarrage du projet Docker
param(
    [string]$Action = "start",
    [switch]$Build = $false,
    [switch]$Reset = $false
)

Write-Host "🐳 Gestion Docker - Projet Climat Sénégal" -ForegroundColor Green
Write-Host "=" * 50

switch ($Action.ToLower()) {
    "start" {
        Write-Host "▶️ Démarrage des services..." -ForegroundColor Yellow
        if ($Build) {
            docker-compose up -d --build
        } else {
            docker-compose up -d
        }
        
        Write-Host "🏥 Vérification de la santé..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        docker-compose ps
        
        Write-Host "🌐 Services disponibles:" -ForegroundColor Cyan
        Write-Host "  API: http://localhost:8000" -ForegroundColor White
        Write-Host "  Dashboard: http://localhost:3000" -ForegroundColor White
        Write-Host "  Grafana: http://localhost:3001" -ForegroundColor White
        Write-Host "  Prometheus: http://localhost:9090" -ForegroundColor White
    }
    
    "stop" {
        Write-Host "⏹️ Arrêt des services..." -ForegroundColor Yellow
        docker-compose down
    }
    
    "logs" {
        Write-Host "📄 Logs des services..." -ForegroundColor Yellow
        docker-compose logs -f
    }
    
    "ml" {
        Write-Host "🤖 Exécution du pipeline ML..." -ForegroundColor Yellow
        docker-compose run --rm ml-pipeline python main.py --only-ml
    }
    
    "reset" {
        Write-Host "🔄 Réinitialisation complète..." -ForegroundColor Red
        docker-compose down -v
        docker system prune -f
        docker-compose up -d --build
    }
    
    default {
        Write-Host "Usage: ./startup.ps1 [start|stop|logs|ml|reset] [-Build] [-Reset]" -ForegroundColor Yellow
    }
}
