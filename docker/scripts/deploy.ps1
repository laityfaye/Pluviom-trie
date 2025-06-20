# startup.ps1 - Script de demarrage du projet Docker
param(
    [string]$Action = "start",
    [switch]$Build = $false,
    [switch]$Reset = $false
)

# Configuration de l'encodage
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "DOCKER - Gestion Projet Climat Senegal" -ForegroundColor Green
Write-Host "=" * 50

switch ($Action.ToLower()) {
    "start" {
        Write-Host ">> Demarrage des services..." -ForegroundColor Yellow
        if ($Build) {
            docker-compose up -d --build
        } else {
            docker-compose up -d
        }
        
        Write-Host ">> Verification de la sante..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        docker-compose ps
        
        Write-Host ">> Services disponibles:" -ForegroundColor Cyan
        Write-Host "  API: http://localhost:8000" -ForegroundColor White
        Write-Host "  Dashboard: http://localhost:3000" -ForegroundColor White
        Write-Host "  Grafana: http://localhost:3001" -ForegroundColor White
        Write-Host "  Prometheus: http://localhost:9090" -ForegroundColor White
    }
    
    "stop" {
        Write-Host ">> Arret des services..." -ForegroundColor Yellow
        docker-compose down
    }
    
    "logs" {
        Write-Host ">> Logs des services..." -ForegroundColor Yellow
        docker-compose logs -f
    }
    
    "ml" {
        Write-Host ">> Execution du pipeline ML..." -ForegroundColor Yellow
        docker-compose run --rm ml-pipeline python main.py --only-ml
    }
    
    "reset" {
        Write-Host ">> Reinitialisation complete..." -ForegroundColor Red
        docker-compose down -v
        docker system prune -f
        docker-compose up -d --build
    }
    
    default {
        Write-Host "Usage: ./startup.ps1 [start|stop|logs|ml|reset] [-Build] [-Reset]" -ForegroundColor Yellow
    }
}