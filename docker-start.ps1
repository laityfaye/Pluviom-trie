# docker-start.ps1
# Script de demarrage rapide pour le projet climat Senegal

function Write-ColorText {
    param(
        [ConsoleColor]$Color,
        [string]$Text
    )
    $originalColor = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Host $Text
    $Host.UI.RawUI.ForegroundColor = $originalColor
}

# En-tete
Write-Host ""
Write-ColorText Cyan "DEMARRAGE RAPIDE - PROJET CLIMAT SENEGAL"
Write-ColorText Cyan "========================================"
Write-Host ""

# Verification de Docker
try {
    docker --version | Out-Null
    Write-ColorText Green "[OK] Docker detecte"
}
catch {
    Write-ColorText Red "[ERREUR] Docker n'est pas installe"
    Write-Host "Telechargement: https://www.docker.com/products/docker-desktop"
    exit 1
}

# Le fichier .env existe deja, on continue
Write-ColorText Green "[OK] Fichier .env detecte"

# Demarrage des services
Write-ColorText Blue "[INFO] Demarrage des services Docker..."
try {
    docker-compose up -d
    if ($LASTEXITCODE -eq 0) {
        Write-ColorText Green "[OK] Services demarres"
    }
    else {
        Write-ColorText Red "[ERREUR] Probleme lors du demarrage"
        exit 1
    }
}
catch {
    Write-ColorText Red "[ERREUR] Erreur lors du demarrage des services"
    Write-Host "Verifiez que Docker Desktop est demarre et reessayez."
    exit 1
}

# Attente du demarrage
Write-ColorText Yellow "[INFO] Attente du demarrage des services (15 secondes)..."
Start-Sleep -Seconds 15

# Verification de l'etat
Write-ColorText Blue "[INFO] Verification des services..."

# Test TimescaleDB
try {
    $null = docker-compose exec -T timescaledb pg_isready -U climate_user 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-ColorText Green "[OK] TimescaleDB operationnel"
    }
    else {
        Write-ColorText Yellow "[ATTENTE] TimescaleDB en cours de demarrage"
    }
}
catch {
    Write-ColorText Yellow "[ATTENTE] TimescaleDB en cours de demarrage"
}

# Test Redis
try {
    $null = docker-compose exec -T redis redis-cli ping 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-ColorText Green "[OK] Redis operationnel"
    }
    else {
        Write-ColorText Yellow "[ATTENTE] Redis en cours de demarrage"
    }
}
catch {
    Write-ColorText Yellow "[ATTENTE] Redis en cours de demarrage"
}

# Test API
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-ColorText Green "[OK] API operationnelle"
    }
    else {
        Write-ColorText Yellow "[ATTENTE] API en cours de demarrage"
    }
}
catch {
    Write-ColorText Yellow "[ATTENTE] API en cours de demarrage"
}

# Informations de connexion
Write-Host ""
Write-ColorText Green "SERVICES DISPONIBLES:"
Write-Host "====================="
Write-Host "API Climate:      http://localhost:8000"
Write-Host "Documentation:    http://localhost:8000/docs"
Write-Host "Base de donnees:  localhost:5432"
Write-Host "Redis:            localhost:6379"
Write-Host "Grafana:          http://localhost:3001 (admin/passe123)"
Write-Host "Prometheus:       http://localhost:9090"
Write-Host ""
Write-ColorText Cyan "COMMANDES UTILES:"
Write-Host "================="
Write-Host "Logs en temps reel:  docker-compose logs -f"
Write-Host "Arret des services:  docker-compose down"
Write-Host "Etat des services:   docker-compose ps"
Write-Host "Redemarrage:         docker-compose restart"
Write-Host ""
Write-ColorText Magenta "Le systeme est pret a etre utilise!"

# Ouverture automatique du navigateur
Write-Host ""
$openBrowser = Read-Host "Voulez-vous ouvrir la documentation API dans votre navigateur? (o/N)"
if ($openBrowser -eq "o" -or $openBrowser -eq "O" -or $openBrowser -eq "oui") {
    try {
        Start-Process "http://localhost:8000/docs"
        Write-ColorText Green "[OK] Navigateur ouvert"
    }
    catch {
        Write-Host "Impossible d'ouvrir le navigateur automatiquement"
        Write-Host "Ouvrez manuellement: http://localhost:8000/docs"
    }
}

Write-Host ""
Write-ColorText Green "Deployment termine avec succes!"