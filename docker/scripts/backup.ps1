# backup.ps1 - Script de sauvegarde Docker
param(
    [string]$BackupName = (Get-Date -Format "yyyy-MM-dd_HH-mm-ss")
)

Write-Host "💾 Sauvegarde Docker - $BackupName" -ForegroundColor Green

$backupDir = "data/backup"
if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir -Force
}

Write-Host "📊 Sauvegarde base de données..." -ForegroundColor Yellow
docker-compose exec -T timescaledb pg_dump -U postgres climatsn_db > "$backupDir/db_$BackupName.sql"

Write-Host "🤖 Sauvegarde modèles ML..." -ForegroundColor Yellow
docker cp climate-ml-pipeline:/app/outputs/models "$backupDir/models_$BackupName"

Write-Host "✅ Sauvegarde terminée: $backupDir" -ForegroundColor Green
