#!/usr/bin/env python3
"""
Script d'adaptation Docker pour projet existant
Adapte la structure existante pour la containerisation Docker

Usage:
    python docker_setup.py

Ce script va:
1. Analyser votre structure existante
2. Créer la structure Docker nécessaire
3. Générer les Dockerfiles adaptés
4. Créer docker-compose.yml
5. Configurer les variables d'environnement
"""

import os
import sys
from pathlib import Path
import shutil
import json

class DockerSetupManager:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.docker_structure = {
            'docker': ['api', 'ml-pipeline', 'dashboard', 'timescaledb'],
            'services': ['api', 'ml-pipeline', 'dashboard', 'database'],
            'config': ['docker']
        }
        
    def analyze_existing_structure(self):
        """Analyse la structure existante du projet."""
        print("🔍 Analyse de la structure existante...")
        print("=" * 60)
        
        existing_structure = {
            'files': [],
            'directories': [],
            'python_files': [],
            'key_components': {}
        }
        
        # Parcourir le projet existant
        for item in self.project_root.iterdir():
            if item.is_file():
                existing_structure['files'].append(item.name)
                if item.suffix == '.py':
                    existing_structure['python_files'].append(item.name)
            elif item.is_dir() and not item.name.startswith('.'):
                existing_structure['directories'].append(item.name)
        
        # Identifier les composants clés
        key_files = {
            'main.py': 'Point d\'entrée principal',
            'requirements.txt': 'Dépendances Python',
            'scripts/': 'Scripts d\'analyse ML',
            'src/': 'Code source modulaire',
            'data/': 'Données du projet',
            'outputs/': 'Résultats et visualisations'
        }
        
        for key_file, description in key_files.items():
            if '/' in key_file:
                # Dossier
                if (self.project_root / key_file.rstrip('/')).exists():
                    existing_structure['key_components'][key_file] = {
                        'type': 'directory',
                        'description': description,
                        'exists': True
                    }
            else:
                # Fichier
                if (self.project_root / key_file).exists():
                    existing_structure['key_components'][key_file] = {
                        'type': 'file',
                        'description': description,
                        'exists': True
                    }
        
        # Affichage du résumé
        print("📁 Structure existante détectée:")
        for component, info in existing_structure['key_components'].items():
            status = "✅" if info['exists'] else "❌"
            print(f"  {status} {component:<20} - {info['description']}")
        
        print(f"\n📊 Statistiques:")
        print(f"  Fichiers Python: {len(existing_structure['python_files'])}")
        print(f"  Dossiers: {len(existing_structure['directories'])}")
        
        return existing_structure
    
    def create_docker_structure(self):
        """Crée la structure Docker nécessaire."""
        print("\n🏗️  Création de la structure Docker...")
        print("=" * 60)
        
        # Créer les dossiers Docker
        docker_dirs = [
            'docker/api',
            'docker/ml-pipeline', 
            'docker/dashboard',
            'docker/timescaledb/init-scripts',
            'services/api',
            'services/ml-pipeline',
            'services/dashboard',
            'services/database',
            'config/docker',
            'monitoring'
        ]
        
        for dir_path in docker_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  📂 Créé: {dir_path}")
        
        return True
    
    def generate_dockerfile_ml_pipeline(self):
        """Génère le Dockerfile pour le pipeline ML."""
        dockerfile_content = """# Dockerfile pour Pipeline ML - Projet Climat Sénégal
FROM python:3.9-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Métadonnées
LABEL maintainer="Laity FAYE"
LABEL description="Pipeline ML pour prédiction événements climatiques extrêmes"
LABEL version="1.0.0"

# Dépendances système pour scientific computing
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    gfortran \\
    libopenblas-dev \\
    liblapack-dev \\
    libhdf5-dev \\
    pkg-config \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copie et installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copie de la structure du projet
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main.py .
COPY quick_test.py .

# Création des dossiers de sortie
RUN mkdir -p data/raw data/processed outputs/models outputs/reports outputs/visualizations

# Permissions d'exécution
RUN chmod +x main.py

# Variables d'environnement pour le projet
ENV PYTHONPATH=/app:/app/src
ENV PROJECT_ROOT=/app
ENV DATA_PATH=/app/data
ENV OUTPUT_PATH=/app/outputs

# Healthcheck
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \\
  CMD python quick_test.py || exit 1

# Point d'entrée par défaut
CMD ["python", "main.py", "--only-ml"]
"""
        
        dockerfile_path = self.project_root / 'docker/ml-pipeline/Dockerfile'
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        
        print("  📄 Dockerfile ML Pipeline créé")
        return dockerfile_path
    
    def generate_dockerfile_api(self):
        """Génère le Dockerfile pour l'API FastAPI."""
        dockerfile_content = """# Dockerfile pour API - Projet Climat Sénégal
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dépendances système légères
RUN apt-get update && apt-get install -y \\
    gcc \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation des dépendances API
COPY services/api/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code API
COPY services/api/ .

# Copie des modèles ML pré-entraînés
COPY outputs/models/ ./models/

# Port d'exposition
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Variables d'environnement
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MODEL_PATH=/app/models

# Commande de démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
"""
        
        dockerfile_path = self.project_root / 'docker/api/Dockerfile'
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        
        print("  📄 Dockerfile API créé")
        return dockerfile_path
    
    def generate_api_service_files(self):
        """Génère les fichiers de base pour le service API."""
        
        # Requirements API
        api_requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
python-multipart==0.0.6
python-dotenv==1.0.0
prometheus-client==0.19.0
"""
        
        api_req_path = self.project_root / 'services/api/requirements.txt'
        with open(api_req_path, 'w', encoding='utf-8') as f:
            f.write(api_requirements)
        
        # API main.py
        api_main_content = '''"""
API FastAPI pour le projet de prédiction climatique
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
from datetime import datetime, timedelta
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Climate Extreme Prediction API",
    description="API pour prédictions d'événements climatiques extrêmes au Sénégal",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des chemins
MODEL_PATH = Path(os.getenv("MODEL_PATH", "./models"))

# Modèles chargés en mémoire
loaded_models = {}

class PredictionRequest(BaseModel):
    features: dict
    model_name: str = "random_forest"

class PredictionResponse(BaseModel):
    prediction: float
    probability: float
    confidence_interval: dict
    model_used: str
    timestamp: datetime

@app.on_event("startup")
async def startup_event():
    """Initialisation de l'API."""
    logger.info("🚀 Démarrage de l'API Climate Prediction")
    
    # Charger les modèles disponibles
    if MODEL_PATH.exists():
        for model_file in MODEL_PATH.glob("*.pkl"):
            try:
                model_name = model_file.stem
                loaded_models[model_name] = joblib.load(model_file)
                logger.info(f"✅ Modèle chargé: {model_name}")
            except Exception as e:
                logger.error(f"❌ Erreur chargement {model_file}: {e}")
    
    logger.info(f"📊 {len(loaded_models)} modèles disponibles")

@app.get("/")
async def root():
    return {
        "message": "API Climate Extreme Prediction",
        "version": "1.0.0",
        "status": "running",
        "models_available": list(loaded_models.keys())
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": len(loaded_models)
    }

@app.get("/models")
async def list_models():
    return {
        "available_models": list(loaded_models.keys()),
        "total_models": len(loaded_models)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_extreme_event(request: PredictionRequest):
    """Prédire la probabilité d'un événement extrême."""
    
    if request.model_name not in loaded_models:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{request.model_name}' non trouvé. Modèles disponibles: {list(loaded_models.keys())}"
        )
    
    try:
        model = loaded_models[request.model_name]
        
        # Convertir les features en format approprié
        features_df = pd.DataFrame([request.features])
        
        # Prédiction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_df)[0]
            prediction = model.predict(features_df)[0]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            prediction = model.predict(features_df)[0]
            probability = float(prediction)
        
        # Intervalle de confiance simulé (à améliorer avec vrais modèles)
        confidence_interval = {
            "lower": max(0, probability - 0.1),
            "upper": min(1, probability + 0.1)
        }
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=probability,
            confidence_interval=confidence_interval,
            model_used=request.model_name,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Métriques Prometheus."""
    # À implémenter avec prometheus_client
    return {"message": "Métriques à implémenter"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        api_main_path = self.project_root / 'services/api/main.py'
        with open(api_main_path, 'w', encoding='utf-8') as f:
            f.write(api_main_content)
        
        print("  📄 Service API créé")
    
    def generate_docker_compose(self):
        """Génère le fichier docker-compose.yml principal."""
        compose_content = """version: '3.8'

services:
  # Base de données TimescaleDB
  timescaledb:
    image: timescale/timescaledb:latest-pg14
    container_name: climate-timescaledb
    environment:
      - POSTGRES_DB=${POSTGRES_DB:-climate_db}
      - POSTGRES_USER=${POSTGRES_USER:-climate_user}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-secure_password}
      - PGDATA=/var/lib/postgresql/data/pgdata
    volumes:
      - timescale_data:/var/lib/postgresql/data
      - ./docker/timescaledb/init-scripts:/docker-entrypoint-initdb.d
      - ./data/backup:/backup
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER:-climate_user} -d $${POSTGRES_DB:-climate_db}"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - climate-network

  # Redis pour cache et files d'attente
  redis:
    image: redis:7-alpine
    container_name: climate-redis
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - climate-network

  # Pipeline ML - Traitement existant
  ml-pipeline:
    build: 
      context: .
      dockerfile: docker/ml-pipeline/Dockerfile
    container_name: climate-ml-pipeline
    environment:
      - DATABASE_URL=postgresql://$${POSTGRES_USER:-climate_user}:$${POSTGRES_PASSWORD:-secure_password}@timescaledb:5432/$${POSTGRES_DB:-climate_db}
      - REDIS_URL=redis://redis:6379
      - PYTHONPATH=/app:/app/src
    depends_on:
      timescaledb:
        condition: service_healthy
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ml_models:/app/outputs/models
    working_dir: /app
    networks:
      - climate-network
    # Note: restart="no" pour exécution manuelle du pipeline
    restart: "no"

  # API FastAPI
  api:
    build:
      context: .
      dockerfile: docker/api/Dockerfile
    container_name: climate-api
    environment:
      - DATABASE_URL=postgresql://$${POSTGRES_USER:-climate_user}:$${POSTGRES_PASSWORD:-secure_password}@timescaledb:5432/$${POSTGRES_DB:-climate_db}
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/app/models
    depends_on:
      timescaledb:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "8000:8000"
    volumes:
      - ml_models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - climate-network

  # Dashboard Web (à développer)
  dashboard:
    image: nginx:alpine
    container_name: climate-dashboard
    ports:
      - "3000:80"
    volumes:
      - ./services/dashboard/dist:/usr/share/nginx/html:ro
      - ./docker/dashboard/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api
    restart: unless-stopped
    networks:
      - climate-network

  # Monitoring - Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: climate-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - climate-network

  # Monitoring - Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: climate-grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - climate-network

volumes:
  timescale_data:
    driver: local
  redis_data:
    driver: local
  ml_models:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  climate-network:
    driver: bridge
    name: senegal-climate-network
"""
        
        compose_path = self.project_root / 'docker-compose.yml'
        with open(compose_path, 'w', encoding='utf-8') as f:
            f.write(compose_content)
        
        print("  📄 docker-compose.yml créé")
        return compose_path
    
    def generate_env_file(self):
        """Génère le fichier .env avec les variables d'environnement."""
        env_content = """# Configuration Docker - Projet Climat Sénégal
# Base de données TimescaleDB
POSTGRES_DB=climate_db
POSTGRES_USER=climate_user
POSTGRES_PASSWORD=ChangeMe_SecurePassword2024!

# API Configuration
API_SECRET_KEY=your_super_secure_api_key_here_change_me
API_DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000

# Monitoring
GRAFANA_PASSWORD=secure_grafana_password_2024

# Pipeline ML
ML_MODEL_PATH=/app/outputs/models
DATA_PATH=/app/data
OUTPUT_PATH=/app/outputs

# Redis Configuration
REDIS_URL=redis://redis:6379

# Projet Configuration
COMPOSE_PROJECT_NAME=senegal-climate
PROJECT_NAME=senegal-extreme-climate-prediction

# Environnement
ENVIRONMENT=development
DEBUG=true

# Logging
LOG_LEVEL=INFO
"""
        
        env_path = self.project_root / '.env'
        if not env_path.exists():  # Ne pas écraser si existe déjà
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(env_content)
            print("  📄 .env créé")
        else:
            print("  📄 .env existe déjà (préservé)")
    
    def generate_dockerignore(self):
        """Génère le fichier .dockerignore."""
        dockerignore_content = """# Git
.git
.gitignore
*.md

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Cache
.cache/
.pytest_cache/
.coverage

# Temporary files
*.tmp
*.temp

# Large data files (à adapter selon votre projet)
data/raw/*.nc
data/raw/*.mat
*.hdf5

# Build artifacts
docker-compose.override.yml
.env.local
.env.*.local

# Documentation
docs/_build/

# Jupyter
.jupyter/
*.ipynb_checkpoints
"""
        
        dockerignore_path = self.project_root / '.dockerignore'
        with open(dockerignore_path, 'w', encoding='utf-8') as f:
            f.write(dockerignore_content)
        
        print("  📄 .dockerignore créé")
    
    def generate_monitoring_config(self):
        """Génère les configurations de monitoring."""
        
        # Prometheus config
        prometheus_config = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'climate-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
  
  - job_name: 'timescaledb'
    static_configs:
      - targets: ['timescaledb:5432']
"""
        
        prometheus_dir = self.project_root / 'monitoring'
        prometheus_dir.mkdir(exist_ok=True)
        
        with open(prometheus_dir / 'prometheus.yml', 'w', encoding='utf-8') as f:
            f.write(prometheus_config)
        
        print("  📄 Configuration monitoring créée")
    
    def generate_scripts(self):
        """Génère les scripts utilitaires Docker."""
        
        # Créer le dossier docker/scripts pour éviter confusion
        docker_scripts_dir = self.project_root / 'docker/scripts'
        docker_scripts_dir.mkdir(exist_ok=True)
        
        # Script de démarrage PowerShell
        startup_script = """# startup.ps1 - Script de démarrage du projet Docker
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
"""
        
        # Sauvegarder dans docker/scripts/
        startup_script_path = docker_scripts_dir / 'startup.ps1'
        with open(startup_script_path, 'w', encoding='utf-8') as f:
            f.write(startup_script)
        
        # Créer aussi un lien dans la racine pour facilité d'usage
        root_startup_path = self.project_root / 'docker-start.ps1'
        with open(root_startup_path, 'w', encoding='utf-8') as f:
            f.write(startup_script)
        
        # Script de déploiement
        deploy_script = """# deploy.ps1 - Script de déploiement Docker
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
"""
        
        deploy_script_path = docker_scripts_dir / 'deploy.ps1'
        with open(deploy_script_path, 'w', encoding='utf-8') as f:
            f.write(deploy_script)
        
        # Script de sauvegarde
        backup_script = """# backup.ps1 - Script de sauvegarde Docker
param(
    [string]$BackupName = (Get-Date -Format "yyyy-MM-dd_HH-mm-ss")
)

Write-Host "💾 Sauvegarde Docker - $BackupName" -ForegroundColor Green

$backupDir = "data/backup"
if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir -Force
}

Write-Host "📊 Sauvegarde base de données..." -ForegroundColor Yellow
docker-compose exec -T timescaledb pg_dump -U climate_user climate_db > "$backupDir/db_$BackupName.sql"

Write-Host "🤖 Sauvegarde modèles ML..." -ForegroundColor Yellow
docker cp climate-ml-pipeline:/app/outputs/models "$backupDir/models_$BackupName"

Write-Host "✅ Sauvegarde terminée: $backupDir" -ForegroundColor Green
"""
        
        backup_script_path = docker_scripts_dir / 'backup.ps1'
        with open(backup_script_path, 'w', encoding='utf-8') as f:
            f.write(backup_script)
        
        print("  📄 Scripts Docker créés dans docker/scripts/")
        print("  📄 Raccourci docker-start.ps1 créé à la racine")
    
    def run_setup(self):
        """Exécute le setup complet."""
        print("🐳 SETUP DOCKER - ADAPTATION STRUCTURE EXISTANTE")
        print("=" * 80)
        print(f"📁 Projet: {self.project_root}")
        print("=" * 80)
        
        # Étape 1: Analyser l'existant
        existing = self.analyze_existing_structure()
        
        # Étape 2: Créer structure Docker
        self.create_docker_structure()
        
        # Étape 3: Générer Dockerfiles
        print("\n🔨 Génération des Dockerfiles...")
        self.generate_dockerfile_ml_pipeline()
        self.generate_dockerfile_api()
        
        # Étape 4: Générer services
        print("\n⚙️ Création des services...")
        self.generate_api_service_files()
        
        # Étape 5: Orchestration
        print("\n🎼 Configuration Docker Compose...")
        self.generate_docker_compose()
        self.generate_env_file()
        self.generate_dockerignore()
        
        # Étape 6: Monitoring
        print("\n📊 Configuration monitoring...")
        self.generate_monitoring_config()
        
        # Étape 7: Scripts utilitaires
        print("\n🛠️ Génération des scripts...")
        self.generate_scripts()
        
        # Résumé final
        print("\n" + "=" * 80)
        print("✅ SETUP DOCKER TERMINÉ!")
        print("=" * 80)
        print("📂 Structure créée:")
        print("  ├── docker/")
        print("  │   ├── api/Dockerfile")
        print("  │   ├── ml-pipeline/Dockerfile")
        print("  │   └── timescaledb/")
        print("  ├── services/")
        print("  │   └── api/")
        print("  ├── docker-compose.yml")
        print("  ├── .env")
        print("  └── startup.ps1")
        
        print("\n🚀 Prochaines étapes:")
        print("  1. Vérifiez et modifiez .env selon vos besoins")
        print("  2. Lancez: docker-compose up -d --build")
        print("  3. Testez le pipeline: docker-compose run --rm ml-pipeline python main.py")
        print("  4. Accédez à l'API: http://localhost:8000")
        
        return True

def main():
    """Point d'entrée principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Docker pour projet existant")
    parser.add_argument("--project-root", help="Chemin racine du projet", default=".")
    
    args = parser.parse_args()
    
    setup_manager = DockerSetupManager(args.project_root)
    
    try:
        setup_manager.run_setup()
        return 0
    except Exception as e:
        print(f"\n❌ Erreur lors du setup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())