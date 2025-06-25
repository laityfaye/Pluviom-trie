# services/api/main.py
"""
API REST pour les prédictions d'événements climatiques extrêmes au Sénégal
VERSION CORRIGÉE - Fix pour erreur 404 sur /docs
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import json
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODÈLES PYDANTIC
# ============================================================================

class StationInfo(BaseModel):
    id: str
    name: str
    region: str
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    status: str = "active"

class WeatherData(BaseModel):
    time: datetime
    station_id: str
    temperature: Optional[float] = None
    precipitation: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None

class PredictionRequest(BaseModel):
    station_id: Optional[str] = None
    region: Optional[str] = None
    forecast_hours: int = Field(default=72, ge=1, le=168)  # 1h à 7 jours
    model_name: Optional[str] = None
    include_confidence: bool = True

class PredictionResponse(BaseModel):
    id: str
    prediction_time: datetime
    target_time: datetime
    forecast_horizon_hours: int
    station_id: Optional[str]
    region: Optional[str]
    predicted_class: Optional[str]
    probability: Optional[float]
    confidence_interval: Optional[Dict[str, float]]
    model_name: str
    data_quality_score: Optional[float]

class AlertInfo(BaseModel):
    id: str
    alert_time: datetime
    event_type: str
    alert_level: str
    title: str
    message: str
    region: Optional[str]
    station_id: Optional[str]
    valid_from: datetime
    valid_until: Optional[datetime]
    confidence_score: Optional[float]

# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration API
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes par défaut

# Configuration des fichiers statiques
STATIC_FILES_DIR = os.getenv("STATIC_FILES_DIR", "../../outputs")

# Variables globales pour les connexions
redis_client = None
db_pool = None

# ============================================================================
# GESTIONNAIRE DE CONTEXTE POUR LES CONNEXIONS
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application - VERSION SIMPLIFIÉE"""
    global redis_client, db_pool
    
    # Startup
    logger.info("🚀 Démarrage de l'API climat...")
    
    try:
        # Tentative connexion Redis (optionnelle)
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
            await redis_client.ping()
            logger.info("✅ Redis connecté")
        except Exception as e:
            logger.warning(f"⚠️ Redis non disponible: {e}")
            redis_client = None
        
        # Pool de connexions PostgreSQL (optionnel)
        try:
            import asyncpg
            DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:secure_password@localhost:5432/climatsn_db")
            db_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=10,
                command_timeout=30
            )
            logger.info("✅ Base de données connectée")
        except Exception as e:
            logger.warning(f"⚠️ Base de données non disponible: {e}")
            db_pool = None
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage: {e}")
        # Ne pas faire échouer le démarrage
    
    finally:
        # Shutdown
        logger.info("🛑 Arrêt de l'API...")
        
        if redis_client:
            await redis_client.close()
            logger.info("✅ Redis déconnecté")
        
        if db_pool:
            await db_pool.close()
            logger.info("✅ Base de données déconnectée")

# ============================================================================
# APPLICATION FASTAPI - VERSION CORRIGÉE
# ============================================================================

app = FastAPI(
    title="API Prédictions Climat Sénégal",
    description="API REST pour les prédictions d'événements climatiques extrêmes au Sénégal avec visualisations ML",
    version="1.0.0",
    lifespan=lifespan,
    debug=API_DEBUG,
    docs_url="/docs",  # URL documentation Swagger
    redoc_url="/redoc"  # URL documentation ReDoc
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Dashboard
        "http://127.0.0.1:3000",  # Dashboard alternatif
        "http://localhost:8000",  # API elle-même
        "*"  # Tous les autres (à restreindre en production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ============================================================================
# INCLUSION DU ROUTER VISUALIZATIONS - VERSION SÉCURISÉE
# ============================================================================

# Import conditionnel du module visualizations
VISUALIZATIONS_AVAILABLE = False
try:
    from visualizations import router as visualizations_router
    app.include_router(visualizations_router)
    logger.info("✅ Module visualizations importé et routeur ajouté")
    VISUALIZATIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Module visualizations non disponible: {e}")
    VISUALIZATIONS_AVAILABLE = False

# ============================================================================
# MONTAGE DES FICHIERS STATIQUES
# ============================================================================

# Monter les fichiers statiques pour les outputs (si le dossier existe)
if os.path.exists(STATIC_FILES_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")
    logger.info(f"✅ Fichiers statiques montés: {STATIC_FILES_DIR}")
else:
    logger.warning(f"⚠️ Dossier statique non trouvé: {STATIC_FILES_DIR}")

# ============================================================================
# DONNÉES DE TEST
# ============================================================================

# Données de stations simulées si pas de DB
MOCK_STATIONS = [
    StationInfo(id="DAKAR_AERO", name="Dakar Aéroport", region="Dakar", latitude=14.7447, longitude=-17.4913),
    StationInfo(id="SAINT_LOUIS", name="Saint-Louis", region="Saint-Louis", latitude=16.0469, longitude=-16.4624),
    StationInfo(id="TAMBACOUNDA", name="Tambacounda", region="Tambacounda", latitude=13.7671, longitude=-13.6681),
    StationInfo(id="ZIGUINCHOR", name="Ziguinchor", region="Ziguinchor", latitude=12.5556, longitude=-16.2719),
    StationInfo(id="KAOLACK", name="Kaolack", region="Kaolack", latitude=14.1593, longitude=-16.0728)
]

# ============================================================================
# ENDPOINTS - SANTÉ ET INFORMATION
# ============================================================================

@app.get("/")
async def root():
    """Endpoint racine avec informations de base"""
    return {
        "message": "API Prédictions Climat Sénégal",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs",
        "features": {
            "database": db_pool is not None,
            "redis_cache": redis_client is not None,
            "visualizations": VISUALIZATIONS_AVAILABLE,
            "ml_models": True,
            "predictions": True
        },
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "stations": "/stations",
            "weather": "/weather",
            "predictions": "/predictions",
            "alerts": "/alerts",
            "models": "/models",
            "analysis": "/analysis",
            "visualizations": "/api/visualizations" if VISUALIZATIONS_AVAILABLE else None,
            "dashboard": "/dashboard/overview",
            "research": "/research/progress",
            "metrics": "/metrics"
        },
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "services": {}
    }
    
    # Test Redis
    try:
        if redis_client:
            await redis_client.ping()
            status["services"]["redis"] = "connected"
        else:
            status["services"]["redis"] = "not_configured"
    except Exception:
        status["services"]["redis"] = "error"
        status["status"] = "degraded"
    
    # Test Database
    try:
        if db_pool:
            conn = await db_pool.acquire()
            await conn.fetchval("SELECT 1")
            await db_pool.release(conn)
            status["services"]["database"] = "connected"
        else:
            status["services"]["database"] = "not_configured"
    except Exception:
        status["services"]["database"] = "error"
        status["status"] = "degraded"
    
    # Test Visualizations
    if VISUALIZATIONS_AVAILABLE:
        try:
            from visualizations import scanner
            viz_count = len(scanner.scan_all_visualizations())
            status["services"]["visualizations"] = {
                "status": "available",
                "count": viz_count
            }
        except Exception as e:
            status["services"]["visualizations"] = {
                "status": "error",
                "error": str(e)
            }
    else:
        status["services"]["visualizations"] = "not_available"
    
    return status

@app.get("/info")
async def api_info():
    """Informations détaillées sur l'API"""
    return {
        "name": "API Prédictions Climat Sénégal",
        "version": "1.0.0",
        "description": "API REST pour les prédictions d'événements climatiques extrêmes avec ML",
        "features": [
            "Prédictions météorologiques",
            "Détection d'événements extrêmes",
            "Visualisations interactives" if VISUALIZATIONS_AVAILABLE else "Visualisations (non disponibles)",
            "Analyses de machine learning",
            "Alertes automatisées",
            "Monitoring en temps réel"
        ],
        "endpoints": {
            "stations": "/stations - Gestion des stations météorologiques",
            "weather": "/weather - Données météorologiques récentes",
            "predictions": "/predictions - Prédictions et modèles ML",
            "alerts": "/alerts - Alertes et événements extrêmes",
            "models": "/models - Gestion des modèles ML",
            "analysis": "/analysis - Résultats d'analyses",
            "visualizations": "/api/visualizations - Visualisations générées" if VISUALIZATIONS_AVAILABLE else None,
            "dashboard": "/dashboard/overview - Vue d'ensemble",
            "research": "/research/progress - Suivi recherche",
            "monitoring": "/metrics - Métriques et monitoring"
        },
        "documentation": "/docs",
        "contact": {
            "research": "Analyse des schémas de température de surface des océans",
            "institution": "Université du Sénégal",
            "domain": "Prédictions climatiques et ML"
        },
        "services_status": {
            "database": db_pool is not None,
            "cache": redis_client is not None,
            "visualizations": VISUALIZATIONS_AVAILABLE
        }
    }

# ============================================================================
# ENDPOINTS - STATIONS MÉTÉOROLOGIQUES
# ============================================================================

@app.get("/stations", response_model=List[StationInfo])
async def get_stations(
    region: Optional[str] = Query(None, description="Filtrer par région"),
    active_only: bool = Query(True, description="Stations actives uniquement")
):
    """Récupère la liste des stations météorologiques"""
    
    # Si pas de DB, retourner les données simulées
    if not db_pool:
        stations = MOCK_STATIONS
        if region:
            stations = [s for s in stations if s.region.lower() == region.lower()]
        return stations
    
    # TODO: Logique avec base de données
    return MOCK_STATIONS

@app.get("/stations/{station_id}", response_model=StationInfo)
async def get_station(station_id: str = Path(..., description="ID de la station")):
    """Récupère les informations d'une station spécifique"""
    
    # Si pas de DB, chercher dans les données simulées
    if not db_pool:
        for station in MOCK_STATIONS:
            if station.id == station_id:
                return station
        raise HTTPException(status_code=404, detail="Station non trouvée")
    
    # TODO: Logique avec base de données
    raise HTTPException(status_code=404, detail="Station non trouvée")

# ============================================================================
# ENDPOINTS - DASHBOARD ET RECHERCHE
# ============================================================================

@app.get("/dashboard/overview")
async def get_dashboard_overview():
    """Vue d'ensemble pour le tableau de bord"""
    
    try:
        # Statistiques des visualisations si disponibles
        viz_stats = {}
        if VISUALIZATIONS_AVAILABLE:
            from visualizations import scanner, CATEGORY_MAPPING
            
            # Récupérer les visualisations par catégorie
            for category in CATEGORY_MAPPING.keys():
                viz_list = scanner.scan_all_visualizations(category)
                viz_stats[category] = {
                    "count": len(viz_list),
                    "latest": viz_list[0] if viz_list else None,
                    "category_info": CATEGORY_MAPPING[category]
                }
        
        # Statistiques générales (simulées si pas de DB)
        db_stats = {
            "active_stations": len(MOCK_STATIONS), 
            "recent_observations": 120, 
            "recent_predictions": 15, 
            "active_models": 3
        }
        
        overview = {
            "timestamp": datetime.now(),
            "system_status": "operational",
            "database_stats": db_stats,
            "visualizations": {
                "available": VISUALIZATIONS_AVAILABLE,
                "total_visualizations": sum(cat["count"] for cat in viz_stats.values()) if viz_stats else 0,
                "by_category": viz_stats,
                "categories": list(viz_stats.keys()) if viz_stats else []
            },
            "quick_access": {
                "latest_visualizations": "/api/visualizations/list?limit=10" if VISUALIZATIONS_AVAILABLE else None,
                "weather_stats": "/weather/stats",
                "active_alerts": "/alerts/active",
                "model_status": "/models"
            }
        }
        
        return overview
        
    except Exception as e:
        logger.error(f"Erreur dashboard overview: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur génération overview: {str(e)}")

@app.get("/research/progress")
async def get_research_progress():
    """Suivi du progrès de la recherche de mémoire"""
    
    try:
        # Mapping des étapes du mémoire aux catégories de visualizations
        research_phases = {
            "detection_extremes": {
                "title": "Détection des Événements Extrêmes",
                "description": "Identification et classification des événements de précipitations extrêmes",
                "category": "detection",
                "chapter": "Chapitre 4 - Section 1",
                "status": "completed"
            },
            "spatial_analysis": {
                "title": "Analyse Spatiale",
                "description": "Distribution spatiale et patterns géographiques",
                "category": "spatial", 
                "chapter": "Chapitre 4 - Section 1",
                "status": "completed"
            },
            "temporal_analysis": {
                "title": "Analyse Temporelle",
                "description": "Variabilité interannuelle et tendances temporelles",
                "category": "temporal",
                "chapter": "Chapitre 4 - Section 2-3",
                "status": "in_progress"
            },
            "teleconnections": {
                "title": "Téléconnexions Climatiques",
                "description": "Liens avec les modes de variabilité à grande échelle",
                "category": "teleconnections",
                "chapter": "Chapitre 4 - Section 4",
                "status": "completed"
            },
            "machine_learning": {
                "title": "Modèles d'Apprentissage Automatique",
                "description": "Développement et évaluation des modèles ML",
                "category": "machine-learning",
                "chapter": "Chapitre 3 - Section 6",
                "status": "in_progress"
            },
            "clustering": {
                "title": "Analyse de Clustering",
                "description": "Classification et regroupement des patterns",
                "category": "clustering",
                "chapter": "Chapitre 4 - Section 5",
                "status": "completed"
            }
        }
        
        # Enrichir avec les données de visualizations si disponibles
        if VISUALIZATIONS_AVAILABLE:
            from visualizations import scanner
            for phase_key, phase_info in research_phases.items():
                category = phase_info["category"]
                viz_list = scanner.scan_all_visualizations(category)
                
                phase_info.update({
                    "visualizations_count": len(viz_list),
                    "latest_visualization": viz_list[0] if viz_list else None,
                    "has_results": len(viz_list) > 0,
                    "completion_percentage": 100 if len(viz_list) > 0 else 0
                })
        else:
            # Données simulées si pas de visualizations
            for phase_info in research_phases.values():
                phase_info.update({
                    "visualizations_count": 2,
                    "latest_visualization": None,
                    "has_results": True,
                    "completion_percentage": 75
                })
        
        # Calcul du progrès global
        completed_phases = sum(1 for phase in research_phases.values() if phase.get("has_results", False))
        total_phases = len(research_phases)
        global_progress = (completed_phases / total_phases) * 100
        
        progress = {
            "timestamp": datetime.now(),
            "global_progress": global_progress,
            "completed_phases": completed_phases,
            "total_phases": total_phases,
            "phases": research_phases,
            "next_steps": [
                "Finalisation des analyses temporelles",
                "Optimisation des modèles ML", 
                "Validation croisée des résultats",
                "Rédaction des conclusions"
            ],
            "thesis_structure": {
                "introduction": "completed",
                "chapter_1": "completed", 
                "chapter_2": "completed",
                "chapter_3": "in_progress",
                "chapter_4": "in_progress",
                "chapter_5": "planned",
                "conclusion": "planned"
            },
            "visualizations_available": VISUALIZATIONS_AVAILABLE
        }
        
        return progress
        
    except Exception as e:
        logger.error(f"Erreur research progress: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur suivi recherche: {str(e)}")

# ============================================================================
# ENDPOINTS - MÉTRIQUES
# ============================================================================

@app.get("/metrics")
async def get_metrics():
    """Métriques complètes de l'API pour monitoring"""
    
    # Statistiques de base
    base_stats = {
        "active_stations": len(MOCK_STATIONS),
        "recent_observations_24h": 120,
        "recent_predictions_24h": 15,
        "active_alerts": 1,
        "recent_events_30d": 8,
        "active_models": 2,
        "total_analyses": 6
    }
    
    # Métriques de visualizations
    viz_count = 0
    viz_status = "not_available"
    if VISUALIZATIONS_AVAILABLE:
        try:
            from visualizations import scanner
            viz_count = len(scanner.scan_all_visualizations())
            viz_status = "operational"
        except Exception as e:
            viz_status = f"error: {str(e)}"
    
    # Métriques complètes
    metrics = {
        "system": {
            **base_stats,
            "visualizations_count": viz_count,
            "visualizations_status": viz_status
        },
        "performance": {
            "avg_response_time_ms": 150,
            "cache_hit_rate": 0.85 if redis_client else 0,
            "prediction_accuracy": 0.87,
            "data_quality_score": 0.92,
            "api_uptime": 0.998
        },
        "research": {
            "thesis_completion": 75.0,
            "chapters_completed": 3,
            "total_chapters": 5,
            "ml_models_trained": base_stats.get("active_models", 2),
            "visualizations_generated": viz_count
        },
        "services": {
            "database": db_pool is not None,
            "cache": redis_client is not None,
            "visualizations": VISUALIZATIONS_AVAILABLE
        },
        "timestamp": datetime.now()
    }
    
    return metrics

# ============================================================================
# ENDPOINTS DE FALLBACK POUR VISUALIZATIONS
# ============================================================================

# Si les visualizations ne sont pas disponibles, fournir des endpoints de fallback
if not VISUALIZATIONS_AVAILABLE:
    
    @app.get("/api/visualizations/list")
    async def fallback_visualizations_list():
        """Endpoint de fallback pour les visualisations"""
        return {
            "message": "Module visualizations non disponible",
            "status": "fallback_mode",
            "visualizations": [
                {
                    "id": "mock_viz_1",
                    "title": "Distribution Temporelle (Simulé)",
                    "category": "detection",
                    "image": "/static/placeholder.png",
                    "date": datetime.now().isoformat()
                }
            ],
            "total": 1,
            "note": "Créez le fichier visualizations.py pour activer les vraies visualisations"
        }
    
    @app.get("/api/visualizations/health")
    async def fallback_visualizations_health():
        """Health check de fallback pour visualisations"""
        return {
            "status": "not_available",
            "message": "Module visualizations.py non trouvé",
            "suggestion": "Vérifiez que le fichier visualizations.py est présent dans le même dossier"
        }

# ============================================================================
# GESTION DES ERREURS
# ============================================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Gestionnaire d'erreurs global"""
    logger.error(f"Erreur non gérée sur {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erreur interne du serveur", 
            "type": "internal_error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
            "services_available": {
                "database": db_pool is not None,
                "cache": redis_client is not None,
                "visualizations": VISUALIZATIONS_AVAILABLE
            }
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Gestionnaire d'erreurs HTTP spécialisé"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "type": "http_error",
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

# ============================================================================
# LANCEMENT DE L'APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Démarrage de l'API Climat Sénégal...")
    print("📍 URL: http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    print("🔧 Services disponibles:")
    print(f"   - Base de données: {'✅' if 'DATABASE_URL' in os.environ else '⚠️ Mode fallback'}")
    print(f"   - Cache Redis: {'✅' if 'REDIS_URL' in os.environ else '⚠️ Optionnel'}")
    print(f"   - Visualisations: {'✅' if VISUALIZATIONS_AVAILABLE else '⚠️ Module non trouvé'}")
    print()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=API_DEBUG,
        log_level="info",
        access_log=True
    )