# services/api/main.py
"""
API REST pour les prédictions d'événements climatiques extrêmes au Sénégal
Utilise FastAPI avec TimescaleDB et Redis pour le cache
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
from pydantic import BaseModel, Field
import asyncpg
import redis.asyncio as redis
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
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

class ExtremeEventInfo(BaseModel):
    id: str
    event_time: datetime
    event_type: str
    event_name: str
    severity_level: int
    intensity: Optional[float]
    duration_hours: Optional[int]
    region: Optional[str]
    confidence_score: Optional[float]

# ============================================================================
# CONFIGURATION ET CONNEXIONS
# ============================================================================

# Configuration de la base de données
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:secure_password@localhost:5432/climatsn_db"
)

# Configuration Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Configuration API
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes par défaut

# Variables globales pour les connexions
redis_client = None
db_pool = None

# ============================================================================
# GESTIONNAIRE DE CONTEXTE POUR LES CONNEXIONS
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application"""
    global redis_client, db_pool
    
    # Startup
    logger.info("🚀 Démarrage de l'API climat...")
    
    try:
        # Connexion Redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("✅ Redis connecté")
        
        # Pool de connexions PostgreSQL
        db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        logger.info("✅ Base de données connectée")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage: {e}")
        raise
    
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
# APPLICATION FASTAPI
# ============================================================================

app = FastAPI(
    title="API Prédictions Climat Sénégal",
    description="API REST pour les prédictions d'événements climatiques extrêmes au Sénégal",
    version="1.0.0",
    lifespan=lifespan,
    debug=API_DEBUG
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# UTILITAIRES
# ============================================================================

async def get_db_connection():
    """Récupère une connexion à la base de données"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Base de données non disponible")
    return await db_pool.acquire()

async def get_cached_data(key: str) -> Optional[Any]:
    """Récupère des données depuis le cache Redis"""
    try:
        if redis_client:
            data = await redis_client.get(key)
            return json.loads(data) if data else None
    except Exception as e:
        logger.warning(f"Erreur cache lecture {key}: {e}")
    return None

async def set_cached_data(key: str, data: Any, ttl: int = CACHE_TTL) -> bool:
    """Stocke des données dans le cache Redis"""
    try:
        if redis_client:
            await redis_client.setex(key, ttl, json.dumps(data, default=str))
            return True
    except Exception as e:
        logger.warning(f"Erreur cache écriture {key}: {e}")
    return False

def create_cache_key(*args) -> str:
    """Crée une clé de cache standardisée"""
    return ":".join(str(arg) for arg in args)

# ============================================================================
# ENDPOINTS - SANTÉ ET INFORMATION
# ============================================================================

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
        conn = await get_db_connection()
        await conn.fetchval("SELECT 1")
        await db_pool.release(conn)
        status["services"]["database"] = "connected"
    except Exception:
        status["services"]["database"] = "error"
        status["status"] = "degraded"
    
    return status

@app.get("/info")
async def api_info():
    """Informations sur l'API"""
    return {
        "name": "API Prédictions Climat Sénégal",
        "version": "1.0.0",
        "description": "API REST pour les prédictions d'événements climatiques extrêmes",
        "endpoints": {
            "stations": "/stations - Gestion des stations météorologiques",
            "weather": "/weather - Données météorologiques récentes",
            "predictions": "/predictions - Prédictions et modèles ML",
            "alerts": "/alerts - Alertes et événements extrêmes",
            "monitoring": "/metrics - Métriques et monitoring"
        },
        "documentation": "/docs"
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
    
    cache_key = create_cache_key("stations", region, active_only)
    cached_data = await get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    conn = await get_db_connection()
    try:
        query = """
            SELECT id, name, region, latitude, longitude, altitude, status
            FROM stations
            WHERE ($1::text IS NULL OR region = $1)
            AND ($2::boolean IS FALSE OR status = 'active')
            ORDER BY region, name
        """
        
        rows = await conn.fetch(query, region, active_only)
        stations = [StationInfo(**dict(row)) for row in rows]
        
        await set_cached_data(cache_key, [s.dict() for s in stations])
        return stations
        
    finally:
        await db_pool.release(conn)

@app.get("/stations/{station_id}", response_model=StationInfo)
async def get_station(station_id: str = Path(..., description="ID de la station")):
    """Récupère les informations d'une station spécifique"""
    
    cache_key = create_cache_key("station", station_id)
    cached_data = await get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    conn = await get_db_connection()
    try:
        query = """
            SELECT id, name, region, latitude, longitude, altitude, status
            FROM stations
            WHERE id = $1
        """
        
        row = await conn.fetchrow(query, station_id)
        if not row:
            raise HTTPException(status_code=404, detail="Station non trouvée")
        
        station = StationInfo(**dict(row))
        await set_cached_data(cache_key, station.dict())
        return station
        
    finally:
        await db_pool.release(conn)

# ============================================================================
# ENDPOINTS - DONNÉES MÉTÉOROLOGIQUES
# ============================================================================

@app.get("/weather/recent", response_model=List[WeatherData])
async def get_recent_weather(
    station_id: Optional[str] = Query(None, description="ID de la station"),
    region: Optional[str] = Query(None, description="Région"),
    hours: int = Query(24, ge=1, le=168, description="Nombre d'heures"),
    limit: int = Query(100, ge=1, le=1000, description="Limite de résultats")
):
    """Récupère les données météorologiques récentes"""
    
    cache_key = create_cache_key("weather_recent", station_id, region, hours, limit)
    cached_data = await get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    conn = await get_db_connection()
    try:
        query = """
            SELECT w.time, w.station_id, w.temperature, w.precipitation,
                   w.humidity, w.pressure, w.wind_speed, w.wind_direction
            FROM weather_data w
            JOIN stations s ON w.station_id = s.id
            WHERE w.time >= NOW() - INTERVAL '%s hours'
            AND ($1::text IS NULL OR w.station_id = $1)
            AND ($2::text IS NULL OR s.region = $2)
            AND w.data_quality <= 2
            ORDER BY w.time DESC
            LIMIT $3
        """ % hours
        
        rows = await conn.fetch(query, station_id, region, limit)
        weather_data = [WeatherData(**dict(row)) for row in rows]
        
        await set_cached_data(cache_key, [w.dict() for w in weather_data], ttl=60)
        return weather_data
        
    finally:
        await db_pool.release(conn)

@app.get("/weather/stats")
async def get_weather_stats(
    station_id: Optional[str] = Query(None, description="ID de la station"),
    days: int = Query(30, ge=1, le=365, description="Période en jours")
):
    """Statistiques météorologiques sur une période"""
    
    cache_key = create_cache_key("weather_stats", station_id, days)
    cached_data = await get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    conn = await get_db_connection()
    try:
        query = """
            SELECT 
                COUNT(*) as total_observations,
                AVG(temperature) as avg_temperature,
                MIN(temperature) as min_temperature,
                MAX(temperature) as max_temperature,
                SUM(precipitation) as total_precipitation,
                AVG(precipitation) as avg_precipitation,
                MAX(precipitation) as max_precipitation,
                AVG(humidity) as avg_humidity,
                AVG(pressure) as avg_pressure,
                AVG(wind_speed) as avg_wind_speed
            FROM weather_data
            WHERE time >= NOW() - INTERVAL '%s days'
            AND ($1::text IS NULL OR station_id = $1)
            AND data_quality <= 2
        """ % days
        
        row = await conn.fetchrow(query, station_id)
        stats = dict(row) if row else {}
        
        # Convertir les Decimal en float pour JSON
        for key, value in stats.items():
            if value is not None and hasattr(value, '__float__'):
                stats[key] = float(value)
        
        await set_cached_data(cache_key, stats)
        return stats
        
    finally:
        await db_pool.release(conn)

# ============================================================================
# ENDPOINTS - PRÉDICTIONS
# ============================================================================

@app.post("/predictions/generate", response_model=List[PredictionResponse])
async def generate_predictions(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Génère de nouvelles prédictions"""
    
    conn = await get_db_connection()
    try:
        # Récupération du modèle actif
        model_query = """
            SELECT id, name, model_type, features
            FROM ml_models
            WHERE status = 'active'
            AND ($1::text IS NULL OR name = $1)
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        model = await conn.fetchrow(model_query, request.model_name)
        if not model:
            raise HTTPException(status_code=404, detail="Aucun modèle actif trouvé")
        
        # Simulation de prédiction (à remplacer par votre modèle ML)
        predictions = []
        target_time = datetime.now() + timedelta(hours=request.forecast_hours)
        
        # Si station spécifique
        if request.station_id:
            stations = [request.station_id]
        else:
            # Récupérer les stations de la région
            station_query = """
                SELECT id FROM stations 
                WHERE status = 'active'
                AND ($1::text IS NULL OR region = $1)
            """
            station_rows = await conn.fetch(station_query, request.region)
            stations = [row['id'] for row in station_rows]
        
        # Générer prédictions pour chaque station
        for station_id in stations[:5]:  # Limiter à 5 stations pour la démo
            # Simulation d'une prédiction
            probability = np.random.uniform(0.1, 0.9)
            predicted_class = "normal" if probability < 0.3 else "extreme"
            
            # Insertion de la prédiction
            insert_query = """
                INSERT INTO predictions (
                    model_id, prediction_time, target_time, station_id,
                    predicted_class, probability, confidence_interval,
                    input_features, data_quality_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id, prediction_time, target_time, forecast_horizon_hours
            """
            
            confidence_interval = {
                "lower": max(0, probability - 0.1),
                "upper": min(1, probability + 0.1)
            } if request.include_confidence else None
            
            pred_row = await conn.fetchrow(
                insert_query,
                model['id'],
                datetime.now(),
                target_time,
                station_id,
                predicted_class,
                probability,
                json.dumps(confidence_interval),
                json.dumps({"temperature": 30.5, "humidity": 75.2}),
                0.85
            )
            
            prediction = PredictionResponse(
                id=str(pred_row['id']),
                prediction_time=pred_row['prediction_time'],
                target_time=pred_row['target_time'],
                forecast_horizon_hours=pred_row['forecast_horizon_hours'],
                station_id=station_id,
                region=request.region,
                predicted_class=predicted_class,
                probability=probability,
                confidence_interval=confidence_interval,
                model_name=model['name'],
                data_quality_score=0.85
            )
            
            predictions.append(prediction)
        
        # Tâche en arrière-plan pour notification
        background_tasks.add_task(log_prediction_request, request, len(predictions))
        
        return predictions
        
    finally:
        await db_pool.release(conn)

@app.get("/predictions/latest", response_model=List[PredictionResponse])
async def get_latest_predictions(
    station_id: Optional[str] = Query(None, description="ID de la station"),
    region: Optional[str] = Query(None, description="Région"),
    limit: int = Query(50, ge=1, le=200, description="Limite de résultats")
):
    """Récupère les dernières prédictions"""
    
    cache_key = create_cache_key("predictions_latest", station_id, region, limit)
    cached_data = await get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    conn = await get_db_connection()
    try:
        query = """
            SELECT 
                p.id::text, p.prediction_time, p.target_time, 
                p.forecast_horizon_hours, p.station_id, s.region,
                p.predicted_class, p.probability, p.confidence_interval,
                m.name as model_name, p.data_quality_score
            FROM predictions p
            JOIN ml_models m ON p.model_id = m.id
            LEFT JOIN stations s ON p.station_id = s.id
            WHERE ($1::text IS NULL OR p.station_id = $1)
            AND ($2::text IS NULL OR s.region = $2)
            AND p.prediction_time >= NOW() - INTERVAL '24 hours'
            ORDER BY p.prediction_time DESC
            LIMIT $3
        """
        
        rows = await conn.fetch(query, station_id, region, limit)
        predictions = []
        
        for row in rows:
            row_dict = dict(row)
            # Parse confidence_interval JSON
            if row_dict['confidence_interval']:
                row_dict['confidence_interval'] = json.loads(row_dict['confidence_interval'])
            
            predictions.append(PredictionResponse(**row_dict))
        
        await set_cached_data(cache_key, [p.dict() for p in predictions], ttl=120)
        return predictions
        
    finally:
        await db_pool.release(conn)

# ============================================================================
# ENDPOINTS - ALERTES ET ÉVÉNEMENTS EXTRÊMES
# ============================================================================

@app.get("/alerts/active", response_model=List[AlertInfo])
async def get_active_alerts(
    region: Optional[str] = Query(None, description="Filtrer par région"),
    alert_level: Optional[str] = Query(None, description="Niveau d'alerte")
):
    """Récupère les alertes actives"""
    
    cache_key = create_cache_key("alerts_active", region, alert_level)
    cached_data = await get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    conn = await get_db_connection()
    try:
        query = """
            SELECT 
                id::text, alert_time, event_type, alert_level,
                title, message, region, station_id,
                valid_from, valid_until, confidence_score
            FROM active_alerts
            WHERE ($1::text IS NULL OR region = $1)
            AND ($2::text IS NULL OR alert_level = $2)
            ORDER BY alert_level DESC, alert_time DESC
        """
        
        rows = await conn.fetch(query, region, alert_level)
        alerts = [AlertInfo(**dict(row)) for row in rows]
        
        await set_cached_data(cache_key, [a.dict() for a in alerts], ttl=60)
        return alerts
        
    finally:
        await db_pool.release(conn)

@app.get("/events/recent", response_model=List[ExtremeEventInfo])
async def get_recent_extreme_events(
    days: int = Query(30, ge=1, le=365, description="Période en jours"),
    event_type: Optional[str] = Query(None, description="Type d'événement"),
    region: Optional[str] = Query(None, description="Région")
):
    """Récupère les événements extrêmes récents"""
    
    cache_key = create_cache_key("events_recent", days, event_type, region)
    cached_data = await get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    conn = await get_db_connection()
    try:
        query = """
            SELECT 
                id::text, event_time, event_type, event_name,
                severity_level, intensity, duration_hours,
                region, confidence_score
            FROM recent_extremes
            WHERE event_time >= NOW() - INTERVAL '%s days'
            AND ($1::text IS NULL OR event_type = $1)
            AND ($2::text IS NULL OR region = $2)
            ORDER BY event_time DESC
        """ % days
        
        rows = await conn.fetch(query, event_type, region)
        events = [ExtremeEventInfo(**dict(row)) for row in rows]
        
        await set_cached_data(cache_key, [e.dict() for e in events])
        return events
        
    finally:
        await db_pool.release(conn)

# ============================================================================
# ENDPOINTS - MONITORING ET MÉTRIQUES
# ============================================================================

@app.get("/metrics")
async def get_metrics():
    """Métriques de l'API pour monitoring"""
    
    conn = await get_db_connection()
    try:
        # Statistiques générales
        stats_query = """
            SELECT 
                (SELECT COUNT(*) FROM stations WHERE status = 'active') as active_stations,
                (SELECT COUNT(*) FROM weather_data WHERE time >= NOW() - INTERVAL '24 hours') as recent_observations,
                (SELECT COUNT(*) FROM predictions WHERE prediction_time >= NOW() - INTERVAL '24 hours') as recent_predictions,
                (SELECT COUNT(*) FROM alerts WHERE status = 'active') as active_alerts,
                (SELECT COUNT(*) FROM extreme_events WHERE event_time >= NOW() - INTERVAL '30 days') as recent_events
        """
        
        stats = await conn.fetchrow(stats_query)
        
        # Métriques de performance (simulées)
        metrics = {
            "system": {
                "active_stations": stats['active_stations'],
                "recent_observations_24h": stats['recent_observations'],
                "recent_predictions_24h": stats['recent_predictions'],
                "active_alerts": stats['active_alerts'],
                "recent_events_30d": stats['recent_events']
            },
            "performance": {
                "avg_response_time_ms": 150,
                "cache_hit_rate": 0.85,
                "prediction_accuracy": 0.87,
                "data_quality_score": 0.92
            },
            "timestamp": datetime.now()
        }
        
        return metrics
        
    finally:
        await db_pool.release(conn)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

async def log_prediction_request(request: PredictionRequest, count: int):
    """Log des requêtes de prédiction en arrière-plan"""
    logger.info(f"Prédiction générée: {count} résultats pour {request.station_id or request.region}")

# ============================================================================
# GESTION DES ERREURS
# ============================================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Gestionnaire d'erreurs global"""
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur", "type": "internal_error"}
    )

# ============================================================================
# LANCEMENT DE L'APPLICATION
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=API_DEBUG,
        log_level="info"
    )