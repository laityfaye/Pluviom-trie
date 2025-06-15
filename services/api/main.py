"""
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
