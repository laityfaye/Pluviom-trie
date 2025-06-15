# Dockerfile pour API - Projet Climat Sénégal
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dépendances système légères
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
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
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Variables d'environnement
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MODEL_PATH=/app/models

# Commande de démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
