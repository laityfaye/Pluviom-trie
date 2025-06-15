# Dockerfile pour Pipeline ML - Projet Climat Sénégal
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
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    curl \
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
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
  CMD python quick_test.py || exit 1

# Point d'entrée par défaut
CMD ["python", "main.py", "--only-ml"]
