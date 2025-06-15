# docker/timescaledb/Dockerfile
FROM timescale/timescaledb:latest-pg14

# Variables d'environnement
ENV POSTGRES_DB=climate_db
ENV POSTGRES_USER=climate_user
ENV POSTGRES_PASSWORD=secure_password

# Scripts d'initialisation
COPY init-scripts/ /docker-entrypoint-initdb.d/

# Configuration PostgreSQL optimisée
COPY postgresql.conf /var/lib/postgresql/data/postgresql.conf