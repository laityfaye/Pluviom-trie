# docker/timescaledb/Dockerfile
FROM timescale/timescaledb:latest-pg14

# Métadonnées
LABEL maintainer="Équipe Climat Sénégal"
LABEL description="TimescaleDB optimisé pour données climatiques"
LABEL version="1.0.0"

# Installation d'extensions supplémentaires
RUN apt-get update && apt-get install -y \
    postgresql-14-postgis-3 \
    postgresql-14-postgis-3-scripts \
    postgresql-contrib-14 \
    && rm -rf /var/lib/apt/lists/*

# Configuration par défaut
ENV POSTGRES_DB=climatsn_db \
    POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=rout \
    PGDATA=/var/lib/postgresql/data/pgdata

# Copie des scripts d'initialisation
COPY docker/timescaledb/init-scripts/ /docker-entrypoint-initdb.d/

# Copie de la configuration PostgreSQL
COPY docker/timescaledb/postgresql.conf /etc/postgresql/postgresql.conf

# Script de démarrage personnalisé
COPY docker/timescaledb/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Volumes pour la persistance
VOLUME ["/var/lib/postgresql/data", "/backup"]

# Port exposé
EXPOSE 5432

# Commande de démarrage
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]