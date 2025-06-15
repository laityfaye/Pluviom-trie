-- docker/timescaledb/init-scripts/01-init.sql

-- Création de l'extension TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Création des tables principales
CREATE TABLE IF NOT EXISTS weather_data (
    time TIMESTAMPTZ NOT NULL,
    station_id VARCHAR(50) NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    wind_direction DOUBLE PRECISION
);

-- Conversion en hypertable
SELECT create_hypertable('weather_data', 'time');

-- Création des index
CREATE INDEX IF NOT EXISTS idx_weather_station_time 
ON weather_data (station_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_weather_location 
ON weather_data (latitude, longitude);

-- Table des événements extrêmes
CREATE TABLE IF NOT EXISTS extreme_events (
    id SERIAL PRIMARY KEY,
    event_time TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    intensity DOUBLE PRECISION,
    duration_hours INTEGER,
    affected_stations TEXT[],
    geom GEOMETRY(POLYGON, 4326)
);

-- Table des prédictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    prediction_time TIMESTAMPTZ NOT NULL,
    target_time TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(100),
    probability DOUBLE PRECISION,
    confidence_interval JSONB,
    features JSONB
);

-- Création des vues pour l'API
CREATE OR REPLACE VIEW recent_extremes AS
SELECT * FROM extreme_events 
WHERE event_time >= NOW() - INTERVAL '30 days'
ORDER BY event_time DESC;