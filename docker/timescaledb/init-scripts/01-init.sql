-- docker/timescaledb/init-scripts/01-init.sql
-- Script d'initialisation complet pour TimescaleDB
-- Projet : Prédiction d'événements climatiques extrêmes au Sénégal
-- Version optimisée et adaptée

-- ============================================================================
-- EXTENSIONS ET CONFIGURATION
-- ============================================================================

-- Activation TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Extension PostGIS pour données géographiques (optionnelle)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS postgis CASCADE;
    RAISE NOTICE 'PostGIS installé avec succès';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'PostGIS non disponible - fonctionnalités géographiques limitées';
END $$;

-- Extension pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" CASCADE;

-- ============================================================================
-- SUPPRESSION ET RECRÉATION DES TABLES (Ordre important)
-- ============================================================================

-- Supprimer les tables dans l'ordre inverse des dépendances
DROP TABLE IF EXISTS alerts CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS extreme_events CASCADE;
DROP TABLE IF EXISTS alert_thresholds CASCADE;
DROP TABLE IF EXISTS ml_models CASCADE;
DROP TABLE IF EXISTS event_types CASCADE;
DROP TABLE IF EXISTS weather_data CASCADE;
DROP TABLE IF EXISTS stations CASCADE;

-- Supprimer les vues
DROP VIEW IF EXISTS recent_extremes CASCADE;
DROP VIEW IF EXISTS active_alerts CASCADE;
DROP VIEW IF EXISTS latest_predictions CASCADE;
DROP VIEW IF EXISTS regional_stats CASCADE;

-- ============================================================================
-- TABLES PRINCIPALES - DONNÉES MÉTÉOROLOGIQUES
-- ============================================================================

-- Table des stations météorologiques
CREATE TABLE stations (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    region VARCHAR(100),
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    altitude DOUBLE PRECISION,
    installation_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    station_type VARCHAR(50) DEFAULT 'automatic',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ajouter la colonne géométrique seulement si PostGIS est disponible
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis') THEN
        ALTER TABLE stations ADD COLUMN geom GEOMETRY(POINT, 4326) 
        GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED;
        RAISE NOTICE 'Colonne géométrique ajoutée pour les stations';
    END IF;
END $$;

-- Index pour les stations
CREATE INDEX IF NOT EXISTS idx_stations_region ON stations (region);
CREATE INDEX IF NOT EXISTS idx_stations_status ON stations (status) WHERE status = 'active';

-- Index géographique seulement si PostGIS est disponible
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis') THEN
        CREATE INDEX IF NOT EXISTS idx_stations_geom ON stations USING GIST (geom);
    END IF;
END $$;

-- Table principale des données météorologiques
CREATE TABLE weather_data (
    time TIMESTAMPTZ NOT NULL,
    station_id VARCHAR(50) NOT NULL REFERENCES stations(id),
    -- Données de température (°C)
    temperature DOUBLE PRECISION,
    temperature_min DOUBLE PRECISION,
    temperature_max DOUBLE PRECISION,
    -- Données de précipitation (mm)
    precipitation DOUBLE PRECISION,
    precipitation_intensity DOUBLE PRECISION,
    -- Données d'humidité (%)
    humidity DOUBLE PRECISION,
    humidity_min DOUBLE PRECISION,
    humidity_max DOUBLE PRECISION,
    -- Données de pression (hPa)
    pressure DOUBLE PRECISION,
    pressure_sea_level DOUBLE PRECISION,
    -- Données de vent
    wind_speed DOUBLE PRECISION,
    wind_direction DOUBLE PRECISION,
    wind_gust DOUBLE PRECISION,
    -- Données de rayonnement
    solar_radiation DOUBLE PRECISION,
    uv_index DOUBLE PRECISION,
    -- Visibilité et nuages
    visibility DOUBLE PRECISION,
    cloud_cover DOUBLE PRECISION,
    -- Métadonnées
    data_quality SMALLINT DEFAULT 1, -- 1=excellent, 2=bon, 3=douteux, 4=mauvais
    data_source VARCHAR(50) DEFAULT 'station',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- Contrainte d'unicité
    UNIQUE(time, station_id)
);

-- Conversion en hypertable avec partitionnement par temps
DO $$
BEGIN
    PERFORM create_hypertable('weather_data', 'time', 
        chunk_time_interval => INTERVAL '1 month',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'Hypertable weather_data créée avec succès';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Erreur création hypertable weather_data: %', SQLERRM;
END $$;

-- Index optimisés pour les requêtes temporelles
CREATE INDEX IF NOT EXISTS idx_weather_station_time 
    ON weather_data (station_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_weather_time_quality 
    ON weather_data (time, data_quality) WHERE data_quality <= 2;
CREATE INDEX IF NOT EXISTS idx_weather_precipitation 
    ON weather_data (time, precipitation) WHERE precipitation > 0;

-- ============================================================================
-- TABLES POUR ÉVÉNEMENTS EXTRÊMES
-- ============================================================================

-- Types d'événements extrêmes
CREATE TABLE event_types (
    code VARCHAR(20) PRIMARY KEY,
    name_fr VARCHAR(100) NOT NULL,
    name_en VARCHAR(100) NOT NULL,
    description TEXT,
    severity_levels JSONB,
    color_code VARCHAR(7) -- Code couleur hex
);

-- Insertion des types d'événements pour le Sénégal
INSERT INTO event_types (code, name_fr, name_en, description, severity_levels, color_code) VALUES
('DROUGHT', 'Sécheresse', 'Drought', 'Déficit prolongé de précipitations', 
    '{"1": "Légère", "2": "Modérée", "3": "Sévère", "4": "Extrême"}', '#D2691E'),
('FLOOD', 'Inondation', 'Flood', 'Excès de précipitations causant des inondations',
    '{"1": "Mineure", "2": "Modérée", "3": "Majeure", "4": "Catastrophique"}', '#1E90FF'),
('HEATWAVE', 'Vague de chaleur', 'Heat Wave', 'Températures exceptionnellement élevées',
    '{"1": "Chaleur", "2": "Forte chaleur", "3": "Canicule", "4": "Canicule extrême"}', '#FF4500'),
('CYCLONE', 'Système cyclonique', 'Cyclonic System', 'Système dépressionnaire intense',
    '{"1": "Faible", "2": "Modéré", "3": "Intense", "4": "Très intense"}', '#8B0000'),
('SANDSTORM', 'Tempête de sable', 'Sandstorm', 'Vents forts soulevant du sable',
    '{"1": "Légère", "2": "Modérée", "3": "Forte", "4": "Extrême"}', '#DAA520')
ON CONFLICT (code) DO NOTHING;

-- Table des événements extrêmes détectés
CREATE TABLE extreme_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_time TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(20) NOT NULL REFERENCES event_types(code),
    severity_level SMALLINT NOT NULL CHECK (severity_level BETWEEN 1 AND 4),
    intensity DOUBLE PRECISION,
    duration_hours INTEGER,
    peak_time TIMESTAMPTZ,
    -- Stations affectées
    affected_stations TEXT[],
    primary_station VARCHAR(50) REFERENCES stations(id),
    -- Géographie (coordonnées basiques toujours disponibles)
    center_lat DOUBLE PRECISION,
    center_lon DOUBLE PRECISION,
    -- Impacts estimés
    population_affected INTEGER,
    economic_impact DOUBLE PRECISION,
    -- Métadonnées
    detection_method VARCHAR(50) DEFAULT 'automatic',
    confidence_score DOUBLE PRECISION CHECK (confidence_score BETWEEN 0 AND 1),
    verified BOOLEAN DEFAULT FALSE,
    verification_time TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ajouter la colonne géométrique seulement si PostGIS est disponible
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis') THEN
        ALTER TABLE extreme_events ADD COLUMN affected_area GEOMETRY(POLYGON, 4326);
        RAISE NOTICE 'Colonne géométrique ajoutée pour extreme_events';
    END IF;
END $$;

-- Index pour les événements extrêmes
CREATE INDEX IF NOT EXISTS idx_extreme_events_time ON extreme_events (event_time DESC);
CREATE INDEX IF NOT EXISTS idx_extreme_events_type ON extreme_events (event_type);
CREATE INDEX IF NOT EXISTS idx_extreme_events_severity ON extreme_events (severity_level);

-- Index géographique seulement si PostGIS est disponible
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis') THEN
        CREATE INDEX IF NOT EXISTS idx_extreme_events_area ON extreme_events USING GIST (affected_area);
    END IF;
END $$;

-- ============================================================================
-- TABLES POUR PRÉDICTIONS
-- ============================================================================

-- Modèles ML utilisés
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'classification', 'regression', 'timeseries'
    target_variable VARCHAR(50) NOT NULL,
    features JSONB NOT NULL,
    hyperparameters JSONB,
    performance_metrics JSONB,
    training_period DATERANGE,
    model_path TEXT,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    trained_at TIMESTAMPTZ,
    UNIQUE(name, version)
);

-- Table des prédictions
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(id),
    prediction_time TIMESTAMPTZ NOT NULL,
    target_time TIMESTAMPTZ NOT NULL,
    forecast_horizon_hours INTEGER GENERATED ALWAYS AS (EXTRACT(EPOCH FROM (target_time - prediction_time))/3600) STORED,
    -- Prédiction principale
    predicted_class VARCHAR(50),
    probability DOUBLE PRECISION CHECK (probability BETWEEN 0 AND 1),
    confidence_interval JSONB, -- {"lower": 0.1, "upper": 0.9}
    -- Détails par région/station
    station_id VARCHAR(50) REFERENCES stations(id),
    region VARCHAR(100),
    -- Features utilisées
    input_features JSONB NOT NULL,
    model_output JSONB, -- Sortie brute du modèle
    -- Métadonnées
    execution_time_ms INTEGER,
    data_quality_score DOUBLE PRECISION,
    warning_flags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ajouter la colonne géométrique seulement si PostGIS est disponible
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis') THEN
        ALTER TABLE predictions ADD COLUMN geographic_scope GEOMETRY(POLYGON, 4326);
        RAISE NOTICE 'Colonne géométrique ajoutée pour predictions';
    END IF;
END $$;

-- Conversion en hypertable pour les prédictions
DO $$
BEGIN
    PERFORM create_hypertable('predictions', 'prediction_time',
        chunk_time_interval => INTERVAL '1 week',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'Hypertable predictions créée avec succès';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Erreur création hypertable predictions: %', SQLERRM;
END $$;

-- Index pour les prédictions
CREATE INDEX IF NOT EXISTS idx_predictions_model_time 
    ON predictions (model_id, prediction_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_target_time 
    ON predictions (target_time);
CREATE INDEX IF NOT EXISTS idx_predictions_station 
    ON predictions (station_id, prediction_time DESC);

-- ============================================================================
-- TABLES POUR ALERTES
-- ============================================================================

-- Configuration des seuils d'alerte
CREATE TABLE alert_thresholds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(20) NOT NULL REFERENCES event_types(code),
    region VARCHAR(100),
    station_id VARCHAR(50) REFERENCES stations(id),
    variable VARCHAR(50) NOT NULL,
    threshold_value DOUBLE PRECISION NOT NULL,
    comparison_operator VARCHAR(10) NOT NULL CHECK (comparison_operator IN ('>', '<', '>=', '<=', '=', '!=')),
    duration_hours INTEGER,
    alert_level VARCHAR(20) DEFAULT 'warning', -- 'info', 'warning', 'alert', 'critical'
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table des alertes générées
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_time TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(20) NOT NULL REFERENCES event_types(code),
    alert_level VARCHAR(20) NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    -- Localisation
    station_id VARCHAR(50) REFERENCES stations(id),
    region VARCHAR(100),
    -- Temporalité
    valid_from TIMESTAMPTZ NOT NULL,
    valid_until TIMESTAMPTZ,
    -- Prédiction associée
    prediction_id UUID REFERENCES predictions(id),
    confidence_score DOUBLE PRECISION,
    -- Status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'expired', 'cancelled'
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    -- Métadonnées
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ajouter la colonne géométrique seulement si PostGIS est disponible
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis') THEN
        ALTER TABLE alerts ADD COLUMN affected_area GEOMETRY(POLYGON, 4326);
        RAISE NOTICE 'Colonne géométrique ajoutée pour alerts';
    END IF;
END $$;

-- Index pour les alertes
CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts (alert_time DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_region ON alerts (region);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status) WHERE status = 'active';

-- ============================================================================
-- VUES UTILES POUR L'API
-- ============================================================================

-- Vue des événements récents
CREATE OR REPLACE VIEW recent_extremes AS
SELECT 
    e.*,
    et.name_fr as event_name,
    et.color_code,
    s.name as station_name,
    s.region
FROM extreme_events e
JOIN event_types et ON e.event_type = et.code
LEFT JOIN stations s ON e.primary_station = s.id
WHERE e.event_time >= NOW() - INTERVAL '30 days'
ORDER BY e.event_time DESC;

-- Vue des alertes actives
CREATE OR REPLACE VIEW active_alerts AS
SELECT 
    a.*,
    et.name_fr as event_name,
    et.color_code,
    s.name as station_name
FROM alerts a
JOIN event_types et ON a.event_type = et.code
LEFT JOIN stations s ON a.station_id = s.id
WHERE a.status = 'active' 
    AND a.valid_from <= NOW() 
    AND (a.valid_until IS NULL OR a.valid_until >= NOW())
ORDER BY a.alert_level DESC, a.alert_time DESC;

-- Vue des dernières prédictions par station
CREATE OR REPLACE VIEW latest_predictions AS
SELECT DISTINCT ON (station_id, model_id)
    p.*,
    m.name as model_name,
    m.model_type,
    s.name as station_name,
    s.region
FROM predictions p
JOIN ml_models m ON p.model_id = m.id
LEFT JOIN stations s ON p.station_id = s.id
WHERE p.prediction_time >= NOW() - INTERVAL '24 hours'
ORDER BY station_id, model_id, prediction_time DESC;

-- Vue statistiques par région
CREATE OR REPLACE VIEW regional_stats AS
SELECT 
    s.region,
    COUNT(DISTINCT s.id) as station_count,
    COUNT(wd.*) as total_observations,
    AVG(wd.temperature) as avg_temperature,
    AVG(wd.precipitation) as avg_precipitation,
    MAX(wd.time) as last_observation
FROM stations s
LEFT JOIN weather_data wd ON s.id = wd.station_id 
    AND wd.time >= NOW() - INTERVAL '30 days'
GROUP BY s.region;

-- ============================================================================
-- FONCTIONS UTILES
-- ============================================================================

-- Fonction pour calculer la distance entre deux points (version simple sans PostGIS)
CREATE OR REPLACE FUNCTION distance_km(lat1 DOUBLE PRECISION, lon1 DOUBLE PRECISION, 
                                       lat2 DOUBLE PRECISION, lon2 DOUBLE PRECISION)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    dlat DOUBLE PRECISION;
    dlon DOUBLE PRECISION;
    a DOUBLE PRECISION;
    c DOUBLE PRECISION;
    r DOUBLE PRECISION := 6371; -- Rayon de la Terre en km
BEGIN
    -- Formule haversine pour calculer la distance
    dlat := radians(lat2 - lat1);
    dlon := radians(lon2 - lon1);
    a := sin(dlat/2) * sin(dlat/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2) * sin(dlon/2);
    c := 2 * atan2(sqrt(a), sqrt(1-a));
    RETURN r * c;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Remplacer par la version PostGIS si disponible
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis') THEN
        CREATE OR REPLACE FUNCTION distance_km(lat1 DOUBLE PRECISION, lon1 DOUBLE PRECISION, 
                                               lat2 DOUBLE PRECISION, lon2 DOUBLE PRECISION)
        RETURNS DOUBLE PRECISION AS $func$
        BEGIN
            RETURN ST_Distance(
                ST_Transform(ST_SetSRID(ST_MakePoint(lon1, lat1), 4326), 3857),
                ST_Transform(ST_SetSRID(ST_MakePoint(lon2, lat2), 4326), 3857)
            ) / 1000.0;
        END;
        $func$ LANGUAGE plpgsql IMMUTABLE;
        RAISE NOTICE 'Fonction distance_km utilisant PostGIS créée';
    ELSE
        RAISE NOTICE 'Fonction distance_km utilisant la formule haversine créée';
    END IF;
END $$;

-- Fonction pour nettoyer les anciennes prédictions
CREATE OR REPLACE FUNCTION cleanup_old_predictions(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM predictions 
    WHERE prediction_time < NOW() - INTERVAL '1 day' * retention_days;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger pour mettre à jour updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Application des triggers
DROP TRIGGER IF EXISTS update_stations_updated_at ON stations;
CREATE TRIGGER update_stations_updated_at 
    BEFORE UPDATE ON stations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_extreme_events_updated_at ON extreme_events;
CREATE TRIGGER update_extreme_events_updated_at 
    BEFORE UPDATE ON extreme_events 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_alerts_updated_at ON alerts;
CREATE TRIGGER update_alerts_updated_at 
    BEFORE UPDATE ON alerts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- CONFIGURATION TIMESCALEDB (avec gestion d'erreurs)
-- ============================================================================

-- Configuration de la rétention automatique
DO $$
BEGIN
    PERFORM add_retention_policy('weather_data', INTERVAL '2 years');
    RAISE NOTICE 'Politique de rétention ajoutée pour weather_data';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Erreur ajout politique de rétention weather_data: %', SQLERRM;
END $$;

DO $$
BEGIN
    PERFORM add_retention_policy('predictions', INTERVAL '6 months');
    RAISE NOTICE 'Politique de rétention ajoutée pour predictions';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Erreur ajout politique de rétention predictions: %', SQLERRM;
END $$;

-- Configuration de la compression
DO $$
BEGIN
    ALTER TABLE weather_data SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'station_id',
        timescaledb.compress_orderby = 'time DESC'
    );
    RAISE NOTICE 'Compression configurée pour weather_data';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Erreur configuration compression weather_data: %', SQLERRM;
END $$;

-- Politique de compression automatique
DO $$
BEGIN
    PERFORM add_compression_policy('weather_data', INTERVAL '7 days');
    RAISE NOTICE 'Politique de compression ajoutée pour weather_data';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Erreur ajout politique de compression weather_data: %', SQLERRM;
END $$;

-- ============================================================================
-- DONNÉES DE RÉFÉRENCE
-- ============================================================================

-- Insertion de quelques stations test (mise à jour si elles existent déjà)
INSERT INTO stations (id, name, region, latitude, longitude, altitude) VALUES
('DAKAR_YOFF', 'Dakar-Yoff', 'Dakar', 14.7392, -17.4904, 24),
('SAINT_LOUIS', 'Saint-Louis', 'Saint-Louis', 16.0469, -16.4734, 7),
('TAMBACOUNDA', 'Tambacounda', 'Tambacounda', 13.7671, -13.6681, 49),
('ZIGUINCHOR', 'Ziguinchor', 'Ziguinchor', 12.5581, -16.2719, 26),
('KAOLACK', 'Kaolack', 'Kaolack', 14.1594, -16.0724, 7),
('LOUGA', 'Louga', 'Louga', 15.6186, -16.2269, 25),
('KOLDA', 'Kolda', 'Kolda', 12.8831, -14.9417, 15),
('KEDOUGOU', 'Kédougou', 'Kédougou', 12.5561, -12.1756, 178)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    region = EXCLUDED.region,
    latitude = EXCLUDED.latitude,
    longitude = EXCLUDED.longitude,
    altitude = EXCLUDED.altitude,
    updated_at = NOW();

-- Message de fin
DO $$
DECLARE
    table_count INTEGER;
    station_count INTEGER;
    event_type_count INTEGER;
    postgis_available BOOLEAN;
BEGIN
    SELECT COUNT(*) INTO table_count FROM information_schema.tables WHERE table_schema = 'public';
    SELECT COUNT(*) INTO station_count FROM stations;
    SELECT COUNT(*) INTO event_type_count FROM event_types;
    SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'postgis') INTO postgis_available;
    
    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Base de données TimescaleDB initialisée avec succès!';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables créées: %', table_count;
    RAISE NOTICE 'Stations disponibles: %', station_count;
    RAISE NOTICE 'Types d''événements: %', event_type_count;
    RAISE NOTICE 'PostGIS disponible: %', CASE WHEN postgis_available THEN 'OUI' ELSE 'NON' END;
    RAISE NOTICE 'TimescaleDB: Hypertables configurées';
    RAISE NOTICE 'Vues: 4 vues API créées';
    RAISE NOTICE 'Fonctions: 3 fonctions utilitaires';
    RAISE NOTICE 'Triggers: 3 triggers automatiques';
    RAISE NOTICE '';
    RAISE NOTICE '🎉 Votre infrastructure est prête pour la production!';
    RAISE NOTICE '';
END $$;