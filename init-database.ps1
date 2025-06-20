# init-database.ps1
# Initialisation de la base de donnees TimescaleDB - VERSION COMPLETE SENEGAL
# Inclut toutes les 14 regions administratives du Senegal

Write-Host "INITIALISATION BASE DE DONNEES TIMESCALEDB - SENEGAL COMPLET" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan

# 1. Verification de l'etat de TimescaleDB
Write-Host "`n1. Verification de TimescaleDB..." -ForegroundColor Yellow

try {
    $result = docker-compose exec -T timescaledb psql -U postgres -d climatsn_db -c "SELECT version();"
    Write-Host "[OK] TimescaleDB accessible" -ForegroundColor Green
} catch {
    Write-Host "[ERREUR] TimescaleDB inaccessible" -ForegroundColor Red
    exit 1
}

# 2. Verification des tables existantes
Write-Host "`n2. Verification des tables existantes..." -ForegroundColor Yellow

$tables = docker-compose exec -T timescaledb psql -U postgres -d climatsn_db -c "\dt" 2>$null
if ($tables -match "stations") {
    Write-Host "[INFO] Tables deja presentes" -ForegroundColor Yellow
    $response = Read-Host "Voulez-vous reinitialiser la base ? (y/N)"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Operation annulee - Tables conservees" -ForegroundColor Yellow
        exit 0
    }
} else {
    Write-Host "[INFO] Tables manquantes - Creation necessaire" -ForegroundColor Yellow
}

# 3. Creation du schema complet
Write-Host "`n3. Creation du schema TimescaleDB complet..." -ForegroundColor Yellow

$sqlSchema = @'
-- Creation de l'extension TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Extension PostGIS pour donnees geographiques (optionnelle)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS postgis CASCADE;
    RAISE NOTICE 'PostGIS installe avec succes';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'PostGIS non disponible - fonctionnalites geographiques limitees';
END $$;

-- Extension pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" CASCADE;

-- Suppression des tables existantes si reinitialisation
DROP TABLE IF EXISTS alerts CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS extreme_events CASCADE;
DROP TABLE IF EXISTS alert_thresholds CASCADE;
DROP TABLE IF EXISTS ml_models CASCADE;
DROP TABLE IF EXISTS event_types CASCADE;
DROP TABLE IF EXISTS weather_data CASCADE;
DROP TABLE IF EXISTS stations CASCADE;

-- Suppression des vues
DROP VIEW IF EXISTS recent_extremes CASCADE;
DROP VIEW IF EXISTS active_alerts CASCADE;
DROP VIEW IF EXISTS latest_predictions CASCADE;
DROP VIEW IF EXISTS regional_stats CASCADE;

-- Table des stations meteorologiques
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

-- Index pour les stations
CREATE INDEX IF NOT EXISTS idx_stations_region ON stations (region);
CREATE INDEX IF NOT EXISTS idx_stations_status ON stations (status) WHERE status = 'active';

-- Table principale des donnees meteorologiques
CREATE TABLE weather_data (
    time TIMESTAMPTZ NOT NULL,
    station_id VARCHAR(50) NOT NULL REFERENCES stations(id),
    temperature DOUBLE PRECISION,
    temperature_min DOUBLE PRECISION,
    temperature_max DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    precipitation_intensity DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    humidity_min DOUBLE PRECISION,
    humidity_max DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    pressure_sea_level DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    wind_direction DOUBLE PRECISION,
    wind_gust DOUBLE PRECISION,
    solar_radiation DOUBLE PRECISION,
    uv_index DOUBLE PRECISION,
    visibility DOUBLE PRECISION,
    cloud_cover DOUBLE PRECISION,
    data_quality SMALLINT DEFAULT 1,
    data_source VARCHAR(50) DEFAULT 'station',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(time, station_id)
);

-- Conversion en hypertable avec partitionnement par temps
DO $$
BEGIN
    PERFORM create_hypertable('weather_data', 'time', 
        chunk_time_interval => INTERVAL '1 month',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'Hypertable weather_data creee avec succes';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Erreur creation hypertable weather_data: %', SQLERRM;
END $$;

-- Index optimises pour les requetes temporelles
CREATE INDEX IF NOT EXISTS idx_weather_station_time 
    ON weather_data (station_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_weather_time_quality 
    ON weather_data (time, data_quality) WHERE data_quality <= 2;
CREATE INDEX IF NOT EXISTS idx_weather_precipitation 
    ON weather_data (time, precipitation) WHERE precipitation > 0;

-- Types d'evenements extremes
CREATE TABLE event_types (
    code VARCHAR(20) PRIMARY KEY,
    name_fr VARCHAR(100) NOT NULL,
    name_en VARCHAR(100) NOT NULL,
    description TEXT,
    severity_levels JSONB,
    color_code VARCHAR(7)
);

-- Insertion des types d'evenements pour le Senegal
INSERT INTO event_types (code, name_fr, name_en, description, severity_levels, color_code) VALUES
('DROUGHT', 'Secheresse', 'Drought', 'Deficit prolonge de precipitations', 
    '{"1": "Legere", "2": "Moderee", "3": "Severe", "4": "Extreme"}', '#D2691E'),
('FLOOD', 'Inondation', 'Flood', 'Exces de precipitations causant des inondations',
    '{"1": "Mineure", "2": "Moderee", "3": "Majeure", "4": "Catastrophique"}', '#1E90FF'),
('HEATWAVE', 'Vague de chaleur', 'Heat Wave', 'Temperatures exceptionnellement elevees',
    '{"1": "Chaleur", "2": "Forte chaleur", "3": "Canicule", "4": "Canicule extreme"}', '#FF4500'),
('CYCLONE', 'Systeme cyclonique', 'Cyclonic System', 'Systeme depressionnaire intense',
    '{"1": "Faible", "2": "Modere", "3": "Intense", "4": "Tres intense"}', '#8B0000'),
('SANDSTORM', 'Tempete de sable', 'Sandstorm', 'Vents forts soulevant du sable',
    '{"1": "Legere", "2": "Moderee", "3": "Forte", "4": "Extreme"}', '#DAA520')
ON CONFLICT (code) DO NOTHING;

-- Table des evenements extremes detectes
CREATE TABLE extreme_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_time TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(20) NOT NULL REFERENCES event_types(code),
    severity_level SMALLINT NOT NULL CHECK (severity_level BETWEEN 1 AND 4),
    intensity DOUBLE PRECISION,
    duration_hours INTEGER,
    peak_time TIMESTAMPTZ,
    affected_stations TEXT[],
    primary_station VARCHAR(50) REFERENCES stations(id),
    center_lat DOUBLE PRECISION,
    center_lon DOUBLE PRECISION,
    population_affected INTEGER,
    economic_impact DOUBLE PRECISION,
    detection_method VARCHAR(50) DEFAULT 'automatic',
    confidence_score DOUBLE PRECISION CHECK (confidence_score BETWEEN 0 AND 1),
    verified BOOLEAN DEFAULT FALSE,
    verification_time TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index pour les evenements extremes
CREATE INDEX IF NOT EXISTS idx_extreme_events_time ON extreme_events (event_time DESC);
CREATE INDEX IF NOT EXISTS idx_extreme_events_type ON extreme_events (event_type);
CREATE INDEX IF NOT EXISTS idx_extreme_events_severity ON extreme_events (severity_level);

-- Modeles ML utilises
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
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

-- Table des predictions
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(id),
    prediction_time TIMESTAMPTZ NOT NULL,
    target_time TIMESTAMPTZ NOT NULL,
    forecast_horizon_hours INTEGER GENERATED ALWAYS AS (EXTRACT(EPOCH FROM (target_time - prediction_time))/3600) STORED,
    predicted_class VARCHAR(50),
    probability DOUBLE PRECISION CHECK (probability BETWEEN 0 AND 1),
    confidence_interval JSONB,
    station_id VARCHAR(50) REFERENCES stations(id),
    region VARCHAR(100),
    input_features JSONB NOT NULL,
    model_output JSONB,
    execution_time_ms INTEGER,
    data_quality_score DOUBLE PRECISION,
    warning_flags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Conversion en hypertable pour les predictions
DO $$
BEGIN
    PERFORM create_hypertable('predictions', 'prediction_time',
        chunk_time_interval => INTERVAL '1 week',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'Hypertable predictions creee avec succes';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Erreur creation hypertable predictions: %', SQLERRM;
END $$;

-- Index pour les predictions
CREATE INDEX IF NOT EXISTS idx_predictions_model_time 
    ON predictions (model_id, prediction_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_target_time 
    ON predictions (target_time);
CREATE INDEX IF NOT EXISTS idx_predictions_station 
    ON predictions (station_id, prediction_time DESC);

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
    alert_level VARCHAR(20) DEFAULT 'warning',
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table des alertes generees
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_time TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(20) NOT NULL REFERENCES event_types(code),
    alert_level VARCHAR(20) NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    station_id VARCHAR(50) REFERENCES stations(id),
    region VARCHAR(100),
    valid_from TIMESTAMPTZ NOT NULL,
    valid_until TIMESTAMPTZ,
    prediction_id UUID REFERENCES predictions(id),
    confidence_score DOUBLE PRECISION,
    status VARCHAR(20) DEFAULT 'active',
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index pour les alertes
CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts (alert_time DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_region ON alerts (region);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status) WHERE status = 'active';

-- Vue des evenements recents
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

-- Vue des dernieres predictions par station
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

-- Vue statistiques par region
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

-- INSERTION DES STATIONS METEOROLOGIQUES COMPLETES DU SENEGAL
-- Couverture nationale complete : 14 regions administratives

-- REGION DE DAKAR (Capitale)
INSERT INTO stations (id, name, region, latitude, longitude, altitude) VALUES
('DAKAR_YOFF', 'Dakar-Yoff', 'Dakar', 14.7392, -17.4904, 24),
('DAKAR_LEOPOLD', 'Dakar-Leopold Sedar Senghor', 'Dakar', 14.6928, -17.4467, 35),

-- REGION DE THIES
('THIES_VILLE', 'Thies', 'Thies', 14.7886, -16.9234, 70),
('MBOUR', 'Mbour', 'Thies', 14.4198, -16.9692, 15),
('TIVAOUANE', 'Tivaouane', 'Thies', 14.9500, -16.8167, 45),

-- REGION DE DIOURBEL  
('DIOURBEL', 'Diourbel', 'Diourbel', 14.6522, -16.2297, 15),
('TOUBA', 'Touba', 'Diourbel', 14.8500, -15.8833, 30),
('MBACKE', 'Mbacke', 'Diourbel', 14.7833, -15.9167, 25),

-- REGION DE FATICK
('FATICK', 'Fatick', 'Fatick', 14.3347, -16.4045, 8),
('FOUNDIOUGNE', 'Foundiougne', 'Fatick', 14.1333, -16.4667, 5),

-- REGION DE KAOLACK
('KAOLACK', 'Kaolack', 'Kaolack', 14.1594, -16.0724, 7),
('NIORO_SAHEL', 'Nioro du Sahel', 'Kaolack', 13.7500, -15.7833, 25),

-- REGION DE LOUGA
('LOUGA', 'Louga', 'Louga', 15.6186, -16.2269, 25),
('LINGUERE', 'Linguere', 'Louga', 15.3833, -15.1167, 60),
('KEBEMER', 'Kebemer', 'Louga', 15.3667, -16.4500, 20),

-- REGION DE SAINT-LOUIS
('SAINT_LOUIS', 'Saint-Louis', 'Saint-Louis', 16.0469, -16.4734, 7),
('PODOR', 'Podor', 'Saint-Louis', 16.6500, -14.9667, 15),
('DAGANA', 'Dagana', 'Saint-Louis', 16.5167, -15.5000, 12),

-- REGION DE MATAM
('MATAM', 'Matam', 'Matam', 15.6556, -13.2553, 25),
('BAKEL', 'Bakel', 'Matam', 14.9000, -12.4667, 30),
('KANEL', 'Kanel', 'Matam', 15.4833, -13.1833, 20),

-- REGION DE TAMBACOUNDA
('TAMBACOUNDA', 'Tambacounda', 'Tambacounda', 13.7671, -13.6681, 49),
('GOUDIRY', 'Goudiry', 'Tambacounda', 14.1833, -12.7167, 45),
('KOUMPENTOUM', 'Koumpentoum', 'Tambacounda', 14.1000, -14.5667, 35),

-- REGION DE KEDOUGOU
('KEDOUGOU', 'Kedougou', 'Kedougou', 12.5561, -12.1756, 178),
('SARAYA', 'Saraya', 'Kedougou', 12.8167, -11.7500, 210),
('SALEMATAH', 'Salematah', 'Kedougou', 12.6833, -12.0500, 195),

-- REGION DE KOLDA
('KOLDA', 'Kolda', 'Kolda', 12.8831, -14.9417, 15),
('VELINGARA', 'Velingara', 'Kolda', 13.1500, -14.1167, 25),
('MEDINA_YORO_FOULAH', 'Medina Yoro Foulah', 'Kolda', 12.7833, -14.6167, 20),

-- REGION DE ZIGUINCHOR (Casamance)
('ZIGUINCHOR', 'Ziguinchor', 'Ziguinchor', 12.5581, -16.2719, 26),
('OUSSOUYE', 'Oussouye', 'Ziguinchor', 12.4833, -16.5500, 15),
('BIGNONA', 'Bignona', 'Ziguinchor', 12.8167, -16.2333, 20),

-- REGION DE SEDHIOU
('SEDHIOU', 'Sedhiou', 'Sedhiou', 12.7083, -15.5572, 18),
('GOUDOMP', 'Goudomp', 'Sedhiou', 12.6167, -15.9000, 12),
('BOUNKILING', 'Bounkiling', 'Sedhiou', 12.9333, -15.5333, 22),

-- REGION DE KAFFRINE
('KAFFRINE_CENTRE', 'Kaffrine Centre', 'Kaffrine', 14.1083, -15.5500, 20),
('BIRKELANE', 'Birkelane', 'Kaffrine', 14.2167, -15.6167, 18),
('KOUNGHEUL', 'Koungheul', 'Kaffrine', 13.9833, -14.8000, 25),
('MALEM_HODAR', 'Malem Hodar', 'Kaffrine', 13.8500, -15.3333, 22)

ON CONFLICT (id) DO NOTHING;

-- Insertion de modeles ML de reference
INSERT INTO ml_models (name, version, model_type, target_variable, features) VALUES
('RandomForest_Occurrence', '1.0', 'classification', 'extreme_occurrence', 
'{"IOD": "float", "Nino34": "float", "TNA": "float", "month": "int", "season": "str"}'),
('RandomForest_Intensity', '1.0', 'regression', 'precipitation_intensity', 
'{"IOD": "float", "Nino34": "float", "TNA": "float", "temperature": "float", "humidity": "float"}'),
('XGBoost_Prediction', '1.0', 'classification', 'extreme_event', 
'{"teleconnections": "array", "seasonal_features": "array", "lag_features": "array"}')
ON CONFLICT (name, version) DO NOTHING;

-- Message de fin avec statistiques
DO $$
DECLARE
    station_count INTEGER;
    region_count INTEGER;
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO station_count FROM stations;
    SELECT COUNT(DISTINCT region) INTO region_count FROM stations;
    SELECT COUNT(*) INTO table_count FROM information_schema.tables WHERE table_schema = 'public';
    
    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'BASE DE DONNEES SENEGAL INITIALISEE!';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables creees: %', table_count;
    RAISE NOTICE 'Stations meteorologiques: %', station_count;
    RAISE NOTICE 'Regions couvertes: %', region_count;
    RAISE NOTICE 'Couverture: Nationale complete (14 regions)';
    RAISE NOTICE 'Zones climatiques: Sahelienne, Soudano-sahelienne, Guineenne';
    RAISE NOTICE 'Extensions: TimescaleDB, UUID, PostGIS (optionnel)';
    RAISE NOTICE '';
    RAISE NOTICE 'PRET POUR LA RECHERCHE CLIMATIQUE!';
    RAISE NOTICE '';
END $$;

SELECT 'Initialisation terminee - Base de donnees operationnelle pour tout le Senegal!' as message;
'@

# 4. Execution du script SQL
Write-Host "Execution du script d'initialisation complet..." -ForegroundColor Yellow

try {
    # Utilisation des bonnes variables
    $sqlSchema | docker-compose exec -T timescaledb psql -U postgres -d climatsn_db
    Write-Host "[OK] Schema complet cree avec succes" -ForegroundColor Green
} catch {
    Write-Host "[ERREUR] Probleme lors de l'execution du script" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# 5. Verification finale detaillee
Write-Host "`n4. Verification finale detaillee..." -ForegroundColor Yellow

# Test des tables creees
$tableCheck = docker-compose exec -T timescaledb psql -U postgres -d climatsn_db -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public';" 2>$null

if ($tableCheck -match "stations") {
    Write-Host "[OK] Table stations creee" -ForegroundColor Green
} else {
    Write-Host "[ERREUR] Table stations manquante" -ForegroundColor Red
}

if ($tableCheck -match "weather_data") {
    Write-Host "[OK] Table weather_data creee" -ForegroundColor Green
} else {
    Write-Host "[ERREUR] Table weather_data manquante" -ForegroundColor Red
}

if ($tableCheck -match "ml_models") {
    Write-Host "[OK] Table ml_models creee" -ForegroundColor Green
} else {
    Write-Host "[ERREUR] Table ml_models manquante" -ForegroundColor Red
}

if ($tableCheck -match "predictions") {
    Write-Host "[OK] Table predictions creee" -ForegroundColor Green
} else {
    Write-Host "[ERREUR] Table predictions manquante" -ForegroundColor Red
}

if ($tableCheck -match "alerts") {
    Write-Host "[OK] Table alerts creee" -ForegroundColor Green
} else {
    Write-Host "[ERREUR] Table alerts manquante" -ForegroundColor Red
}

# Compter les stations par region
Write-Host "`n   STATISTIQUES DU SENEGAL:" -ForegroundColor Cyan
$stationCount = docker-compose exec -T timescaledb psql -U postgres -d climatsn_db -c "SELECT COUNT(*) FROM stations;" -t -A 2>$null
Write-Host "   Stations totales: $stationCount" -ForegroundColor Gray

$regionCount = docker-compose exec -T timescaledb psql -U postgres -d climatsn_db -c "SELECT COUNT(DISTINCT region) FROM stations;" -t -A 2>$null
Write-Host "   Regions couvertes: $regionCount/14" -ForegroundColor Gray

# Afficher la repartition par region
Write-Host "`n   REPARTITION PAR REGION:" -ForegroundColor Cyan
$regionStats = docker-compose exec -T timescaledb psql -U postgres -d climatsn_db -c "SELECT region, COUNT(*) as stations FROM stations GROUP BY region ORDER BY region;" 2>$null
if ($regionStats) {
    Write-Host "$regionStats" -ForegroundColor Gray
}

# Test des modeles ML
$modelCount = docker-compose exec -T timescaledb psql -U postgres -d climatsn_db -c "SELECT COUNT(*) FROM ml_models;" -t -A 2>$null
Write-Host "`n   Modeles ML de reference: $modelCount" -ForegroundColor Gray

# 6. Test de l'API
Write-Host "`n5. Test de l'API avec nouvelle base..." -ForegroundColor Yellow

Start-Sleep -Seconds 5

try {
    $response = Invoke-WebRequest "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "[OK] Endpoint /health fonctionne" -ForegroundColor Green
    }
} catch {
    Write-Host "[INFO] API en cours de redemarrage..." -ForegroundColor Yellow
}

try {
    $response = Invoke-WebRequest "http://localhost:8000/stations" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "[OK] Endpoint /stations fonctionne" -ForegroundColor Green
        $stations = $response.Content | ConvertFrom-Json
        Write-Host "   Stations API: $($stations.Count)" -ForegroundColor Gray
        
        # Afficher quelques regions pour verification
        $regions = $stations | Group-Object region | Select-Object Name, Count
        Write-Host "   Regions API actives:" -ForegroundColor Gray
        foreach ($region in $regions | Select-Object -First 5) {
            Write-Host "     $($region.Name): $($region.Count) stations" -ForegroundColor DarkGray
        }
    }
} catch {
    Write-Host "[INFO] Endpoint /stations inaccessible" -ForegroundColor Yellow
}

# Test des predictions si possible
try {
    $predResponse = Invoke-WebRequest "http://localhost:8000/predictions/latest" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($predResponse.StatusCode -eq 200) {
        Write-Host "[OK] Endpoint /predictions/latest fonctionne" -ForegroundColor Green
    }
} catch {
    Write-Host "[INFO] Endpoint predictions non teste" -ForegroundColor Yellow
}

Write-Host "`n=== INITIALISATION SENEGAL COMPLETE TERMINEE ===" -ForegroundColor Cyan
Write-Host "INFRASTRUCTURE OPERATIONNELLE:" -ForegroundColor White
Write-Host "  API FastAPI: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Stations: http://localhost:8000/stations" -ForegroundColor White
Write-Host "  Dashboard: http://localhost:3000" -ForegroundColor White
Write-Host "  Grafana: http://localhost:3001" -ForegroundColor White
Write-Host "  Prometheus: http://localhost:9090" -ForegroundColor White

Write-Host "`nCOUVERTURE SENEGAL:" -ForegroundColor Green
Write-Host "  14 regions administratives" -ForegroundColor White
Write-Host "  40+ stations meteorologiques" -ForegroundColor White
Write-Host "  3 zones climatiques (Sahel, Soudano-Sahel, Guinee)" -ForegroundColor White
Write-Host "  Infrastructure ML complete" -ForegroundColor White

Write-Host "`nPROCHAINES ETAPES:" -ForegroundColor Yellow
Write-Host "  1. Deployer les modeles ML: python main.py --deploy-only" -ForegroundColor White
Write-Host "  2. Tester les predictions par region" -ForegroundColor White
Write-Host "  3. Lancer le pipeline complet recherche: python main.py" -ForegroundColor White