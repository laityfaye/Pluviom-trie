# src/config/settings.py
"""
Configuration centralisée pour l'analyse des précipitations extrêmes au Sénégal.
"""

import os
from pathlib import Path

# ============================================================================
# CHEMINS ET DOSSIERS
# ============================================================================

# Dossier racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Dossiers de données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dossiers de sortie
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Sous-dossiers de visualisation
DETECTION_VIZ_DIR = VISUALIZATION_DIR / "detection"
SPATIAL_VIZ_DIR = VISUALIZATION_DIR / "spatial"
TEMPORAL_VIZ_DIR = VISUALIZATION_DIR / "temporal"

# Fichiers de données
CHIRPS_FILENAME = "chirps_WA_1981_2023_dayly.mat"
CHIRPS_FILEPATH = RAW_DATA_DIR / CHIRPS_FILENAME

# ============================================================================
# PARAMÈTRES GÉOGRAPHIQUES
# ============================================================================

# Limites géographiques du Sénégal
SENEGAL_BOUNDS = {
    'lat_min': 12.0,
    'lat_max': 17.0,
    'lon_min': -18.0,
    'lon_max': -11.0
}

# ============================================================================
# PARAMÈTRES DE DÉTECTION D'ÉVÉNEMENTS EXTRÊMES
# ============================================================================

DETECTION_CRITERIA = {
    'threshold_anomaly': 2.0,          # Seuil d'anomalie standardisée
    'min_grid_points': 40,             # Nombre minimum de points de grille
    'min_precipitation': 5.0,          # Précipitation minimale (mm)
    'min_std_threshold': 0.001         # Seuil minimal d'écart-type
}

# Paramètres pour le calcul de la climatologie
CLIMATOLOGY_PARAMS = {
    'smoothing_window': 15,            # Taille de la fenêtre de lissage
    'min_observations': 5,             # Nombre minimum d'observations
    'n_days_year': 366                 # Nombre de jours dans l'année
}

# ============================================================================
# CLASSIFICATION SAISONNIÈRE
# ============================================================================

SEASONS_SENEGAL = {
    'saison_seche': {
        'months': [11, 12, 1, 2, 3, 4],
        'name_fr': 'Saison sèche',
        'description': 'Novembre à Avril - Période de faibles précipitations'
    },
    'saison_des_pluies': {
        'months': [5, 6, 7, 8, 9, 10],
        'name_fr': 'Saison des pluies',
        'description': 'Mai à Octobre - Période de précipitations importantes'
    }
}

# ============================================================================
# PARAMÈTRES DE VISUALISATION
# ============================================================================

SEASON_COLORS = {
    'saison_seche': '#E74C3C',         # Rouge
    'saison_des_pluies': '#27AE60',    # Vert
    'neutre': '#3498DB'                # Bleu
}

PLOT_PARAMS = {
    'figure_size': (14, 6),
    'dpi': 300,
    'style': 'default',
    'alpha_scatter': 0.6,
    'alpha_hist': 0.7,
    'line_width': 2,
    'marker_size': 4
}

# ============================================================================
# PARAMÈTRES NUMÉRIQUES
# ============================================================================

NUMERICAL_PARAMS = {
    'pos_inf_replacement': 15,
    'neg_inf_replacement': -15,
    'nan_replacement': 0,
    'float_precision': 1e-10
}

# ============================================================================
# NOMS DE FICHIERS DE SORTIE
# ============================================================================

OUTPUT_FILENAMES = {
    'extreme_events': 'extreme_events_senegal_final.csv',
    'climatology': 'climatology_senegal.npz',
    'anomalies': 'standardized_anomalies_senegal.npz',
    'detection_report': 'rapport_detection_evenements.txt',
    'summary_stats': 'statistiques_resume.json'
}

VISUALIZATION_FILENAMES = {
    'temporal_distribution': '01_distribution_temporelle.png',
    'intensity_coverage': '02_intensite_couverture.png',
    'evolution_anomalies': '03_evolution_anomalies.png',
    'spatial_distribution': '04_distribution_spatiale.png'
}

# ============================================================================
# MÉTADONNÉES DU PROJET
# ============================================================================

PROJECT_INFO = {
    'title': 'Analyse des précipitations extrêmes au Sénégal',
    'author': '[Votre nom]',
    'version': '1.0.0',
    'data_source': 'CHIRPS',
    'temporal_coverage': '1981-2023',
    'spatial_coverage': 'Sénégal (12°N-17°N, 18°W-11°W)'
}

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def create_output_directories():
    """Crée tous les dossiers de sortie nécessaires."""
    directories = [
        OUTPUT_DIR, VISUALIZATION_DIR, DETECTION_VIZ_DIR,
        SPATIAL_VIZ_DIR, TEMPORAL_VIZ_DIR, REPORTS_DIR, PROCESSED_DATA_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Dossiers de sortie créés")

def get_season_from_month(month: int) -> str:
    """Détermine la saison à partir du mois pour le Sénégal."""
    if month in SEASONS_SENEGAL['saison_seche']['months']:
        return 'Saison_seche'
    elif month in SEASONS_SENEGAL['saison_des_pluies']['months']:
        return 'Saison_des_pluies'
    else:
        return 'Indetermine'

def get_output_path(filename_key: str) -> Path:
    """Génère le chemin complet pour un fichier de sortie."""
    if filename_key in OUTPUT_FILENAMES:
        filename = OUTPUT_FILENAMES[filename_key]
        if filename.endswith('.txt'):
            return REPORTS_DIR / filename
        else:
            return PROCESSED_DATA_DIR / filename
    elif filename_key in VISUALIZATION_FILENAMES:
        filename = VISUALIZATION_FILENAMES[filename_key]
        return DETECTION_VIZ_DIR / filename
    else:
        raise ValueError(f"Clé de fichier inconnue: {filename_key}")

def print_project_info():
    """Affiche les informations du projet."""
    print("=" * 80)
    print(f"🎯 {PROJECT_INFO['title']}")
    print("=" * 80)
    print(f"Version: {PROJECT_INFO['version']}")
    print(f"Source des données: {PROJECT_INFO['data_source']}")
    print(f"Couverture temporelle: {PROJECT_INFO['temporal_coverage']}")
    print(f"Couverture spatiale: {PROJECT_INFO['spatial_coverage']}")
    print("=" * 80)

if __name__ == "__main__":
    print("Configuration du projet chargée avec succès")
    print_project_info()
    create_output_directories()