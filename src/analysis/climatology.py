# src/analysis/climatology.py
"""
Module d'analyse climatologique pour les données de précipitations.
"""

import numpy as np
from typing import Tuple, Dict, Any, List
from tqdm import tqdm
import warnings
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.settings import CLIMATOLOGY_PARAMS, NUMERICAL_PARAMS
except ImportError:
    # Valeurs par défaut
    CLIMATOLOGY_PARAMS = {
        'smoothing_window': 15,
        'min_observations': 5,
        'n_days_year': 366
    }
    NUMERICAL_PARAMS = {
        'pos_inf_replacement': 15,
        'neg_inf_replacement': -15,
        'nan_replacement': 0
    }

warnings.filterwarnings('ignore')


def calculate_daily_climatology_robust(precip_data: np.ndarray, dates: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule la climatologie quotidienne avec gestion robuste des NaN.
    
    Args:
        precip_data (np.ndarray): Données de précipitation (temps, lat, lon)
        dates (List): Liste des dates correspondantes
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (climatologie_lissée, ecart_type_lissé)
    """
    print("\n🔄 CALCUL DE LA CLIMATOLOGIE QUOTIDIENNE")
    print("-" * 50)
    
    # Jours de l'année (1-366 pour années bissextiles)
    doy = np.array([d.timetuple().tm_yday for d in dates])
    
    n_days = CLIMATOLOGY_PARAMS['n_days_year']
    n_lat, n_lon = precip_data.shape[1], precip_data.shape[2]
    
    # Initialisation
    daily_climatology = np.full((n_days, n_lat, n_lon), np.nan)
    daily_std = np.full((n_days, n_lat, n_lon), np.nan)
    
    print("Calcul de la climatologie par jour de l'année...")
    
    # Calcul pour chaque jour de l'année
    for day in tqdm(range(1, n_days + 1), desc="Climatologie"):
        day_indices = np.where(doy == day)[0]
        
        if len(day_indices) >= CLIMATOLOGY_PARAMS['min_observations']:
            day_data = precip_data[day_indices, :, :]
            
            # Moyenne et écart-type avec gestion robuste des NaN
            daily_climatology[day-1, :, :] = np.nanmean(day_data, axis=0)
            daily_std[day-1, :, :] = np.nanstd(day_data, axis=0, ddof=1)
    
    # Lissage avec fenêtre glissante
    print(f"Application du lissage (fenêtre de {CLIMATOLOGY_PARAMS['smoothing_window']} jours)...")
    
    smoothed_climatology = np.full_like(daily_climatology, np.nan)
    smoothed_std = np.full_like(daily_std, np.nan)
    
    window_size = CLIMATOLOGY_PARAMS['smoothing_window']
    half_window = window_size // 2
    
    for day in tqdm(range(n_days), desc="Lissage"):
        # Fenêtre cyclique (début/fin d'année)
        window_indices = [(day - half_window + i) % n_days for i in range(window_size)]
        
        # Lissage de la climatologie et des écarts-types
        window_clim = daily_climatology[window_indices, :, :]
        window_std = daily_std[window_indices, :, :]
        
        # Moyennes lissées avec gestion robuste des NaN
        smoothed_climatology[day, :, :] = np.nanmean(window_clim, axis=0)
        smoothed_std[day, :, :] = np.nanmean(window_std, axis=0)
    
    # Statistiques de validation
    valid_clim = ~np.isnan(smoothed_climatology).all(axis=(1, 2))
    valid_std = ~np.isnan(smoothed_std).all(axis=(1, 2))
    
    print(f"✅ Climatologie calculée:")
    print(f"   Jours valides (climatologie): {valid_clim.sum()}/366")
    print(f"   Jours valides (écart-type): {valid_std.sum()}/366")
    
    return smoothed_climatology, smoothed_std


def calculate_standardized_anomalies_robust(precip_data: np.ndarray, dates: List, 
                                          climatology: np.ndarray, std_dev: np.ndarray) -> np.ndarray:
    """
    Calcule les anomalies standardisées avec gestion robuste des NaN.
    
    Args:
        precip_data (np.ndarray): Données de précipitation
        dates (List): Liste des dates
        climatology (np.ndarray): Climatologie quotidienne
        std_dev (np.ndarray): Écarts-types quotidiens
        
    Returns:
        np.ndarray: Anomalies standardisées
    """
    print("\n🔄 CALCUL DES ANOMALIES STANDARDISÉES")
    print("-" * 50)
    
    n_time, n_lat, n_lon = precip_data.shape
    standardized_anomalies = np.full_like(precip_data, np.nan)
    doy_idx = np.array([d.timetuple().tm_yday - 1 for d in dates])  # 0-365 indexing
    
    print("Calcul des anomalies pour chaque jour...")
    
    for i in tqdm(range(len(dates)), desc="Anomalies"):
        day_idx = doy_idx[i]
        
        # Climatologie et écart-type pour ce jour
        clim_day = climatology[day_idx, :, :]
        std_day = std_dev[day_idx, :, :]
        day_precip = precip_data[i, :, :]
        
        # Masque des valeurs valides (gestion robuste des NaN)
        valid_mask = (
            (std_day > 0.001) & 
            ~np.isnan(std_day) & 
            ~np.isnan(clim_day) & 
            ~np.isnan(day_precip)
        )
        
        if valid_mask.any():
            # Calcul de l'anomalie standardisée seulement où c'est valide
            standardized_anomalies[i, valid_mask] = (
                (day_precip[valid_mask] - clim_day[valid_mask]) / std_day[valid_mask]
            )
    
    # Remplacer les valeurs infinies par des valeurs extrêmes mais finies
    standardized_anomalies = np.nan_to_num(standardized_anomalies, 
                                         nan=NUMERICAL_PARAMS['nan_replacement'], 
                                         posinf=NUMERICAL_PARAMS['pos_inf_replacement'], 
                                         neginf=NUMERICAL_PARAMS['neg_inf_replacement'])
    
    # Statistiques de validation
    valid_anomalies = ~np.isnan(standardized_anomalies).all(axis=(1, 2))
    
    print(f"✅ Anomalies calculées:")
    print(f"   Jours avec anomalies valides: {valid_anomalies.sum()}/{len(dates)}")
    print(f"   Anomalie moyenne: {np.nanmean(standardized_anomalies):.3f}")
    print(f"   Anomalie max: {np.nanmax(standardized_anomalies):.1f}σ")
    print(f"   Anomalie min: {np.nanmin(standardized_anomalies):.1f}σ")
    
    return standardized_anomalies


def calculate_climatology_and_anomalies(precip_data: np.ndarray, dates: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fonction de convenance pour calculer climatologie et anomalies en une fois.
    
    Args:
        precip_data (np.ndarray): Données de précipitation
        dates (List): Liste des dates
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (climatologie, écart_type, anomalies)
    """
    print("🔄 CALCUL COMPLET DE LA CLIMATOLOGIE ET DES ANOMALIES")
    print("=" * 60)
    
    # Calculer la climatologie
    climatology, std_dev = calculate_daily_climatology_robust(precip_data, dates)
    
    # Calculer les anomalies
    anomalies = calculate_standardized_anomalies_robust(precip_data, dates, climatology, std_dev)
    
    print("✅ Calcul terminé avec succès")
    return climatology, std_dev, anomalies


if __name__ == "__main__":
    print("Module d'analyse climatologique")
    print("=" * 50)
    print("Ce module contient les outils pour:")
    print("• Calculer la climatologie quotidienne lissée")
    print("• Calculer les anomalies standardisées")
    print("• Valider la qualité des calculs")