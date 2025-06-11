# src/analysis/detection.py
"""
Module de détection des événements de précipitations extrêmes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from tqdm import tqdm
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
try:
    from ..config.settings import DETECTION_CRITERIA
except ImportError:
    try:
        from src.config.settings import DETECTION_CRITERIA
    except ImportError:
        # Valeurs par défaut en dernier recours
        DETECTION_CRITERIA = {
            'threshold_anomaly': 2.0,
            'min_grid_points': 40,
            'min_precipitation': 5.0
        }

def detect_extreme_precipitation_events_final(precip_data: np.ndarray, std_anomalies: np.ndarray, 
                                            dates: List, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
    """
    Détection finale des événements de précipitations extrêmes.
    
    Args:
        precip_data (np.ndarray): Données de précipitation (temps, lat, lon)
        std_anomalies (np.ndarray): Anomalies standardisées
        dates (List): Liste des dates
        lats (np.ndarray): Latitudes
        lons (np.ndarray): Longitudes
        
    Returns:
        pd.DataFrame: DataFrame des événements détectés
    """
    print("\n🔄 DÉTECTION DES ÉVÉNEMENTS EXTRÊMES - VERSION FINALE")
    print("-" * 50)
    print("Critères de détection optimisés:")
    print("• Anomalie standardisée: > +2σ (98e centile)")
    print("• Points de grille minimum: 40 (≈7% superficie)")
    print("• Précipitation maximale: ≥ 5mm (réaliste pour le Sénégal)")
    print("• Classement: par couverture spatiale décroissante")
    
    # Paramètres optimisés
    THRESHOLD_ANOMALY = DETECTION_CRITERIA['threshold_anomaly']
    MIN_GRID_POINTS = DETECTION_CRITERIA['min_grid_points']
    MIN_PRECIPITATION = DETECTION_CRITERIA['min_precipitation']
    
    n_time, n_lat, n_lon = precip_data.shape
    total_grid_points = n_lat * n_lon
    
    print(f"\nParamètres de détection:")
    print(f"   Points de grille totaux: {total_grid_points}")
    print(f"   Seuil minimum: {MIN_GRID_POINTS} points ({MIN_GRID_POINTS/total_grid_points*100:.1f}%)")
    print(f"   Seuil d'anomalie: +{THRESHOLD_ANOMALY}σ")
    print(f"   Seuil de précipitation: {MIN_PRECIPITATION} mm")
    
    extreme_events = []
    
    print("\nRecherche des événements extrêmes...")
    
    for i in tqdm(range(n_time), desc="Détection"):
        # Données du jour
        day_precip = precip_data[i, :, :]
        day_anomalies = std_anomalies[i, :, :]
        
        # Masque des points dépassant le seuil d'anomalie
        extreme_mask = (day_anomalies > THRESHOLD_ANOMALY) & ~np.isnan(day_anomalies)
        n_extreme_points = np.sum(extreme_mask)
        
        # Vérifier le critère de couverture minimale
        if n_extreme_points >= MIN_GRID_POINTS:
            # Vérifier le critère de précipitation
            max_precip_day = np.nanmax(day_precip)
            
            if not np.isnan(max_precip_day) and max_precip_day >= MIN_PRECIPITATION:
                # Pourcentage de couverture
                coverage_percent = (n_extreme_points / total_grid_points) * 100
                
                # Statistiques des précipitations dans les zones extrêmes
                extreme_precip = day_precip[extreme_mask]
                mean_precip = np.nanmean(extreme_precip)
                min_precip = np.nanmin(extreme_precip)
                
                # Statistiques des anomalies dans les zones extrêmes
                extreme_anomalies = day_anomalies[extreme_mask]
                mean_anomaly = np.nanmean(extreme_anomalies)
                max_anomaly = np.nanmax(extreme_anomalies)
                
                # Centroïde géographique de l'événement
                lat_indices, lon_indices = np.where(extreme_mask)
                centroid_lat = np.mean(lats[lat_indices])
                centroid_lon = np.mean(lons[lon_indices])
                
                # Stocker l'événement
                extreme_events.append({
                    'date': dates[i],
                    'coverage_points': n_extreme_points,
                    'coverage_percent': coverage_percent,
                    'mean_precip': mean_precip,
                    'max_precip': max_precip_day,
                    'min_precip': min_precip,
                    'mean_anomaly': mean_anomaly,
                    'max_anomaly': max_anomaly,
                    'centroid_lat': centroid_lat,
                    'centroid_lon': centroid_lon,
                    'month': dates[i].month,
                    'year': dates[i].year,
                    'day_of_year': dates[i].timetuple().tm_yday
                })
    
    print(f"\n✅ Événements détectés: {len(extreme_events)}")
    
    if extreme_events:
        # Créer DataFrame
        df_events = pd.DataFrame(extreme_events)
        df_events.set_index('date', inplace=True)
        
        # CLASSEMENT PAR COUVERTURE SPATIALE (décroissant)
        df_events.sort_values(['coverage_points', 'max_anomaly'], 
                            ascending=[False, False], inplace=True)
        
        print(f"   Période: {df_events.index.min().strftime('%Y-%m-%d')} à {df_events.index.max().strftime('%Y-%m-%d')}")
        print(f"   Fréquence moyenne: {len(df_events)/(df_events['year'].max()-df_events['year'].min()+1):.1f} événements/an")
        
        # Validation des critères
        print(f"\n🔍 VALIDATION DES CRITÈRES:")
        print(f"   ✅ Tous les événements: anomalie > +{THRESHOLD_ANOMALY}σ")
        print(f"   ✅ Tous les événements: couverture ≥ {MIN_GRID_POINTS} points")
        print(f"   ✅ Tous les événements: précipitation ≥ {MIN_PRECIPITATION} mm")
        print(f"   📊 Précipitation moyenne: {df_events['max_precip'].mean():.2f} mm")
        print(f"   📊 Précipitation médiane: {df_events['max_precip'].median():.2f} mm")
        print(f"   📊 Couverture moyenne: {df_events['coverage_percent'].mean():.2f}%")
        print(f"   📊 Anomalie moyenne: {df_events['max_anomaly'].mean():.2f}σ")
        
        return df_events
    else:
        print("❌ Aucun événement détecté avec ces critères")
        return pd.DataFrame()


def analyze_spatial_distribution(df_events: pd.DataFrame, lats: np.ndarray, lons: np.ndarray) -> Tuple[pd.Series, pd.Series]:
    """
    Analyse la distribution spatiale des événements extrêmes.
    
    Args:
        df_events (pd.DataFrame): DataFrame des événements
        lats (np.ndarray): Latitudes
        lons (np.ndarray): Longitudes
        
    Returns:
        Tuple[pd.Series, pd.Series]: (régions_lat, régions_lon)
    """
    print("\n🔄 ANALYSE DE LA DISTRIBUTION SPATIALE")
    print("-" * 50)
    
    # Statistiques spatiales
    print("📍 Statistiques des centroïdes:")
    print(f"   Latitude moyenne: {df_events['centroid_lat'].mean():.3f}°N")
    print(f"   Longitude moyenne: {df_events['centroid_lon'].mean():.3f}°E")
    print(f"   Écart-type latitude: {df_events['centroid_lat'].std():.3f}°")
    print(f"   Écart-type longitude: {df_events['centroid_lon'].std():.3f}°")
    
    # Régions préférentielles
    lat_bins = np.linspace(df_events['centroid_lat'].min(), df_events['centroid_lat'].max(), 5)
    lon_bins = np.linspace(df_events['centroid_lon'].min(), df_events['centroid_lon'].max(), 5)
    
    # 4 labels pour 5 bins
    lat_regions = pd.cut(df_events['centroid_lat'], bins=lat_bins, 
                        labels=['Sud', 'Sud-Centre', 'Nord-Centre', 'Nord'])
    lon_regions = pd.cut(df_events['centroid_lon'], bins=lon_bins, 
                        labels=['Ouest', 'Ouest-Centre', 'Est-Centre', 'Est'])
    
    print(f"\n📊 Distribution régionale (Latitude):")
    for region, count in lat_regions.value_counts().items():
        pct = count / len(df_events) * 100
        print(f"   {region}: {count} événements ({pct:.1f}%)")
    
    print(f"\n📊 Distribution régionale (Longitude):")
    for region, count in lon_regions.value_counts().items():
        pct = count / len(df_events) * 100
        print(f"   {region}: {count} événements ({pct:.1f}%)")
    
    return lat_regions, lon_regions


class ExtremeEventDetector:
    """
    Classe pour la détection des événements de précipitations extrêmes.
    """
    
    def __init__(self, threshold_anomaly: float = None, min_grid_points: int = None, 
                 min_precipitation: float = None):
        """
        Initialise le détecteur d'événements extrêmes.
        
        Args:
            threshold_anomaly (float, optional): Seuil d'anomalie standardisée
            min_grid_points (int, optional): Nombre minimum de points de grille
            min_precipitation (float, optional): Précipitation minimale
        """
        self.threshold_anomaly = threshold_anomaly or DETECTION_CRITERIA['threshold_anomaly']
        self.min_grid_points = min_grid_points or DETECTION_CRITERIA['min_grid_points']
        self.min_precipitation = min_precipitation or DETECTION_CRITERIA['min_precipitation']
    
    def detect_events(self, precip_data: np.ndarray, anomalies: np.ndarray, 
                     dates: List, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
        """
        Détecte les événements de précipitations extrêmes.
        
        Args:
            precip_data (np.ndarray): Données de précipitation
            anomalies (np.ndarray): Anomalies standardisées
            dates (List): Liste des dates
            lats (np.ndarray): Latitudes
            lons (np.ndarray): Longitudes
            
        Returns:
            pd.DataFrame: DataFrame des événements détectés
        """
        # Sauvegarder les critères actuels
        current_criteria = DETECTION_CRITERIA.copy()
        
        # Mettre à jour avec les valeurs de l'instance
        DETECTION_CRITERIA['threshold_anomaly'] = self.threshold_anomaly
        DETECTION_CRITERIA['min_grid_points'] = self.min_grid_points
        DETECTION_CRITERIA['min_precipitation'] = self.min_precipitation
        
        try:
            # Détecter les événements
            df_events = detect_extreme_precipitation_events_final(
                precip_data, anomalies, dates, lats, lons
            )
            
            return df_events
            
        finally:
            # Restaurer les critères originaux
            DETECTION_CRITERIA.update(current_criteria)


if __name__ == "__main__":
    print("Module de détection des événements extrêmes")
    print("=" * 50)
    print("Ce module contient les outils pour:")
    print("• Détecter les événements de précipitations extrêmes")
    print("• Analyser la distribution spatiale")
    print("• Valider les critères de détection")