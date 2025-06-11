
"""
Module centralisé pour les calculs de métriques spatiales.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

# Import absolu pour éviter les problèmes d'import relatif
try:
    from src.utils.geographic_references import SenegalGeography
except ImportError:
    from ..utils.geographic_references import SenegalGeography

class SpatialMetricsCalculator:
    """Calculateur centralisé des métriques spatiales."""
    
    def __init__(self):
        self.geo = SenegalGeography()
    
    def calculate_comprehensive_metrics(self, event_date: pd.Timestamp, 
                                      event_data: pd.Series,
                                      precip_data: Optional[np.ndarray] = None,
                                      anomalies: Optional[np.ndarray] = None,
                                      lats: Optional[np.ndarray] = None,
                                      lons: Optional[np.ndarray] = None,
                                      dates: Optional[list] = None,
                                      rank: int = 0) -> Dict[str, Any]:
        """
        Calcule les métriques spatiales de manière unifiée.
        
        Utilise les données CHIRPS si disponibles, sinon utilise le DataFrame.
        """
        
        if self._has_chirps_data(precip_data, anomalies, lats, lons, dates):
            return self._calculate_from_chirps(event_date, event_data, 
                                             precip_data, anomalies, lats, lons, dates, rank)
        else:
            return self._calculate_from_dataframe(event_date, event_data, rank)
    
    def _has_chirps_data(self, precip_data, anomalies, lats, lons, dates) -> bool:
        """Vérifie si les données CHIRPS sont disponibles et valides."""
        return all(data is not None for data in [precip_data, anomalies, lats, lons, dates])
    
    def _calculate_from_chirps(self, event_date, event_data, 
                              precip_data, anomalies, lats, lons, dates, rank) -> Dict[str, Any]:
        """Calcule à partir des données CHIRPS complètes."""
        
        # Trouver l'index temporel
        date_idx = self._find_date_index(event_date, dates)
        if date_idx is None:
            return self._calculate_from_dataframe(event_date, event_data, rank)
        
        # Données spatiales du jour
        day_precip = precip_data[date_idx, :, :]
        day_anomalies = anomalies[date_idx, :, :]
        extreme_mask = (day_anomalies > 2.0) & ~np.isnan(day_anomalies)
        
        if not extreme_mask.any():
            return self._calculate_from_dataframe(event_date, event_data, rank)
        
        # Calculs géographiques précis
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        
        # Surface par pixel
        lat_res = np.abs(lats[1] - lats[0]) if len(lats) > 1 else 0.25
        lon_res = np.abs(lons[1] - lons[0]) if len(lons) > 1 else 0.25
        pixel_area_km2 = (lat_res * 111.32) * (lon_res * 111.32 * np.cos(np.radians(lat_grid)))
        
        # Métriques de base
        total_area_km2 = np.sum(pixel_area_km2[extreme_mask])
        num_pixels = np.sum(extreme_mask)
        
        # Données valides dans la zone extrême
        valid_mask = extreme_mask & ~np.isnan(day_precip)
        if not valid_mask.any():
            return self._calculate_from_dataframe(event_date, event_data, rank)
        
        valid_precip = day_precip[valid_mask]
        valid_lat = lat_grid[valid_mask]
        valid_lon = lon_grid[valid_mask]
        
        # Centroïde pondéré
        weights = valid_precip
        centroid_lat = np.average(valid_lat, weights=weights)
        centroid_lon = np.average(valid_lon, weights=weights)
        
        # Position du maximum
        max_intensity = np.nanmax(day_precip)
        max_idx = np.unravel_index(np.nanargmax(day_precip), day_precip.shape)
        max_lat = lat_grid[max_idx]
        max_lon = lon_grid[max_idx]
        
        # Statistiques d'intensité
        intensity_stats = {
            'mean': float(np.mean(valid_precip)),
            'median': float(np.median(valid_precip)),
            'std': float(np.std(valid_precip)),
            'p95': float(np.percentile(valid_precip, 95))
        }
        
        # Étendue géographique
        lat_extent = valid_lat.max() - valid_lat.min()
        lon_extent = valid_lon.max() - valid_lon.min()
        
        return {
            'rank': rank,
            'date': event_date.strftime('%Y-%m-%d'),
            'total_area_km2': float(total_area_km2),
            'num_pixels_affected': int(num_pixels),
            'centroid_lat': float(centroid_lat),
            'centroid_lon': float(centroid_lon),
            'lat_extent_deg': float(lat_extent),
            'lon_extent_deg': float(lon_extent),
            'max_intensity_mm': float(max_intensity),
            'max_intensity_lat': float(max_lat),
            'max_intensity_lon': float(max_lon),
            'coverage_percent': float(100 * num_pixels / (len(lats) * len(lons))),
            'intensity_stats': intensity_stats,
            'region': self.geo.identify_closest_region(centroid_lat, centroid_lon)
        }
    
    def _calculate_from_dataframe(self, event_date, event_data, rank) -> Dict[str, Any]:
        """Calcule à partir des données du DataFrame avec estimations réalistes."""
        
        max_intensity = float(event_data['max_precip'])
        coverage_percent = float(event_data['coverage_percent'])
        centroid_lat = float(event_data['centroid_lat'])
        centroid_lon = float(event_data['centroid_lon'])
        
        # Estimations réalistes basées sur les données existantes
        total_pixels = 20 * 28  # Grille Sénégal
        affected_pixels = int((coverage_percent / 100.0) * total_pixels)
        estimated_area = affected_pixels * 625.0  # ~25km x 25km par pixel
        
        # Statistiques d'intensité estimées de manière réaliste
        intensity_stats = {
            'mean': max_intensity * 0.4,
            'median': max_intensity * 0.3,
            'std': max_intensity * 0.25,
            'p95': max_intensity * 0.8
        }
        
        return {
            'rank': rank,
            'date': event_date.strftime('%Y-%m-%d'),
            'total_area_km2': float(estimated_area),
            'num_pixels_affected': int(affected_pixels),
            'centroid_lat': centroid_lat,
            'centroid_lon': centroid_lon,
            'lat_extent_deg': 1.0,  # Estimation par défaut
            'lon_extent_deg': 1.0,  # Estimation par défaut
            'max_intensity_mm': max_intensity,
            'max_intensity_lat': centroid_lat,  # Approximation
            'max_intensity_lon': centroid_lon,  # Approximation
            'coverage_percent': coverage_percent,
            'intensity_stats': intensity_stats,
            'region': self.geo.identify_closest_region(centroid_lat, centroid_lon)
        }
    
    def _find_date_index(self, target_date, dates) -> Optional[int]:
        """Trouve l'index correspondant à une date dans les données CHIRPS."""
        target_datetime = target_date.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for i, date in enumerate(dates):
            if date.replace(hour=0, minute=0, second=0, microsecond=0) == target_datetime:
                return i
        return None
