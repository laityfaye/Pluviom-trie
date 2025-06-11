"""
Module centralisé pour les références géographiques du Sénégal.
"""

from typing import Dict, Any, List

class SenegalGeography:
    """Classe centralisée pour toutes les références géographiques du Sénégal."""
    
    # Principales villes du Sénégal avec coordonnées précises
    CITIES = {
        'Dakar': {'lat': 14.6928, 'lon': -17.4467, 'type': 'capitale', 'size': 12},
        'Thiès': {'lat': 14.7886, 'lon': -16.9260, 'type': 'ville', 'size': 8},
        'Kaolack': {'lat': 14.1500, 'lon': -16.0667, 'type': 'ville', 'size': 8},
        'Saint-Louis': {'lat': 16.0333, 'lon': -16.5167, 'type': 'ville', 'size': 8},
        'Ziguinchor': {'lat': 12.5833, 'lon': -16.2667, 'type': 'ville', 'size': 8},
        'Diourbel': {'lat': 14.6594, 'lon': -16.2311, 'type': 'ville', 'size': 6},
        'Tambacounda': {'lat': 13.7667, 'lon': -13.6667, 'type': 'ville', 'size': 8},
        'Kolda': {'lat': 12.8833, 'lon': -14.9500, 'type': 'ville', 'size': 6},
        'Fatick': {'lat': 14.3333, 'lon': -16.4167, 'type': 'ville', 'size': 6},
        'Louga': {'lat': 15.6181, 'lon': -16.2314, 'type': 'ville', 'size': 6},
        'Matam': {'lat': 15.6558, 'lon': -13.2553, 'type': 'ville', 'size': 6},
        'Kédougou': {'lat': 12.5597, 'lon': -12.1750, 'type': 'ville', 'size': 6},
        'Sédhiou': {'lat': 12.7089, 'lon': -15.5647, 'type': 'ville', 'size': 6},
        'Kaffrine': {'lat': 14.1058, 'lon': -15.5472, 'type': 'ville', 'size': 6}
    }
    
    # Régions administratives du Sénégal
    REGIONS = {
        'Dakar': {
            'lat': 14.6928, 'lon': -17.4467,
            'bounds': {'lat_min': 14.53, 'lat_max': 14.85, 'lon_min': -17.54, 'lon_max': -17.10}
        },
        'Thiès': {
            'lat': 14.7886, 'lon': -16.9260,
            'bounds': {'lat_min': 14.40, 'lat_max': 15.15, 'lon_min': -17.20, 'lon_max': -16.50}
        },
        'Diourbel': {
            'lat': 14.6594, 'lon': -16.2311,
            'bounds': {'lat_min': 14.30, 'lat_max': 15.00, 'lon_min': -16.50, 'lon_max': -15.80}
        },
        'Fatick': {
            'lat': 14.3333, 'lon': -16.4167,
            'bounds': {'lat_min': 13.75, 'lat_max': 14.75, 'lon_min': -16.80, 'lon_max': -15.90}
        },
        'Kaolack': {
            'lat': 14.1500, 'lon': -16.0667,
            'bounds': {'lat_min': 13.75, 'lat_max': 14.50, 'lon_min': -16.40, 'lon_max': -15.40}
        },
        'Kaffrine': {
            'lat': 14.1058, 'lon': -15.5472,
            'bounds': {'lat_min': 13.60, 'lat_max': 14.60, 'lon_min': -16.00, 'lon_max': -15.00}
        },
        'Tambacounda': {
            'lat': 13.7667, 'lon': -13.6667,
            'bounds': {'lat_min': 13.00, 'lat_max': 14.80, 'lon_min': -14.50, 'lon_max': -11.50}
        },
        'Kédougou': {
            'lat': 12.5597, 'lon': -12.1750,
            'bounds': {'lat_min': 12.10, 'lat_max': 13.00, 'lon_min': -13.00, 'lon_max': -11.50}
        },
        'Kolda': {
            'lat': 12.8833, 'lon': -14.9500,
            'bounds': {'lat_min': 12.40, 'lat_max': 13.40, 'lon_min': -15.50, 'lon_max': -14.00}
        },
        'Sédhiou': {
            'lat': 12.7089, 'lon': -15.5647,
            'bounds': {'lat_min': 12.30, 'lat_max': 13.20, 'lon_min': -16.20, 'lon_max': -15.00}
        },
        'Ziguinchor': {
            'lat': 12.5833, 'lon': -16.2667,
            'bounds': {'lat_min': 12.25, 'lat_max': 13.00, 'lon_min': -16.75, 'lon_max': -15.80}
        },
        'Saint-Louis': {
            'lat': 16.0333, 'lon': -16.5167,
            'bounds': {'lat_min': 15.50, 'lat_max': 16.70, 'lon_min': -17.00, 'lon_max': -15.50}
        },
        'Louga': {
            'lat': 15.6181, 'lon': -16.2314,
            'bounds': {'lat_min': 15.00, 'lat_max': 16.30, 'lon_min': -16.80, 'lon_max': -15.20}
        },
        'Matam': {
            'lat': 15.6558, 'lon': -13.2553,
            'bounds': {'lat_min': 15.20, 'lat_max': 16.70, 'lon_min': -14.00, 'lon_max': -12.00}
        }
    }
    
    # Limites du Sénégal
    SENEGAL_BOUNDS = {
        'lat_min': 12.0,
        'lat_max': 17.0,
        'lon_min': -18.0,
        'lon_max': -11.0
    }
    
    @classmethod
    def identify_closest_region(cls, lat: float, lon: float) -> str:
        """Identifie la région la plus proche d'un point."""
        min_distance = float('inf')
        closest_region = "Zone indéterminée"
        
        for region_name, region_info in cls.REGIONS.items():
            # Vérifier si le point est dans les limites de la région
            bounds = region_info['bounds']
            if (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
                bounds['lon_min'] <= lon <= bounds['lon_max']):
                return region_name
            
            # Sinon, calculer la distance au centre de la région
            distance = ((lat - region_info['lat'])**2 + (lon - region_info['lon'])**2)**0.5
            if distance < min_distance:
                min_distance = distance
                closest_region = region_name
        
        return closest_region
    
    @classmethod
    def identify_nearby_cities(cls, lat: float, lon: float, max_distance: float = 1.0) -> List[Dict[str, Any]]:
        """Identifie les villes proches d'un point."""
        nearby_cities = []
        
        for city_name, city_info in cls.CITIES.items():
            distance = ((lat - city_info['lat'])**2 + (lon - city_info['lon'])**2)**0.5
            if distance <= max_distance:
                nearby_cities.append({
                    'name': city_name,
                    'distance_km': distance * 111,  # Conversion approximative en km
                    'direction': cls._get_direction(lat, lon, city_info['lat'], city_info['lon'])
                })
        
        return sorted(nearby_cities, key=lambda x: x['distance_km'])
    
    @staticmethod
    def _get_direction(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
        """Détermine la direction relative entre deux points."""
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        if abs(dlat) > abs(dlon):
            return "Nord" if dlat > 0 else "Sud"
        else:
            return "Est" if dlon > 0 else "Ouest"