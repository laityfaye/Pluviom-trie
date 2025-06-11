"""
Module de visualisation géographique centralisé.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional

# Import absolu pour éviter les problèmes d'import relatif
try:
    from src.utils.geographic_references import SenegalGeography
except ImportError:
    # Fallback pour les tests
    from ..utils.geographic_references import SenegalGeography

class SenegalMapVisualizer:
    """Visualiseur centralisé pour les cartes du Sénégal."""
    
    def __init__(self):
        self.geo = SenegalGeography()
        self.bounds = SenegalGeography.SENEGAL_BOUNDS
    
    def add_geographic_references(self, ax, event_lat: Optional[float] = None, 
                                event_lon: Optional[float] = None):
        """Ajoute les références géographiques à une carte - VERSION UNIFIÉE."""
        
        # Ajouter les villes principales
        for city_name, city_info in self.geo.CITIES.items():
            if (self.bounds['lat_min'] <= city_info['lat'] <= self.bounds['lat_max'] and
                self.bounds['lon_min'] <= city_info['lon'] <= self.bounds['lon_max']):
                
                color = 'red' if city_info['type'] == 'capitale' else 'blue'
                size = city_info['size']
                
                ax.plot(city_info['lon'], city_info['lat'], 'o', 
                       color=color, markersize=size, markeredgecolor='white', 
                       markeredgewidth=1, alpha=0.8)
                
                # Nom de la ville
                ax.annotate(city_name, (city_info['lon'], city_info['lat']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='black',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # Ajouter les limites des régions principales
        for region_name, region_info in self.geo.REGIONS.items():
            bounds = region_info['bounds']
            if (bounds['lat_min'] >= self.bounds['lat_min'] and
                bounds['lat_max'] <= self.bounds['lat_max']):
                
                # Rectangle pour délimiter la région
                rect = patches.Rectangle(
                    (bounds['lon_min'], bounds['lat_min']),
                    bounds['lon_max'] - bounds['lon_min'],
                    bounds['lat_max'] - bounds['lat_min'],
                    linewidth=1, edgecolor='gray', facecolor='none', 
                    linestyle=':', alpha=0.5
                )
                ax.add_patch(rect)
        
        # Frontières du Sénégal
        rect = patches.Rectangle(
            (self.bounds['lon_min'], self.bounds['lat_min']),
            self.bounds['lon_max'] - self.bounds['lon_min'],
            self.bounds['lat_max'] - self.bounds['lat_min'],
            linewidth=2.5, edgecolor='black', facecolor='none', linestyle='-'
        )
        ax.add_patch(rect)
        
        # Marquer l'événement si fourni
        if event_lat is not None and event_lon is not None:
            ax.plot(event_lon, event_lat, '*', color='yellow', markersize=15, 
                   markeredgecolor='black', markeredgewidth=1.5, label='Centroïde événement')
    
    def create_reference_map(self, title: str = "Carte de référence du Sénégal", 
                           figsize: tuple = (12, 10)):
        """Crée une carte de référence complète."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Ajouter un fond de carte simple
        ax.add_patch(patches.Rectangle(
            (self.bounds['lon_min'], self.bounds['lat_min']),
            self.bounds['lon_max'] - self.bounds['lon_min'],
            self.bounds['lat_max'] - self.bounds['lat_min'],
            facecolor='lightblue', alpha=0.1, edgecolor='none'
        ))
        
        self.add_geographic_references(ax)
        
        # Configuration des axes
        ax.set_xlabel('Longitude (°)', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(self.bounds['lon_min']-0.1, self.bounds['lon_max']+0.1)
        ax.set_ylim(self.bounds['lat_min']-0.1, self.bounds['lat_max']+0.1)
        
        # Légendes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, label='Capitale (Dakar)'),
            Patch(facecolor='blue', alpha=0.8, label='Villes principales'),
            Patch(facecolor='gray', alpha=0.5, label='Limites régionales'),
            Patch(facecolor='black', alpha=0.8, label='Frontières nationales')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        return fig, ax