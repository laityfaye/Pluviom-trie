#!/usr/bin/env python3
# scripts/02_spatial_analysis_top10.py
"""
Analyse spatiale détaillée des 10 événements de précipitations extrêmes les plus étendus au Sénégal
avec références géographiques précises (villes, régions, départements).
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import seaborn as sns

# ============================================================================
# CONFIGURATION DES IMPORTS
# ============================================================================

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.config.settings import get_output_path, SENEGAL_BOUNDS
    from src.data.loader import ChirpsDataLoader
    print("✅ Tous les modules importés avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

# ============================================================================
# RÉFÉRENCES GÉOGRAPHIQUES DU SÉNÉGAL
# ============================================================================

# Principales villes du Sénégal avec coordonnées précises
SENEGAL_CITIES = {
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
SENEGAL_REGIONS = {
    'Dakar': {'lat': 14.6928, 'lon': -17.4467, 'bounds': {'lat_min': 14.53, 'lat_max': 14.85, 'lon_min': -17.54, 'lon_max': -17.10}},
    'Thiès': {'lat': 14.7886, 'lon': -16.9260, 'bounds': {'lat_min': 14.40, 'lat_max': 15.15, 'lon_min': -17.20, 'lon_max': -16.50}},
    'Diourbel': {'lat': 14.6594, 'lon': -16.2311, 'bounds': {'lat_min': 14.30, 'lat_max': 15.00, 'lon_min': -16.50, 'lon_max': -15.80}},
    'Fatick': {'lat': 14.3333, 'lon': -16.4167, 'bounds': {'lat_min': 13.75, 'lat_max': 14.75, 'lon_min': -16.80, 'lon_max': -15.90}},
    'Kaolack': {'lat': 14.1500, 'lon': -16.0667, 'bounds': {'lat_min': 13.75, 'lat_max': 14.50, 'lon_min': -16.40, 'lon_max': -15.40}},
    'Kaffrine': {'lat': 14.1058, 'lon': -15.5472, 'bounds': {'lat_min': 13.60, 'lat_max': 14.60, 'lon_min': -16.00, 'lon_max': -15.00}},
    'Tambacounda': {'lat': 13.7667, 'lon': -13.6667, 'bounds': {'lat_min': 13.00, 'lat_max': 14.80, 'lon_min': -14.50, 'lon_max': -11.50}},
    'Kédougou': {'lat': 12.5597, 'lon': -12.1750, 'bounds': {'lat_min': 12.10, 'lat_max': 13.00, 'lon_min': -13.00, 'lon_max': -11.50}},
    'Kolda': {'lat': 12.8833, 'lon': -14.9500, 'bounds': {'lat_min': 12.40, 'lat_max': 13.40, 'lon_min': -15.50, 'lon_max': -14.00}},
    'Sédhiou': {'lat': 12.7089, 'lon': -15.5647, 'bounds': {'lat_min': 12.30, 'lat_max': 13.20, 'lon_min': -16.20, 'lon_max': -15.00}},
    'Ziguinchor': {'lat': 12.5833, 'lon': -16.2667, 'bounds': {'lat_min': 12.25, 'lat_max': 13.00, 'lon_min': -16.75, 'lon_max': -15.80}},
    'Saint-Louis': {'lat': 16.0333, 'lon': -16.5167, 'bounds': {'lat_min': 15.50, 'lat_max': 16.70, 'lon_min': -17.00, 'lon_max': -15.50}},
    'Louga': {'lat': 15.6181, 'lon': -16.2314, 'bounds': {'lat_min': 15.00, 'lat_max': 16.30, 'lon_min': -16.80, 'lon_max': -15.20}},
    'Matam': {'lat': 15.6558, 'lon': -13.2553, 'bounds': {'lat_min': 15.20, 'lat_max': 16.70, 'lon_min': -14.00, 'lon_max': -12.00}}
}

class SpatialAnalysisWithReferences:
    """Classe pour l'analyse spatiale avec références géographiques du Sénégal."""
    
    def __init__(self):
        """Initialise l'analyseur spatial."""
        self.df_events = None
        self.precip_data = None
        self.anomalies = None
        self.dates = None
        self.lats = None
        self.lons = None
        self.top10_events = None
        
    def load_existing_data(self):
        """Charge les données existantes."""
        print("🔄 Chargement des données existantes...")
        
        try:
            # Charger le dataset principal
            events_file = project_root / "data/processed/extreme_events_senegal_final.csv"
            if events_file.exists():
                self.df_events = pd.read_csv(events_file, index_col=0, parse_dates=True)
                print(f"✅ Dataset chargé: {len(self.df_events)} événements")
            else:
                print(f"❌ Dataset non trouvé: {events_file}")
                return False
            
            # Charger les données CHIRPS
            chirps_file = project_root / "data/raw/chirps_WA_1981_2023_dayly.mat"
            if chirps_file.exists():
                loader = ChirpsDataLoader(str(chirps_file))
                self.precip_data, self.dates, self.lats, self.lons = loader.load_senegal_data()
                print(f"✅ Données CHIRPS chargées: {self.precip_data.shape}")
            else:
                return False
            
            # Charger les anomalies
            anom_file = project_root / "data/processed/standardized_anomalies_senegal.npz"
            if anom_file.exists():
                anom_data = np.load(anom_file)
                self.anomalies = anom_data['anomalies']
                print(f"✅ Anomalies chargées: {self.anomalies.shape}")
            else:
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def get_top10_events(self):
        """Récupère les 10 événements les plus étendus."""
        print("\n📊 SÉLECTION DES TOP 10 ÉVÉNEMENTS")
        print("=" * 50)
        
        self.top10_events = self.df_events.head(10).copy()
        
        print("Top 10 événements par couverture spatiale:")
        for i, (date, event) in enumerate(self.top10_events.iterrows(), 1):
            region = self.identify_closest_region(event['centroid_lat'], event['centroid_lon'])
            saison_label = "Pluies" if event['saison'] == 'Saison_des_pluies' else "Sèche"
            print(f"{i:2d}. {date.strftime('%Y-%m-%d')} - Région: {region}")
            print(f"    Couverture: {event['coverage_percent']:5.1f}%, Précip: {event['max_precip']:6.1f} mm")
            print(f"    Centroïde: ({event['centroid_lat']:.3f}°N, {event['centroid_lon']:.3f}°E)")
            print(f"    Saison: {saison_label}")
            print()
        
        return self.top10_events
    
    def identify_closest_region(self, lat, lon):
        """Identifie la région la plus proche d'un point."""
        min_distance = float('inf')
        closest_region = "Zone indéterminée"
        
        for region_name, region_info in SENEGAL_REGIONS.items():
            # Vérifier si le point est dans les limites de la région
            bounds = region_info['bounds']
            if (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
                bounds['lon_min'] <= lon <= bounds['lon_max']):
                return region_name
            
            # Sinon, calculer la distance au centre de la région
            distance = np.sqrt((lat - region_info['lat'])**2 + (lon - region_info['lon'])**2)
            if distance < min_distance:
                min_distance = distance
                closest_region = region_name
        
        return closest_region
    
    def identify_nearby_cities(self, lat, lon, max_distance=1.0):
        """Identifie les villes proches d'un point."""
        nearby_cities = []
        
        for city_name, city_info in SENEGAL_CITIES.items():
            distance = np.sqrt((lat - city_info['lat'])**2 + (lon - city_info['lon'])**2)
            if distance <= max_distance:
                nearby_cities.append({
                    'name': city_name,
                    'distance_km': distance * 111,  # Conversion approximative en km
                    'direction': self.get_direction(lat, lon, city_info['lat'], city_info['lon'])
                })
        
        return sorted(nearby_cities, key=lambda x: x['distance_km'])
    
    def get_direction(self, lat1, lon1, lat2, lon2):
        """Détermine la direction relative entre deux points."""
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        if abs(dlat) > abs(dlon):
            return "Nord" if dlat > 0 else "Sud"
        else:
            return "Est" if dlon > 0 else "Ouest"
    
    def get_spatial_mask_for_event(self, event_date):
        """Récupère le masque spatial pour un événement donné."""
        date_idx = None
        for i, date in enumerate(self.dates):
            if date.strftime('%Y-%m-%d') == event_date.strftime('%Y-%m-%d'):
                date_idx = i
                break
        
        if date_idx is None:
            return None, None, None
        
        day_precip = self.precip_data[date_idx, :, :]
        day_anomalies = self.anomalies[date_idx, :, :]
        extreme_mask = (day_anomalies > 2.0) & ~np.isnan(day_anomalies)
        
        return day_precip, day_anomalies, extreme_mask
    
    def add_geographic_references_to_map(self, ax, event_lat=None, event_lon=None):
        """Ajoute les références géographiques à une carte."""
        
        # Ajouter les villes principales
        for city_name, city_info in SENEGAL_CITIES.items():
            if (SENEGAL_BOUNDS['lat_min'] <= city_info['lat'] <= SENEGAL_BOUNDS['lat_max'] and
                SENEGAL_BOUNDS['lon_min'] <= city_info['lon'] <= SENEGAL_BOUNDS['lon_max']):
                
                color = 'red' if city_info['type'] == 'capitale' else 'blue'
                size = city_info['size']
                
                ax.plot(city_info['lon'], city_info['lat'], 'o', 
                       color=color, markersize=size, markeredgecolor='white', 
                       markeredgewidth=1, alpha=0.8)
                
                # Ajouter le nom de la ville
                ax.annotate(city_name, (city_info['lon'], city_info['lat']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='black',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # Ajouter les limites des régions principales
        for region_name, region_info in SENEGAL_REGIONS.items():
            bounds = region_info['bounds']
            if (bounds['lat_min'] >= SENEGAL_BOUNDS['lat_min'] and
                bounds['lat_max'] <= SENEGAL_BOUNDS['lat_max']):
                
                # Rectangle pour délimiter la région
                rect = patches.Rectangle(
                    (bounds['lon_min'], bounds['lat_min']),
                    bounds['lon_max'] - bounds['lon_min'],
                    bounds['lat_max'] - bounds['lat_min'],
                    linewidth=1, edgecolor='gray', facecolor='none', 
                    linestyle=':', alpha=0.5
                )
                ax.add_patch(rect)
        
        # Ajouter les frontières du Sénégal
        senegal_bounds = SENEGAL_BOUNDS
        rect = patches.Rectangle(
            (senegal_bounds['lon_min'], senegal_bounds['lat_min']),
            senegal_bounds['lon_max'] - senegal_bounds['lon_min'],
            senegal_bounds['lat_max'] - senegal_bounds['lat_min'],
            linewidth=2.5, edgecolor='black', facecolor='none', linestyle='-'
        )
        ax.add_patch(rect)
        
        # Si un événement spécifique est fourni, marquer sa position
        if event_lat is not None and event_lon is not None:
            ax.plot(event_lon, event_lat, '*', color='yellow', markersize=15, 
                   markeredgecolor='black', markeredgewidth=1.5, label='Centroïde événement')
    
    def create_individual_event_map(self, event_idx, event_date, event_data):
        """Crée une carte détaillée pour un événement avec références géographiques."""
        
        print(f"   Création carte événement #{event_idx}: {event_date.strftime('%Y-%m-%d')}")
        
        # Récupérer les données spatiales
        day_precip, day_anomalies, extreme_mask = self.get_spatial_mask_for_event(event_date)
        
        if extreme_mask is None:
            print(f"❌ Données non trouvées pour {event_date}")
            return
        
        # Identifier la région et les villes proches
        region = self.identify_closest_region(event_data['centroid_lat'], event_data['centroid_lon'])
        nearby_cities = self.identify_nearby_cities(event_data['centroid_lat'], event_data['centroid_lon'])
        
        # Créer la figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        saison_label = "Saison sèche" if event_data['saison'] == 'Saison_seche' else "Saison des pluies"
        
        # Titre avec références géographiques
        title = f'Événement #{event_idx} - {event_date.strftime("%Y-%m-%d")} ({saison_label})\n'
        title += f'Région: {region} - Couverture: {event_data["coverage_percent"]:.1f}%'
        if nearby_cities:
            title += f' - Proche de: {nearby_cities[0]["name"]} ({nearby_cities[0]["distance_km"]:.0f} km)'
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 1. Carte des précipitations avec références
        im1 = axes[0].contourf(self.lons, self.lats, day_precip, 
                              levels=20, cmap='Blues', extend='max')
        axes[0].contour(self.lons, self.lats, extreme_mask.astype(int), 
                       levels=[0.5], colors='red', linewidths=2.5)
        
        self.add_geographic_references_to_map(axes[0], event_data['centroid_lat'], event_data['centroid_lon'])
        
        plt.colorbar(im1, ax=axes[0], label='Précipitation (mm)', shrink=0.8)
        axes[0].set_title(f'Précipitations\nMax: {event_data["max_precip"]:.1f} mm')
        axes[0].set_xlabel('Longitude (°)')
        axes[0].set_ylabel('Latitude (°)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right', fontsize=8)
        
        # 2. Carte des anomalies avec références
        vmax = max(5, np.nanmax(day_anomalies[extreme_mask]) if extreme_mask.any() else 5)
        im2 = axes[1].contourf(self.lons, self.lats, day_anomalies, 
                              levels=np.linspace(-2, vmax, 20), cmap='RdYlBu_r', extend='both')
        axes[1].contour(self.lons, self.lats, extreme_mask.astype(int), 
                       levels=[0.5], colors='black', linewidths=2.5)
        
        self.add_geographic_references_to_map(axes[1], event_data['centroid_lat'], event_data['centroid_lon'])
        
        plt.colorbar(im2, ax=axes[1], label='Anomalie (σ)', shrink=0.8)
        axes[1].set_title(f'Anomalies Standardisées\nMax: {event_data["max_anomaly"]:.1f}σ')
        axes[1].set_xlabel('Longitude (°)')
        axes[1].set_ylabel('Latitude (°)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Zone de couverture avec analyse détaillée
        coverage_map = np.zeros_like(extreme_mask, dtype=float)
        coverage_map[extreme_mask] = day_precip[extreme_mask]
        
        if coverage_map.max() > 0:
            im3 = axes[2].contourf(self.lons, self.lats, coverage_map, 
                                  levels=15, cmap='Reds', extend='max')
            plt.colorbar(im3, ax=axes[2], label='Précipitation (mm)', shrink=0.8)
        
        self.add_geographic_references_to_map(axes[2], event_data['centroid_lat'], event_data['centroid_lon'])
        
        axes[2].set_title(f'Zone de Couverture - Région: {region}\n{event_data["coverage_points"]} points affectés')
        axes[2].set_xlabel('Longitude (°)')
        axes[2].set_ylabel('Latitude (°)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        filename = f"top10_event_{event_idx:02d}_{event_date.strftime('%Y%m%d')}_{region.replace(' ', '_')}.png"
        output_path = project_root / "outputs/visualizations/spatial" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def create_reference_map_senegal(self):
        """Crée une carte de référence complète du Sénégal."""
        
        print("\n🗺️  CRÉATION DE LA CARTE DE RÉFÉRENCE DU SÉNÉGAL")
        print("-" * 50)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Titre
        ax.set_title('Carte de Référence du Sénégal\nVilles principales, Régions et Événements Extrêmes', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Ajouter un fond de carte simple
        ax.add_patch(patches.Rectangle(
            (SENEGAL_BOUNDS['lon_min'], SENEGAL_BOUNDS['lat_min']),
            SENEGAL_BOUNDS['lon_max'] - SENEGAL_BOUNDS['lon_min'],
            SENEGAL_BOUNDS['lat_max'] - SENEGAL_BOUNDS['lat_min'],
            facecolor='lightblue', alpha=0.1, edgecolor='none'
        ))
        
        # Ajouter toutes les références géographiques
        self.add_geographic_references_to_map(ax)
        
        # Ajouter les centroïdes des top 10 événements
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        for i, (event_date, event_data) in enumerate(self.top10_events.iterrows(), 1):
            ax.scatter(event_data['centroid_lon'], event_data['centroid_lat'], 
                      c=[colors[i-1]], s=event_data['coverage_percent']*5, 
                      alpha=0.8, edgecolors='black', linewidth=1)
            
            # Numéroter les événements
            ax.annotate(f'{i}', (event_data['centroid_lon'], event_data['centroid_lat']),
                       xytext=(0, 0), textcoords='offset points', 
                       fontsize=10, fontweight='bold', color='white',
                       ha='center', va='center')
        
        # Légendes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, label='Capitale (Dakar)'),
            Patch(facecolor='blue', alpha=0.8, label='Villes principales'),
            Patch(facecolor='gray', alpha=0.5, label='Limites régionales'),
            Patch(facecolor='black', alpha=0.8, label='Frontières nationales')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # Configuration des axes
        ax.set_xlabel('Longitude (°)', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(SENEGAL_BOUNDS['lon_min']-0.1, SENEGAL_BOUNDS['lon_max']+0.1)
        ax.set_ylim(SENEGAL_BOUNDS['lat_min']-0.1, SENEGAL_BOUNDS['lat_max']+0.1)
        
        # Sauvegarder
        output_path = project_root / "outputs/visualizations/spatial/carte_reference_senegal.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Carte de référence sauvegardée: carte_reference_senegal.png")
    
    def create_synthesis_map_with_references(self):
        """Crée une carte de synthèse avec références géographiques."""
        
        print("\n🗺️  CRÉATION DE LA CARTE DE SYNTHÈSE AVEC RÉFÉRENCES")
        print("-" * 50)
        
        # Créer une grille de fréquence d'impact
        frequency_grid = np.zeros((len(self.lats), len(self.lons)))
        intensity_grid = np.zeros((len(self.lats), len(self.lons)))
        
        valid_events = 0
        for i, (event_date, event_data) in enumerate(self.top10_events.iterrows(), 1):
            day_precip, day_anomalies, extreme_mask = self.get_spatial_mask_for_event(event_date)
            
            if extreme_mask is not None:
                frequency_grid += extreme_mask.astype(float)
                intensity_grid += np.where(extreme_mask, day_precip, 0)
                valid_events += 1
        
        print(f"   Événements intégrés dans la synthèse: {valid_events}/10")
        
        # Créer la figure de synthèse
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Synthèse Spatiale des Top 10 Événements Extrêmes - Sénégal (1981-2023)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Fréquence d'impact avec références
        max_freq = max(1, int(frequency_grid.max()))
        levels_freq = range(max_freq + 1)
        im1 = axes[0,0].contourf(self.lons, self.lats, frequency_grid, 
                                levels=levels_freq, cmap='YlOrRd', extend='max')
        
        self.add_geographic_references_to_map(axes[0,0])
        
        cbar1 = plt.colorbar(im1, ax=axes[0,0], label='Nombre d\'événements', shrink=0.8)
        axes[0,0].set_title('Fréquence d\'Impact par Zone\n(Avec villes et régions)')
        axes[0,0].set_xlabel('Longitude (°)')
        axes[0,0].set_ylabel('Latitude (°)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Intensité moyenne avec références
        intensity_avg = np.where(frequency_grid > 0, intensity_grid / frequency_grid, 0)
        if intensity_avg.max() > 0:
            im2 = axes[0,1].contourf(self.lons, self.lats, intensity_avg, 
                                    levels=15, cmap='Blues', extend='max')
            cbar2 = plt.colorbar(im2, ax=axes[0,1], label='Précipitation moyenne (mm)', shrink=0.8)
        
        self.add_geographic_references_to_map(axes[0,1])
        
        axes[0,1].set_title('Intensité Moyenne par Zone\n(Avec villes et régions)')
        axes[0,1].set_xlabel('Longitude (°)')
        axes[0,1].set_ylabel('Latitude (°)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Centroïdes avec références régionales
        for i, (event_date, event_data) in enumerate(self.top10_events.iterrows(), 1):
            region = self.identify_closest_region(event_data['centroid_lat'], event_data['centroid_lon'])
            color = 'red' if event_data['saison'] == 'Saison_seche' else 'blue'
            size = event_data['coverage_percent'] * 3
            
            axes[1,0].scatter(event_data['centroid_lon'], event_data['centroid_lat'], 
                            c=color, s=size, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Annoter avec numéro et région
            axes[1,0].annotate(f'{i}\n{region[:4]}', 
                             (event_data['centroid_lon'], event_data['centroid_lat']),
                             xytext=(3, 3), textcoords='offset points', 
                             fontsize=7, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        self.add_geographic_references_to_map(axes[1,0])
        
        axes[1,0].set_title('Localisation des Centroïdes par Région\n(Taille ∝ Couverture spatiale)')
        axes[1,0].set_xlabel('Longitude (°)')
        axes[1,0].set_ylabel('Latitude (°)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Légende pour les saisons
        axes[1,0].scatter([], [], c='blue', s=100, label='Saison des pluies', alpha=0.7)
        axes[1,0].scatter([], [], c='red', s=100, label='Saison sèche', alpha=0.7)
        axes[1,0].legend(loc='upper right')
        
        # 4. Zones de vulnérabilité avec références
        vulnerability_map = frequency_grid * intensity_avg
        if vulnerability_map.max() > 0:
            im4 = axes[1,1].contourf(self.lons, self.lats, vulnerability_map, 
                                    levels=15, cmap='Spectral_r', extend='max')
            cbar4 = plt.colorbar(im4, ax=axes[1,1], label='Indice de Vulnérabilité', shrink=0.8)
            
            # Contours des zones très vulnérables
            high_impact_mask = frequency_grid >= 3
            if high_impact_mask.any():
                axes[1,1].contour(self.lons, self.lats, high_impact_mask.astype(int), 
                                 levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        self.add_geographic_references_to_map(axes[1,1])
        
        axes[1,1].set_title('Zones de Vulnérabilité Maximale\n(Fréquence × Intensité)')
        axes[1,1].set_xlabel('Longitude (°)')
        axes[1,1].set_ylabel('Latitude (°)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        output_path = project_root / "outputs/visualizations/spatial/synthesis_map_with_references.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Carte de synthèse avec références sauvegardée: synthesis_map_with_references.png")
        
        return frequency_grid, intensity_avg, vulnerability_map
    
    def generate_geographic_report(self):
        """Génère un rapport avec localisation géographique précise."""
        
        print("\n📄 GÉNÉRATION DU RAPPORT GÉOGRAPHIQUE")
        print("-" * 50)
        
        report_path = project_root / "outputs/reports/rapport_geographique_top10.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RAPPORT GÉOGRAPHIQUE - TOP 10 ÉVÉNEMENTS EXTRÊMES\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Période d'analyse: 1981-2023\n")
            f.write(f"Zone d'étude: Sénégal ({SENEGAL_BOUNDS['lat_min']}°N-{SENEGAL_BOUNDS['lat_max']}°N, ")
            f.write(f"{SENEGAL_BOUNDS['lon_min']}°E-{SENEGAL_BOUNDS['lon_max']}°E)\n\n")
            
            f.write("LOCALISATION GÉOGRAPHIQUE DÉTAILLÉE\n")
            f.write("-" * 45 + "\n\n")
            
            for i, (event_date, event_data) in enumerate(self.top10_events.iterrows(), 1):
                region = self.identify_closest_region(event_data['centroid_lat'], event_data['centroid_lon'])
                nearby_cities = self.identify_nearby_cities(event_data['centroid_lat'], event_data['centroid_lon'])
                
                f.write(f"ÉVÉNEMENT #{i} - {event_date.strftime('%Y-%m-%d')}\n")
                f.write(f"Région administrative: {region}\n")
                f.write(f"Coordonnées centroïde: {event_data['centroid_lat']:.4f}°N, {event_data['centroid_lon']:.4f}°E\n")
                f.write(f"Couverture spatiale: {event_data['coverage_percent']:.2f}%\n")
                f.write(f"Précipitation maximale: {event_data['max_precip']:.2f} mm\n")
                f.write(f"Anomalie maximale: {event_data['max_anomaly']:.2f}σ\n")
                f.write(f"Saison: {event_data['saison'].replace('_', ' ').title()}\n")
                
                if nearby_cities:
                    f.write(f"Villes proches:\n")
                    for city in nearby_cities[:3]:  # Top 3 villes les plus proches
                        f.write(f"  - {city['name']}: {city['distance_km']:.1f} km au {city['direction']}\n")
                else:
                    f.write(f"Zone rurale/isolée (pas de villes majeures dans un rayon de 111 km)\n")
                
                # Calcul de l'étendue géographique
                day_precip, day_anomalies, extreme_mask = self.get_spatial_mask_for_event(event_date)
                if extreme_mask is not None and extreme_mask.any():
                    lat_indices, lon_indices = np.where(extreme_mask)
                    if len(lat_indices) > 0:
                        affected_lats = self.lats[lat_indices]
                        affected_lons = self.lons[lon_indices]
                        
                        lat_span = np.max(affected_lats) - np.min(affected_lats)
                        lon_span = np.max(affected_lons) - np.min(affected_lons)
                        lat_extent_km = lat_span * 111
                        lon_extent_km = lon_span * 111 * np.cos(np.radians(np.mean(affected_lats)))
                        
                        f.write(f"Étendue géographique:\n")
                        f.write(f"  - Latitude: {np.min(affected_lats):.3f}° à {np.max(affected_lats):.3f}° (span: {lat_span:.3f}°)\n")
                        f.write(f"  - Longitude: {np.min(affected_lons):.3f}° à {np.max(affected_lons):.3f}° (span: {lon_span:.3f}°)\n")
                        f.write(f"  - Dimensions: {lat_extent_km:.0f} km × {lon_extent_km:.0f} km\n")
                        f.write(f"  - Points de grille affectés: {len(lat_indices)}\n")
                
                f.write("\n" + "-" * 60 + "\n\n")
            
            # Synthèse par région
            f.write("SYNTHÈSE PAR RÉGION ADMINISTRATIVE\n")
            f.write("-" * 40 + "\n\n")
            
            region_count = {}
            region_stats = {}
            
            for _, event_data in self.top10_events.iterrows():
                region = self.identify_closest_region(event_data['centroid_lat'], event_data['centroid_lon'])
                region_count[region] = region_count.get(region, 0) + 1
                
                if region not in region_stats:
                    region_stats[region] = {
                        'events': [],
                        'total_coverage': 0,
                        'total_precip': 0,
                        'max_precip': 0
                    }
                
                region_stats[region]['events'].append(event_data)
                region_stats[region]['total_coverage'] += event_data['coverage_percent']
                region_stats[region]['total_precip'] += event_data['max_precip']
                region_stats[region]['max_precip'] = max(region_stats[region]['max_precip'], event_data['max_precip'])
            
            for region, count in sorted(region_count.items(), key=lambda x: x[1], reverse=True):
                stats = region_stats[region]
                avg_coverage = stats['total_coverage'] / count
                avg_precip = stats['total_precip'] / count
                
                f.write(f"{region}: {count} événement(s)\n")
                f.write(f"  - Couverture moyenne: {avg_coverage:.2f}%\n")
                f.write(f"  - Précipitation moyenne: {avg_precip:.2f} mm\n")
                f.write(f"  - Précipitation maximale: {stats['max_precip']:.2f} mm\n")
                f.write(f"  - Ville principale: {SENEGAL_REGIONS[region]['lat']:.2f}°N, {SENEGAL_REGIONS[region]['lon']:.2f}°E\n\n")
            
            # Recommandations
            f.write("RECOMMANDATIONS POUR LA GESTION DES RISQUES\n")
            f.write("-" * 50 + "\n\n")
            
            priority_regions = [r for r, c in region_count.items() if c >= 2]
            if priority_regions:
                f.write(f"RÉGIONS PRIORITAIRES (≥2 événements):\n")
                for region in priority_regions:
                    f.write(f"- {region}: surveillance météorologique renforcée recommandée\n")
                f.write("\n")
            
            f.write("MESURES RECOMMANDÉES:\n")
            f.write("- Installation de pluviomètres supplémentaires dans les zones identifiées\n")
            f.write("- Planification d'urgence adaptée aux spécificités régionales\n")
            f.write("- Systèmes d'alerte précoce ciblés par région\n")
            f.write("- Infrastructure de drainage renforcée dans les zones vulnérables\n")
            f.write("- Coordination inter-régionale pour la gestion des événements transfrontaliers\n")
        
        print(f"✅ Rapport géographique sauvegardé: {report_path}")
        
        return region_count, region_stats
    
    def run_complete_analysis_with_references(self):
        """Lance l'analyse spatiale complète avec références géographiques."""
        
        print("ANALYSE SPATIALE AVEC RÉFÉRENCES GÉOGRAPHIQUES - SÉNÉGAL")
        print("=" * 70)
        
        # Charger les données
        if not self.load_existing_data():
            print("❌ Impossible de charger les données")
            return False
        
        # Sélectionner les top 10
        self.get_top10_events()
        
        # Créer la carte de référence du Sénégal
        self.create_reference_map_senegal()
        
        # Créer les cartes individuelles avec références
        print(f"\n🗺️  CRÉATION DES CARTES INDIVIDUELLES AVEC RÉFÉRENCES")
        print("-" * 60)
        
        generated_files = []
        for i, (event_date, event_data) in enumerate(self.top10_events.iterrows(), 1):
            filename = self.create_individual_event_map(i, event_date, event_data)
            if filename:
                generated_files.append(filename)
        
        print(f"✅ {len(generated_files)} cartes individuelles créées avec références géographiques")
        
        # Créer la carte de synthèse avec références
        frequency_grid, intensity_avg, vulnerability_map = self.create_synthesis_map_with_references()
        
        # Générer le rapport géographique
        region_count, region_stats = self.generate_geographic_report()
        
        # Résumé final
        print(f"\n" + "=" * 70)
        print("✅ ANALYSE SPATIALE AVEC RÉFÉRENCES TERMINÉE")
        print("=" * 70)
        print(f"📊 Événements analysés: {len(self.top10_events)}")
        print(f"🗺️  Cartes avec références: {len(generated_files)} fichiers individuels")
        print(f"📋 Cartes de synthèse: 2 fichiers (référence + synthèse)")
        print(f"📄 Rapport géographique: 1 fichier détaillé")
        print(f"📁 Dossier de sortie: outputs/visualizations/spatial/")
        
        print(f"\n🏛️  RÉPARTITION PAR RÉGION:")
        for region, count in sorted(region_count.items(), key=lambda x: x[1], reverse=True):
            stats = region_stats[region]
            avg_coverage = stats['total_coverage'] / count
            print(f"   • {region}: {count} événement(s) - Couverture moy: {avg_coverage:.1f}%")
        
        print(f"\n🎯 FICHIERS GÉNÉRÉS:")
        print(f"   • carte_reference_senegal.png - Carte complète du Sénégal")
        print(f"   • synthesis_map_with_references.png - Synthèse avec villes/régions")
        print(f"   • {len(generated_files)} cartes individuelles avec localisation précise")
        print(f"   • rapport_geographique_top10.txt - Rapport avec coordonnées exactes")
        
        return True

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale."""
    print("ANALYSE SPATIALE AVEC RÉFÉRENCES GÉOGRAPHIQUES DU SÉNÉGAL")
    print("=" * 70)
    
    analyzer = SpatialAnalysisWithReferences()
    success = analyzer.run_complete_analysis_with_references()
    
    if success:
        print("\n🎉 ANALYSE SPATIALE AVEC RÉFÉRENCES RÉUSSIE!")
        print("\n📍 LOCALISATION PRÉCISE OBTENUE:")
        print("• Chaque événement localisé dans sa région administrative")
        print("• Distances calculées aux villes principales")
        print("• Coordonnées GPS exactes des centroïdes")
        print("• Dimensions géographiques en kilomètres")
        print("• Zones de vulnérabilité identifiées par région")
        print("\nConsultez le dossier outputs/visualizations/spatial/")
        return 0
    else:
        print("\n❌ ÉCHEC DE L'ANALYSE SPATIALE")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)