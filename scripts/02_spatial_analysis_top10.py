#!/usr/bin/env python3
# scripts/02_spatial_analysis_top10.py
"""
Analyse spatiale détaillée des 10 événements de précipitations extrêmes les plus étendus au Sénégal
avec références géographiques précises (villes, régions, départements).
VERSION REFACTORISÉE utilisant les modules centralisés.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    from src.utils.geographic_references import SenegalGeography
    from src.analysis.spatial_metrics import SpatialMetricsCalculator
    from src.visualization.geographic_maps import SenegalMapVisualizer
    from src.reports.spatial_reports import SpatialReportGenerator
    print("✅ Tous les modules importés avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

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
        
        # Modules centralisés
        self.metrics_calculator = SpatialMetricsCalculator()
        self.map_visualizer = SenegalMapVisualizer()
        self.report_generator = SpatialReportGenerator(
            project_root / "outputs/reports"
        )
        
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
                print("⚠️ Données CHIRPS non trouvées")
                return False
            
            # Charger les anomalies
            anom_file = project_root / "data/processed/standardized_anomalies_senegal.npz"
            if anom_file.exists():
                anom_data = np.load(anom_file)
                self.anomalies = anom_data['anomalies']
                print(f"✅ Anomalies chargées: {self.anomalies.shape}")
            else:
                print("⚠️ Anomalies non trouvées")
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
            region = self.metrics_calculator.geo.identify_closest_region(
                event['centroid_lat'], event['centroid_lon']
            )
            saison_label = "Pluies" if event['saison'] == 'Saison_des_pluies' else "Sèche"
            print(f"{i:2d}. {date.strftime('%Y-%m-%d')} - Région: {region}")
            print(f"    Couverture: {event['coverage_percent']:5.1f}%, Précip: {event['max_precip']:6.1f} mm")
            print(f"    Centroïde: ({event['centroid_lat']:.3f}°N, {event['centroid_lon']:.3f}°E)")
            print(f"    Saison: {saison_label}")
            print()
        
        return self.top10_events
    
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
    
    def create_individual_event_map(self, event_idx, event_date, event_data):
        """Crée une carte détaillée pour un événement avec références géographiques."""
        
        print(f"   Création carte événement #{event_idx}: {event_date.strftime('%Y-%m-%d')}")
        
        # Récupérer les données spatiales
        day_precip, day_anomalies, extreme_mask = self.get_spatial_mask_for_event(event_date)
        
        if extreme_mask is None:
            print(f"❌ Données non trouvées pour {event_date}")
            return
        
        # Calculer les métriques spatiales avec le module centralisé
        spatial_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            event_date, event_data, self.precip_data, self.anomalies, 
            self.lats, self.lons, self.dates, event_idx
        )
        
        # Créer la figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        saison_label = "Saison sèche" if event_data['saison'] == 'Saison_seche' else "Saison des pluies"
        
        # Titre avec références géographiques
        title = f'Événement #{event_idx} - {event_date.strftime("%Y-%m-%d")} ({saison_label})\n'
        title += f'Région: {spatial_metrics["region"]} - Couverture: {event_data["coverage_percent"]:.1f}%'
        
        # Identifier les villes proches
        nearby_cities = self.metrics_calculator.geo.identify_nearby_cities(
            event_data['centroid_lat'], event_data['centroid_lon']
        )
        if nearby_cities:
            title += f' - Proche de: {nearby_cities[0]["name"]} ({nearby_cities[0]["distance_km"]:.0f} km)'
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 1. Carte des précipitations avec références
        im1 = axes[0].contourf(self.lons, self.lats, day_precip, 
                              levels=20, cmap='Blues', extend='max')
        axes[0].contour(self.lons, self.lats, extreme_mask.astype(int), 
                       levels=[0.5], colors='red', linewidths=2.5)
        
        self.map_visualizer.add_geographic_references(
            axes[0], event_data['centroid_lat'], event_data['centroid_lon']
        )
        
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
        
        self.map_visualizer.add_geographic_references(
            axes[1], event_data['centroid_lat'], event_data['centroid_lon']
        )
        
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
        
        self.map_visualizer.add_geographic_references(
            axes[2], event_data['centroid_lat'], event_data['centroid_lon']
        )
        
        axes[2].set_title(f'Zone de Couverture - Région: {spatial_metrics["region"]}\n{event_data["coverage_points"]} points affectés')
        axes[2].set_xlabel('Longitude (°)')
        axes[2].set_ylabel('Latitude (°)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        filename = f"top10_event_{event_idx:02d}_{event_date.strftime('%Y%m%d')}_{spatial_metrics['region'].replace(' ', '_')}.png"
        output_path = project_root / "outputs/visualizations/spatial" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename, spatial_metrics
    
    def create_synthesis_map_with_references(self, all_spatial_metrics):
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
        
        self.map_visualizer.add_geographic_references(axes[0,0])
        
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
        
        self.map_visualizer.add_geographic_references(axes[0,1])
        
        axes[0,1].set_title('Intensité Moyenne par Zone\n(Avec villes et régions)')
        axes[0,1].set_xlabel('Longitude (°)')
        axes[0,1].set_ylabel('Latitude (°)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Centroïdes avec références régionales
        for i, metrics in enumerate(all_spatial_metrics, 1):
            color = 'red' if self.top10_events.iloc[i-1]['saison'] == 'Saison_seche' else 'blue'
            size = metrics['coverage_percent'] * 3
            
            axes[1,0].scatter(metrics['centroid_lon'], metrics['centroid_lat'], 
                            c=color, s=size, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Annoter avec numéro et région
            axes[1,0].annotate(f'{i}\n{metrics["region"][:4]}', 
                             (metrics['centroid_lon'], metrics['centroid_lat']),
                             xytext=(3, 3), textcoords='offset points', 
                             fontsize=7, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        self.map_visualizer.add_geographic_references(axes[1,0])
        
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
        
        self.map_visualizer.add_geographic_references(axes[1,1])
        
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
        self.map_visualizer.create_reference_map()
        
        # Créer les cartes individuelles avec références
        print(f"\n🗺️  CRÉATION DES CARTES INDIVIDUELLES AVEC RÉFÉRENCES")
        print("-" * 60)
        
        generated_files = []
        all_spatial_metrics = []
        
        for i, (event_date, event_data) in enumerate(self.top10_events.iterrows(), 1):
            filename, spatial_metrics = self.create_individual_event_map(i, event_date, event_data)
            if filename:
                generated_files.append(filename)
                all_spatial_metrics.append(spatial_metrics)
        
        print(f"✅ {len(generated_files)} cartes individuelles créées avec références géographiques")
        
        # Créer la carte de synthèse avec références
        frequency_grid, intensity_avg, vulnerability_map = self.create_synthesis_map_with_references(all_spatial_metrics)
        
        # Générer le rapport géographique avec le module centralisé
        report_path = self.report_generator.generate_comprehensive_report(
            all_spatial_metrics, 
            "top10_coverage_events",
            "Top 10 Événements par Couverture Spatiale"
        )
        
        # Résumé final
        print(f"\n" + "=" * 70)
        print("✅ ANALYSE SPATIALE AVEC RÉFÉRENCES TERMINÉE")
        print("=" * 70)
        print(f"📊 Événements analysés: {len(self.top10_events)}")
        print(f"🗺️  Cartes avec références: {len(generated_files)} fichiers individuels")
        print(f"📋 Cartes de synthèse: 2 fichiers (référence + synthèse)")
        print(f"📄 Rapport géographique: {report_path}")
        print(f"📁 Dossier de sortie: outputs/visualizations/spatial/")
        
        # Distribution régionale
        region_counts = {}
        for metrics in all_spatial_metrics:
            region = metrics['region']
            region_counts[region] = region_counts.get(region, 0) + 1
        
        print(f"\n🏛️  RÉPARTITION PAR RÉGION:")
        for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
            avg_coverage = np.mean([m['coverage_percent'] for m in all_spatial_metrics if m['region'] == region])
            print(f"   • {region}: {count} événement(s) - Couverture moy: {avg_coverage:.1f}%")
        
        print(f"\n🎯 FICHIERS GÉNÉRÉS:")
        print(f"   • carte_reference_senegal.png - Carte complète du Sénégal")
        print(f"   • synthesis_map_with_references.png - Synthèse avec villes/régions")
        print(f"   • {len(generated_files)} cartes individuelles avec localisation précise")
        print(f"   • {report_path.split('/')[-1]} - Rapport avec coordonnées exactes")
        
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