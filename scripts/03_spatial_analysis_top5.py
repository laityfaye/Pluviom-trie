#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/03_spatial_analysis_top5.py
"""
Analyse spatiale des 5 précipitations extrêmes les plus intenses au Sénégal.
VERSION REFACTORISÉE utilisant les modules centralisés.

Auteur: Laity FAYE  
Date: 2025
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import seaborn as sns
import json

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
    from src.visualization.geographic_maps import SenegalMapVisualizer
    from src.analysis.spatial_metrics import SpatialMetricsCalculator
    from src.reports.spatial_reports import SpatialReportGenerator
    print("✅ Tous les modules importés avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

class SpatialAnalysisTop5:
    """Classe pour l'analyse spatiale des TOP 5 événements les plus intenses."""
    
    def __init__(self):
        """Initialise l'analyseur spatial."""
        self.df_events = None
        self.precip_data = None
        self.anomalies = None
        self.dates = None
        self.lats = None
        self.lons = None
        self.top5_events = None
        
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
                
                # Debug - afficher les colonnes disponibles
                print(f"📊 Colonnes disponibles: {list(self.df_events.columns)}")
                print(f"📋 Exemple de données:")
                print(self.df_events.iloc[0])
                
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
                print("⚠️ Données CHIRPS non trouvées, utilisation des données du DataFrame uniquement")
                self.precip_data = None
            
            # Charger les anomalies
            anom_file = project_root / "data/processed/standardized_anomalies_senegal.npz"
            if anom_file.exists():
                anom_data = np.load(anom_file)
                self.anomalies = anom_data['anomalies']
                print(f"✅ Anomalies chargées: {self.anomalies.shape}")
            else:
                print("⚠️ Anomalies non trouvées")
                self.anomalies = None
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def get_top5_events_by_intensity(self):
        """Récupère les 5 événements les plus intenses (par précipitation max)."""
        print("\n📊 SÉLECTION DES TOP 5 ÉVÉNEMENTS LES PLUS INTENSES")
        print("=" * 60)
        
        # Trier par précipitation maximale décroissante
        self.top5_events = self.df_events.nlargest(5, 'max_precip').copy()
        
        print("Top 5 événements par intensité des précipitations:")
        for i, (date, event) in enumerate(self.top5_events.iterrows(), 1):
            region = self.metrics_calculator.geo.identify_closest_region(
                event['centroid_lat'], event['centroid_lon']
            )
            saison_label = "Pluies" if event['saison'] == 'Saison_des_pluies' else "Sèche"
            print(f"{i:2d}. {date.strftime('%Y-%m-%d')} - Région: {region}")
            print(f"    Intensité: {event['max_precip']:6.1f} mm, Couverture: {event['coverage_percent']:5.1f}%")
            print(f"    Centroïde: ({event['centroid_lat']:.3f}°N, {event['centroid_lon']:.3f}°E)")
            print(f"    Saison: {saison_label}")
            print()
        
        return self.top5_events
    
    def get_spatial_mask_for_event(self, event_date):
        """Récupère le masque spatial pour un événement donné."""
        if self.precip_data is None or self.anomalies is None:
            return None, None, None
            
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
    
    def create_individual_event_map_top5(self, event_idx, event_date, event_data, spatial_metrics):
        """Crée une carte détaillée pour un événement TOP 5."""
        
        print(f"   Création carte TOP 5 #{event_idx}: {event_date.strftime('%Y-%m-%d')}")
        
        # Essayer de récupérer les données spatiales
        day_precip, day_anomalies, extreme_mask = self.get_spatial_mask_for_event(event_date)
        
        # Créer la figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        saison_label = "Saison sèche" if event_data['saison'] == 'Saison_seche' else "Saison des pluies"
        
        # Titre avec métriques précises
        title = f'TOP 5 Rang #{event_idx} - {event_date.strftime("%Y-%m-%d")} ({saison_label})\n'
        title += f'Région: {spatial_metrics["region"]} - Intensité: {spatial_metrics["max_intensity_mm"]:.1f} mm - Surface: {spatial_metrics["total_area_km2"]:.0f} km²'
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 1. Carte des précipitations avec références
        if day_precip is not None and extreme_mask is not None:
            im1 = axes[0].contourf(self.lons, self.lats, day_precip, 
                                  levels=20, cmap='Blues', extend='max')
            axes[0].contour(self.lons, self.lats, extreme_mask.astype(int), 
                           levels=[0.5], colors='red', linewidths=2.5)
        else:
            # Carte simplifiée si pas de données CHIRPS
            axes[0].text(0.5, 0.5, f'Événement du {event_date.strftime("%Y-%m-%d")}\n'
                        f'Intensité: {spatial_metrics["max_intensity_mm"]:.1f} mm\n'
                        f'Données CHIRPS non disponibles',
                        ha='center', va='center', transform=axes[0].transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        self.map_visualizer.add_geographic_references(
            axes[0], spatial_metrics['centroid_lat'], spatial_metrics['centroid_lon']
        )
        
        if day_precip is not None:
            plt.colorbar(im1, ax=axes[0], label='Précipitation (mm)', shrink=0.8)
        axes[0].set_title(f'Précipitations\nMax: {spatial_metrics["max_intensity_mm"]:.1f} mm')
        axes[0].set_xlabel('Longitude (°)')
        axes[0].set_ylabel('Latitude (°)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right', fontsize=8)
        
        # 2. Carte des anomalies avec références
        if day_anomalies is not None and extreme_mask is not None:
            vmax = max(5, np.nanmax(day_anomalies[extreme_mask]) if extreme_mask.any() else 5)
            im2 = axes[1].contourf(self.lons, self.lats, day_anomalies, 
                                  levels=np.linspace(-2, vmax, 20), cmap='RdYlBu_r', extend='both')
            axes[1].contour(self.lons, self.lats, extreme_mask.astype(int), 
                           levels=[0.5], colors='black', linewidths=2.5)
            plt.colorbar(im2, ax=axes[1], label='Anomalie (σ)', shrink=0.8)
        else:
            axes[1].text(0.5, 0.5, f'Anomalies\n'
                        f'Max anomalie: {event_data.get("max_anomaly", "N/A")}\n'
                        f'Données non disponibles',
                        ha='center', va='center', transform=axes[1].transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        self.map_visualizer.add_geographic_references(
            axes[1], spatial_metrics['centroid_lat'], spatial_metrics['centroid_lon']
        )
        
        axes[1].set_title(f'Anomalies Standardisées\nMax: {event_data.get("max_anomaly", "N/A")}')
        axes[1].set_xlabel('Longitude (°)')
        axes[1].set_ylabel('Latitude (°)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Métriques spatiales précises
        axes[2].text(0.1, 0.9, f'MÉTRIQUES SPATIALES PRÉCISES', fontsize=14, fontweight='bold', 
                    transform=axes[2].transAxes)
        
        metrics_text = f"""
LOCALISATION:
• Région: {spatial_metrics["region"]}
• Centroïde: {spatial_metrics["centroid_lat"]:.3f}°N, {abs(spatial_metrics["centroid_lon"]):.3f}°W
• Position max: {spatial_metrics["max_intensity_lat"]:.3f}°N, {abs(spatial_metrics["max_intensity_lon"]):.3f}°W

ÉTENDUE SPATIALE:
• Surface totale: {spatial_metrics["total_area_km2"]:.0f} km²
• Couverture: {spatial_metrics["coverage_percent"]:.2f}% du territoire
• Pixels affectés: {spatial_metrics["num_pixels_affected"]} points
• Étendue: {spatial_metrics["lat_extent_deg"]:.2f}° × {spatial_metrics["lon_extent_deg"]:.2f}°

INTENSITÉ:
• Maximum: {spatial_metrics["max_intensity_mm"]:.1f} mm/jour
• Moyenne: {spatial_metrics["intensity_stats"]["mean"]:.1f} mm/jour
• Médiane: {spatial_metrics["intensity_stats"]["median"]:.1f} mm/jour
• 95e percentile: {spatial_metrics["intensity_stats"]["p95"]:.1f} mm/jour
• Écart-type: {spatial_metrics["intensity_stats"]["std"]:.1f} mm/jour
        """
        
        axes[2].text(0.05, 0.8, metrics_text, fontsize=10, fontfamily='monospace',
                    transform=axes[2].transAxes, verticalalignment='top')
        
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Sauvegarder
        filename = f"top5_intense_event_{event_idx:02d}_{event_date.strftime('%Y%m%d')}_{spatial_metrics['region'].replace(' ', '_')}.png"
        output_path = project_root / "outputs/visualizations/spatial_top5" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def create_comparative_analysis_top5(self, spatial_results):
        """Crée une analyse comparative des TOP 5."""
        
        print("\n📊 CRÉATION DE L'ANALYSE COMPARATIVE TOP 5")
        print("-" * 50)
        
        df = pd.DataFrame(spatial_results)
        
        # Vérifier les données avant de créer les graphiques
        print(f"Debug - Données disponibles:")
        print(f"  Intensités: {df['max_intensity_mm'].values}")
        print(f"  Surfaces: {df['total_area_km2'].values}")
        print(f"  Couvertures: {df['coverage_percent'].values}")
        print(f"  Régions: {df['region'].values}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analyse Comparative des 5 Précipitations Extrêmes les Plus Intenses', 
                     fontsize=16, fontweight='bold')
        
        colors = plt.cm.viridis(np.linspace(0, 1, 5))
        
        # 1. Intensité vs Surface
        ax1 = axes[0, 0]
        
        # Filtrer les valeurs valides
        valid_mask = ~(np.isnan(df['max_intensity_mm']) | np.isnan(df['total_area_km2']))
        if valid_mask.any():
            x_vals = df['max_intensity_mm'][valid_mask]
            y_vals = df['total_area_km2'][valid_mask]
            ranks = df['rank'][valid_mask]
            
            scatter = ax1.scatter(x_vals, y_vals, s=150, c=colors[:len(x_vals)], 
                                 alpha=0.8, edgecolors='black')
            
            for i, (x, y, rank) in enumerate(zip(x_vals, y_vals, ranks)):
                ax1.annotate(f'#{rank}', (x, y), xytext=(5, 5), 
                            textcoords='offset points', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'Données non disponibles', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        ax1.set_xlabel('Intensité Maximale (mm/jour)', fontweight='bold')
        ax1.set_ylabel('Surface Affectée (km²)', fontweight='bold')
        ax1.set_title('Relation Intensité-Surface', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Couverture par rang d'intensité
        ax2 = axes[0, 1]
        valid_coverage = ~np.isnan(df['coverage_percent'])
        if valid_coverage.any():
            bars = ax2.bar(df['rank'][valid_coverage], df['coverage_percent'][valid_coverage], 
                          color=colors[:len(df[valid_coverage])], alpha=0.8, edgecolor='black')
            ax2.set_xticks(df['rank'][valid_coverage])
            
            # Annotations sur les barres
            for bar, pct in zip(bars, df['coverage_percent'][valid_coverage]):
                if not np.isnan(pct):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{pct:.2f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Données non disponibles', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        ax2.set_xlabel('Rang d\'Intensité', fontweight='bold')
        ax2.set_ylabel('Couverture Spatiale (%)', fontweight='bold')
        ax2.set_title('Couverture par Rang d\'Intensité', fontweight='bold')
        
        # 3. Distribution géographique
        ax3 = axes[1, 0]
        valid_coords = ~(np.isnan(df['centroid_lat']) | np.isnan(df['centroid_lon']) | np.isnan(df['max_intensity_mm']))
        if valid_coords.any():
            x_vals = df['centroid_lon'][valid_coords]
            y_vals = df['centroid_lat'][valid_coords]
            sizes = df['max_intensity_mm'][valid_coords] * 3
            ranks = df['rank'][valid_coords]
            regions = df['region'][valid_coords]
            
            scatter = ax3.scatter(x_vals, y_vals, s=sizes, c=colors[:len(x_vals)], 
                                 alpha=0.8, edgecolors='black')
            
            for i, (x, y, rank, region) in enumerate(zip(x_vals, y_vals, ranks, regions)):
                ax3.annotate(f'#{rank}\n{region[:4]}', (x, y), xytext=(5, 5), 
                            textcoords='offset points', fontweight='bold', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'Données non disponibles', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        ax3.set_xlabel('Longitude (°)', fontweight='bold')
        ax3.set_ylabel('Latitude (°)', fontweight='bold')
        ax3.set_title('Distribution Géographique\n(Taille ∝ Intensité)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistiques d'intensité
        ax4 = axes[1, 1]
        valid_stats = [r for r in spatial_results if not np.isnan(r.get('intensity_stats', {}).get('mean', np.nan))]
        
        if valid_stats:
            intensity_means = [r['intensity_stats']['mean'] for r in valid_stats]
            intensity_p95 = [r['intensity_stats']['p95'] for r in valid_stats]
            ranks = [r['rank'] for r in valid_stats]
            
            x_pos = np.arange(len(ranks))
            width = 0.35
            
            bars1 = ax4.bar(x_pos - width/2, intensity_means, width, 
                           label='Intensité Moyenne', color='lightblue', alpha=0.8, edgecolor='black')
            bars2 = ax4.bar(x_pos + width/2, intensity_p95, width, 
                           label='95e Percentile', color='darkblue', alpha=0.8, edgecolor='black')
            
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'#{r}' for r in ranks])
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Données non disponibles', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        ax4.set_xlabel('Rang d\'Intensité', fontweight='bold')
        ax4.set_ylabel('Intensité (mm/jour)', fontweight='bold')
        ax4.set_title('Statistiques d\'Intensité', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        output_path = project_root / "outputs/visualizations/spatial_top5/comparative_analysis_top5.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Analyse comparative sauvegardée: comparative_analysis_top5.png")
    
    def create_synthesis_map_safe(self, spatial_results):
        """Version sécurisée de la carte de synthèse des TOP 5."""
        
        print("\n🗺️  CRÉATION DE LA CARTE DE SYNTHÈSE TOP 5 (VERSION SÉCURISÉE)")
        print("-" * 50)
        
        # Créer une figure simple sans éléments complexes
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Titre
        ax.set_title('Synthèse Géographique des 5 Précipitations Extrêmes les Plus Intenses\n'
                     'Sénégal (1981-2023) - Distribution Spatiale Précise', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Couleurs pour chaque événement
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
        
        # Ajouter les références géographiques avec le module centralisé
        self.map_visualizer.add_geographic_references(ax)
        
        # Ajouter les événements TOP 5
        valid_events = 0
        for i, result in enumerate(spatial_results):
            # Vérifier les coordonnées
            if (np.isnan(result['centroid_lat']) or np.isnan(result['centroid_lon']) or
                not (SENEGAL_BOUNDS['lat_min'] <= result['centroid_lat'] <= SENEGAL_BOUNDS['lat_max']) or
                not (SENEGAL_BOUNDS['lon_min'] <= result['centroid_lon'] <= SENEGAL_BOUNDS['lon_max'])):
                print(f"⚠️  Coordonnées invalides pour l'événement #{result['rank']}")
                continue
            
            # Taille proportionnelle à l'intensité
            size = min(500, max(50, result['max_intensity_mm'] * 1.5))
            
            # Point principal
            ax.scatter(result['centroid_lon'], result['centroid_lat'], 
                      s=size, c=colors[i], alpha=0.8, 
                      edgecolors='black', linewidth=2, zorder=10)
            
            # Annotation détaillée
            ax.annotate(f"#{result['rank']}\n{result['max_intensity_mm']:.0f}mm", 
                       (result['centroid_lon'], result['centroid_lat']),
                       xytext=(8, 8), textcoords='offset points',
                       fontweight='bold', fontsize=10, color='darkred',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                       zorder=11)
            
            valid_events += 1
        
        # Configuration des axes
        ax.set_xlabel('Longitude (°)', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(SENEGAL_BOUNDS['lon_min'], SENEGAL_BOUNDS['lon_max'])
        ax.set_ylim(SENEGAL_BOUNDS['lat_min'], SENEGAL_BOUNDS['lat_max'])
        
        # Statistiques dans un encadré
        df = pd.DataFrame(spatial_results)
        stats_text = f"""TOP 5 ÉVÉNEMENTS LES PLUS INTENSES:

• Intensité maximale: {df['max_intensity_mm'].max():.0f} mm/jour
• Intensité minimale: {df['max_intensity_mm'].min():.0f} mm/jour
• Surface totale affectée: {df['total_area_km2'].sum():.0f} km²
• Couverture moyenne: {df['coverage_percent'].mean():.1f}%
• Dispersion lat: {df['centroid_lat'].max() - df['centroid_lat'].min():.2f}°
• Dispersion lon: {df['centroid_lon'].max() - df['centroid_lon'].min():.2f}°

RÉGIONS AFFECTÉES:"""
        
        # Compter les événements par région
        region_counts = df['region'].value_counts()
        for region, count in region_counts.items():
            stats_text += f"\n• {region}: {count} événement(s)"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
               fontsize=9, fontfamily='monospace')
        
        plt.tight_layout()
        
        # Sauvegarder avec gestion d'erreur robuste
        output_path = project_root / "outputs/visualizations/spatial_top5/synthesis_map_top5_safe.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Carte de synthèse TOP 5 sauvegardée: synthesis_map_top5_safe.png")
            success = True
        except Exception as e:
            print(f"⚠️  Erreur haute résolution: {e}")
            try:
                plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"✅ Carte de synthèse TOP 5 sauvegardée (résolution réduite)")
                success = True
            except Exception as e2:
                print(f"❌ Erreur critique lors de la sauvegarde: {e2}")
                success = False
        finally:
            plt.close()
        
        return success
    
    def save_spatial_data_top5(self, spatial_results):
        """Sauvegarde les données spatiales TOP 5."""
        
        # DataFrame pour analyse
        df = pd.DataFrame(spatial_results)
        csv_path = project_root / "outputs/data/spatial_analysis_top5_intense.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, float_format='%.4f')
        
        # JSON pour traitement automatisé
        json_path = project_root / "outputs/data/spatial_summary_top5_intense.json"
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'analysis_type': 'top5_most_intense_precipitation_events',
            'total_events': len(spatial_results),
            'summary_statistics': {
                'max_intensity_mm': float(df['max_intensity_mm'].max()),
                'min_intensity_mm': float(df['max_intensity_mm'].min()),
                'mean_intensity_mm': float(df['max_intensity_mm'].mean()),
                'total_area_km2': float(df['total_area_km2'].sum()),
                'mean_coverage_percent': float(df['coverage_percent'].mean()),
                'latitudinal_extent_deg': float(df['centroid_lat'].max() - df['centroid_lat'].min()),
                'longitudinal_extent_deg': float(df['centroid_lon'].max() - df['centroid_lon'].min()),
                'regions_affected': df['region'].unique().tolist(),
                'intensity_area_correlation': float(df['max_intensity_mm'].corr(df['total_area_km2']))
            },
            'events': spatial_results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Données CSV: {csv_path}")
        print(f"📄 Résumé JSON: {json_path}")
    
    def run_complete_top5_analysis(self):
        """Lance l'analyse spatiale complète des TOP 5."""
        
        print("🌧️  ANALYSE SPATIALE - TOP 5 PRÉCIPITATIONS EXTRÊMES")
        print("=" * 70)
        print("Distribution spatiale précise et concise de la zone de couverture")
        print("=" * 70)
        
        # Charger les données
        if not self.load_existing_data():
            print("❌ Impossible de charger les données")
            return False
        
        # Sélectionner les TOP 5 par intensité
        self.get_top5_events_by_intensity()
        
        # Calculer les métriques spatiales détaillées
        print("\n🔢 CALCUL DES MÉTRIQUES SPATIALES DÉTAILLÉES")
        print("-" * 50)
        
        spatial_results = []
        for i, (event_date, event_data) in enumerate(self.top5_events.iterrows(), 1):
            print(f"   Analyse événement #{i}: {event_date.strftime('%Y-%m-%d')}")
            
            # Utiliser le module centralisé pour calculer les métriques
            metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                event_date, event_data, self.precip_data, self.anomalies,
                self.lats, self.lons, self.dates, i
            )
            spatial_results.append(metrics)
        
        # Afficher debug des résultats
        print(f"\n🔍 DEBUG - Vérification des métriques calculées:")
        for i, result in enumerate(spatial_results):
            print(f"   Événement #{result['rank']}: Intensité = {result['max_intensity_mm']:.1f} mm, "
                  f"Surface = {result['total_area_km2']:.0f} km², Région = {result['region']}")
        
        # Créer les cartes individuelles
        print(f"\n🗺️  CRÉATION DES CARTES INDIVIDUELLES TOP 5")
        print("-" * 50)
        
        generated_files = []
        for i, (event_date, event_data) in enumerate(self.top5_events.iterrows(), 1):
            try:
                filename = self.create_individual_event_map_top5(i, event_date, event_data, spatial_results[i-1])
                if filename:
                    generated_files.append(filename)
            except Exception as e:
                print(f"⚠️  Erreur création carte #{i}: {e}")
        
        print(f"✅ {len(generated_files)} cartes individuelles créées")
        
        # Créer l'analyse comparative
        try:
            self.create_comparative_analysis_top5(spatial_results)
        except Exception as e:
            print(f"⚠️  Erreur analyse comparative: {e}")
        
        # Créer la carte de synthèse (version sécurisée)
        try:
            self.create_synthesis_map_safe(spatial_results)
        except Exception as e:
            print(f"⚠️  Carte de synthèse ignorée (problème technique): {e}")
            print("   Les autres résultats sont disponibles")
        
        # Générer les rapports et sauvegardes avec le module centralisé
        try:
            report_path = self.report_generator.generate_comprehensive_report(
                spatial_results,
                "top5_intensity_events", 
                "Top 5 Événements par Intensité des Précipitations"
            )
            print(f"✅ Rapport détaillé généré: {report_path}")
            
            self.save_spatial_data_top5(spatial_results)
        except Exception as e:
            print(f"⚠️  Erreur génération rapports: {e}")
        
        # Résumé final
        print(f"\n" + "=" * 70)
        print("✅ ANALYSE SPATIALE TOP 5 TERMINÉE AVEC SUCCÈS!")
        print("=" * 70)
        
        df = pd.DataFrame(spatial_results)
        print(f"📊 Événements analysés: {len(spatial_results)}")
        print(f"🗺️  Cartes générées: {len(generated_files)} individuelles + analyses")
        print(f"📄 Rapports: Données et analyses disponibles")
        print(f"📁 Dossier: outputs/visualizations/spatial_top5/")
        
        print(f"\n🎯 RÉSULTATS CLÉS:")
        print(f"   • Intensité maximale: {df['max_intensity_mm'].max():.1f} mm/jour")
        print(f"   • Surface totale: {df['total_area_km2'].sum():.0f} km²")
        print(f"   • Couverture moyenne: {df['coverage_percent'].mean():.2f}%")
        print(f"   • Distribution: {len(df['region'].unique())} régions affectées")
        
        region_counts = df['region'].value_counts()
        print(f"\n🏛️  RÉPARTITION RÉGIONALE:")
        for region, count in region_counts.items():
            print(f"   • {region}: {count} événement(s)")
        
        return True

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale."""
    print("🌧️  ANALYSE SPATIALE - TOP 5 PRÉCIPITATIONS EXTRÊMES")
    print("=" * 70)
    print("Distribution spatiale très précise et concise de la zone de couverture")
    print("=" * 70)
    
    analyzer = SpatialAnalysisTop5()
    success = analyzer.run_complete_top5_analysis()
    
    if success:
        print("\n🎉 ANALYSE SPATIALE TOP 5 RÉUSSIE!")
        print("\n💡 UTILISATION DES RÉSULTATS:")
        print("• Intégrez les cartes dans le Chapitre 4 (Distribution spatiale)")
        print("• Utilisez les métriques CSV pour l'analyse statistique")
        print("• Le rapport détaillé fournit le contexte pour la rédaction")
        print("• Toutes les visualisations sont prêtes pour publication (300 DPI)")
        print("\nConsultez le dossier outputs/visualizations/spatial_top5/")
        return 0
    else:
        print("\n❌ ÉCHEC DE L'ANALYSE SPATIALE TOP 5")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)