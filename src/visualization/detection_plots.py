# src/visualization/detection_plots.py
"""
Module de visualisation pour les événements de précipitations extrêmes.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
try:
    from ..config.settings import (
        SEASON_COLORS, PLOT_PARAMS, get_output_path,
        VISUALIZATION_FILENAMES
    )
except ImportError:
    try:
        from src.config.settings import (
            SEASON_COLORS, PLOT_PARAMS, get_output_path,
            VISUALIZATION_FILENAMES
        )
    except ImportError:
        # Valeurs par défaut
        SEASON_COLORS = {
            'saison_seche': '#E74C3C',
            'saison_des_pluies': '#27AE60'
        }
        PLOT_PARAMS = {
            'figure_size': (14, 6),
            'dpi': 300,
            'style': 'default'
        }
        
        def get_output_path(key):
            return f"outputs/visualizations/{key}.png"

def create_detection_visualizations_part1(df_events: pd.DataFrame):
    """
    Crée les visualisations de détection - Partie 1: Distribution saisonnière et mensuelle.
    
    Args:
        df_events (pd.DataFrame): DataFrame des événements extrêmes
    """
    print("\n🔄 CRÉATION DES VISUALISATIONS - PARTIE 1")
    print("-" * 50)
    
    plt.style.use(PLOT_PARAMS['style'])
    sns.set_palette("husl")
    
    # Figure 1: Distribution saisonnière et mensuelle
    fig, axes = plt.subplots(1, 2, figsize=PLOT_PARAMS['figure_size'])
    fig.suptitle('Distribution temporelle des événements extrêmes - Sénégal\n'
                 'Critères: Anomalie > +2σ, Couverture ≥ 40 points, Précipitation ≥ 5mm', 
                 fontsize=14, fontweight='bold')
    
    # 1. Distribution saisonnière
    season_counts = df_events['saison'].value_counts()
    colors = [SEASON_COLORS['saison_seche'], SEASON_COLORS['saison_des_pluies']]
    
    labels_clean = [s.replace('_', ' ').title() for s in season_counts.index]
    
    wedges, texts, autotexts = axes[0].pie(season_counts.values, 
                                         labels=labels_clean, 
                                         autopct='%1.1f%%',
                                         colors=colors, 
                                         startangle=90)
    axes[0].set_title('Distribution saisonnière\n(Saison des pluies: Mai-Oct)', fontweight='bold')
    
    # 2. Distribution mensuelle
    monthly_counts = df_events.groupby('month').size()
    month_colors = []
    for month in range(1, 13):
        if month in [5, 6, 7, 8, 9, 10]:  # Saison des pluies
            month_colors.append(SEASON_COLORS['saison_des_pluies'])
        else:  # Saison sèche
            month_colors.append(SEASON_COLORS['saison_seche'])
    
    bars = axes[1].bar(monthly_counts.index, monthly_counts.values, 
                      color=[month_colors[m-1] for m in monthly_counts.index], 
                      alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[1].set_title('Distribution mensuelle\n(Vert=Saison des pluies, Rouge=Saison sèche)', 
                     fontweight='bold')
    axes[1].set_xlabel('Mois')
    axes[1].set_ylabel('Nombre d\'événements')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(monthly_counts.index)
    
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, monthly_counts.values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    try:
        output_path = get_output_path('temporal_distribution')
        plt.savefig(output_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        print(f"✅ Partie 1 sauvegardée: {output_path}")
    except:
        plt.savefig('outputs/visualizations/01_distribution_temporelle.png', 
                   dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        print("✅ Partie 1 sauvegardée: 01_distribution_temporelle.png")
    
    plt.close()


def create_detection_visualizations_part2(df_events: pd.DataFrame):
    """
    Crée les visualisations de détection - Partie 2: Intensité et couverture.
    
    Args:
        df_events (pd.DataFrame): DataFrame des événements extrêmes
    """
    print("\n🔄 CRÉATION DES VISUALISATIONS - PARTIE 2")
    print("-" * 50)
    
    # Figure 2: Relation couverture-intensité et distribution des précipitations
    fig, axes = plt.subplots(1, 2, figsize=PLOT_PARAMS['figure_size'])
    fig.suptitle('Caractéristiques des événements extrêmes - Sénégal', 
                 fontsize=14, fontweight='bold')
    
    # 1. Relation couverture-intensité
    colors_scatter = [SEASON_COLORS['saison_seche'] if s == 'Saison_seche' 
                     else SEASON_COLORS['saison_des_pluies'] for s in df_events['saison']]
    
    scatter = axes[0].scatter(df_events['coverage_percent'], df_events['max_precip'],
                             c=colors_scatter, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    axes[0].set_xlabel('Couverture spatiale (%)')
    axes[0].set_ylabel('Précipitation maximale (mm)')
    axes[0].set_title('Relation couverture-intensité', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Légende personnalisée
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=SEASON_COLORS['saison_seche'], alpha=0.6, label='Saison sèche'),
                      Patch(facecolor=SEASON_COLORS['saison_des_pluies'], alpha=0.6, label='Saison des pluies')]
    axes[0].legend(handles=legend_elements)
    
    # 2. Distribution des précipitations
    axes[1].hist(df_events['max_precip'], bins=30, alpha=0.7, color='skyblue', 
                edgecolor='black', linewidth=0.5)
    
    # Ligne verticale pour le seuil de 5mm
    axes[1].axvline(x=5, color='red', linestyle='--', linewidth=2, label='Seuil 5mm')
    
    axes[1].set_title('Distribution des précipitations maximales', fontweight='bold')
    axes[1].set_xlabel('Précipitation maximale (mm)')
    axes[1].set_ylabel('Nombre d\'événements')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    try:
        output_path = get_output_path('intensity_coverage')
        plt.savefig(output_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        print(f"✅ Partie 2 sauvegardée: {output_path}")
    except:
        plt.savefig('outputs/visualizations/02_intensite_couverture.png', 
                   dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        print("✅ Partie 2 sauvegardée: 02_intensite_couverture.png")
    
    plt.close()


def create_detection_visualizations_part3(df_events: pd.DataFrame):
    """
    Crée les visualisations de détection - Partie 3: Évolution temporelle et anomalies.
    
    Args:
        df_events (pd.DataFrame): DataFrame des événements extrêmes
    """
    print("\n🔄 CRÉATION DES VISUALISATIONS - PARTIE 3")
    print("-" * 50)
    
    # Figure 3: Évolution temporelle et distribution des anomalies
    fig, axes = plt.subplots(1, 2, figsize=PLOT_PARAMS['figure_size'])
    fig.suptitle('Évolution temporelle et anomalies - Sénégal', 
                 fontsize=14, fontweight='bold')
    
    # 1. Évolution temporelle
    yearly_counts = df_events.groupby('year').size()
    yearly_seasonal = df_events.groupby(['year', 'saison']).size().unstack(fill_value=0)
    
    if 'Saison_des_pluies' in yearly_seasonal.columns:
        axes[0].plot(yearly_seasonal.index, yearly_seasonal['Saison_des_pluies'], 
                    color=SEASON_COLORS['saison_des_pluies'], marker='o', 
                    label='Saison des pluies', linewidth=2, markersize=4)
    
    if 'Saison_seche' in yearly_seasonal.columns:
        axes[0].plot(yearly_seasonal.index, yearly_seasonal['Saison_seche'], 
                    color=SEASON_COLORS['saison_seche'], marker='s', 
                    label='Saison sèche', linewidth=2, markersize=4)
    
    axes[0].set_title('Évolution temporelle par saison', fontweight='bold')
    axes[0].set_xlabel('Année')
    axes[0].set_ylabel('Nombre d\'événements')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Distribution des anomalies
    axes[1].hist(df_events['max_anomaly'], bins=30, alpha=0.7, color='orange', 
                edgecolor='black', linewidth=0.5)
    
    # Ligne verticale pour le seuil de +2σ
    axes[1].axvline(x=2, color='red', linestyle='--', linewidth=2, label='Seuil +2σ')
    
    axes[1].set_title('Distribution des anomalies maximales', fontweight='bold')
    axes[1].set_xlabel('Anomalie maximale (σ)')
    axes[1].set_ylabel('Nombre d\'événements')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    try:
        output_path = get_output_path('evolution_anomalies')
        plt.savefig(output_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        print(f"✅ Partie 3 sauvegardée: {output_path}")
    except:
        plt.savefig('outputs/visualizations/03_evolution_anomalies.png', 
                   dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        print("✅ Partie 3 sauvegardée: 03_evolution_anomalies.png")
    
    plt.close()


def create_spatial_distribution_visualization(df_events: pd.DataFrame, lats: np.ndarray, lons: np.ndarray):
    """
    Crée la visualisation de la distribution spatiale des événements extrêmes.
    
    Args:
        df_events (pd.DataFrame): DataFrame des événements extrêmes
        lats (np.ndarray): Latitudes
        lons (np.ndarray): Longitudes
    """
    print("\n🔄 CRÉATION DE LA VISUALISATION SPATIALE")
    print("-" * 50)
    
    # Figure 4: Distribution spatiale
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Distribution spatiale des événements extrêmes - Sénégal', 
                 fontsize=14, fontweight='bold')
    
    # 1. Carte des centroïdes
    # Créer une grille pour la densité
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(lons.min(), lons.max(), 20),
        np.linspace(lats.min(), lats.max(), 20)
    )
    
    # Calculer la densité d'événements
    try:
        from scipy.stats import gaussian_kde
        if len(df_events) > 1:
            positions = np.vstack([df_events['centroid_lon'], df_events['centroid_lat']])
            kernel = gaussian_kde(positions)
            density = kernel(np.vstack([lon_grid.ravel(), lat_grid.ravel()])).reshape(lon_grid.shape)
            
            # Contours de densité
            contour = axes[0].contourf(lon_grid, lat_grid, density, levels=10, cmap='YlOrRd', alpha=0.6)
            plt.colorbar(contour, ax=axes[0], label='Densité d\'événements')
    except ImportError:
        print("Note: scipy non disponible pour les contours de densité")
    
    # Points des événements
    colors_season = [SEASON_COLORS['saison_seche'] if s == 'Saison_seche' 
                    else SEASON_COLORS['saison_des_pluies'] for s in df_events['saison']]
    sizes = df_events['coverage_percent'] * 2  # Taille proportionnelle à la couverture
    
    scatter = axes[0].scatter(df_events['centroid_lon'], df_events['centroid_lat'],
                             c=colors_season, s=sizes, alpha=0.7, 
                             edgecolors='black', linewidth=0.5)
    
    axes[0].set_xlabel('Longitude (°)')
    axes[0].set_ylabel('Latitude (°)')
    axes[0].set_title('Localisation des événements\n(Taille = Couverture spatiale)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SEASON_COLORS['saison_seche'], alpha=0.7, label='Saison sèche'),
        Patch(facecolor=SEASON_COLORS['saison_des_pluies'], alpha=0.7, label='Saison des pluies')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')
    
    # 2. Histogrammes des positions
    axes[1].hist(df_events['centroid_lat'], bins=15, alpha=0.7, color='skyblue', 
                orientation='horizontal', label='Latitude', density=True)
    axes[1].set_ylabel('Latitude (°)')
    axes[1].set_xlabel('Densité de probabilité')
    axes[1].set_title('Distribution latitudinale\ndes événements', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Ajouter les limites du Sénégal
    axes[0].axhline(y=12.0, color='black', linestyle=':', alpha=0.5, label='Limites Sénégal')
    axes[0].axhline(y=17.0, color='black', linestyle=':', alpha=0.5)
    axes[0].axvline(x=-18.0, color='black', linestyle=':', alpha=0.5)
    axes[0].axvline(x=-11.0, color='black', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    try:
        output_path = get_output_path('spatial_distribution')
        plt.savefig(output_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        print(f"✅ Visualisation spatiale sauvegardée: {output_path}")
    except:
        plt.savefig('outputs/visualizations/04_distribution_spatiale.png', 
                   dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        print("✅ Visualisation spatiale sauvegardée: 04_distribution_spatiale.png")
    
    plt.close()


class DetectionVisualizer:
    """
    Classe pour créer toutes les visualisations des événements extrêmes.
    """
    
    def __init__(self):
        """Initialise le visualiseur."""
        plt.style.use(PLOT_PARAMS.get('style', 'default'))
        
    def create_temporal_distribution_plot(self, df_events: pd.DataFrame):
        """Crée le graphique de distribution temporelle."""
        create_detection_visualizations_part1(df_events)
    
    def create_intensity_coverage_plot(self, df_events: pd.DataFrame):
        """Crée le graphique intensité-couverture."""
        create_detection_visualizations_part2(df_events)
    
    def create_evolution_anomalies_plot(self, df_events: pd.DataFrame):
        """Crée le graphique d'évolution et anomalies."""
        create_detection_visualizations_part3(df_events)
    
    def create_spatial_plot(self, df_events: pd.DataFrame, lats: np.ndarray, lons: np.ndarray):
        """Crée la carte de distribution spatiale."""
        create_spatial_distribution_visualization(df_events, lats, lons)
    
    def create_all_plots(self, df_events: pd.DataFrame, lats: np.ndarray, lons: np.ndarray):
        """
        Génère toutes les visualisations.
        
        Args:
            df_events (pd.DataFrame): DataFrame des événements
            lats (np.ndarray): Latitudes
            lons (np.ndarray): Longitudes
        """
        print("🎨 GÉNÉRATION DE TOUTES LES VISUALISATIONS")
        print("=" * 50)
        
        self.create_temporal_distribution_plot(df_events)
        self.create_intensity_coverage_plot(df_events)
        self.create_evolution_anomalies_plot(df_events)
        self.create_spatial_plot(df_events, lats, lons)
        
        print("✅ Toutes les visualisations ont été créées avec succès!")


if __name__ == "__main__":
    print("Module de visualisation des événements extrêmes")
    print("=" * 50)
    print("Ce module contient les outils pour:")
    print("• Créer les graphiques de distribution temporelle")
    print("• Visualiser les relations intensité-couverture")
    print("• Cartographier la distribution spatiale")
    print("• Analyser l'évolution temporelle")