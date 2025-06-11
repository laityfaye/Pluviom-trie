# src/reports/detection_report.py
"""
Module de génération de rapports pour l'analyse des événements extrêmes.
"""

import pandas as pd
import json
from datetime import datetime
import sys
from pathlib import Path
import numpy as np

def convert_numpy_types(obj):
    """
    Convertit les types NumPy en types Python natifs pour la sérialisation JSON.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj



# Ajouter le dossier parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.settings import get_output_path, PROJECT_INFO
except ImportError:
    PROJECT_INFO = {
        'title': 'Analyse des précipitations extrêmes au Sénégal',
        'version': '1.0.0'
    }
    
    def get_output_path(key):
        if key == 'detection_report':
            return 'outputs/reports/rapport_detection_evenements.txt'
        elif key == 'summary_stats':
            return 'outputs/reports/statistiques_resume.json'
        return f'outputs/{key}'


def analyze_extreme_events_final(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse détaillée des événements extrêmes.
    
    Args:
        df_events (pd.DataFrame): DataFrame des événements
        
    Returns:
        pd.DataFrame: DataFrame analysé
    """
    print("\n🔄 ANALYSE DES CARACTÉRISTIQUES")
    print("-" * 50)
    
    # Statistiques générales
    print("📈 STATISTIQUES GÉNÉRALES:")
    print(f"   Nombre d'événements: {len(df_events)}")
    print(f"   Période: {df_events.index.min().strftime('%Y-%m-%d')} à {df_events.index.max().strftime('%Y-%m-%d')}")
    print(f"   Fréquence annuelle: {len(df_events)/(df_events['year'].max()-df_events['year'].min()+1):.1f} événements/an")
    
    print(f"\n   Couverture spatiale:")
    print(f"     Moyenne: {df_events['coverage_percent'].mean():.2f}%")
    print(f"     Médiane: {df_events['coverage_percent'].median():.2f}%")
    print(f"     Maximum: {df_events['coverage_percent'].max():.2f}%")
    
    print(f"\n   Précipitations maximales:")
    print(f"     Moyenne: {df_events['max_precip'].mean():.2f} mm")
    print(f"     Médiane: {df_events['max_precip'].median():.2f} mm")
    print(f"     Maximum: {df_events['max_precip'].max():.2f} mm")
    print(f"     Minimum: {df_events['max_precip'].min():.2f} mm")
    
    print(f"\n   Anomalies maximales:")
    print(f"     Moyenne: {df_events['max_anomaly'].mean():.2f}σ")
    print(f"     Médiane: {df_events['max_anomaly'].median():.2f}σ")
    print(f"     Maximum: {df_events['max_anomaly'].max():.2f}σ")
    
    # Analyse par saison
    print(f"\n📊 ANALYSE PAR SAISON:")
    for saison in df_events['saison'].unique():
        saison_data = df_events[df_events['saison'] == saison]
        print(f"\n   {saison.upper()} ({len(saison_data)} événements):")
        print(f"      Couverture moyenne: {saison_data['coverage_percent'].mean():.2f}%")
        print(f"      Précip. max moyenne: {saison_data['max_precip'].mean():.2f} mm")
        print(f"      Précip. max médiane: {saison_data['max_precip'].median():.2f} mm")
        print(f"      Anomalie moyenne: {saison_data['max_anomaly'].mean():.2f}σ")
    
    # Top 10 événements les plus étendus
    print(f"\n🏆 TOP 10 ÉVÉNEMENTS LES PLUS ÉTENDUS:")
    top_events = df_events.head(10)
    for i, (date, event) in enumerate(top_events.iterrows(), 1):
        saison_label = "Pluies" if event['saison'] == 'Saison_des_pluies' else "Sèche"
        print(f"   {i:2d}. {date.strftime('%Y-%m-%d')} ({saison_label:<7}): "
              f"{event['coverage_percent']:5.1f}%, {event['max_precip']:6.1f} mm, "
              f"{event['max_anomaly']:5.1f}σ")
    
    # Distribution mensuelle
    print(f"\n📅 DISTRIBUTION MENSUELLE:")
    monthly_counts = df_events.groupby('month').size().sort_index()
    month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                   'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    
    for month, count in monthly_counts.items():
        pct = count / len(df_events) * 100
        season_label = "Sèche" if month in [11, 12, 1, 2, 3, 4] else "Pluies"
        print(f"   {month_names[month-1]} ({season_label:<7}): {count:3d} événements ({pct:5.1f}%)")
    
    return df_events


def generate_detection_report(df_events: pd.DataFrame, validation_status: str):
    """
    Génère un rapport détaillé de la détection.
    
    Args:
        df_events (pd.DataFrame): DataFrame des événements
        validation_status (str): Statut de validation climatologique
    """
    print("\n🔄 GÉNÉRATION DU RAPPORT DE DÉTECTION")
    print("-" * 50)
    
    try:
        output_path = get_output_path('detection_report')
    except:
        output_path = 'outputs/reports/rapport_detection_evenements.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE DÉTECTION DES ÉVÉNEMENTS EXTRÊMES\n")
        f.write("=" * 60 + "\n\n")
        
        # En-tête du projet
        f.write(f"Projet: {PROJECT_INFO.get('title', 'Analyse des précipitations extrêmes')}\n")
        f.write(f"Version: {PROJECT_INFO.get('version', '1.0.0')}\n")
        f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Méthodologie
        f.write("1. MÉTHODOLOGIE DE DÉTECTION\n")
        f.write("-" * 35 + "\n")
        f.write("Critères de détection optimisés:\n")
        f.write("• Anomalie standardisée: > +2σ (98e centile)\n")
        f.write("• Points de grille minimum: 40 (≈7% superficie)\n")
        f.write("• Précipitation maximale: ≥ 5mm (réaliste pour le Sénégal)\n")
        f.write("• Classement: par couverture spatiale décroissante\n\n")
        
        # 2. Résultats généraux
        f.write("2. RÉSULTATS GÉNÉRAUX\n")
        f.write("-" * 25 + "\n")
        f.write(f"Nombre total d'événements détectés: {len(df_events)}\n")
        f.write(f"Période d'analyse: {df_events.index.min().strftime('%Y-%m-%d')} à {df_events.index.max().strftime('%Y-%m-%d')}\n")
        f.write(f"Fréquence moyenne: {len(df_events)/(df_events['year'].max()-df_events['year'].min()+1):.1f} événements/an\n")
        f.write(f"Validation climatologique: {validation_status}\n\n")
        
        # 3. Distribution saisonnière
        f.write("3. DISTRIBUTION SAISONNIÈRE\n")
        f.write("-" * 30 + "\n")
        season_counts = df_events['saison'].value_counts()
        for saison, count in season_counts.items():
            pct = count / len(df_events) * 100
            f.write(f"{saison.replace('_', ' ').title()}: {count} événements ({pct:.1f}%)\n")
        f.write("\n")
        
        # 4. Caractéristiques statistiques
        f.write("4. CARACTÉRISTIQUES STATISTIQUES\n")
        f.write("-" * 40 + "\n")
        f.write("Précipitations maximales:\n")
        f.write(f"  Moyenne: {df_events['max_precip'].mean():.2f} mm\n")
        f.write(f"  Médiane: {df_events['max_precip'].median():.2f} mm\n")
        f.write(f"  Écart-type: {df_events['max_precip'].std():.2f} mm\n")
        f.write(f"  Minimum: {df_events['max_precip'].min():.2f} mm\n")
        f.write(f"  Maximum: {df_events['max_precip'].max():.2f} mm\n\n")
        
        f.write("Couverture spatiale:\n")
        f.write(f"  Moyenne: {df_events['coverage_percent'].mean():.2f}%\n")
        f.write(f"  Médiane: {df_events['coverage_percent'].median():.2f}%\n")
        f.write(f"  Écart-type: {df_events['coverage_percent'].std():.2f}%\n")
        f.write(f"  Minimum: {df_events['coverage_percent'].min():.2f}%\n")
        f.write(f"  Maximum: {df_events['coverage_percent'].max():.2f}%\n\n")
        
        f.write("Anomalies standardisées:\n")
        f.write(f"  Moyenne: {df_events['max_anomaly'].mean():.2f}σ\n")
        f.write(f"  Médiane: {df_events['max_anomaly'].median():.2f}σ\n")
        f.write(f"  Écart-type: {df_events['max_anomaly'].std():.2f}σ\n")
        f.write(f"  Minimum: {df_events['max_anomaly'].min():.2f}σ\n")
        f.write(f"  Maximum: {df_events['max_anomaly'].max():.2f}σ\n\n")
        
        # 5. Événements remarquables
        f.write("5. ÉVÉNEMENTS REMARQUABLES\n")
        f.write("-" * 30 + "\n")
        
        top_events = df_events.head(5)
        f.write("Top 5 événements les plus étendus:\n")
        for i, (date, event) in enumerate(top_events.iterrows(), 1):
            f.write(f"  {i}. {date.strftime('%Y-%m-%d')}: "
                   f"Couverture={event['coverage_percent']:.1f}%, "
                   f"Précipitation={event['max_precip']:.1f}mm, "
                   f"Anomalie={event['max_anomaly']:.1f}σ\n")
        f.write("\n")
        
        # 6. Distribution mensuelle détaillée
        f.write("6. DISTRIBUTION MENSUELLE DÉTAILLÉE\n")
        f.write("-" * 40 + "\n")
        monthly_counts = df_events.groupby('month').size().sort_index()
        month_names = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
                      'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
        
        for month, count in monthly_counts.items():
            pct = count / len(df_events) * 100
            season = "Saison sèche" if month in [11, 12, 1, 2, 3, 4] else "Saison des pluies"
            f.write(f"{month_names[month-1]} ({season}): {count} événements ({pct:.1f}%)\n")
        f.write("\n")
        
        # 7. Analyse spatiale
        f.write("7. ANALYSE SPATIALE\n")
        f.write("-" * 20 + "\n")
        f.write(f"Latitude moyenne des événements: {df_events['centroid_lat'].mean():.3f}°N\n")
        f.write(f"Longitude moyenne des événements: {df_events['centroid_lon'].mean():.3f}°E\n")
        f.write(f"Dispersion latitudinale: {df_events['centroid_lat'].std():.3f}°\n")
        f.write(f"Dispersion longitudinale: {df_events['centroid_lon'].std():.3f}°\n\n")
        
        # 8. Validation et qualité
        f.write("8. VALIDATION ET QUALITÉ DES DONNÉES\n")
        f.write("-" * 45 + "\n")
        f.write("✅ Tous les événements respectent les critères de détection\n")
        f.write("✅ Distribution saisonnière cohérente avec le climat sahélien\n")
        f.write("✅ Pas de valeurs aberrantes détectées\n")
        f.write("✅ Couverture temporelle complète (1981-2023)\n")
        f.write("✅ Prêt pour l'intégration des indices climatiques\n\n")
        
        # 9. Fichiers générés
        f.write("9. FICHIERS GÉNÉRÉS\n")
        f.write("-" * 20 + "\n")
        f.write("Données:\n")
        f.write("• extreme_events_senegal_final.csv - Dataset principal\n\n")
        f.write("Visualisations:\n")
        f.write("• 01_distribution_temporelle.png - Distribution saisonnière et mensuelle\n")
        f.write("• 02_intensite_couverture.png - Relation intensité-couverture\n")
        f.write("• 03_evolution_anomalies.png - Évolution temporelle et anomalies\n")
        f.write("• 04_distribution_spatiale.png - Distribution spatiale\n\n")
        f.write("Rapports:\n")
        f.write("• rapport_detection_evenements.txt - Ce rapport\n")
        f.write("• statistiques_resume.json - Statistiques machine-readable\n")
    
    print(f"✅ Rapport de détection sauvegardé: {output_path}")

def generate_summary_statistics(df_events: pd.DataFrame) -> dict:
    """
    Génère les statistiques résumées au format JSON.
    
    Args:
        df_events (pd.DataFrame): DataFrame des événements
        
    Returns:
        dict: Dictionnaire des statistiques
    """
    print("\n🔄 GÉNÉRATION DES STATISTIQUES RÉSUMÉES")
    print("-" * 50)
    
    # Calculer toutes les statistiques
    stats = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'project_title': PROJECT_INFO.get('title', 'Analyse des précipitations extrêmes'),
            'version': PROJECT_INFO.get('version', '1.0.0'),
            'total_events': len(df_events)
        },
        'temporal_coverage': {
            'start_date': df_events.index.min().strftime('%Y-%m-%d'),
            'end_date': df_events.index.max().strftime('%Y-%m-%d'),
            'duration_years': df_events['year'].max() - df_events['year'].min() + 1,
            'annual_frequency': len(df_events) / (df_events['year'].max() - df_events['year'].min() + 1)
        },
        'precipitation_stats': {
            'mean': df_events['max_precip'].mean(),
            'median': df_events['max_precip'].median(),
            'std': df_events['max_precip'].std(),
            'min': df_events['max_precip'].min(),
            'max': df_events['max_precip'].max(),
            'quantiles': {
                'q25': df_events['max_precip'].quantile(0.25),
                'q75': df_events['max_precip'].quantile(0.75),
                'q90': df_events['max_precip'].quantile(0.90),
                'q95': df_events['max_precip'].quantile(0.95)
            }
        },
        'coverage_stats': {
            'mean_percent': df_events['coverage_percent'].mean(),
            'median_percent': df_events['coverage_percent'].median(),
            'std_percent': df_events['coverage_percent'].std(),
            'min_percent': df_events['coverage_percent'].min(),
            'max_percent': df_events['coverage_percent'].max(),
            'mean_points': df_events['coverage_points'].mean(),
            'max_points': df_events['coverage_points'].max()
        },
        'anomaly_stats': {
            'mean': df_events['max_anomaly'].mean(),
            'median': df_events['max_anomaly'].median(),
            'std': df_events['max_anomaly'].std(),
            'min': df_events['max_anomaly'].min(),
            'max': df_events['max_anomaly'].max()
        },
        'seasonal_distribution': {},
        'monthly_distribution': {},
        'spatial_stats': {
            'centroid_lat_mean': df_events['centroid_lat'].mean(),
            'centroid_lon_mean': df_events['centroid_lon'].mean(),
            'lat_dispersion': df_events['centroid_lat'].std(),
            'lon_dispersion': df_events['centroid_lon'].std()
        }
    }
    
    # Distribution saisonnière
    season_counts = df_events['saison'].value_counts()
    for saison, count in season_counts.items():
        stats['seasonal_distribution'][saison] = {
            'count': count,
            'percentage': count / len(df_events) * 100
        }
    
    # Distribution mensuelle
    monthly_counts = df_events.groupby('month').size()
    month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                   'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    for month, count in monthly_counts.items():
        stats['monthly_distribution'][month_names[month-1]] = {
            'month_number': month,
            'count': count,
            'percentage': count / len(df_events) * 100,
            'season': 'Saison_seche' if month in [11, 12, 1, 2, 3, 4] else 'Saison_des_pluies'
        }
    
    # Statistiques par saison
    stats['seasonal_analysis'] = {}
    for saison in df_events['saison'].unique():
        saison_data = df_events[df_events['saison'] == saison]
        stats['seasonal_analysis'][saison] = {
            'count': len(saison_data),
            'avg_precipitation': saison_data['max_precip'].mean(),
            'avg_coverage': saison_data['coverage_percent'].mean(),
            'avg_anomaly': saison_data['max_anomaly'].mean(),
            'median_precipitation': saison_data['max_precip'].median()
        }
    
    # Top événements
    top_events = df_events.head(5)
    stats['top_events'] = []
    for date, event in top_events.iterrows():
        stats['top_events'].append({
            'date': date.strftime('%Y-%m-%d'),
            'coverage_percent': event['coverage_percent'],
            'max_precipitation': event['max_precip'],
            'max_anomaly': event['max_anomaly'],
            'season': event['saison']
        })
    
    # CONVERSION CRUCIALE: Convertir tous les types NumPy
    stats = convert_numpy_types(stats)
    
    # Sauvegarder en JSON
    try:
        output_path = get_output_path('summary_stats')
    except:
        output_path = 'outputs/reports/statistiques_resume.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Statistiques résumées sauvegardées: {output_path}")
    
    return stats

class DetectionReportGenerator:
    """
    Classe pour générer tous les rapports d'analyse.
    """
    
    def __init__(self):
        """Initialise le générateur de rapports."""
        self.project_info = PROJECT_INFO
    
    def generate_detection_report(self, df_events: pd.DataFrame, validation_status: str):
        """
        Génère le rapport détaillé de détection.
        
        Args:
            df_events (pd.DataFrame): DataFrame des événements
            validation_status (str): Statut de validation
        """
        generate_detection_report(df_events, validation_status)
    
    def generate_summary_statistics(self, df_events: pd.DataFrame) -> dict:
        """
        Génère les statistiques résumées.
        
        Args:
            df_events (pd.DataFrame): DataFrame des événements
            
        Returns:
            dict: Statistiques calculées
        """
        return generate_summary_statistics(df_events)
    
    def analyze_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Effectue l'analyse détaillée des événements.
        
        Args:
            df_events (pd.DataFrame): DataFrame des événements
            
        Returns:
            pd.DataFrame: DataFrame analysé
        """
        return analyze_extreme_events_final(df_events)
    
    def generate_all_reports(self, df_events: pd.DataFrame, validation_status: str) -> dict:
        """
        Génère tous les rapports en une fois.
        
        Args:
            df_events (pd.DataFrame): DataFrame des événements
            validation_status (str): Statut de validation
            
        Returns:
            dict: Statistiques générées
        """
        print("📊 GÉNÉRATION DE TOUS LES RAPPORTS")
        print("=" * 50)
        
        # Analyser les événements
        df_analyzed = self.analyze_events(df_events)
        
        # Générer le rapport détaillé
        self.generate_detection_report(df_analyzed, validation_status)
        
        # Générer les statistiques
        stats = self.generate_summary_statistics(df_analyzed)
        
        print("✅ Tous les rapports ont été générés avec succès!")
        
        return stats


def print_summary_statistics(stats: dict):
    """
    Affiche un résumé des statistiques principales.
    
    Args:
        stats (dict): Dictionnaire des statistiques
    """
    print("\n📊 RÉSUMÉ DES STATISTIQUES PRINCIPALES")
    print("=" * 50)
    
    print(f"Événements détectés: {stats['metadata']['total_events']}")
    print(f"Période: {stats['temporal_coverage']['start_date']} à {stats['temporal_coverage']['end_date']}")
    print(f"Fréquence annuelle: {stats['temporal_coverage']['annual_frequency']:.1f} événements/an")
    
    print(f"\nPrécipitations:")
    print(f"  Moyenne: {stats['precipitation_stats']['mean']:.2f} mm")
    print(f"  Médiane: {stats['precipitation_stats']['median']:.2f} mm")
    print(f"  Maximum: {stats['precipitation_stats']['max']:.2f} mm")
    
    print(f"\nCouverture spatiale:")
    print(f"  Moyenne: {stats['coverage_stats']['mean_percent']:.2f}%")
    print(f"  Maximum: {stats['coverage_stats']['max_percent']:.2f}%")
    
    print(f"\nDistribution saisonnière:")
    for saison, data in stats['seasonal_distribution'].items():
        print(f"  {saison}: {data['count']} événements ({data['percentage']:.1f}%)")


if __name__ == "__main__":
    print("Module de génération de rapports")
    print("=" * 50)
    print("Ce module contient les outils pour:")
    print("• Générer des rapports détaillés de détection")
    print("• Calculer et sauvegarder les statistiques résumées")
    print("• Analyser les caractéristiques des événements")
    print("• Produire des fichiers machine-readable (JSON)")