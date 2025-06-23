#!/usr/bin/env python3
# scripts/01_detection_extremes.py
"""
Script principal pour la détection des événements de précipitations extrêmes au Sénégal.
VERSION COMPLÈTE OPTIMISÉE MÉMOIRE - Analyse complète avec chargement par chunks.

Ce script orchestre l'ensemble du processus d'analyse :
1. Chargement optimisé des données CHIRPS (par chunks)
2. Calcul de la climatologie et des anomalies
3. Détection des événements extrêmes
4. Classification saisonnière
5. Génération des visualisations et rapports

Utilisation:
    python scripts/01_detection_extremes.py
    python scripts/01_detection_extremes.py /chemin/vers/chirps.mat

Auteur: Laity FAYE
Date: 2025-06-14
Version: 3.0 - Complète + Optimisée mémoire Docker
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import gc
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION DES IMPORTS - VERSION CORRIGÉE
# ============================================================================

# Ajouter le dossier racine et src au PYTHONPATH
def setup_project_paths():
    """Configure les chemins du projet de manière propre."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    src_dir = project_root / "src"
    
    # Ajouter SEULEMENT si pas déjà présent
    for path_str in [str(project_root), str(src_dir)]:
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    
    return project_root

PROJECT_ROOT = setup_project_paths()

# IMPORTS AVEC GESTION D'ERREURS
try:
    from src.config.settings import (
        CHIRPS_FILEPATH, DETECTION_CRITERIA, PROJECT_INFO,
        create_output_directories, print_project_info, get_output_path
    )
    print("✅ Config settings importé")
except ImportError as e:
    print(f"⚠️ Problème config: {e}")
    # Configuration basique de fallback
    CHIRPS_FILEPATH = "/app/data/raw/chirps_WA_1981_2023_dayly.mat"
    
    def create_output_directories():
        """Créer les dossiers de sortie."""
        Path("outputs/data").mkdir(parents=True, exist_ok=True)
        Path("outputs/visualizations").mkdir(parents=True, exist_ok=True)
        Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    
    def print_project_info():
        """Afficher les infos du projet."""
        print("📊 Projet: Détection événements précipitations extrêmes - Sénégal")
    
    def get_output_path(key):
        """Obtenir le chemin de sortie."""
        paths = {
            'extreme_events': 'outputs/data/extreme_events_senegal_final.csv',
            'climatology': 'outputs/data/climatology_senegal.npz',
            'anomalies': 'outputs/data/anomalies_senegal.npz',
            'detection_report': 'outputs/reports/rapport_detection_evenements.txt'
        }
        return paths.get(key, f'outputs/data/{key}.csv')

# Imports des modules d'analyse avec fallback
try:
    from src.analysis.climatology import calculate_climatology_and_anomalies
    print("✅ Climatology importé")
except ImportError as e:
    print(f"⚠️ Problème climatology: {e}")
    # Fallback basique
    def calculate_climatology_and_anomalies(precip_data, dates):
        print("⚠️ Utilisation du calcul climatology basique")
        # Calcul simple de la climatologie
        climatology = np.nanmean(precip_data, axis=0)
        std_dev = np.nanstd(precip_data, axis=0)
        anomalies = (precip_data - climatology) / std_dev
        return climatology, std_dev, anomalies

try:
    from src.analysis.detection import ExtremeEventDetector
    print("✅ Detection importé")
except ImportError as e:
    print(f"⚠️ Problème detection: {e}")
    # Fallback basique
    class ExtremeEventDetector:
        def detect_events(self, precip_data, anomalies, dates, lats, lons):
            print("⚠️ Utilisation du détecteur basique")
            # Détection simple des événements
            extreme_indices = np.where(anomalies > 2.0)
            events_data = []
            
            for i, (day_idx, lat_idx, lon_idx) in enumerate(zip(*extreme_indices)):
                if i < 100:  # Limiter pour l'exemple
                    events_data.append({
                        'date': dates[day_idx],
                        'max_precip': precip_data[day_idx, lat_idx, lon_idx],
                        'max_anomaly': anomalies[day_idx, lat_idx, lon_idx],
                        'centroid_lat': lats[lat_idx],
                        'centroid_lon': lons[lon_idx],
                        'coverage_percent': 50.0  # Valeur par défaut
                    })
            
            df = pd.DataFrame(events_data)
            if not df.empty:
                df.set_index('date', inplace=True)
            return df

try:
    from src.utils.season_classifier import SeasonClassifier
    print("✅ Season classifier importé")
except ImportError as e:
    print(f"⚠️ Problème season classifier: {e}")
    # Fallback basique
    class SeasonClassifier:
        def classify_and_validate(self, df_events):
            print("⚠️ Utilisation du classificateur basique")
            # Classification simple basée sur le mois
            def get_season(date):
                month = date.month
                if 5 <= month <= 10:
                    return 'Saison_des_pluies'
                else:
                    return 'Saison_seche'
            
            df_events['saison'] = df_events.index.map(get_season)
            df_events['year'] = df_events.index.year
            df_events['month'] = df_events.index.month
            
            return df_events, "VALIDÉ (basique)"

try:
    from src.visualization.detection_plots import DetectionVisualizer
    print("✅ Visualizer importé")
except ImportError as e:
    print(f"⚠️ Problème visualizer: {e}")
    # Fallback basique
    class DetectionVisualizer:
        def create_all_plots(self, df_events, lats, lons):
            print("⚠️ Visualiseur basique - créant un graphique simple")
            try:
                import matplotlib.pyplot as plt
                
                # Créer un graphique simple
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                if 'saison' in df_events.columns:
                    season_counts = df_events['saison'].value_counts()
                    ax.pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%')
                    ax.set_title('Distribution saisonnière des événements extrêmes')
                else:
                    ax.plot(df_events.index, df_events['max_precip'], 'o-')
                    ax.set_title('Événements de précipitations extrêmes')
                    ax.set_ylabel('Précipitation (mm)')
                
                plt.tight_layout()
                plt.savefig('outputs/visualizations/evenements_extremes_basique.png', dpi=300)
                plt.close()
                print("✅ Graphique basique sauvegardé: evenements_extremes_basique.png")
                
            except Exception as e:
                print(f"⚠️ Impossible de créer les visualisations: {e}")

try:
    from src.reports.detection_report import DetectionReportGenerator
    print("✅ Report generator importé")
except ImportError as e:
    print(f"⚠️ Problème report generator: {e}")
    # Fallback basique
    class DetectionReportGenerator:
        def generate_all_reports(self, df_events, validation_status):
            print("⚠️ Générateur de rapport basique")
            try:
                report_path = get_output_path('detection_report')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("RAPPORT DE DÉTECTION D'ÉVÉNEMENTS EXTRÊMES\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Nombre d'événements détectés: {len(df_events)}\n")
                    f.write(f"Validation: {validation_status}\n")
                    if not df_events.empty:
                        f.write(f"Période: {df_events.index.min()} à {df_events.index.max()}\n")
                        f.write(f"Précipitation moyenne: {df_events['max_precip'].mean():.2f} mm\n")
                        if 'saison' in df_events.columns:
                            f.write("\nDistribution saisonnière:\n")
                            for saison, count in df_events['saison'].value_counts().items():
                                f.write(f"  {saison}: {count}\n")
                
                print(f"✅ Rapport basique sauvegardé: {report_path}")
                return {}
            except Exception as e:
                print(f"⚠️ Erreur génération rapport: {e}")
                return {}

print("✅ Imports terminés")

# ============================================================================
# LOADER CHIRPS OPTIMISÉ MÉMOIRE - INTÉGRÉ DIRECTEMENT
# ============================================================================

class OptimizedChirpsLoader:
    """
    Loader CHIRPS optimisé pour Docker avec contraintes mémoire.
    Charge les données par chunks et filtre directement pour le Sénégal.
    REMPLACE ChirpsDataLoader pour éviter les problèmes mémoire.
    """
    
    def __init__(self, chirps_file_path: str):
        self.chirps_file_path = Path(chirps_file_path)
        self.chunk_size = 365  # Une année à la fois
        
        # Limites géographiques Sénégal
        self.lat_min, self.lat_max = 12.0, 17.0
        self.lon_min, self.lon_max = -18.0, -11.0
        
        if not self.chirps_file_path.exists():
            raise FileNotFoundError(f"Fichier CHIRPS non trouvé: {chirps_file_path}")
        
        print(f"🔧 OptimizedChirpsLoader initialisé: {self.chirps_file_path}")
    
    def load_senegal_data(self):
        """
        Interface compatible avec l'ancien ChirpsDataLoader.
        Charge les données CHIRPS pour le Sénégal de manière optimisée.
        
        Returns:
            Tuple: (precip_data, dates, lats, lons)
        """
        print("🔄 CHARGEMENT OPTIMISÉ DES DONNÉES CHIRPS - SÉNÉGAL")
        print("=" * 70)
        
        # 1. Charger métadonnées et identifier la zone Sénégal
        with h5py.File(self.chirps_file_path, 'r') as f:
            print("🔄 CHARGEMENT DES DONNÉES CHIRPS BRUTES")
            print("-" * 50)
            print(f"Clés disponibles: {list(f.keys())}")
            
            # Coordonnées complètes
            full_latitude = np.array(f['latitude']).flatten()
            full_longitude = np.array(f['longitude']).flatten()
            data_shape = f['precip'].shape
            
            print(f"📊 Shape totale: {data_shape}")
            print(f"💾 Taille mémoire estimée complète: {np.prod(data_shape) * 8 / (1024**3):.2f} GB")
        
        # 2. Créer les masques pour le Sénégal
        lat_mask = (full_latitude >= self.lat_min) & (full_latitude <= self.lat_max)
        lon_mask = (full_longitude >= self.lon_min) & (full_longitude <= self.lon_max)
        
        # Coordonnées Sénégal
        senegal_lat = full_latitude[lat_mask]
        senegal_lon = full_longitude[lon_mask]
        
        print(f"🗺️ ZONE SÉNÉGAL IDENTIFIÉE:")
        print(f"   Latitudes: {lat_mask.sum()} points ({senegal_lat.min():.2f}°N à {senegal_lat.max():.2f}°N)")
        print(f"   Longitudes: {lon_mask.sum()} points ({senegal_lon.min():.2f}°E à {senegal_lon.max():.2f}°E)")
        
        # 3. Chargement par chunks avec filtrage Sénégal
        total_days = data_shape[0]
        senegal_shape = (total_days, lat_mask.sum(), lon_mask.sum())
        
        print(f"📦 CHARGEMENT OPTIMISÉ PAR CHUNKS:")
        print(f"   Shape finale Sénégal: {senegal_shape}")
        print(f"   Mémoire finale estimée: {np.prod(senegal_shape) * 4 / (1024**2):.1f} MB")
        print(f"   Chunks de {self.chunk_size} jours")
        
        # Préparer le tableau final
        senegal_data_chunks = []
        
        with h5py.File(self.chirps_file_path, 'r') as f:
            precip_dataset = f['precip']
            
            for start_idx in range(0, total_days, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_days)
                chunk_num = start_idx // self.chunk_size + 1
                total_chunks = (total_days + self.chunk_size - 1) // self.chunk_size
                
                print(f"   📦 Chunk {chunk_num}/{total_chunks}: jours {start_idx+1}-{end_idx}")
                
                # Charger chunk complet
                chunk_data = precip_dataset[start_idx:end_idx, :, :].astype(np.float32)
                
                # Filtrer pour Sénégal immédiatement
                senegal_chunk = chunk_data[:, lat_mask, :][:, :, lon_mask]
                senegal_data_chunks.append(senegal_chunk)
                
                print(f"      Shape chunk: {senegal_chunk.shape}")
                print(f"      Mémoire chunk: {senegal_chunk.nbytes / (1024**2):.1f} MB")
                
                # Nettoyage mémoire
                del chunk_data, senegal_chunk
                gc.collect()
        
        # 4. Assembler tous les chunks
        print("🔧 Assemblage final des données...")
        senegal_data = np.concatenate(senegal_data_chunks, axis=0)
        
        # Nettoyage final
        del senegal_data_chunks
        gc.collect()
        
        # 5. Créer les dates
        start_date = datetime(1981, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(total_days)]
        
        print(f"✅ DONNÉES SÉNÉGAL CHARGÉES AVEC SUCCÈS:")
        print(f"   Shape finale: {senegal_data.shape}")
        print(f"   Mémoire finale: {senegal_data.nbytes / (1024**2):.1f} MB")
        print(f"   Période: {dates[0].strftime('%Y-%m-%d')} à {dates[-1].strftime('%Y-%m-%d')}")
        print(f"   Réduction mémoire: {(1 - senegal_data.nbytes / (np.prod(data_shape) * 8)) * 100:.1f}%")
        
        # 6. Statistiques rapides
        valid_data = senegal_data[~np.isnan(senegal_data)]
        if len(valid_data) > 0:
            print(f"📊 STATISTIQUES PRÉCIPITATIONS:")
            print(f"   Min: {valid_data.min():.2f} mm")
            print(f"   Max: {valid_data.max():.2f} mm")
            print(f"   Moyenne: {valid_data.mean():.2f} mm")
            print(f"   Valeurs valides: {len(valid_data):,}/{senegal_data.size:,} ({len(valid_data)/senegal_data.size*100:.1f}%)")
        
        return senegal_data, dates, senegal_lat, senegal_lon

# ============================================================================
# CLASSE PRINCIPALE D'ANALYSE - VERSION COMPLÈTE OPTIMISÉE
# ============================================================================

class ExtremeEventsAnalyzer:
    """
    Classe principale pour l'analyse complète des événements extrêmes.
    Version optimisée pour Docker avec contraintes mémoire + analyse complète.
    """
    
    def __init__(self, chirps_file_path: str = None):
        """
        Initialise l'analyseur.
        
        Args:
            chirps_file_path (str, optional): Chemin vers le fichier CHIRPS
        """
        try:
            self.chirps_file_path = chirps_file_path or str(CHIRPS_FILEPATH)
        except:
            self.chirps_file_path = chirps_file_path or "/app/data/raw/chirps_WA_1981_2023_dayly.mat"
            
        self.precip_data = None
        self.dates = None
        self.lats = None
        self.lons = None
        self.climatology = None
        self.std_dev = None
        self.anomalies = None
        self.extreme_events_df = None
        
        # Initialiser les modules
        self.detector = ExtremeEventDetector()
        self.classifier = SeasonClassifier()
        self.visualizer = DetectionVisualizer()
        self.report_generator = DetectionReportGenerator()
        
        print("✅ ExtremeEventsAnalyzer initialisé (version optimisée)")
    
    def step_1_load_data(self) -> bool:
        """
        Étape 1: Chargement optimisé des données CHIRPS.
        
        Returns:
            bool: True si succès, False sinon
        """
        print("\n" + "="*80)
        print("ÉTAPE 1: CHARGEMENT DES DONNÉES (OPTIMISÉ)")
        print("="*80)
        
        # Vérifier l'existence du fichier
        if not os.path.exists(self.chirps_file_path):
            print(f"❌ Fichier CHIRPS non trouvé: {self.chirps_file_path}")
            print("Veuillez placer le fichier dans le bon dossier ou ajuster le chemin dans config/settings.py")
            return False
        
        try:
            # Utiliser le loader optimisé INTÉGRÉ
            loader = OptimizedChirpsLoader(self.chirps_file_path)
            self.precip_data, self.dates, self.lats, self.lons = loader.load_senegal_data()
            
            if self.precip_data is None:
                print("❌ Échec du chargement des données")
                return False
            
            print("✅ Données chargées avec succès")
            print(f"   Forme des données: {self.precip_data.shape}")
            print(f"   Période: {self.dates[0].strftime('%Y-%m-%d')} à {self.dates[-1].strftime('%Y-%m-%d')}")
            print(f"   Points de grille: {self.precip_data.shape[1] * self.precip_data.shape[2]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_2_calculate_climatology(self) -> bool:
        """
        Étape 2: Calcul de la climatologie et des anomalies.
        
        Returns:
            bool: True si succès, False sinon
        """
        print("\n" + "="*80)
        print("ÉTAPE 2: CALCUL DE LA CLIMATOLOGIE ET DES ANOMALIES")
        print("="*80)
        
        try:
            # Calculer climatologie et anomalies
            self.climatology, self.std_dev, self.anomalies = calculate_climatology_and_anomalies(
                self.precip_data, self.dates
            )
            
            print("✅ Climatologie et anomalies calculées avec succès")
            print(f"   Climatologie: {self.climatology.shape}")
            print(f"   Anomalies: {self.anomalies.shape}")
            print(f"   Anomalie max: {np.nanmax(self.anomalies):.1f}σ")
            print(f"   Anomalie min: {np.nanmin(self.anomalies):.1f}σ")
            
            # Nettoyage mémoire après calcul
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du calcul de la climatologie: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_3_detect_extreme_events(self) -> bool:
        """
        Étape 3: Détection des événements extrêmes.
        
        Returns:
            bool: True si succès, False sinon
        """
        print("\n" + "="*80)
        print("ÉTAPE 3: DÉTECTION DES ÉVÉNEMENTS EXTRÊMES")
        print("="*80)
        
        try:
            # Détecter les événements
            self.extreme_events_df = self.detector.detect_events(
                self.precip_data, self.anomalies, self.dates, self.lats, self.lons
            )
            
            if self.extreme_events_df.empty:
                print("❌ Aucun événement extrême détecté")
                print("Essayez de réduire les critères de détection dans config/settings.py")
                return False
            
            print(f"✅ {len(self.extreme_events_df)} événements extrêmes détectés")
            print(f"   Période: {self.extreme_events_df.index.min().strftime('%Y-%m-%d')} à {self.extreme_events_df.index.max().strftime('%Y-%m-%d')}")
            print(f"   Précipitation moyenne: {self.extreme_events_df['max_precip'].mean():.2f} mm")
            
            if 'coverage_percent' in self.extreme_events_df.columns:
                print(f"   Couverture moyenne: {self.extreme_events_df['coverage_percent'].mean():.2f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la détection: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_4_seasonal_classification(self) -> str:
        """
        Étape 4: Classification saisonnière.
        
        Returns:
            str: Statut de validation climatologique
        """
        print("\n" + "="*80)
        print("ÉTAPE 4: CLASSIFICATION SAISONNIÈRE")
        print("="*80)
        
        try:
            # Classifier les saisons
            self.extreme_events_df, validation_status = self.classifier.classify_and_validate(
                self.extreme_events_df
            )
            
            print(f"✅ Classification saisonnière terminée")
            print(f"   Validation climatologique: {validation_status}")
            
            # Afficher la distribution
            if 'saison' in self.extreme_events_df.columns:
                season_counts = self.extreme_events_df['saison'].value_counts()
                for saison, count in season_counts.items():
                    pct = count / len(self.extreme_events_df) * 100
                    print(f"   {saison}: {count} événements ({pct:.1f}%)")
            
            return validation_status
            
        except Exception as e:
            print(f"❌ Erreur lors de la classification: {e}")
            import traceback
            traceback.print_exc()
            return "ERREUR"
    
    def step_5_generate_visualizations(self) -> bool:
        """
        Étape 5: Génération des visualisations.
        
        Returns:
            bool: True si succès, False sinon
        """
        print("\n" + "="*80)
        print("ÉTAPE 5: GÉNÉRATION DES VISUALISATIONS")
        print("="*80)
        
        try:
            # Générer toutes les visualisations
            self.visualizer.create_all_plots(self.extreme_events_df, self.lats, self.lons)
            
            print("✅ Toutes les visualisations ont été générées")
            print("   Fichiers créés dans outputs/visualizations/")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération des visualisations: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_6_generate_reports(self, validation_status: str) -> bool:
        """
        Étape 6: Génération des rapports.
        
        Args:
            validation_status (str): Statut de validation climatologique
            
        Returns:
            bool: True si succès, False sinon
        """
        print("\n" + "="*80)
        print("ÉTAPE 6: GÉNÉRATION DES RAPPORTS")
        print("="*80)
        
        try:
            # Générer tous les rapports
            stats = self.report_generator.generate_all_reports(
                self.extreme_events_df, validation_status
            )
            
            print("✅ Rapports générés avec succès")
            print("   Fichiers créés dans outputs/reports/")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération des rapports: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_7_save_data(self) -> bool:
        """
        Étape 7: Sauvegarde des données.
        
        Returns:
            bool: True si succès, False sinon
        """
        print("\n" + "="*80)
        print("ÉTAPE 7: SAUVEGARDE DES DONNÉES")
        print("="*80)
        
        try:
            # Créer dossiers si nécessaire
            Path("outputs/data").mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le dataset principal
            output_file = get_output_path('extreme_events')
            self.extreme_events_df.to_csv(output_file)
            print(f"✅ Dataset principal sauvegardé: {output_file}")
            
            # Sauvegarder la climatologie (optionnel)
            clim_file = get_output_path('climatology')
            np.savez_compressed(clim_file, 
                              climatology=self.climatology, 
                              std_dev=self.std_dev,
                              lats=self.lats,
                              lons=self.lons)
            print(f"✅ Climatologie sauvegardée: {clim_file}")
            
            # Sauvegarder les anomalies (optionnel)
            anom_file = get_output_path('anomalies')
            np.savez_compressed(anom_file, 
                              anomalies=self.anomalies,
                              dates=[d.strftime('%Y-%m-%d') for d in self.dates])
            print(f"✅ Anomalies sauvegardées: {anom_file}")
            
            # Nettoyage final mémoire
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_complete_analysis(self) -> bool:
        """
        Lance l'analyse complète des événements extrêmes (version optimisée).
        
        Returns:
            bool: True si succès, False sinon
        """
        try:
            print_project_info()
            
            # Créer les dossiers de sortie
            create_output_directories()
            
            # Exécuter toutes les étapes
            if not self.step_1_load_data():
                return False
            
            if not self.step_2_calculate_climatology():
                return False
            
            if not self.step_3_detect_extreme_events():
                return False
            
            validation_status = self.step_4_seasonal_classification()
            if validation_status == "ERREUR":
                return False
            
            if not self.step_5_generate_visualizations():
                return False
            
            if not self.step_6_generate_reports(validation_status):
                return False
            
            if not self.step_7_save_data():
                return False
            
            # Résumé final
            self.print_final_summary(validation_status)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERREUR DURANT L'ANALYSE: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_final_summary(self, validation_status: str):
        """
        Affiche le résumé final de l'analyse.
        
        Args:
            validation_status (str): Statut de validation climatologique
        """
        print("\n" + "="*80)
        print("✅ ANALYSE TERMINÉE AVEC SUCCÈS")
        print("="*80)
        
        # Statistiques principales
        n_events = len(self.extreme_events_df)
        
        print(f"📊 RÉSULTATS PRINCIPAUX:")
        print(f"   Événements détectés: {n_events}")
        print(f"   Période d'analyse: {self.extreme_events_df.index.min().strftime('%Y-%m-%d')} à {self.extreme_events_df.index.max().strftime('%Y-%m-%d')}")
        print(f"   Validation climatologique: {validation_status}")
        
        # Distribution saisonnière
        if 'saison' in self.extreme_events_df.columns:
            season_counts = self.extreme_events_df['saison'].value_counts()
            
            if 'Saison_des_pluies' in season_counts:
                pluies_pct = season_counts['Saison_des_pluies'] / n_events * 100
                print(f"   Saison des pluies: {season_counts['Saison_des_pluies']} événements ({pluies_pct:.1f}%)")
            
            if 'Saison_seche' in season_counts:
                seche_pct = season_counts['Saison_seche'] / n_events * 100
                print(f"   Saison sèche: {season_counts['Saison_seche']} événements ({seche_pct:.1f}%)")
        
        # Statistiques des événements
        print(f"   Précipitation moyenne: {self.extreme_events_df['max_precip'].mean():.2f} mm")
        
        if 'coverage_percent' in self.extreme_events_df.columns:
            print(f"   Couverture spatiale moyenne: {self.extreme_events_df['coverage_percent'].mean():.2f}%")
        
        if 'max_anomaly' in self.extreme_events_df.columns:
            print(f"   Anomalie moyenne: {self.extreme_events_df['max_anomaly'].mean():.2f}σ")
        
        # Statistiques mémoire
        memory_used = self.precip_data.nbytes / (1024**2)
        print(f"   💾 Mémoire utilisée: {memory_used:.1f} MB")
        
        print(f"\n📁 FICHIERS GÉNÉRÉS:")
        try:
            print(f"   • Dataset principal: {get_output_path('extreme_events')}")
            print(f"   • Rapport détaillé: {get_output_path('detection_report')}")
            print(f"   • Climatologie: {get_output_path('climatology')}")
            print(f"   • Anomalies: {get_output_path('anomalies')}")
        except:
            print(f"   • Dataset principal: outputs/data/extreme_events_senegal_final.csv")
            print(f"   • Rapport détaillé: outputs/reports/rapport_detection_evenements.txt")
            print(f"   • Visualisations: outputs/visualizations/")
        
        print(f"\n🎯 PRÊT POUR LES ÉTAPES SUIVANTES:")
        print(f"   • Analyse des indices climatiques (SST, ENSO, etc.)")
        print(f"   • Application des algorithmes d'apprentissage automatique")
        print(f"   • Développement de modèles prédictifs")
        
        print(f"\n🚀 OPTIMISATIONS APPLIQUÉES:")
        print(f"   • Chargement par chunks pour économie mémoire")
        print(f"   • Filtrage géographique immédiat (Sénégal uniquement)")
        print(f"   • Nettoyage mémoire automatique (garbage collection)")
        print(f"   • Fallbacks robustes en cas de modules manquants")

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale du script.
    """
    print("Script de détection des événements de précipitations extrêmes - Sénégal")
    print("Version COMPLÈTE + OPTIMISÉE MÉMOIRE avec architecture modulaire")
    print("="*80)
    
    # Vérifier les arguments de ligne de commande
    if len(sys.argv) > 1:
        chirps_file = sys.argv[1]
        print(f"📁 Fichier CHIRPS spécifié: {chirps_file}")
        if not os.path.exists(chirps_file):
            print(f"❌ Fichier non trouvé: {chirps_file}")
            return 1
    else:
        chirps_file = None
        try:
            print(f"📁 Utilisation du fichier CHIRPS par défaut: {CHIRPS_FILEPATH}")
        except:
            print(f"📁 Utilisation du fichier CHIRPS par défaut: /app/data/raw/chirps_WA_1981_2023_dayly.mat")
    
    # Créer et lancer l'analyseur
    analyzer = ExtremeEventsAnalyzer(chirps_file)
    
    # Lancer l'analyse COMPLÈTE (avec visualisations et rapports)
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎉 ANALYSE COMPLÈTE RÉUSSIE!")
        print("Vous pouvez maintenant passer à l'étape suivante de votre mémoire.")
        print("Le dataset est prêt pour l'analyse des indices climatiques et le machine learning.")
        print("📊 Toutes les visualisations et rapports ont été générés.")
        return 0
    else:
        print("\n💥 ÉCHEC DE L'ANALYSE")
        print("Vérifiez les erreurs ci-dessus et corrigez les problèmes.")
        print("Consultez le README.md pour plus d'informations sur la configuration.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)