#!/usr/bin/env python3
# scripts/01_detection_extremes.py
"""
Script principal pour la détection des événements de précipitations extrêmes au Sénégal.
VERSION OPTIMISÉE MÉMOIRE - Chargement par chunks pour éviter les problèmes de mémoire Docker.

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
Version: 2.1 - Optimisée mémoire Docker (Import corrigé)
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

# IMPORTS SÉLECTIFS - ÉVITER ChirpsDataLoader
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

try:
    from src.analysis.climatology import calculate_climatology_and_anomalies
    print("✅ Climatology importé")
except ImportError as e:
    print(f"⚠️ Problème climatology: {e}")

try:
    from src.analysis.detection import ExtremeEventDetector
    print("✅ Detection importé")
except ImportError as e:
    print(f"⚠️ Problème detection: {e}")

try:
    from src.utils.season_classifier import SeasonClassifier
    print("✅ Season classifier importé")
except ImportError as e:
    print(f"⚠️ Problème season classifier: {e}")

try:
    from src.visualization.detection_plots import DetectionVisualizer
    print("✅ Visualizer importé")
except ImportError as e:
    print(f"⚠️ Problème visualizer: {e}")

try:
    from src.reports.detection_report import DetectionReportGenerator
    print("✅ Report generator importé")
except ImportError as e:
    print(f"⚠️ Problème report generator: {e}")

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
# CLASSE PRINCIPALE D'ANALYSE - VERSION CORRIGÉE
# ============================================================================

class ExtremeEventsAnalyzer:
    """
    Classe principale pour l'analyse complète des événements extrêmes.
    Version optimisée pour Docker avec contraintes mémoire.
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
        
        # Initialiser les modules disponibles
        try:
            self.detector = ExtremeEventDetector()
            print("✅ ExtremeEventDetector initialisé")
        except:
            self.detector = None
            print("⚠️ ExtremeEventDetector non disponible")
            
        try:
            self.classifier = SeasonClassifier()
            print("✅ SeasonClassifier initialisé")
        except:
            self.classifier = None
            print("⚠️ SeasonClassifier non disponible")
            
        try:
            self.visualizer = DetectionVisualizer()
            print("✅ DetectionVisualizer initialisé")
        except:
            self.visualizer = None
            print("⚠️ DetectionVisualizer non disponible")
            
        try:
            self.report_generator = DetectionReportGenerator()
            print("✅ DetectionReportGenerator initialisé")
        except:
            self.report_generator = None
            print("⚠️ DetectionReportGenerator non disponible")
        
    def step_1_load_data(self) -> bool:
        """
        Étape 1: Chargement optimisé des données CHIRPS.
        
        Returns:
            bool: True si succès, False sinon
        """
        print("\n" + "="*80)
        print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
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
    
    def step_2_basic_analysis(self) -> bool:
        """
        Étape 2: Analyse basique des données (si modules avancés non disponibles).
        
        Returns:
            bool: True si succès, False sinon
        """
        print("\n" + "="*80)
        print("ÉTAPE 2: ANALYSE BASIQUE DES DONNÉES")
        print("="*80)
        
        try:
            # Créer dossiers de sortie
            output_dir = Path("/app/outputs/data")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder données de base
            output_file = output_dir / "senegal_precip_data_optimized.npz"
            np.savez_compressed(
                output_file,
                precip_data=self.precip_data,
                dates=[d.strftime('%Y-%m-%d') for d in self.dates],
                latitude=self.lats,
                longitude=self.lons
            )
            
            print(f"✅ Données sauvegardées: {output_file}")
            print(f"   📁 Taille fichier: {output_file.stat().st_size / (1024**2):.1f} MB")
            
            # Statistiques basiques
            valid_data = self.precip_data[~np.isnan(self.precip_data)]
            extreme_threshold = np.percentile(valid_data[valid_data > 0], 95)  # 95e percentile
            
            print(f"📊 Analyse statistique:")
            print(f"   Seuil extrême (95e percentile): {extreme_threshold:.2f} mm")
            print(f"   Jours avec précipitations: {(valid_data > 0.1).sum():,}")
            print(f"   Jours extrêmes: {(valid_data > extreme_threshold).sum():,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse basique: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_basic_analysis(self) -> bool:
        """
        Lance l'analyse basique (chargement + statistiques).
        
        Returns:
            bool: True si succès, False sinon
        """
        try:
            # Affichage info projet si disponible
            try:
                print_project_info()
                create_output_directories()
            except:
                print("📁 Configuration basique utilisée")
            
            # Exécuter les étapes de base
            if not self.step_1_load_data():
                return False
            
            if not self.step_2_basic_analysis():
                return False
            
            # Résumé final
            print("\n" + "="*80)
            print("✅ ANALYSE BASIQUE TERMINÉE AVEC SUCCÈS")
            print("="*80)
            print(f"📊 Données Sénégal chargées: {self.precip_data.shape}")
            print(f"💾 Mémoire utilisée: {self.precip_data.nbytes / (1024**2):.1f} MB")
            print(f"🎯 Données prêtes pour l'analyse avancée")
            print(f"🚀 Pipeline ML peut maintenant continuer")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERREUR DURANT L'ANALYSE: {e}")
            import traceback
            traceback.print_exc()
            return False

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale du script.
    """
    print("Script de détection des événements de précipitations extrêmes - Sénégal")
    print("Version refactorisée avec architecture modulaire - OPTIMISÉE MÉMOIRE v2.1")
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
    
    # Lancer l'analyse basique (qui inclut le chargement optimisé)
    success = analyzer.run_basic_analysis()
    
    if success:
        print("\n🎉 ANALYSE RÉUSSIE!")
        print("Chargement optimisé terminé - données prêtes pour l'analyse complète.")
        print("Le dataset est prêt pour l'analyse des indices climatiques et le machine learning.")
        return 0
    else:
        print("\n💥 ÉCHEC DE L'ANALYSE")
        print("Vérifiez les erreurs ci-dessus et corrigez les problèmes.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)