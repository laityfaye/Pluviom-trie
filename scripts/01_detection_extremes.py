#!/usr/bin/env python3
# scripts/01_detection_extremes.py
"""
Script principal pour la détection des événements de précipitations extrêmes au Sénégal.

Ce script orchestre l'ensemble du processus d'analyse :
1. Chargement des données CHIRPS
2. Calcul de la climatologie et des anomalies
3. Détection des événements extrêmes
4. Classification saisonnière
5. Génération des visualisations et rapports

Utilisation:
    python scripts/01_detection_extremes.py
    python scripts/01_detection_extremes.py /chemin/vers/chirps.mat

Auteur: [Votre nom]
Date: [Date]
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION DES IMPORTS - VERSION CORRIGÉE
# ============================================================================

# Ajouter le dossier racine et src au PYTHONPATH
project_root = Path(__file__).parent.parent  # Remonte de scripts/ à la racine
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Imports des modules du projet avec préfixe src. explicite
try:
    from src.config.settings import (
        CHIRPS_FILEPATH, DETECTION_CRITERIA, PROJECT_INFO,
        create_output_directories, print_project_info, get_output_path
    )
    from src.data.loader import ChirpsDataLoader
    from src.analysis.climatology import calculate_climatology_and_anomalies
    from src.analysis.detection import ExtremeEventDetector
    from src.utils.season_classifier import SeasonClassifier
    from src.visualization.detection_plots import DetectionVisualizer
    from src.reports.detection_report import DetectionReportGenerator
    print("✅ Tous les modules importés avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Vérifiez que tous les modules sont présents dans le dossier src/")
    print("Structure attendue:")
    print("  src/")
    print("    config/settings.py")
    print("    data/loader.py")
    print("    analysis/climatology.py")
    print("    analysis/detection.py")
    print("    utils/season_classifier.py")
    print("    visualization/detection_plots.py")
    print("    reports/detection_report.py")
    print("\nVérifiez aussi que tous les fichiers __init__.py sont présents")
    sys.exit(1)

# ============================================================================
# CLASSE PRINCIPALE D'ANALYSE
# ============================================================================

class ExtremeEventsAnalyzer:
    """
    Classe principale pour l'analyse complète des événements extrêmes.
    """
    
    def __init__(self, chirps_file_path: str = None):
        """
        Initialise l'analyseur.
        
        Args:
            chirps_file_path (str, optional): Chemin vers le fichier CHIRPS
        """
        self.chirps_file_path = chirps_file_path or str(CHIRPS_FILEPATH)
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
        
    def step_1_load_data(self) -> bool:
        """
        Étape 1: Chargement des données CHIRPS.
        
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
            print("Structure attendue:")
            print(f"  {project_root}/data/raw/chirps_WA_1981_2023_dayly.mat")
            return False
        
        # Charger les données
        loader = ChirpsDataLoader(self.chirps_file_path)
        self.precip_data, self.dates, self.lats, self.lons = loader.load_senegal_data()
        
        if self.precip_data is None:
            print("❌ Échec du chargement des données")
            return False
        
        print("✅ Données chargées avec succès")
        print(f"   Forme des données: {self.precip_data.shape}")
        print(f"   Période: {self.dates[0].strftime('%Y-%m-%d')} à {self.dates[-1].strftime('%Y-%m-%d')}")
        print(f"   Points de grille: {self.precip_data.shape[1] * self.precip_data.shape[2]}")
        
        return True
    
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
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_complete_analysis(self) -> bool:
        """
        Lance l'analyse complète des événements extrêmes.
        
        Returns:
            bool: True si succès, False sinon
        """
        print_project_info()
        
        # Créer les dossiers de sortie
        create_output_directories()
        
        try:
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
        season_counts = self.extreme_events_df['saison'].value_counts()
        
        print(f"📊 RÉSULTATS PRINCIPAUX:")
        print(f"   Événements détectés: {n_events}")
        print(f"   Période d'analyse: {self.extreme_events_df.index.min().strftime('%Y-%m-%d')} à {self.extreme_events_df.index.max().strftime('%Y-%m-%d')}")
        print(f"   Validation climatologique: {validation_status}")
        
        if 'Saison_des_pluies' in season_counts:
            pluies_pct = season_counts['Saison_des_pluies'] / n_events * 100
            print(f"   Saison des pluies: {season_counts['Saison_des_pluies']} événements ({pluies_pct:.1f}%)")
        
        if 'Saison_seche' in season_counts:
            seche_pct = season_counts['Saison_seche'] / n_events * 100
            print(f"   Saison sèche: {season_counts['Saison_seche']} événements ({seche_pct:.1f}%)")
        
        print(f"   Précipitation moyenne: {self.extreme_events_df['max_precip'].mean():.2f} mm")
        print(f"   Couverture spatiale moyenne: {self.extreme_events_df['coverage_percent'].mean():.2f}%")
        print(f"   Anomalie moyenne: {self.extreme_events_df['max_anomaly'].mean():.2f}σ")
        
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

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale du script.
    """
    print("Script de détection des événements de précipitations extrêmes - Sénégal")
    print("Version refactorisée avec architecture modulaire")
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
        print(f"📁 Utilisation du fichier CHIRPS par défaut: {CHIRPS_FILEPATH}")
    
    # Créer et lancer l'analyseur
    analyzer = ExtremeEventsAnalyzer(chirps_file)
    
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎉 ANALYSE RÉUSSIE!")
        print("Vous pouvez maintenant passer à l'étape suivante de votre mémoire.")
        print("Le dataset est prêt pour l'analyse des indices climatiques et le machine learning.")
        return 0
    else:
        print("\n💥 ÉCHEC DE L'ANALYSE")
        print("Vérifiez les erreurs ci-dessus et corrigez les problèmes.")
        print("Consultez le README.md pour plus d'informations sur la configuration.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)