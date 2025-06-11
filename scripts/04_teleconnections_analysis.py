#!/usr/bin/env python3
# scripts/04_teleconnections_analysis.py
"""
Script principal pour l'analyse des téléconnexions - VERSION CORRIGÉE
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION DES IMPORTS
# ============================================================================

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.data.climate_indices_loader import ClimateIndicesLoader
    from src.analysis.teleconnections import TeleconnectionsAnalyzer
    from src.config.settings import create_output_directories
    print("✅ Tous les modules importés avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Vérifiez que tous les modules sont présents dans le dossier src/")
    sys.exit(1)

# ============================================================================
# CLASSE PRINCIPALE D'ANALYSE DES TÉLÉCONNEXIONS - VERSION CORRIGÉE
# ============================================================================

class TeleconnectionsAnalysisMain:
    """
    Classe principale pour l'analyse des téléconnexions.
    """
    
    def __init__(self, max_lag: int = 12):
        self.max_lag = max_lag
        
        # CORRECTION: Définir tous les chemins nécessaires
        self.events_file = project_root / "data/processed/extreme_events_senegal_final.csv"
        self.indices_raw_path = project_root / "data/raw/climate_indices"  # AJOUTÉ
        self.indices_processed_file = project_root / "data/processed/climate_indices_combined.csv"
        
        # Initialisation des objets
        self.indices_loader = None  # Sera initialisé dans step_1
        self.teleconnections_analyzer = None  # Sera initialisé dans step_3
            
    def step_1_prepare_climate_indices(self) -> bool:
        """
        Étape 1: Préparation des indices climatiques.
        
        Returns:
            bool: True si succès, False sinon
        """
        print("\n" + "="*80)
        print("ÉTAPE 1: PRÉPARATION DES INDICES CLIMATIQUES")
        print("="*80)
        
        try:
            # CORRECTION: Vérifier que le dossier existe
            if not self.indices_raw_path.exists():
                print(f"❌ Dossier des indices climatiques non trouvé: {self.indices_raw_path}")
                print("Créez le dossier et placez-y vos fichiers d'indices (IOD, Nino34, TNA)")
                return False
            
            # Initialisation du loader avec le bon chemin
            self.indices_loader = ClimateIndicesLoader(str(self.indices_raw_path))
            
            # Chargement des indices individuels
            indices_dict = self.indices_loader.load_all_indices()
            
            if not indices_dict:
                print("❌ Échec du chargement des indices climatiques")
                print("Vérifiez que vos fichiers d'indices sont dans le bon format :")
                print(f"   - {self.indices_raw_path}/IOD_index.xlsx")
                print(f"   - {self.indices_raw_path}/Nino34_index.csv")
                print(f"   - {self.indices_raw_path}/TNA_index.csv")
                return False
            
            # Création du dataset combiné
            combined_df = self.indices_loader.create_combined_dataset()
            
            if combined_df.empty:
                print("❌ Échec de la création du dataset combiné")
                return False
            
            # Sauvegarde
            self.indices_loader.save_processed_data()
            
            print(f"✅ Indices climatiques préparés avec succès")
            print(f"   Période: {combined_df.index.min().strftime('%Y-%m')} à {combined_df.index.max().strftime('%Y-%m')}")
            print(f"   Indices: {list(combined_df.columns)}")
            print(f"   Observations: {len(combined_df)} mois")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la préparation des indices: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_2_verify_extreme_events(self) -> bool:
        """
        Étape 2: Vérification des événements extrêmes.
        """
        print("\n" + "="*80)
        print("ÉTAPE 2: VÉRIFICATION DES ÉVÉNEMENTS EXTRÊMES")
        print("="*80)
        
        try:
            if not self.events_file.exists():
                print(f"❌ Fichier des événements non trouvé: {self.events_file}")
                print("Lancez d'abord le script de détection des événements:")
                print("   python scripts/01_detection_extremes.py")
                return False
            
            # Chargement rapide pour vérification
            df_events = pd.read_csv(self.events_file, index_col=0, parse_dates=True)
            
            print(f"✅ Événements extrêmes vérifiés")
            print(f"   Nombre d'événements: {len(df_events)}")
            print(f"   Période: {df_events.index.min().strftime('%Y-%m-%d')} à {df_events.index.max().strftime('%Y-%m-%d')}")
            print(f"   Colonnes disponibles: {list(df_events.columns)}")
            
            # Vérification de la cohérence temporelle
            if self.indices_processed_file.exists():
                df_indices = pd.read_csv(self.indices_processed_file, index_col=0, parse_dates=True)
                
                # Vérification de l'overlap temporel
                events_start = df_events.index.min()
                events_end = df_events.index.max()
                indices_start = df_indices.index.min()
                indices_end = df_indices.index.max()
                
                overlap_start = max(events_start, indices_start)
                overlap_end = min(events_end, indices_end)
                
                if overlap_start < overlap_end:
                    overlap_years = (overlap_end - overlap_start).days / 365.25
                    print(f"   Période de chevauchement: {overlap_start.strftime('%Y-%m')} à {overlap_end.strftime('%Y-%m')} ({overlap_years:.1f} ans)")
                else:
                    print("⚠️  Aucun chevauchement temporel détecté entre événements et indices")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la vérification des événements: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_3_analyze_teleconnections(self) -> bool:
        """
        Étape 3: Analyse des téléconnexions.
        """
        print("\n" + "="*80)
        print("ÉTAPE 3: ANALYSE DES TÉLÉCONNEXIONS")
        print("="*80)
        
        try:
            # Initialisation de l'analyseur
            self.teleconnections_analyzer = TeleconnectionsAnalyzer()
            
            # Lancement de l'analyse complète
            results = self.teleconnections_analyzer.run_complete_analysis(
                events_file=str(self.events_file),
                indices_file=str(self.indices_processed_file)
            )
            
            if not results:
                print("❌ Échec de l'analyse des téléconnexions")
                return False
            
            print(f"✅ Analyse des téléconnexions terminée avec succès")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse des téléconnexions: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_4_generate_ml_features(self) -> bool:
        """
        Étape 4: Génération des features ML - VERSION SIMPLIFIÉE.
        """
        print("\n" + "="*80)
        print("ÉTAPE 4: GÉNÉRATION DES FEATURES MACHINE LEARNING")
        print("="*80)
        
        try:
            # CORRECTION: Vérifier que l'indices_loader existe
            if self.indices_loader is None:
                print("⚠️  Loader d'indices non initialisé, initialisation...")
                self.indices_loader = ClimateIndicesLoader(str(self.indices_raw_path))
                self.indices_loader.load_all_indices()
                self.indices_loader.create_combined_dataset()
            
            # Vérifier que combined_data existe
            if not hasattr(self.indices_loader, 'combined_data') or self.indices_loader.combined_data is None:
                print("🔄 Rechargement des indices...")
                self.indices_loader.load_all_indices()
                self.indices_loader.create_combined_dataset()
            
            # Création des features avec décalages
            lagged_features = self.indices_loader.create_lagged_features(max_lag=self.max_lag)
            
            if lagged_features.empty:
                print("❌ Échec de la création des features")
                return False
            
            print(f"✅ Features ML générées avec succès")
            print(f"   Nombre de features: {lagged_features.shape[1]}")
            print(f"   Nombre d'observations: {lagged_features.shape[0]}")
            print(f"   Période: {lagged_features.index.min().strftime('%Y-%m')} à {lagged_features.index.max().strftime('%Y-%m')}")
            
            # Sauvegarde des features
            output_dir = project_root / "data/processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            features_file = output_dir / f"climate_features_lag{self.max_lag}.csv"
            lagged_features.to_csv(features_file)
            
            print(f"   💾 Features sauvegardées: {features_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération des features: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_5_prepare_ml_dataset(self) -> bool:
        """
        Étape 5: Préparation du dataset final pour ML.
        """
        print("\n" + "="*80)
        print("ÉTAPE 5: PRÉPARATION DATASET MACHINE LEARNING")
        print("="*80)
        
        try:
            # Chargement des événements extrêmes
            df_events = pd.read_csv(self.events_file, index_col=0, parse_dates=True)
            
            # Chargement des features climatiques
            features_file = project_root / "data/processed" / f"climate_features_lag{self.max_lag}.csv"
            
            if not features_file.exists():
                print(f"❌ Fichier des features non trouvé: {features_file}")
                print("Exécutez d'abord l'étape 4 (génération des features)")
                return False
            
            df_features = pd.read_csv(features_file, index_col=0, parse_dates=True)
            
            # Création de la série mensuelle d'événements
            monthly_events = df_events.groupby(pd.Grouper(freq='MS')).size()
            monthly_events = monthly_events.reindex(df_features.index, fill_value=0)
            
            # Variable cible binaire (présence d'événements extrêmes dans le mois)
            target_binary = (monthly_events > 0).astype(int)
            
            # Variable cible continue (nombre d'événements par mois)
            target_count = monthly_events
            
            # Variable cible intensité (précipitation maximale du mois)
            monthly_max_precip = df_events.groupby(pd.Grouper(freq='MS'))['max_precip'].max()
            target_intensity = monthly_max_precip.reindex(df_features.index, fill_value=0)
            
            # Dataset ML final
            ml_dataset = pd.DataFrame(index=df_features.index)
            
            # Ajout des variables cibles
            ml_dataset['target_occurrence'] = target_binary
            ml_dataset['target_count'] = target_count
            ml_dataset['target_intensity'] = target_intensity
            
            # Ajout des features climatiques
            for col in df_features.columns:
                ml_dataset[f'feature_{col}'] = df_features[col]
            
            # Ajout de features temporelles
            ml_dataset['month'] = ml_dataset.index.month
            ml_dataset['season'] = ml_dataset['month'].apply(
                lambda x: 1 if x in [5, 6, 7, 8, 9, 10] else 0  # 1=pluies, 0=sèche
            )
            
            # Suppression des lignes avec trop de valeurs manquantes
            ml_dataset_clean = ml_dataset.dropna(thresh=int(len(ml_dataset.columns) * 0.8))
            
            print(f"✅ Dataset ML préparé avec succès")
            print(f"   Dimensions: {ml_dataset_clean.shape}")
            print(f"   Variables cibles: 3 (occurrence, count, intensity)")
            print(f"   Features climatiques: {len([c for c in ml_dataset_clean.columns if c.startswith('feature_')])}")
            print(f"   Features temporelles: 2 (month, season)")
            print(f"   Période: {ml_dataset_clean.index.min().strftime('%Y-%m')} à {ml_dataset_clean.index.max().strftime('%Y-%m')}")
            
            # Statistiques des variables cibles
            print(f"\n📊 STATISTIQUES DES VARIABLES CIBLES:")
            print(f"   Occurrence:")
            print(f"     Mois avec événements: {ml_dataset_clean['target_occurrence'].sum()}/{len(ml_dataset_clean)} ({ml_dataset_clean['target_occurrence'].mean()*100:.1f}%)")
            print(f"   Count:")
            print(f"     Événements/mois (moyenne): {ml_dataset_clean['target_count'].mean():.2f}")
            print(f"     Événements/mois (max): {ml_dataset_clean['target_count'].max()}")
            print(f"   Intensity:")
            print(f"     Intensité moyenne: {ml_dataset_clean['target_intensity'].mean():.2f} mm")
            print(f"     Intensité maximale: {ml_dataset_clean['target_intensity'].max():.2f} mm")
            
            # Sauvegarde
            output_file = project_root / "data/processed" / "ml_dataset_teleconnections.csv"
            ml_dataset_clean.to_csv(output_file)
            
            print(f"\n💾 Dataset ML sauvegardé: {output_file}")
            
            # Création d'un fichier de métadonnées
            metadata = {
                'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source_events': str(self.events_file),
                'source_indices': str(features_file),
                'n_observations': len(ml_dataset_clean),
                'n_features_climate': len([c for c in ml_dataset_clean.columns if c.startswith('feature_')]),
                'n_features_temporal': 2,
                'max_lag_months': self.max_lag,
                'period_start': ml_dataset_clean.index.min().strftime('%Y-%m'),
                'period_end': ml_dataset_clean.index.max().strftime('%Y-%m'),
                'target_variables': ['target_occurrence', 'target_count', 'target_intensity'],
                'ready_for_ml': True
            }
            
            metadata_file = project_root / "data/processed" / "ml_dataset_metadata.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"📋 Métadonnées sauvegardées: {metadata_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la préparation du dataset ML: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_complete_analysis(self) -> bool:
        """
        Lance l'analyse complète des téléconnexions.
        """
        print("🌊 ANALYSE COMPLÈTE DES TÉLÉCONNEXIONS OCÉAN-ATMOSPHÈRE")
        print("=" * 80)
        print("Analyse des relations entre indices climatiques (IOD, Nino34, TNA)")
        print("et événements de précipitations extrêmes au Sénégal")
        print("=" * 80)
        
        # Créer les dossiers de sortie
        create_output_directories()
        
        try:
            # Étape 1: Préparation des indices climatiques
            if not self.step_1_prepare_climate_indices():
                return False
            
            # Étape 2: Vérification des événements extrêmes
            if not self.step_2_verify_extreme_events():
                return False
            
            # Étape 3: Analyse des téléconnexions
            if not self.step_3_analyze_teleconnections():
                return False
            
            # Étape 4: Génération des features ML
            if not self.step_4_generate_ml_features():
                return False
            
            # Étape 5: Préparation du dataset ML final
            if not self.step_5_prepare_ml_dataset():
                return False
            
            # Résumé final
            self.print_final_summary()
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERREUR DURANT L'ANALYSE: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_final_summary(self):
        """
        Affiche le résumé final de l'analyse.
        """
        print("\n" + "="*80)
        print("✅ ANALYSE DES TÉLÉCONNEXIONS TERMINÉE AVEC SUCCÈS")
        print("="*80)
        
        # Vérification des fichiers générés
        output_files = [
            ("Indices climatiques", project_root / "data/processed/climate_indices_combined.csv"),
            ("Features ML", project_root / "data/processed" / f"climate_features_lag{self.max_lag}.csv"),
            ("Dataset ML", project_root / "data/processed/ml_dataset_teleconnections.csv"),
            ("Métadonnées", project_root / "data/processed/ml_dataset_metadata.json"),
            ("Rapport téléconnexions", project_root / "outputs/reports/rapport_teleconnexions.txt")
        ]
        
        print(f"📊 FICHIERS GÉNÉRÉS:")
        for name, filepath in output_files:
            if filepath.exists():
                print(f"   ✅ {name}: {filepath.name}")
            else:
                print(f"   ❌ {name}: MANQUANT")
        
        # Vérification des visualisations
        viz_dir = project_root / "outputs/visualizations/teleconnections"
        if viz_dir.exists():
            viz_files = list(viz_dir.glob("*.png"))
            print(f"\n🎨 VISUALISATIONS CRÉÉES: {len(viz_files)} fichiers")
            for viz_file in viz_files:
                print(f"   📈 {viz_file.name}")
        
        print(f"\n🎯 PRÊT POUR LE MACHINE LEARNING:")
        print(f"   • Dataset structuré avec variables cibles multiples")
        print(f"   • Features climatiques avec décalages optimaux")
        print(f"   • Période d'entraînement/test définie")
        print(f"   • Téléconnexions quantifiées et documentées")

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale du script.
    """
    parser = argparse.ArgumentParser(
        description="Analyse des téléconnexions océan-atmosphère pour les précipitations extrêmes au Sénégal"
    )
    parser.add_argument(
        "--max-lag", 
        type=int, 
        default=12,
        help="Décalage temporel maximum à analyser en mois (défaut: 12)"
    )
    parser.add_argument(
        "--correlation-type",
        choices=['pearson', 'spearman'],
        default='pearson',
        help="Type de corrélation à calculer (défaut: pearson)"
    )
    
    args = parser.parse_args()
    
    print("Script d'analyse des téléconnexions océan-atmosphère")
    print("Version intégrée avec architecture modulaire - CORRIGÉE")
    print("="*80)
    print(f"Configuration:")
    print(f"   Décalage maximum: {args.max_lag} mois")
    print(f"   Type de corrélation: {args.correlation_type}")
    print("="*80)
    
    # Créer et lancer l'analyseur
    analyzer = TeleconnectionsAnalysisMain(max_lag=args.max_lag)
    
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎉 ANALYSE RÉUSSIE!")
        print("Les téléconnexions ont été quantifiées et le dataset ML est prêt.")
        print("Vous pouvez maintenant passer au développement des modèles prédictifs.")
        return 0
    else:
        print("\n💥 ÉCHEC DE L'ANALYSE")
        print("Vérifiez les erreurs ci-dessus et corrigez les problèmes.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)