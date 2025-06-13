#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/06_prediction_tool.py
"""
Outil de prédiction utilisant les modèles ML entraînés.
Permet de faire des prédictions sur de nouvelles données climatiques.

Auteur: Analyse Précipitations Extrêmes
Date: 2025
"""

import sys
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration des chemins
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ExtremeEventPredictor:
    """Outil de prédiction des événements de précipitations extrêmes."""
    
    def __init__(self):
        """Initialise le prédicteur."""
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.metadata = {}
        
        # Chemins
        self.model_dir = project_root / "outputs/models"
        self.data_dir = project_root / "data/processed"
        
    def load_models(self):
        """Charge les modèles pré-entraînés."""
        print("🔄 CHARGEMENT DES MODÈLES PRÉ-ENTRAÎNÉS")
        print("=" * 50)
        
        try:
            # Charger le scaler
            scaler_path = self.model_dir / "feature_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print("✅ Scaler chargé")
            else:
                print("❌ Scaler non trouvé")
                return False
            
            # Charger les métadonnées
            metadata_path = project_root / "outputs/data/ml_results_summary.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print("✅ Métadonnées chargées")
            
            # Chercher et charger les modèles disponibles
            model_files = list(self.model_dir.glob("*.pkl"))
            model_files = [f for f in model_files if f.name != "feature_scaler.pkl"]
            
            for model_file in model_files:
                try:
                    model = joblib.load(model_file)
                    model_name = model_file.stem
                    self.models[model_name] = model
                    print(f"✅ Modèle chargé: {model_name}")
                except Exception as e:
                    print(f"⚠️  Erreur chargement {model_file.name}: {e}")
            
            if not self.models:
                print("❌ Aucun modèle trouvé")
                return False
            
            # Charger les noms des features depuis le dataset ML
            ml_dataset_path = self.data_dir / "ml_dataset_teleconnections.csv"
            if ml_dataset_path.exists():
                sample_data = pd.read_csv(ml_dataset_path, nrows=1)
                target_cols = ['occurrence', 'count', 'intensity']
                self.feature_names = [col for col in sample_data.columns if col not in target_cols]
                print(f"✅ Features identifiées: {len(self.feature_names)} variables")
            
            print(f"\n📊 MODÈLES DISPONIBLES:")
            for name in self.models.keys():
                model_type = "Classification" if "classifier" in name else "Régression"
                print(f"   • {name}: {model_type}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def create_sample_input(self):
        """Crée un exemple d'entrée basé sur les données historiques."""
        print("\n📋 CRÉATION D'UN EXEMPLE D'ENTRÉE")
        print("=" * 50)
        
        try:
            # Charger un échantillon du dataset
            ml_dataset_path = self.data_dir / "ml_dataset_teleconnections.csv"
            if not ml_dataset_path.exists():
                print("❌ Dataset ML non trouvé")
                return None
            
            df = pd.read_csv(ml_dataset_path, index_col=0, parse_dates=True)
            
            # Prendre les données du mois le plus récent
            latest_data = df.iloc[-1]
            
            # Extraire seulement les features
            sample_input = latest_data[self.feature_names].to_dict()
            
            print("✅ Exemple d'entrée créé basé sur les données les plus récentes")
            print(f"   Date de référence: {df.index[-1]}")
            print(f"   Nombre de features: {len(sample_input)}")
            
            # Afficher quelques features importantes
            print(f"\n📊 APERÇU DES FEATURES:")
            for i, (feature, value) in enumerate(list(sample_input.items())[:10]):
                print(f"   {feature}: {value:.3f}")
            if len(sample_input) > 10:
                print(f"   ... et {len(sample_input) - 10} autres features")
            
            return sample_input
            
        except Exception as e:
            print(f"❌ Erreur lors de la création: {e}")
            return None
    
    def predict_from_features(self, features_dict):
        """Fait des prédictions à partir d'un dictionnaire de features."""
        print(f"\n🔮 PRÉDICTIONS À PARTIR DES FEATURES")
        print("=" * 50)
        
        try:
            # Convertir en DataFrame
            features_df = pd.DataFrame([features_dict])
            
            # Vérifier que toutes les features sont présentes
            missing_features = set(self.feature_names) - set(features_df.columns)
            if missing_features:
                print(f"⚠️  Features manquantes: {len(missing_features)} features")
                # Ajouter des valeurs par défaut (0) pour les features manquantes
                for feature in missing_features:
                    features_df[feature] = 0
            
            # Réorganiser les colonnes selon l'ordre attendu
            features_df = features_df[self.feature_names]
            
            # Normaliser avec le scaler
            features_scaled = self.scaler.transform(features_df)
            
            print(f"✅ Features préparées et normalisées")
            
            # Faire les prédictions avec tous les modèles
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    if "classifier" in model_name:
                        # Prédiction de classification (occurrence)
                        prob = model.predict_proba(features_scaled)[0, 1]
                        pred = model.predict(features_scaled)[0]
                        
                        predictions[model_name] = {
                            'type': 'classification',
                            'prediction': bool(pred),
                            'probability': float(prob),
                            'confidence': 'Élevée' if abs(prob - 0.5) > 0.3 else 'Modérée'
                        }
                        
                    elif "regressor" in model_name:
                        # Prédiction de régression (intensité)
                        intensity = model.predict(features_scaled)[0]
                        
                        predictions[model_name] = {
                            'type': 'regression',
                            'predicted_intensity': float(max(0, intensity)),  # Pas d'intensité négative
                            'unit': 'mm/jour'
                        }
                        
                except Exception as e:
                    print(f"⚠️  Erreur avec {model_name}: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            return {}
    
    def predict_from_climate_indices(self, iod=0, nino34=0, tna=0, month=None):
        """Fait des prédictions à partir des indices climatiques principaux."""
        print(f"\n🌊 PRÉDICTION À PARTIR DES INDICES CLIMATIQUES")
        print("=" * 50)
        
        if month is None:
            month = datetime.now().month
        
        print(f"📊 Indices climatiques:")
        print(f"   IOD: {iod:.3f}")
        print(f"   Nino34: {nino34:.3f}")
        print(f"   TNA: {tna:.3f}")
        print(f"   Mois: {month}")
        
        # Créer un dictionnaire de features simplifié
        # En pratique, on devrait avoir tous les lags, mais on simplifie ici
        features_dict = {}
        
        # Ajouter les indices principaux (lag 0)
        features_dict['IOD_lag0'] = iod
        features_dict['Nino34_lag0'] = nino34
        features_dict['TNA_lag0'] = tna
        
        # Ajouter quelques lags importants (on utilise les mêmes valeurs par simplicité)
        for lag in range(1, 13):
            features_dict[f'IOD_lag{lag}'] = iod * (0.9**lag)  # Décroissance artificielle
            features_dict[f'Nino34_lag{lag}'] = nino34 * (0.9**lag)
            features_dict[f'TNA_lag{lag}'] = tna * (0.9**lag)
        
        # Ajouter les features temporelles
        features_dict['month'] = month
        features_dict['season'] = 1 if month in [5, 6, 7, 8, 9, 10] else 0  # Saison des pluies
        
        # Compléter avec des zéros pour les features manquantes
        for feature in self.feature_names:
            if feature not in features_dict:
                features_dict[feature] = 0
        
        return self.predict_from_features(features_dict)
    
    def display_predictions(self, predictions):
        """Affiche les prédictions de manière formatée."""
        print(f"\n🎯 RÉSULTATS DES PRÉDICTIONS")
        print("=" * 50)
        
        if not predictions:
            print("❌ Aucune prédiction disponible")
            return
        
        # Séparer classification et régression
        classification_results = {k: v for k, v in predictions.items() if v['type'] == 'classification'}
        regression_results = {k: v for k, v in predictions.items() if v['type'] == 'regression'}
        
        # Afficher les résultats de classification
        if classification_results:
            print("🎯 CLASSIFICATION (Occurrence d'événements):")
            print("-" * 40)
            
            for model_name, result in classification_results.items():
                model_display = model_name.replace('_classifier', '').replace('_', ' ').title()
                status = "🔴 ÉVÉNEMENT PRÉDIT" if result['prediction'] else "🟢 PAS D'ÉVÉNEMENT"
                
                print(f"   {model_display}:")
                print(f"     • Prédiction: {status}")
                print(f"     • Probabilité: {result['probability']:.1%}")
                print(f"     • Confiance: {result['confidence']}")
                print()
        
        # Afficher les résultats de régression
        if regression_results:
            print("📊 RÉGRESSION (Intensité prédite):")
            print("-" * 35)
            
            for model_name, result in regression_results.items():
                model_display = model_name.replace('_regressor', '').replace('_', ' ').title()
                intensity = result['predicted_intensity']
                
                # Catégoriser l'intensité
                if intensity < 10:
                    category = "Faible"
                    color = "🟢"
                elif intensity < 30:
                    category = "Modérée"
                    color = "🟡"
                elif intensity < 50:
                    category = "Forte"
                    color = "🟠"
                else:
                    category = "Très forte"
                    color = "🔴"
                
                print(f"   {model_display}:")
                print(f"     • Intensité: {intensity:.1f} {result['unit']}")
                print(f"     • Catégorie: {color} {category}")
                print()
        
        # Synthèse et recommandations
        print("📋 SYNTHÈSE ET RECOMMANDATIONS:")
        print("-" * 35)
        
        # Consensus des modèles de classification
        if classification_results:
            predictions_positive = sum(1 for r in classification_results.values() if r['prediction'])
            consensus_prob = np.mean([r['probability'] for r in classification_results.values()])
            
            if predictions_positive > len(classification_results) / 2:
                print("   🔴 RISQUE ÉLEVÉ: Majorité des modèles prédisent un événement")
                print("   💡 Recommandation: Surveillance renforcée conseillée")
            else:
                print("   🟢 RISQUE FAIBLE: Majorité des modèles ne prédisent pas d'événement")
                print("   💡 Recommandation: Surveillance normale")
            
            print(f"   📊 Probabilité moyenne: {consensus_prob:.1%}")
        
        # Intensité maximale prédite
        if regression_results:
            max_intensity = max(r['predicted_intensity'] for r in regression_results.values())
            print(f"   🌧️  Intensité maximale prédite: {max_intensity:.1f} mm/jour")
            
            if max_intensity > 50:
                print("   ⚠️  Intensité potentiellement dangereuse")
            elif max_intensity > 20:
                print("   ⚡ Intensité significative possible")
    
    def interactive_prediction(self):
        """Mode interactif pour les prédictions."""
        print(f"\n🔮 MODE PRÉDICTION INTERACTIF")
        print("=" * 50)
        
        while True:
            print(f"\nOPTIONS DISPONIBLES:")
            print("1. Prédiction avec indices climatiques simplifiés")
            print("2. Prédiction avec exemple basé sur données historiques")
            print("3. Quitter")
            
            try:
                choice = input("\nVotre choix (1-3): ").strip()
                
                if choice == "1":
                    print(f"\n📊 SAISIE DES INDICES CLIMATIQUES:")
                    try:
                        iod = float(input("IOD (Indian Ocean Dipole, ex: -0.5 à 1.0): ") or "0")
                        nino34 = float(input("Nino34 (ENSO, ex: -2.0 à 2.0): ") or "0")
                        tna = float(input("TNA (Tropical North Atlantic, ex: -1.0 à 1.0): ") or "0")
                        month = int(input("Mois (1-12): ") or str(datetime.now().month))
                        
                        predictions = self.predict_from_climate_indices(iod, nino34, tna, month)
                        self.display_predictions(predictions)
                        
                    except ValueError:
                        print("❌ Valeurs invalides, veuillez réessayer")
                
                elif choice == "2":
                    sample_input = self.create_sample_input()
                    if sample_input:
                        predictions = self.predict_from_features(sample_input)
                        self.display_predictions(predictions)
                
                elif choice == "3":
                    print("👋 Au revoir!")
                    break
                
                else:
                    print("❌ Choix invalide")
                    
            except KeyboardInterrupt:
                print("\n👋 Au revoir!")
                break
    
    def run_prediction_tool(self):
        """Lance l'outil de prédiction."""
        print("🔮 OUTIL DE PRÉDICTION - ÉVÉNEMENTS EXTRÊMES")
        print("=" * 70)
        print("Utilise les modèles ML entraînés pour prédire les précipitations extrêmes")
        print("=" * 70)
        
        # Charger les modèles
        if not self.load_models():
            print("❌ Impossible de charger les modèles")
            return False
        
        # Afficher les informations des modèles
        if self.metadata:
            print(f"\n📊 INFORMATIONS DES MODÈLES:")
            if 'data_info' in self.metadata:
                data_info = self.metadata['data_info']
                print(f"   • Période d'entraînement: {data_info.get('train_period', 'N/A')}")
                print(f"   • Période de test: {data_info.get('test_period', 'N/A')}")
                print(f"   • Nombre de features: {data_info.get('n_features', 'N/A')}")
            
            # Meilleur modèle de classification
            if 'classification_results' in self.metadata:
                best_classifier = max(
                    self.metadata['classification_results'].items(),
                    key=lambda x: x[1]['test_f1']
                )
                print(f"   • Meilleur classificateur: {best_classifier[0]} (F1: {best_classifier[1]['test_f1']:.3f})")
            
            # Meilleur modèle de régression
            if 'regression_results' in self.metadata:
                best_regressor = max(
                    self.metadata['regression_results'].items(),
                    key=lambda x: x[1]['test_r2']
                )
                print(f"   • Meilleur régresseur: {best_regressor[0]} (R²: {best_regressor[1]['test_r2']:.3f})")
        
        # Lancer le mode interactif
        self.interactive_prediction()
        
        return True

def main():
    """Fonction principale."""
    print("🔮 OUTIL DE PRÉDICTION ML - PRÉCIPITATIONS EXTRÊMES")
    print("=" * 70)
    
    predictor = ExtremeEventPredictor()
    success = predictor.run_prediction_tool()
    
    if success:
        print("\n✅ OUTIL DE PRÉDICTION TERMINÉ")
        print("\n💡 UTILISATION RECOMMANDÉE:")
        print("• Intégrez cet outil dans un système de monitoring")
        print("• Utilisez les prédictions pour l'alerte précoce")
        print("• Validez les prédictions avec les observations")
        print("• Mettez à jour les modèles régulièrement")
        return 0
    else:
        print("\n❌ ÉCHEC DE L'OUTIL DE PRÉDICTION")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)