#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/05_machine_learning_pipeline.py
"""
Pipeline complet d'apprentissage automatique pour la prédiction des événements de précipitations extrêmes.
Utilise K-Means, Random Forest, XGBoost, SVM et réseaux de neurones.

Auteur: Analyse Précipitations Extrêmes
Date: 2025
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (classification_report, confusion_matrix, 
                           mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, precision_score, recall_score, f1_score,
                           silhouette_score, roc_curve, auc)
from sklearn.decomposition import PCA
import xgboost as xgb
from scipy import stats
import joblib

# Configuration des chemins
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class MachineLearningPipeline:
    """Pipeline complet d'apprentissage automatique pour les précipitations extrêmes."""
    
    def __init__(self):
        """Initialise le pipeline ML."""
        self.data = None
        self.features = None
        self.targets = None
        self.features_scaled = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.classification_results = {}
        self.regression_results = {}
        
        # Variables pour train/test split
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Créer les dossiers de sortie
        self.output_dir = project_root / "outputs"
        self.model_dir = self.output_dir / "models"
        self.viz_dir = self.output_dir / "visualizations" / "machine_learning"
        self.report_dir = self.output_dir / "reports"
        
        for directory in [self.model_dir, self.viz_dir, self.report_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Charge le dataset ML préparé."""
        print("🔄 CHARGEMENT DES DONNÉES")
        print("=" * 50)
        
        try:
            # Charger le dataset ML
            ml_file = project_root / "data/processed/ml_dataset_teleconnections.csv"
            if not ml_file.exists():
                raise FileNotFoundError(f"Dataset ML non trouvé: {ml_file}")
            
            self.data = pd.read_csv(ml_file, index_col=0, parse_dates=True)
            print(f"✅ Dataset chargé: {self.data.shape}")
            print(f"   Période: {self.data.index.min()} à {self.data.index.max()}")
            
            # Identifier les features et targets
            target_cols = ['occurrence', 'count', 'intensity']
            feature_cols = [col for col in self.data.columns if col not in target_cols]
            
            self.features = self.data[feature_cols]
            self.targets = self.data[target_cols]
            
            print(f"✅ Features: {self.features.shape[1]} variables")
            print(f"   Climatiques: {len([c for c in feature_cols if any(idx in c for idx in ['IOD', 'Nino34', 'TNA'])])}")
            print(f"   Temporelles: {len([c for c in feature_cols if c in ['month', 'season']])}")
            print(f"✅ Targets: {len(target_cols)} variables")
            print(f"   {', '.join(target_cols)}")
            
            # Statistiques descriptives
            print(f"\n📊 STATISTIQUES DES TARGETS:")
            for col in target_cols:
                if col == 'occurrence':
                    print(f"   {col}: {self.targets[col].sum()}/{len(self.targets)} mois avec événements ({100*self.targets[col].mean():.1f}%)")
                else:
                    print(f"   {col}: μ={self.targets[col].mean():.2f}, σ={self.targets[col].std():.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def preprocess_data(self):
        """Prétraite les données pour le ML."""
        print(f"\n🔧 PRÉTRAITEMENT DES DONNÉES")
        print("=" * 50)
        
        # Gérer les valeurs manquantes
        missing_before = self.features.isnull().sum().sum()
        self.features = self.features.fillna(self.features.mean())
        self.targets = self.targets.fillna(0)
        
        print(f"✅ Valeurs manquantes traitées: {missing_before} → 0")
        
        # Encoder les variables catégorielles
        if 'season' in self.features.columns:
            le = LabelEncoder()
            self.features['season'] = le.fit_transform(self.features['season'].astype(str))
            print(f"✅ Variable catégorielle encodée: season")
        
        # Normaliser les features
        feature_cols = self.features.columns
        self.features_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.features),
            columns=feature_cols,
            index=self.features.index
        )
        
        print(f"✅ Features normalisées: StandardScaler appliqué")
        print(f"   Shape finale: {self.features_scaled.shape}")
        
        return True
    
    def analyze_feature_importance(self):
        """Analyse l'importance des features avec Random Forest."""
        print(f"\n🎯 ANALYSE DE L'IMPORTANCE DES FEATURES")
        print("=" * 50)
        
        # Random Forest pour l'occurrence
        rf_occurrence = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_occurrence.fit(self.features_scaled, self.targets['occurrence'])
        
        # Importance des features
        feature_importance = pd.DataFrame({
            'feature': self.features_scaled.columns,
            'importance': rf_occurrence.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Visualisation
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(top_features)), top_features['importance'])
        plt.xticks(range(len(top_features)), top_features['feature'], rotation=45, ha='right')
        plt.title('Top 15 Features les Plus Importantes (Random Forest)', fontweight='bold')
        plt.ylabel('Importance')
        plt.grid(axis='y', alpha=0.3)
        
        # Colorer les barres par type de feature
        colors = []
        for feat in top_features['feature']:
            if 'TNA' in feat:
                colors.append('#1f77b4')  # Bleu pour TNA
            elif 'Nino34' in feat:
                colors.append('#ff7f0e')  # Orange pour Nino34
            elif 'IOD' in feat:
                colors.append('#2ca02c')  # Vert pour IOD
            else:
                colors.append('#d62728')  # Rouge pour autres
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Corrélations entre top features et occurrence
        plt.subplot(2, 1, 2)
        top_10_features = top_features.head(10)['feature']
        correlations = []
        for feat in top_10_features:
            corr = self.features_scaled[feat].corr(self.targets['occurrence'])
            correlations.append(corr)
        
        bars2 = plt.bar(range(len(correlations)), correlations)
        plt.xticks(range(len(correlations)), top_10_features, rotation=45, ha='right')
        plt.title('Corrélations des Top 10 Features avec l\'Occurrence', fontweight='bold')
        plt.ylabel('Corrélation')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='y', alpha=0.3)
        
        # Colorer selon le signe
        for i, (bar, corr) in enumerate(zip(bars2, correlations)):
            bar.set_color('#2ca02c' if corr > 0 else '#d62728')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "feature_importance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Analyse terminée")
        print(f"   Top 3 features: {', '.join(top_features.head(3)['feature'].tolist())}")
        print(f"   Feature la plus importante: {top_features.iloc[0]['feature']} ({top_features.iloc[0]['importance']:.3f})")
        
        return feature_importance
    
    def clustering_analysis(self):
        """Analyse de clustering avec K-Means et DBSCAN."""
        print(f"\n🎯 ANALYSE DE CLUSTERING")
        print("=" * 50)
        
        # Réduction de dimensionnalité avec PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(self.features_scaled)
        
        print(f"✅ PCA appliqué: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%} = {pca.explained_variance_ratio_.sum():.1%} variance expliquée")
        
        # K-Means clustering
        print(f"\n🔄 K-MEANS CLUSTERING")
        print("-" * 30)
        
        # Déterminer le nombre optimal de clusters (méthode du coude)
        inertias = []
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.features_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.features_scaled, labels))
        
        # Choisir le nombre optimal (silhouette score max)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"✅ Nombre optimal de clusters: {optimal_k} (silhouette score: {max(silhouette_scores):.3f})")
        
        # Appliquer K-Means optimal
        kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters_kmeans = kmeans_optimal.fit_predict(self.features_scaled)
        
        # Calculer les taux d'occurrence par cluster
        cluster_occurrence_rates = []
        for cluster in range(optimal_k):
            mask = clusters_kmeans == cluster
            occurrence_rate = self.targets.loc[mask, 'occurrence'].mean()
            cluster_occurrence_rates.append(occurrence_rate)
        
        # Sauvegarder les résultats de clustering
        self.clustering_results = {
            'kmeans_labels': clusters_kmeans,
            'optimal_k': optimal_k,
            'silhouette_score': max(silhouette_scores),
            'cluster_occurrence_rates': cluster_occurrence_rates
        }
        
        print(f"✅ Clustering terminé - {optimal_k} clusters identifiés")
        
        return self.clustering_results
    
    def prepare_train_test_split(self):
        """Prépare les données d'entraînement et de test avec validation temporelle."""
        print(f"\n📊 PRÉPARATION TRAIN/TEST")
        print("=" * 50)
        
        # Split temporel : 80% train, 20% test
        split_date = self.data.index[int(0.8 * len(self.data))]
        
        self.X_train = self.features_scaled[self.features_scaled.index <= split_date]
        self.X_test = self.features_scaled[self.features_scaled.index > split_date]
        self.y_train = self.targets[self.targets.index <= split_date]
        self.y_test = self.targets[self.targets.index > split_date]
        
        print(f"✅ Split temporel réalisé:")
        print(f"   Date de coupure: {split_date}")
        print(f"   Train: {len(self.X_train)} observations")
        print(f"   Test: {len(self.X_test)} observations")
        
        # Statistiques par période
        print(f"\n📈 Statistiques par période:")
        print(f"   Train - Occurrence: {self.y_train['occurrence'].mean():.1%}")
        print(f"   Test  - Occurrence: {self.y_test['occurrence'].mean():.1%}")
        print(f"   Train - Intensité moyenne: {self.y_train['intensity'].mean():.1f} mm")
        print(f"   Test  - Intensité moyenne: {self.y_test['intensity'].mean():.1f} mm")
        
        return True
    
    def train_classification_models(self):
        """Entraîne les modèles de classification pour l'occurrence."""
        print(f"\n🤖 ENTRAÎNEMENT DES MODÈLES DE CLASSIFICATION")
        print("=" * 50)
        
        # Définir les modèles
        models_config = {
            'RandomForest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Neural_Network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        self.classification_results = {}
        
        # Entraîner chaque modèle
        for name, config in models_config.items():
            print(f"\n🔄 Entraînement {name}...")
            
            try:
                # Optimisation des hyperparamètres avec TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=5)
                
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'],
                    cv=tscv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(self.X_train, self.y_train['occurrence'])
                
                # Meilleur modèle
                best_model = grid_search.best_estimator_
                
                # Prédictions
                y_pred_train = best_model.predict(self.X_train)
                y_pred_test = best_model.predict(self.X_test)
                y_proba_test = best_model.predict_proba(self.X_test)[:, 1]
                
                # Métriques
                train_accuracy = accuracy_score(self.y_train['occurrence'], y_pred_train)
                test_accuracy = accuracy_score(self.y_test['occurrence'], y_pred_test)
                test_precision = precision_score(self.y_test['occurrence'], y_pred_test, zero_division=0)
                test_recall = recall_score(self.y_test['occurrence'], y_pred_test, zero_division=0)
                test_f1 = f1_score(self.y_test['occurrence'], y_pred_test, zero_division=0)
                
                # Sauvegarder les résultats
                self.classification_results[name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1,
                    'predictions': y_pred_test,
                    'probabilities': y_proba_test
                }
                
                print(f"   ✅ {name} terminé:")
                print(f"      Train Accuracy: {train_accuracy:.3f}")
                print(f"      Test Accuracy: {test_accuracy:.3f}")
                print(f"      Test F1-Score: {test_f1:.3f}")
                
                # Sauvegarder le modèle
                joblib.dump(best_model, self.model_dir / f"{name.lower()}_classifier.pkl")
                
            except Exception as e:
                print(f"   ❌ Erreur avec {name}: {e}")
                continue
        
        return self.classification_results
    
    def train_regression_models(self):
        """Entraîne les modèles de régression pour l'intensité."""
        print(f"\n🎯 ENTRAÎNEMENT DES MODÈLES DE RÉGRESSION")
        print("=" * 50)
        
        # Filtrer seulement les mois avec événements pour la régression
        mask_train = self.y_train['occurrence'] == 1
        mask_test = self.y_test['occurrence'] == 1
        
        X_train_reg = self.X_train[mask_train]
        y_train_reg = self.y_train.loc[mask_train, 'intensity']
        X_test_reg = self.X_test[mask_test]
        y_test_reg = self.y_test.loc[mask_test, 'intensity']
        
        print(f"   Données de régression:")
        print(f"   Train: {len(X_train_reg)} observations avec événements")
        print(f"   Test: {len(X_test_reg)} observations avec événements")
        
        if len(X_train_reg) == 0 or len(X_test_reg) == 0:
            print("   ❌ Pas assez de données pour la régression")
            return {}
        
        # Définir les modèles de régression
        regression_models = {
            'RandomForest_Reg': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'XGBoost_Reg': {
                'model': xgb.XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Neural_Network_Reg': {
                'model': MLPRegressor(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        self.regression_results = {}
        
        # Entraîner chaque modèle
        for name, config in regression_models.items():
            print(f"\n🔄 Entraînement {name}...")
            
            try:
                # Optimisation avec validation croisée
                tscv = TimeSeriesSplit(n_splits=3)  # Moins de splits pour les petits datasets
                
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_reg, y_train_reg)
                
                # Meilleur modèle
                best_model = grid_search.best_estimator_
                
                # Prédictions
                y_pred_train_reg = best_model.predict(X_train_reg)
                y_pred_test_reg = best_model.predict(X_test_reg)
                
                # Métriques
                train_mse = mean_squared_error(y_train_reg, y_pred_train_reg)
                test_mse = mean_squared_error(y_test_reg, y_pred_test_reg)
                test_mae = mean_absolute_error(y_test_reg, y_pred_test_reg)
                test_r2 = r2_score(y_test_reg, y_pred_test_reg)
                
                # Sauvegarder les résultats
                self.regression_results[name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'test_r2': test_r2,
                    'predictions': y_pred_test_reg,
                    'actual': y_test_reg
                }
                
                print(f"   ✅ {name} terminé:")
                print(f"      Train MSE: {train_mse:.2f}")
                print(f"      Test MSE: {test_mse:.2f}")
                print(f"      Test R²: {test_r2:.3f}")
                
                # Sauvegarder le modèle
                joblib.dump(best_model, self.model_dir / f"{name.lower()}_regressor.pkl")
                
            except Exception as e:
                print(f"   ❌ Erreur avec {name}: {e}")
                continue
        
        return self.regression_results
    
    def create_model_comparison_visualization(self):
        """Crée des visualisations comparatives des modèles."""
        print(f"\n📊 CRÉATION DES VISUALISATIONS COMPARATIVES")
        print("=" * 50)
        
        # Graphique de comparaison des modèles de classification
        if self.classification_results:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Comparaison des Modèles d\'Apprentissage Automatique', fontsize=16, fontweight='bold')
            
            # 1. Métriques de classification
            models = list(self.classification_results.keys())
            metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                if i < 3:  # Utiliser seulement les 3 premiers sous-graphiques de la première ligne
                    values = [self.classification_results[model][metric] for model in models]
                    bars = axes[0, i].bar(models, values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
                    axes[0, i].set_title(f'{name} (Classification)', fontweight='bold')
                    axes[0, i].set_ylabel(name)
                    axes[0, i].set_ylim(0, 1)
                    axes[0, i].tick_params(axis='x', rotation=45)
                    
                    # Ajouter les valeurs sur les barres
                    for bar, value in zip(bars, values):
                        axes[0, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Cacher le dernier subplot de la première ligne s'il n'est pas utilisé
            if len(metric_names) < 3:
                axes[0, 2].axis('off')
            
            # 2. Matrice de confusion du meilleur modèle de classification
            if self.classification_results:
                best_model_name = max(self.classification_results.keys(), 
                                    key=lambda x: self.classification_results[x]['test_f1'])
                
                y_pred_best = self.classification_results[best_model_name]['predictions']
                cm = confusion_matrix(self.y_test['occurrence'], y_pred_best)
                
                im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                axes[1, 0].set_title(f'Matrice de Confusion\n({best_model_name})', fontweight='bold')
                
                # Ajouter les labels
                classes = ['Pas d\'événement', 'Événement']
                tick_marks = np.arange(len(classes))
                axes[1, 0].set_xticks(tick_marks)
                axes[1, 0].set_yticks(tick_marks)
                axes[1, 0].set_xticklabels(classes)
                axes[1, 0].set_yticklabels(classes)
                
                # Ajouter les valeurs dans les cellules
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                                       ha="center", va="center",
                                       color="white" if cm[i, j] > thresh else "black",
                                       fontweight='bold')
                
                axes[1, 0].set_ylabel('Vrai Label')
                axes[1, 0].set_xlabel('Label Prédit')
            
            # 3. ROC Curves
            axes[1, 1].set_title('Courbes ROC', fontweight='bold')
            colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
            
            for model_name, color in zip(models, colors):
                if 'probabilities' in self.classification_results[model_name]:
                    y_prob = self.classification_results[model_name]['probabilities']
                    fpr, tpr, _ = roc_curve(self.y_test['occurrence'], y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    axes[1, 1].plot(fpr, tpr, color=color, lw=2,
                                   label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            axes[1, 1].plot([0, 1], [0, 1], 'k--', lw=1)
            axes[1, 1].set_xlim([0.0, 1.0])
            axes[1, 1].set_ylim([0.0, 1.05])
            axes[1, 1].set_xlabel('Taux de Faux Positifs')
            axes[1, 1].set_ylabel('Taux de Vrais Positifs')
            axes[1, 1].legend(loc="lower right", fontsize=8)
            axes[1, 1].grid(True, alpha=0.3)
            
            # 4. Importance des features (meilleur modèle)
            if hasattr(self.classification_results[best_model_name]['model'], 'feature_importances_'):
                importances = self.classification_results[best_model_name]['model'].feature_importances_
                indices = np.argsort(importances)[::-1][:10]  # Top 10
                
                axes[1, 2].bar(range(10), importances[indices])
                axes[1, 2].set_title(f'Top 10 Features\n({best_model_name})', fontweight='bold')
                axes[1, 2].set_xlabel('Features')
                axes[1, 2].set_ylabel('Importance')
                
                # Labels des features (raccourcis)
                feature_names = [self.features_scaled.columns[i][:8] + '...' if len(self.features_scaled.columns[i]) > 8 
                               else self.features_scaled.columns[i] for i in indices]
                axes[1, 2].set_xticks(range(10))
                axes[1, 2].set_xticklabels(feature_names, rotation=45, ha='right')
            else:
                axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / "model_comparison_complete.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Visualisation comparative sauvegardée")
    
    def create_time_series_predictions(self):
        """Crée des prédictions et visualisations temporelles."""
        print(f"\n📈 PRÉDICTIONS TEMPORELLES")
        print("=" * 50)
        
        if not self.classification_results:
            print("❌ Pas de modèles de classification disponibles")
            return None
        
        # Meilleur modèle de classification
        best_model_name = max(self.classification_results.keys(), 
                            key=lambda x: self.classification_results[x]['test_f1'])
        
        best_model = self.classification_results[best_model_name]['model']
        
        # Prédictions sur toute la période de test
        test_predictions = best_model.predict_proba(self.X_test)[:, 1]
        test_binary = best_model.predict(self.X_test)
        
        # Créer la visualisation temporelle
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Prédictions Temporelles - {best_model_name}', fontsize=16, fontweight='bold')
        
        # 1. Probabilités vs observations
        axes[0].plot(self.X_test.index, test_predictions, 'b-', alpha=0.7, label='Probabilité prédite')
        axes[0].scatter(self.X_test.index, self.y_test['occurrence'], c='red', s=20, alpha=0.8, label='Événements observés')
        axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Seuil de décision')
        axes[0].set_ylabel('Probabilité/Occurrence')
        axes[0].set_title('Probabilités de Prédiction vs Événements Observés')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Prédictions binaires vs observations
        axes[1].plot(self.X_test.index, test_binary, 'g-', linewidth=2, label='Prédictions binaires')
        axes[1].scatter(self.X_test.index, self.y_test['occurrence'], c='red', s=20, alpha=0.8, label='Événements observés')
        axes[1].set_ylabel('Occurrence (0/1)')
        axes[1].set_title('Prédictions Binaires vs Événements Observés')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Erreurs de prédiction dans le temps
        errors = np.abs(test_binary - self.y_test['occurrence'])
        axes[2].bar(self.X_test.index, errors, color='orange', alpha=0.7, width=20)
        axes[2].set_ylabel('Erreur de Prédiction')
        axes[2].set_xlabel('Date')
        axes[2].set_title('Erreurs de Prédiction dans le Temps')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "time_series_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyse des erreurs par saison
        test_data_with_errors = pd.DataFrame({
            'date': self.X_test.index,
            'predicted': test_binary,
            'observed': self.y_test['occurrence'].values,
            'error': errors,
            'month': self.X_test.index.month
        })
        
        # Regrouper par saison
        test_data_with_errors['season'] = test_data_with_errors['month'].apply(
            lambda x: 'Dry Season' if x in [11, 12, 1, 2, 3, 4] else 'Rainy Season'
        )
        
        seasonal_errors = test_data_with_errors.groupby('season')['error'].agg(['mean', 'sum', 'count'])
        
        print(f"✅ Analyse temporelle terminée")
        print(f"   Meilleur modèle: {best_model_name}")
        print(f"   F1-Score: {self.classification_results[best_model_name]['test_f1']:.3f}")
        print(f"   Erreurs par saison:")
        for season in seasonal_errors.index:
            error_rate = seasonal_errors.loc[season, 'mean']
            print(f"     {season}: {error_rate:.1%} d'erreur")
        
        return test_data_with_errors
    
    def generate_comprehensive_report(self):
        """Génère un rapport complet des résultats ML."""
        print(f"\n📄 GÉNÉRATION DU RAPPORT COMPLET")
        print("=" * 50)
        
        report_path = self.report_dir / "rapport_machine_learning.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RAPPORT COMPLET - APPRENTISSAGE AUTOMATIQUE\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Période d'analyse: {self.data.index.min()} à {self.data.index.max()}\n")
            f.write(f"Données d'entraînement: {len(self.X_train)} observations\n")
            f.write(f"Données de test: {len(self.X_test)} observations\n\n")
            
            # Section clustering
            if hasattr(self, 'clustering_results'):
                f.write("ANALYSE DE CLUSTERING\n")
                f.write("-" * 30 + "\n")
                f.write(f"Nombre optimal de clusters (K-Means): {self.clustering_results['optimal_k']}\n")
                f.write(f"Score de silhouette: {self.clustering_results['silhouette_score']:.3f}\n")
                f.write(f"Taux d'occurrence par cluster:\n")
                for i, rate in enumerate(self.clustering_results['cluster_occurrence_rates']):
                    f.write(f"  Cluster {i}: {rate:.1%}\n")
                f.write("\n")
            
            # Section classification
            if self.classification_results:
                f.write("MODÈLES DE CLASSIFICATION (OCCURRENCE)\n")
                f.write("-" * 40 + "\n")
                
                # Tableau comparatif
                f.write(f"{'Modèle':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
                f.write("-" * 70 + "\n")
                
                for model_name, results in self.classification_results.items():
                    f.write(f"{model_name:<20} {results['test_accuracy']:<10.3f} "
                           f"{results['test_precision']:<10.3f} {results['test_recall']:<10.3f} "
                           f"{results['test_f1']:<10.3f}\n")
                
                # Meilleur modèle
                best_model = max(self.classification_results.keys(), 
                               key=lambda x: self.classification_results[x]['test_f1'])
                f.write(f"\nMeilleur modèle: {best_model}\n")
                f.write(f"F1-Score: {self.classification_results[best_model]['test_f1']:.3f}\n")
                f.write(f"Paramètres optimaux: {self.classification_results[best_model]['best_params']}\n\n")
            
            # Section régression
            if self.regression_results:
                f.write("MODÈLES DE RÉGRESSION (INTENSITÉ)\n")
                f.write("-" * 35 + "\n")
                
                f.write(f"{'Modèle':<20} {'MSE':<10} {'MAE':<10} {'R²':<10}\n")
                f.write("-" * 50 + "\n")
                
                for model_name, results in self.regression_results.items():
                    f.write(f"{model_name:<20} {results['test_mse']:<10.2f} "
                           f"{results['test_mae']:<10.2f} {results['test_r2']:<10.3f}\n")
                
                # Meilleur modèle de régression
                best_reg_model = max(self.regression_results.keys(), 
                                   key=lambda x: self.regression_results[x]['test_r2'])
                f.write(f"\nMeilleur modèle: {best_reg_model}\n")
                f.write(f"R²: {self.regression_results[best_reg_model]['test_r2']:.3f}\n")
                f.write(f"Paramètres optimaux: {self.regression_results[best_reg_model]['best_params']}\n\n")
            
            # Recommandations
            f.write("RECOMMANDATIONS ET CONCLUSIONS\n")
            f.write("-" * 35 + "\n")
            
            if self.classification_results:
                best_f1 = max(results['test_f1'] for results in self.classification_results.values())
                if best_f1 > 0.7:
                    f.write("• Excellente performance de classification (F1 > 0.7)\n")
                elif best_f1 > 0.5:
                    f.write("• Performance de classification satisfaisante (F1 > 0.5)\n")
                else:
                    f.write("• Performance de classification limitée (F1 < 0.5)\n")
                    f.write("  → Recommandation: enrichir les features ou ajuster les seuils\n")
            
            if self.regression_results:
                best_r2 = max(results['test_r2'] for results in self.regression_results.values())
                if best_r2 > 0.6:
                    f.write("• Bonne capacité de prédiction de l'intensité (R² > 0.6)\n")
                elif best_r2 > 0.3:
                    f.write("• Capacité modérée de prédiction de l'intensité (R² > 0.3)\n")
                else:
                    f.write("• Difficulté à prédire l'intensité (R² < 0.3)\n")
                    f.write("  → Recommandation: considérer des features supplémentaires\n")
            
            f.write("\n")
            f.write("APPLICATIONS OPÉRATIONNELLES SUGGÉRÉES:\n")
            f.write("• Système d'alerte précoce basé sur les téléconnexions\n")
            f.write("• Classification automatique des conditions à risque\n")
            f.write("• Intégration dans les modèles de prévision météorologique\n")
            f.write("• Aide à la décision pour la gestion des ressources en eau\n")
        
        print(f"✅ Rapport complet généré: {report_path}")
        return report_path
    
    def save_models_and_results(self):
        """Sauvegarde tous les modèles et résultats."""
        print(f"\n💾 SAUVEGARDE DES MODÈLES ET RÉSULTATS")
        print("=" * 50)
        
        # Sauvegarder le scaler
        joblib.dump(self.scaler, self.model_dir / "feature_scaler.pkl")
        print(f"✅ Scaler sauvegardé")
        
        # Sauvegarder les résultats en JSON
        import json
        
        # Préparer les résultats pour JSON (sans les objets modèles)
        json_results = {
            'classification_results': {},
            'regression_results': {},
            'data_info': {
                'n_features': self.features_scaled.shape[1],
                'n_observations': len(self.data),
                'train_size': len(self.X_train),
                'test_size': len(self.X_test),
                'train_period': f"{self.X_train.index.min()} - {self.X_train.index.max()}",
                'test_period': f"{self.X_test.index.min()} - {self.X_test.index.max()}"
            }
        }
        
        # Classification results (sans les modèles)
        for name, results in self.classification_results.items():
            json_results['classification_results'][name] = {
                'test_accuracy': float(results['test_accuracy']),
                'test_precision': float(results['test_precision']),
                'test_recall': float(results['test_recall']),
                'test_f1': float(results['test_f1']),
                'best_params': results['best_params']
            }
        
        # Regression results (sans les modèles)
        for name, results in self.regression_results.items():
            json_results['regression_results'][name] = {
                'test_mse': float(results['test_mse']),
                'test_mae': float(results['test_mae']),
                'test_r2': float(results['test_r2']),
                'best_params': results['best_params']
            }
        
        # Clustering results
        if hasattr(self, 'clustering_results'):
            json_results['clustering_results'] = {
                'optimal_k': int(self.clustering_results['optimal_k']),
                'silhouette_score': float(self.clustering_results['silhouette_score']),
                'cluster_occurrence_rates': [float(x) for x in self.clustering_results['cluster_occurrence_rates']]
            }
        
        # Sauvegarder en JSON
        json_path = self.output_dir / "data" / "ml_results_summary.json"
        json_path.parent.mkdir(exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Résultats JSON sauvegardés: {json_path}")
        
        # Sauvegarder les prédictions
        if self.classification_results:
            best_model_name = max(self.classification_results.keys(), 
                                key=lambda x: self.classification_results[x]['test_f1'])
            
            predictions_df = pd.DataFrame({
                'date': self.X_test.index,
                'observed_occurrence': self.y_test['occurrence'].values,
                'predicted_occurrence': self.classification_results[best_model_name]['predictions'],
                'predicted_probability': self.classification_results[best_model_name]['probabilities']
            })
            
            # Ajouter les prédictions de régression si disponibles
            if self.regression_results:
                best_reg_name = max(self.regression_results.keys(), 
                                  key=lambda x: self.regression_results[x]['test_r2'])
                
                # Créer un dataframe pour les prédictions d'intensité
                reg_data = self.regression_results[best_reg_name]
                intensity_df = pd.DataFrame({
                    'date': reg_data['actual'].index,
                    'observed_intensity': reg_data['actual'].values,
                    'predicted_intensity': reg_data['predictions']
                })
                
                # Merger avec les prédictions de classification
                predictions_df = predictions_df.merge(intensity_df, on='date', how='left')
            
            predictions_path = self.output_dir / "data" / "ml_predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            print(f"✅ Prédictions sauvegardées: {predictions_path}")
        
        print(f"✅ Sauvegarde terminée")
        print(f"   Modèles: {len(list(self.model_dir.glob('*.pkl')))} fichiers")
        print(f"   Visualisations: {len(list(self.viz_dir.glob('*.png')))} fichiers")
        
        return True
    
    def run_complete_pipeline(self):
        """Exécute le pipeline complet d'apprentissage automatique."""
        print("🤖 PIPELINE COMPLET D'APPRENTISSAGE AUTOMATIQUE")
        print("=" * 70)
        print("Classification et prédiction des événements de précipitations extrêmes")
        print("Utilisation de K-Means, Random Forest, XGBoost, SVM et réseaux de neurones")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Étape 1: Chargement des données
        if not self.load_data():
            return False
        
        # Étape 2: Prétraitement
        if not self.preprocess_data():
            return False
        
        # Étape 3: Analyse des features
        feature_importance = self.analyze_feature_importance()
        
        # Étape 4: Clustering
        clustering_results = self.clustering_analysis()
        
        # Étape 5: Préparation train/test
        if not self.prepare_train_test_split():
            return False
        
        # Étape 6: Entraînement des modèles de classification
        classification_results = self.train_classification_models()
        
        # Étape 7: Entraînement des modèles de régression
        regression_results = self.train_regression_models()
        
        # Étape 8: Visualisations comparatives
        self.create_model_comparison_visualization()
        
        # Étape 9: Prédictions temporelles
        time_series_results = self.create_time_series_predictions()
        
        # Étape 10: Rapport complet
        report_path = self.generate_comprehensive_report()
        
        # Étape 11: Sauvegarde
        self.save_models_and_results()
        
        # Résumé final
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\n" + "=" * 70)
        print("✅ PIPELINE D'APPRENTISSAGE AUTOMATIQUE TERMINÉ AVEC SUCCÈS!")
        print("=" * 70)
        
        print(f"⏱️  Durée totale: {duration:.1f} secondes")
        print(f"📊 Dataset: {self.data.shape[0]} observations, {self.features_scaled.shape[1]} features")
        
        if self.classification_results:
            best_classifier = max(self.classification_results.keys(), 
                                key=lambda x: self.classification_results[x]['test_f1'])
            best_f1 = self.classification_results[best_classifier]['test_f1']
            print(f"🏆 Meilleur classificateur: {best_classifier} (F1: {best_f1:.3f})")
        
        if self.regression_results:
            best_regressor = max(self.regression_results.keys(), 
                               key=lambda x: self.regression_results[x]['test_r2'])
            best_r2 = self.regression_results[best_regressor]['test_r2']
            print(f"🎯 Meilleur régresseur: {best_regressor} (R²: {best_r2:.3f})")
        
        if hasattr(self, 'clustering_results'):
            print(f"🎯 Clustering: {self.clustering_results['optimal_k']} clusters optimaux")
        
        print(f"\n📁 FICHIERS GÉNÉRÉS:")
        print(f"   🤖 Modèles: {len(list(self.model_dir.glob('*.pkl')))} modèles entraînés")
        print(f"   📊 Visualisations: {len(list(self.viz_dir.glob('*.png')))} graphiques")
        print(f"   📄 Rapport: {report_path}")
        print(f"   💾 Données: ml_results_summary.json, ml_predictions.csv")
        
        print(f"\n🚀 APPLICATIONS POSSIBLES:")
        print(f"   • Classification automatique des conditions climatiques à risque")
        print(f"   • Prédiction saisonnière pour la planification agricole")
        print(f"   • Aide à la décision pour la gestion des ressources hydriques")
        
        return True

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale du pipeline ML."""
    print("🤖 APPRENTISSAGE AUTOMATIQUE - PRÉCIPITATIONS EXTRÊMES")
    print("=" * 70)
    
    # Initialiser et lancer le pipeline
    ml_pipeline = MachineLearningPipeline()
    success = ml_pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 PIPELINE ML RÉUSSI!")
        print("\n💡 PROCHAINES ÉTAPES SUGGÉRÉES:")
        print("• Validation opérationnelle des modèles")
        print("• Intégration dans un système de monitoring")
        print("• Tests sur d'autres régions d'Afrique de l'Ouest")
        print("• Développement d'une interface utilisateur")
        print("\nConsultez les dossiers outputs/models/ et outputs/visualizations/machine_learning/")
        return 0
    else:
        print("\n❌ ÉCHEC DU PIPELINE ML")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)