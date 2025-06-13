#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/07_advanced_clustering_analysis.py
"""
Analyse de clustering avancée avec comparaison de multiples algorithmes.
K-Means, DBSCAN, Clustering Hiérarchique, Gaussian Mixture Models, etc.

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

# ML et clustering imports
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# Configuration des chemins
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class AdvancedClusteringAnalysis:
    """Analyse de clustering avancée avec multiples algorithmes."""
    
    def __init__(self):
        """Initialise l'analyseur de clustering."""
        self.data = None
        self.features = None
        self.targets = None
        self.features_scaled = None
        self.scaler = StandardScaler()
        
        # Résultats des différents algorithmes
        self.clustering_results = {}
        self.evaluation_metrics = {}
        
        # Dossiers de sortie
        self.output_dir = project_root / "outputs"
        self.viz_dir = self.output_dir / "visualizations" / "clustering"
        self.report_dir = self.output_dir / "reports"
        
        for directory in [self.viz_dir, self.report_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_and_prepare_data(self):
        """Charge et prépare les données pour le clustering."""
        print("🔄 CHARGEMENT ET PRÉPARATION DES DONNÉES")
        print("=" * 50)
        
        try:
            # Charger le dataset ML
            ml_file = project_root / "data/processed/ml_dataset_teleconnections.csv"
            if not ml_file.exists():
                raise FileNotFoundError(f"Dataset ML non trouvé: {ml_file}")
            
            self.data = pd.read_csv(ml_file, index_col=0, parse_dates=True)
            print(f"✅ Dataset chargé: {self.data.shape}")
            
            # Séparer features et targets
            target_cols = ['occurrence', 'count', 'intensity']
            feature_cols = [col for col in self.data.columns if col not in target_cols]
            
            self.features = self.data[feature_cols]
            self.targets = self.data[target_cols]
            
            # Prétraitement
            self.features = self.features.fillna(self.features.mean())
            
            # Encoder les variables catégorielles si nécessaire
            if 'season' in self.features.columns:
                le = LabelEncoder()
                self.features['season'] = le.fit_transform(self.features['season'].astype(str))
            
            # Normalisation
            self.features_scaled = pd.DataFrame(
                self.scaler.fit_transform(self.features),
                columns=self.features.columns,
                index=self.features.index
            )
            
            print(f"✅ Données préparées:")
            print(f"   Features: {self.features_scaled.shape[1]} variables")
            print(f"   Observations: {len(self.features_scaled)}")
            print(f"   Période: {self.data.index.min()} à {self.data.index.max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def perform_kmeans_analysis(self):
        """Analyse K-Means avec optimisation du nombre de clusters."""
        print(f"\n🎯 ANALYSE K-MEANS APPROFONDIE")
        print("=" * 50)
        
        # Méthode du coude et silhouette
        k_range = range(2, 16)
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        print("🔄 Optimisation du nombre de clusters...")
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.features_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.features_scaled, labels))
            calinski_scores.append(calinski_harabasz_score(self.features_scaled, labels))
        
        # Déterminer le nombre optimal
        # Méthode du coude automatisée
        def find_elbow(inertias):
            """Trouve le coude automatiquement."""
            # Calculer les dérivées secondes
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            # Le coude est là où la dérivée seconde est maximale
            return np.argmax(diffs2) + 2  # +2 car on commence à k=2
        
        elbow_k = find_elbow(inertias)
        silhouette_k = k_range[np.argmax(silhouette_scores)]
        calinski_k = k_range[np.argmax(calinski_scores)]
        
        # Choisir le meilleur k (consensus ou silhouette)
        optimal_k = silhouette_k
        
        print(f"✅ Optimisation terminée:")
        print(f"   Méthode du coude: k={elbow_k}")
        print(f"   Meilleur silhouette: k={silhouette_k} (score: {max(silhouette_scores):.3f})")
        print(f"   Meilleur Calinski-Harabasz: k={calinski_k}")
        print(f"   K optimal choisi: {optimal_k}")
        
        # Entraîner le modèle final
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans_final.fit_predict(self.features_scaled)
        
        self.clustering_results['KMeans'] = {
            'algorithm': kmeans_final,
            'labels': kmeans_labels,
            'n_clusters': optimal_k,
            'optimization_scores': {
                'k_range': list(k_range),
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores
            }
        }
        
        return optimal_k
    
    def perform_dbscan_analysis(self):
        """Analyse DBSCAN avec optimisation des paramètres."""
        print(f"\n🔍 ANALYSE DBSCAN")
        print("=" * 30)
        
        # Optimiser eps avec k-distance
        print("🔄 Optimisation du paramètre eps...")
        
        # Calculer les distances k-NN pour différentes valeurs de k
        k_values = [3, 4, 5, 6]
        eps_candidates = []
        
        for k in k_values:
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.features_scaled)
            distances, indices = nbrs.kneighbors(self.features_scaled)
            # Prendre la k-ième distance (exclure la distance à soi-même)
            k_distances = np.sort(distances[:, k])
            # Utiliser le 95e percentile comme candidat eps
            eps_candidate = np.percentile(k_distances, 95)
            eps_candidates.append(eps_candidate)
        
        # Tester différentes combinaisons eps/min_samples
        eps_range = eps_candidates
        min_samples_range = [3, 4, 5, 6, 8, 10]
        
        best_score = -1
        best_params = None
        dbscan_results = []
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.features_scaled)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters > 1:  # Au moins 2 clusters
                    # Calculer silhouette sans les points de bruit
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 10:  # Assez de points
                        score = silhouette_score(
                            self.features_scaled[non_noise_mask], 
                            labels[non_noise_mask]
                        )
                        
                        dbscan_results.append({
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'silhouette_score': score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = (eps, min_samples)
        
        if best_params:
            eps_opt, min_samples_opt = best_params
            print(f"✅ Paramètres optimaux: eps={eps_opt:.3f}, min_samples={min_samples_opt}")
            
            # Entraîner le modèle final
            dbscan_final = DBSCAN(eps=eps_opt, min_samples=min_samples_opt)
            dbscan_labels = dbscan_final.fit_predict(self.features_scaled)
            
            n_clusters_final = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise_final = list(dbscan_labels).count(-1)
            
            print(f"   Clusters trouvés: {n_clusters_final}")
            print(f"   Points aberrants: {n_noise_final}")
            print(f"   Silhouette score: {best_score:.3f}")
            
            self.clustering_results['DBSCAN'] = {
                'algorithm': dbscan_final,
                'labels': dbscan_labels,
                'n_clusters': n_clusters_final,
                'n_noise': n_noise_final,
                'best_params': best_params,
                'optimization_results': dbscan_results
            }
        else:
            print("❌ Aucune configuration DBSCAN satisfaisante trouvée")
    
    def perform_hierarchical_clustering(self):
        """Analyse de clustering hiérarchique."""
        print(f"\n🌳 CLUSTERING HIÉRARCHIQUE")
        print("=" * 35)
        
        # Tester différentes méthodes de linkage
        linkage_methods = ['ward', 'complete', 'average', 'single']
        best_score = -1
        best_method = None
        best_n_clusters = None
        
        hierarchical_results = {}
        
        for method in linkage_methods:
            print(f"🔄 Test méthode: {method}")
            
            # Tester différents nombres de clusters
            scores = []
            n_clusters_range = range(2, 11)
            
            for n_clusters in n_clusters_range:
                try:
                    agg = AgglomerativeClustering(
                        n_clusters=n_clusters, 
                        linkage=method
                    )
                    
                    labels = agg.fit_predict(self.features_scaled)
                    score = silhouette_score(self.features_scaled, labels)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_method = method
                        best_n_clusters = n_clusters
                        
                except Exception as e:
                    scores.append(-1)  # Score invalide
                    continue
            
            hierarchical_results[method] = {
                'n_clusters_range': list(n_clusters_range),
                'silhouette_scores': scores,
                'best_n_clusters': n_clusters_range[np.argmax(scores)] if max(scores) > -1 else None,
                'best_score': max(scores) if max(scores) > -1 else -1
            }
        
        if best_method is not None:
            print(f"✅ Meilleure configuration: {best_method} avec {best_n_clusters} clusters")
            print(f"   Silhouette score: {best_score:.3f}")
            
            # Entraîner le modèle final
            agg_final = AgglomerativeClustering(
                n_clusters=best_n_clusters, 
                linkage=best_method
            )
            agg_labels = agg_final.fit_predict(self.features_scaled)
            
            # Calculer le dendrogramme pour visualisation
            if best_method == 'ward':
                linkage_matrix = linkage(self.features_scaled, method=best_method)
            else:
                # Pour les autres méthodes, calculer la matrice de distance
                distance_matrix = pdist(self.features_scaled)
                linkage_matrix = linkage(distance_matrix, method=best_method)
            
            self.clustering_results['Hierarchical'] = {
                'algorithm': agg_final,
                'labels': agg_labels,
                'n_clusters': best_n_clusters,
                'linkage_method': best_method,
                'linkage_matrix': linkage_matrix,
                'optimization_results': hierarchical_results
            }
        else:
            print("❌ Aucune configuration hiérarchique satisfaisante trouvée")
    
    def perform_gaussian_mixture_analysis(self):
        """Analyse avec Gaussian Mixture Models."""
        print(f"\n🎭 GAUSSIAN MIXTURE MODELS")
        print("=" * 35)
        
        # Tester différents nombres de composantes
        n_components_range = range(2, 11)
        covariance_types = ['full', 'tied', 'diag', 'spherical']
        
        best_bic = float('inf')
        best_aic = float('inf')
        best_params = None
        gmm_results = []
        
        print("🔄 Optimisation des paramètres...")
        
        for n_components in n_components_range:
            for cov_type in covariance_types:
                try:
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=cov_type,
                        random_state=42
                    )
                    gmm.fit(self.features_scaled)
                    
                    bic = gmm.bic(self.features_scaled)
                    aic = gmm.aic(self.features_scaled)
                    
                    labels = gmm.predict(self.features_scaled)
                    silhouette = silhouette_score(self.features_scaled, labels)
                    
                    gmm_results.append({
                        'n_components': n_components,
                        'covariance_type': cov_type,
                        'bic': bic,
                        'aic': aic,
                        'silhouette_score': silhouette
                    })
                    
                    # Utiliser BIC pour la sélection (plus bas = mieux)
                    if bic < best_bic:
                        best_bic = bic
                        best_params = (n_components, cov_type)
                        
                except Exception as e:
                    # Certaines configurations peuvent échouer
                    continue
        
        if best_params:
            n_comp_opt, cov_type_opt = best_params
            print(f"✅ Paramètres optimaux: {n_comp_opt} composantes, covariance {cov_type_opt}")
            print(f"   BIC: {best_bic:.2f}")
            
            # Entraîner le modèle final
            gmm_final = GaussianMixture(
                n_components=n_comp_opt,
                covariance_type=cov_type_opt,
                random_state=42
            )
            gmm_final.fit(self.features_scaled)
            gmm_labels = gmm_final.predict(self.features_scaled)
            
            self.clustering_results['GaussianMixture'] = {
                'algorithm': gmm_final,
                'labels': gmm_labels,
                'n_clusters': n_comp_opt,
                'covariance_type': cov_type_opt,
                'bic': best_bic,
                'optimization_results': gmm_results
            }
        else:
            print("❌ Aucune configuration GMM satisfaisante trouvée")
    
    def perform_spectral_clustering(self):
        """Analyse avec Spectral Clustering."""
        print(f"\n🌈 SPECTRAL CLUSTERING")
        print("=" * 30)
        
        # Tester différents nombres de clusters et paramètres
        n_clusters_range = range(2, 11)
        affinity_types = ['rbf', 'nearest_neighbors']
        
        best_score = -1
        best_params = None
        spectral_results = []
        
        print("🔄 Optimisation des paramètres...")
        
        for n_clusters in n_clusters_range:
            for affinity in affinity_types:
                try:
                    spectral = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity=affinity,
                        random_state=42
                    )
                    labels = spectral.fit_predict(self.features_scaled)
                    
                    score = silhouette_score(self.features_scaled, labels)
                    
                    spectral_results.append({
                        'n_clusters': n_clusters,
                        'affinity': affinity,
                        'silhouette_score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = (n_clusters, affinity)
                        
                except Exception as e:
                    continue
        
        if best_params:
            n_clusters_opt, affinity_opt = best_params
            print(f"✅ Paramètres optimaux: {n_clusters_opt} clusters, affinité {affinity_opt}")
            print(f"   Silhouette score: {best_score:.3f}")
            
            # Entraîner le modèle final
            spectral_final = SpectralClustering(
                n_clusters=n_clusters_opt,
                affinity=affinity_opt,
                random_state=42
            )
            spectral_labels = spectral_final.fit_predict(self.features_scaled)
            
            self.clustering_results['Spectral'] = {
                'algorithm': spectral_final,
                'labels': spectral_labels,
                'n_clusters': n_clusters_opt,
                'affinity': affinity_opt,
                'optimization_results': spectral_results
            }
        else:
            print("❌ Aucune configuration Spectral satisfaisante trouvée")
    
    def evaluate_clustering_methods(self):
        """Évalue et compare tous les méthodes de clustering."""
        print(f"\n📊 ÉVALUATION COMPARATIVE DES MÉTHODES")
        print("=" * 50)
        
        evaluation_metrics = {}
        
        for method_name, result in self.clustering_results.items():
            labels = result['labels']
            
            # Exclure les points de bruit pour DBSCAN
            if method_name == 'DBSCAN':
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) < 10:
                    continue
                features_eval = self.features_scaled[non_noise_mask]
                labels_eval = labels[non_noise_mask]
            else:
                features_eval = self.features_scaled
                labels_eval = labels
            
            # Calculer les métriques
            if len(set(labels_eval)) > 1:
                silhouette = silhouette_score(features_eval, labels_eval)
                calinski = calinski_harabasz_score(features_eval, labels_eval)
                davies_bouldin = davies_bouldin_score(features_eval, labels_eval)
                
                # Relation avec les targets
                occurrence_by_cluster = {}
                intensity_by_cluster = {}
                
                # Créer un masque pour les données évaluées
                if method_name == 'DBSCAN':
                    eval_targets = self.targets[non_noise_mask]
                else:
                    eval_targets = self.targets
                
                for cluster_id in set(labels_eval):
                    cluster_mask = labels_eval == cluster_id
                    
                    # Taux d'occurrence par cluster
                    occurrence_rate = eval_targets.loc[cluster_mask, 'occurrence'].mean()
                    occurrence_by_cluster[cluster_id] = occurrence_rate
                    
                    # Intensité moyenne par cluster (pour les événements)
                    event_mask = cluster_mask & (eval_targets['occurrence'] == 1)
                    if event_mask.sum() > 0:
                        intensity_mean = eval_targets.loc[event_mask, 'intensity'].mean()
                        intensity_by_cluster[cluster_id] = intensity_mean
                    else:
                        intensity_by_cluster[cluster_id] = 0
                
                # Variance des taux d'occurrence (mesure de séparation)
                occurrence_variance = np.var(list(occurrence_by_cluster.values()))
                
                evaluation_metrics[method_name] = {
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski,
                    'davies_bouldin_score': davies_bouldin,
                    'n_clusters': len(set(labels_eval)),
                    'occurrence_variance': occurrence_variance,
                    'occurrence_by_cluster': occurrence_by_cluster,
                    'intensity_by_cluster': intensity_by_cluster
                }
                
                print(f"✅ {method_name}:")
                print(f"   Clusters: {len(set(labels_eval))}")
                print(f"   Silhouette: {silhouette:.3f}")
                print(f"   Calinski-Harabasz: {calinski:.2f}")
                print(f"   Davies-Bouldin: {davies_bouldin:.3f}")
                print(f"   Variance occurrence: {occurrence_variance:.4f}")
                
                if method_name == 'DBSCAN':
                    n_noise = result['n_noise']
                    print(f"   Points aberrants: {n_noise}")
        
        self.evaluation_metrics = evaluation_metrics
        return evaluation_metrics
    
    def create_comprehensive_visualizations(self):
        """Crée toutes les visualisations pour l'analyse de clustering."""
        print(f"\n🎨 CRÉATION DES VISUALISATIONS")
        print("=" * 50)
        
        if not self.clustering_results:
            print("❌ Aucun résultat de clustering disponible")
            return
        
        # 1. Comparaison PCA des différents algorithmes
        if len(self.clustering_results) > 0:
            # Calculer PCA une seule fois
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(self.features_scaled)
            
            n_algorithms = len(self.clustering_results)
            fig, axes = plt.subplots(1, n_algorithms, figsize=(6*n_algorithms, 6))
            if n_algorithms == 1:
                axes = [axes]
            
            fig.suptitle('Comparaison des Algorithmes de Clustering (PCA)', 
                        fontsize=16, fontweight='bold')
            
            for i, (method_name, result) in enumerate(self.clustering_results.items()):
                labels = result['labels']
                
                # Scatter plot avec PCA
                scatter = axes[i].scatter(features_pca[:, 0], features_pca[:, 1], 
                                        c=labels, cmap='viridis', alpha=0.7, s=30)
                
                # Points de bruit en noir pour DBSCAN
                if method_name == 'DBSCAN':
                    noise_mask = labels == -1
                    if noise_mask.sum() > 0:
                        axes[i].scatter(features_pca[noise_mask, 0], features_pca[noise_mask, 1], 
                                      c='black', s=20, alpha=0.5, label='Bruit')
                        axes[i].legend()
                
                title = f"{method_name}\n({result['n_clusters']} clusters"
                if method_name == 'DBSCAN' and 'n_noise' in result:
                    title += f", {result['n_noise']} bruit"
                title += ")"
                
                axes[i].set_title(title, fontweight='bold')
                axes[i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                axes[i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                axes[i].grid(True, alpha=0.3)
                
                # Colorbar pour chaque subplot
                plt.colorbar(scatter, ax=axes[i])
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / "clustering_comparison_pca.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Métriques comparatives
        if self.evaluation_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Métriques Comparatives des Algorithmes de Clustering', 
                        fontsize=16, fontweight='bold')
            
            methods = list(self.evaluation_metrics.keys())
            
            # Silhouette scores
            silhouette_scores = [self.evaluation_metrics[m]['silhouette_score'] for m in methods]
            bars1 = axes[0, 0].bar(methods, silhouette_scores, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
            axes[0, 0].set_title('Silhouette Score (plus haut = mieux)', fontweight='bold')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars1, silhouette_scores):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Calinski-Harabasz scores
            calinski_scores = [self.evaluation_metrics[m]['calinski_harabasz_score'] for m in methods]
            bars2 = axes[0, 1].bar(methods, calinski_scores, color=plt.cm.plasma(np.linspace(0, 1, len(methods))))
            axes[0, 1].set_title('Calinski-Harabasz Score (plus haut = mieux)', fontweight='bold')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars2, calinski_scores):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(calinski_scores)*0.01,
                               f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Davies-Bouldin scores
            davies_scores = [self.evaluation_metrics[m]['davies_bouldin_score'] for m in methods]
            bars3 = axes[1, 0].bar(methods, davies_scores, color=plt.cm.coolwarm(np.linspace(0, 1, len(methods))))
            axes[1, 0].set_title('Davies-Bouldin Score (plus bas = mieux)', fontweight='bold')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars3, davies_scores):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(davies_scores)*0.01,
                               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Variance des taux d'occurrence
            occurrence_variances = [self.evaluation_metrics[m]['occurrence_variance'] for m in methods]
            bars4 = axes[1, 1].bar(methods, occurrence_variances, color=plt.cm.Spectral(np.linspace(0, 1, len(methods))))
            axes[1, 1].set_title('Variance des Taux d\'Occurrence\n(plus haut = meilleure séparation)', fontweight='bold')
            axes[1, 1].set_ylabel('Variance')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            for bar, var in zip(bars4, occurrence_variances):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(occurrence_variances)*0.01,
                               f'{var:.4f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / "clustering_metrics_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Optimisation K-Means détaillée
        if 'KMeans' in self.clustering_results:
            opt_scores = self.clustering_results['KMeans']['optimization_scores']
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Optimisation K-Means - Méthodes de Sélection du Nombre de Clusters', 
                        fontsize=16, fontweight='bold')
            
            k_range = opt_scores['k_range']
            
            # Méthode du coude
            axes[0].plot(k_range, opt_scores['inertias'], 'bo-', linewidth=2, markersize=8)
            axes[0].set_title('Méthode du Coude', fontweight='bold')
            axes[0].set_xlabel('Nombre de clusters (k)')
            axes[0].set_ylabel('Inertie')
            axes[0].grid(True, alpha=0.3)
            
            # Silhouette scores
            axes[1].plot(k_range, opt_scores['silhouette_scores'], 'ro-', linewidth=2, markersize=8)
            optimal_k = self.clustering_results['KMeans']['n_clusters']
            axes[1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                           label=f'Optimal k={optimal_k}')
            axes[1].set_title('Silhouette Score', fontweight='bold')
            axes[1].set_xlabel('Nombre de clusters (k)')
            axes[1].set_ylabel('Silhouette Score')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Calinski-Harabasz scores
            axes[2].plot(k_range, opt_scores['calinski_scores'], 'go-', linewidth=2, markersize=8)
            axes[2].set_title('Calinski-Harabasz Score', fontweight='bold')
            axes[2].set_xlabel('Nombre de clusters (k)')
            axes[2].set_ylabel('Calinski-Harabasz Score')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / "kmeans_optimization.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Dendrogramme pour clustering hiérarchique
        if 'Hierarchical' in self.clustering_results:
            plt.figure(figsize=(15, 8))
            linkage_matrix = self.clustering_results['Hierarchical']['linkage_matrix']
            
            dendrogram(linkage_matrix, 
                      truncate_mode='level', 
                      p=10,  # Limiter le nombre de niveaux affichés
                      show_leaf_counts=True)
            
            plt.title('Dendrogramme - Clustering Hiérarchique', fontsize=16, fontweight='bold')
            plt.xlabel('Nombre d\'échantillons dans le nœud')
            plt.ylabel('Distance')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / "hierarchical_dendrogram.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Toutes les visualisations ont été créées")
    
    def generate_comprehensive_report(self):
        """Génère un rapport complet de l'analyse de clustering."""
        print(f"\n📄 GÉNÉRATION DU RAPPORT COMPLET")
        print("=" * 50)
        
        report_path = self.report_dir / "rapport_clustering_avance.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RAPPORT COMPLET - ANALYSE DE CLUSTERING AVANCÉE\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Période d'analyse: {self.data.index.min()} à {self.data.index.max()}\n")
            f.write(f"Nombre d'observations: {len(self.features_scaled)}\n")
            f.write(f"Nombre de features: {self.features_scaled.shape[1]}\n\n")
            
            # Section algorithmes testés
            f.write("ALGORITHMES TESTÉS\n")
            f.write("-" * 25 + "\n")
            for method_name, result in self.clustering_results.items():
                f.write(f"• {method_name}: {result['n_clusters']} clusters\n")
                if method_name == 'DBSCAN' and 'n_noise' in result:
                    f.write(f"  - Points aberrants détectés: {result['n_noise']}\n")
            f.write("\n")
            
            # Section métriques comparatives
            if self.evaluation_metrics:
                f.write("MÉTRIQUES COMPARATIVES\n")
                f.write("-" * 30 + "\n")
                
                # Tableau comparatif
                f.write(f"{'Algorithme':<15} {'Clusters':<10} {'Silhouette':<12} {'Calinski-H':<12} {'Davies-B':<12} {'Var.Occur.':<12}\n")
                f.write("-" * 85 + "\n")
                
                for method_name, metrics in self.evaluation_metrics.items():
                    f.write(f"{method_name:<15} {metrics['n_clusters']:<10} "
                           f"{metrics['silhouette_score']:<12.3f} "
                           f"{metrics['calinski_harabasz_score']:<12.1f} "
                           f"{metrics['davies_bouldin_score']:<12.3f} "
                           f"{metrics['occurrence_variance']:<12.4f}\n")
                
                f.write("\n")
                
                # Meilleur algorithme selon différents critères
                best_silhouette = max(self.evaluation_metrics.items(), 
                                    key=lambda x: x[1]['silhouette_score'])
                best_calinski = max(self.evaluation_metrics.items(), 
                                  key=lambda x: x[1]['calinski_harabasz_score'])
                best_davies = min(self.evaluation_metrics.items(), 
                                key=lambda x: x[1]['davies_bouldin_score'])
                best_occurrence = max(self.evaluation_metrics.items(), 
                                    key=lambda x: x[1]['occurrence_variance'])
                
                f.write("CLASSEMENT PAR CRITÈRE\n")
                f.write("-" * 25 + "\n")
                f.write(f"Meilleur Silhouette: {best_silhouette[0]} ({best_silhouette[1]['silhouette_score']:.3f})\n")
                f.write(f"Meilleur Calinski-Harabasz: {best_calinski[0]} ({best_calinski[1]['calinski_harabasz_score']:.1f})\n")
                f.write(f"Meilleur Davies-Bouldin: {best_davies[0]} ({best_davies[1]['davies_bouldin_score']:.3f})\n")
                f.write(f"Meilleure séparation événements: {best_occurrence[0]} ({best_occurrence[1]['occurrence_variance']:.4f})\n\n")
            
            # Section analyse par algorithme
            f.write("ANALYSE DÉTAILLÉE PAR ALGORITHME\n")
            f.write("-" * 40 + "\n\n")
            
            for method_name, result in self.clustering_results.items():
                f.write(f"{method_name.upper()}\n")
                f.write("=" * len(method_name) + "\n")
                
                f.write(f"Nombre de clusters: {result['n_clusters']}\n")
                
                if method_name == 'KMeans':
                    f.write(f"Méthode d'optimisation: Silhouette + Coude + Calinski-Harabasz\n")
                elif method_name == 'DBSCAN':
                    if 'best_params' in result:
                        eps, min_samples = result['best_params']
                        f.write(f"Paramètres optimaux: eps={eps:.3f}, min_samples={min_samples}\n")
                    if 'n_noise' in result:
                        f.write(f"Points aberrants: {result['n_noise']}\n")
                elif method_name == 'Hierarchical':
                    f.write(f"Méthode de linkage: {result['linkage_method']}\n")
                elif method_name == 'GaussianMixture':
                    if 'covariance_type' in result:
                        f.write(f"Type de covariance: {result['covariance_type']}\n")
                        f.write(f"BIC: {result['bic']:.2f}\n")
                elif method_name == 'Spectral':
                    if 'affinity' in result:
                        f.write(f"Type d'affinité: {result['affinity']}\n")
                
                # Analyse des clusters par rapport aux événements
                if method_name in self.evaluation_metrics:
                    metrics = self.evaluation_metrics[method_name]
                    f.write(f"\nRelation avec les événements extrêmes:\n")
                    
                    for cluster_id, occurrence_rate in metrics['occurrence_by_cluster'].items():
                        intensity = metrics['intensity_by_cluster'].get(cluster_id, 0)
                        f.write(f"  Cluster {cluster_id}: {occurrence_rate:.1%} d'occurrence, "
                               f"{intensity:.1f} mm d'intensité moyenne\n")
                
                f.write("\n" + "-" * 60 + "\n\n")
            
            # Recommandations
            f.write("RECOMMANDATIONS\n")
            f.write("-" * 20 + "\n")
            
            if self.evaluation_metrics:
                # Algorithme recommandé basé sur un score composite
                composite_scores = {}
                for method_name, metrics in self.evaluation_metrics.items():
                    # Score composite (normalisé)
                    silhouette_norm = metrics['silhouette_score']  # [0, 1]
                    occurrence_var_norm = min(1, metrics['occurrence_variance'] * 10)  # Ajuster échelle
                    
                    composite_score = (silhouette_norm + occurrence_var_norm) / 2
                    composite_scores[method_name] = composite_score
                
                best_overall = max(composite_scores.items(), key=lambda x: x[1])
                
                f.write(f"ALGORITHME RECOMMANDÉ: {best_overall[0]}\n")
                f.write(f"Score composite: {best_overall[1]:.3f}\n\n")
                
                f.write("Justification:\n")
                best_metrics = self.evaluation_metrics[best_overall[0]]
                f.write(f"• Bon équilibre entre qualité du clustering (Silhouette: {best_metrics['silhouette_score']:.3f})\n")
                f.write(f"• Bonne séparation des conditions à risque (Variance: {best_metrics['occurrence_variance']:.4f})\n")
                f.write(f"• {best_metrics['n_clusters']} clusters identifiés\n\n")
            
            f.write("APPLICATIONS OPÉRATIONNELLES:\n")
            f.write("• Classification automatique des conditions climatiques\n")
            f.write("• Identification des régimes météorologiques à risque\n")
            f.write("• Amélioration des modèles de prévision\n")
            f.write("• Stratification des données pour l'apprentissage automatique\n")
            f.write("• Analyse des téléconnexions par type de conditions\n\n")
            
            f.write("PROCHAINES ÉTAPES SUGGÉRÉES:\n")
            f.write("• Validation temporelle des clusters\n")
            f.write("• Analyse saisonnière du clustering\n")
            f.write("• Intégration dans les modèles prédictifs\n")
            f.write("• Tests sur d'autres régions d'Afrique de l'Ouest\n")
        
        print(f"✅ Rapport complet généré: {report_path}")
        return report_path
    
    def run_complete_clustering_analysis(self):
        """Exécute l'analyse complète de clustering."""
        print("🎯 ANALYSE DE CLUSTERING AVANCÉE")
        print("=" * 70)
        print("Comparaison de K-Means, DBSCAN, Hiérarchique, GMM et Spectral")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Étape 1: Chargement des données
        if not self.load_and_prepare_data():
            return False
        
        # Étape 2: K-Means (référence)
        optimal_k = self.perform_kmeans_analysis()
        
        # Étape 3: DBSCAN
        self.perform_dbscan_analysis()
        
        # Étape 4: Clustering hiérarchique
        self.perform_hierarchical_clustering()
        
        # Étape 5: Gaussian Mixture Models
        self.perform_gaussian_mixture_analysis()
        
        # Étape 6: Spectral Clustering
        self.perform_spectral_clustering()
        
        # Étape 7: Évaluation comparative
        evaluation_metrics = self.evaluate_clustering_methods()
        
        # Étape 8: Visualisations
        self.create_comprehensive_visualizations()
        
        # Étape 9: Rapport complet
        report_path = self.generate_comprehensive_report()
        
        # Résumé final
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\n" + "=" * 70)
        print("✅ ANALYSE DE CLUSTERING AVANCÉE TERMINÉE!")
        print("=" * 70)
        
        print(f"⏱️  Durée totale: {duration:.1f} secondes")
        print(f"🎯 Algorithmes testés: {len(self.clustering_results)}")
        print(f"📊 Métriques calculées: {len(evaluation_metrics)}")
        
        if evaluation_metrics:
            # Meilleur algorithme
            best_method = max(evaluation_metrics.items(), 
                            key=lambda x: (x[1]['silhouette_score'] + min(1, x[1]['occurrence_variance']*10))/2)
            
            print(f"🏆 Meilleur algorithme: {best_method[0]}")
            print(f"   Silhouette: {best_method[1]['silhouette_score']:.3f}")
            print(f"   Clusters: {best_method[1]['n_clusters']}")
            print(f"   Séparation événements: {best_method[1]['occurrence_variance']:.4f}")
        
        print(f"\n📁 FICHIERS GÉNÉRÉS:")
        viz_files = list(self.viz_dir.glob("*.png"))
        print(f"   🎨 Visualisations: {len(viz_files)} fichiers")
        for viz_file in viz_files:
            print(f"      • {viz_file.name}")
        
        print(f"   📄 Rapport: {report_path}")
        
        print(f"\n🚀 APPLICATIONS RECOMMANDÉES:")
        print(f"   • Utiliser le meilleur algorithme pour stratifier les données")
        print(f"   • Améliorer les modèles ML avec les labels de clusters")
        print(f"   • Développer des alertes spécifiques par cluster")
        print(f"   • Analyser les téléconnexions par régime climatique")
        
        return True

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale."""
    print("🎯 ANALYSE DE CLUSTERING AVANCÉE - PRÉCIPITATIONS EXTRÊMES")
    print("=" * 70)
    
    analyzer = AdvancedClusteringAnalysis()
    success = analyzer.run_complete_clustering_analysis()
    
    if success:
        print("\n🎉 ANALYSE DE CLUSTERING RÉUSSIE!")
        print("\n💡 INTÉGRATION DANS VOTRE MÉMOIRE:")
        print("• Chapitre 3: Méthodologie - Justification du choix d'algorithme")
        print("• Chapitre 4: Résultats - Classification des régimes climatiques")
        print("• Chapitre 5: Discussion - Patterns identifiés par clustering")
        print("• Annexes: Comparaison détaillée des algorithmes")
        print("\nConsultez le dossier outputs/visualizations/clustering/")
        return 0
    else:
        print("\n❌ ÉCHEC DE L'ANALYSE DE CLUSTERING")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)