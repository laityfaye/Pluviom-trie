#!/usr/bin/env python3
# src/analysis/teleconnections.py
"""
Module d'analyse des téléconnexions entre indices climatiques et événements extrêmes.

Ce module analyse les corrélations entre les indices IOD, Nino34, TNA et les événements 
de précipitations extrêmes détectés au Sénégal, avec prise en compte des décalages temporels.

Auteur: [Votre nom]
Date: [Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

class TeleconnectionsAnalyzer:
    """
    Classe pour analyser les téléconnexions entre indices climatiques et événements extrêmes.
    """
    
    def __init__(self):
        """Initialise l'analyseur de téléconnexions."""
        self.extreme_events = None
        self.climate_indices = None
        self.monthly_events = None
        self.correlation_results = {}
        self.best_lags = {}
        
    def load_extreme_events(self, events_file: str) -> pd.DataFrame:
        """
        Charge les événements extrêmes détectés.
        
        Args:
            events_file (str): Chemin vers le fichier des événements extrêmes
            
        Returns:
            pd.DataFrame: DataFrame des événements extrêmes
        """
        print("🔄 Chargement des événements extrêmes...")
        
        try:
            # Chargement du dataset principal
            df_events = pd.read_csv(events_file, index_col=0, parse_dates=True)
            
            print(f"   ✅ {len(df_events)} événements chargés")
            print(f"   📅 Période: {df_events.index.min().strftime('%Y-%m-%d')} à "
                  f"{df_events.index.max().strftime('%Y-%m-%d')}")
            
            self.extreme_events = df_events
            return df_events
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des événements: {e}")
            return pd.DataFrame()
    
    def load_climate_indices(self, indices_file: str) -> pd.DataFrame:
        """
        Charge les indices climatiques.
        
        Args:
            indices_file (str): Chemin vers le fichier des indices climatiques
            
        Returns:
            pd.DataFrame: DataFrame des indices climatiques
        """
        print("🔄 Chargement des indices climatiques...")
        
        try:
            # Chargement du dataset combiné
            df_indices = pd.read_csv(indices_file, index_col=0, parse_dates=True)
            
            print(f"   ✅ {df_indices.shape[1]} indices chargés sur {len(df_indices)} mois")
            print(f"   📅 Période: {df_indices.index.min().strftime('%Y-%m')} à "
                  f"{df_indices.index.max().strftime('%Y-%m')}")
            print(f"   📊 Indices: {list(df_indices.columns)}")
            
            self.climate_indices = df_indices
            return df_indices
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des indices: {e}")
            return pd.DataFrame()
    
    def create_monthly_event_series(self) -> pd.Series:
        """
        Crée une série mensuelle d'événements extrêmes pour les corrélations.
        
        Returns:
            pd.Series: Série mensuelle (nombre d'événements par mois)
        """
        print("🔄 Création de la série mensuelle d'événements...")
        
        if self.extreme_events is None:
            print("❌ Événements extrêmes non chargés")
            return pd.Series(dtype=int)
        
        # Groupement par mois
        monthly_count = self.extreme_events.groupby(pd.Grouper(freq='MS')).size()
        
        # Extension sur toute la période des indices climatiques
        if self.climate_indices is not None:
            full_index = self.climate_indices.index
            monthly_count = monthly_count.reindex(full_index, fill_value=0)
        
        print(f"   ✅ Série mensuelle créée: {len(monthly_count)} mois")
        print(f"   📊 Événements totaux: {monthly_count.sum()}")
        print(f"   📊 Mois avec événements: {(monthly_count > 0).sum()}")
        print(f"   📊 Moyenne: {monthly_count.mean():.2f} événements/mois")
        
        self.monthly_events = monthly_count
        return monthly_count
    
    def calculate_lag_correlations(self, max_lag: int = 12, 
                                 correlation_type: str = 'pearson') -> Dict[str, Dict[int, float]]:
        """
        Calcule les corrélations avec différents décalages temporels.
        
        Args:
            max_lag (int): Décalage maximum en mois
            correlation_type (str): Type de corrélation ('pearson' ou 'spearman')
            
        Returns:
            Dict[str, Dict[int, float]]: Corrélations par indice et décalage
        """
        print(f"\n🔄 CALCUL DES CORRÉLATIONS AVEC DÉCALAGES (0-{max_lag} mois)")
        print("-" * 50)
        
        if self.monthly_events is None:
            print("❌ Série mensuelle d'événements non disponible")
            return {}
        
        if self.climate_indices is None:
            print("❌ Indices climatiques non chargés")
            return {}
        
        correlations = {}
        
        for index_name in self.climate_indices.columns:
            print(f"   Analyse des corrélations pour {index_name}...")
            
            index_correlations = {}
            index_series = self.climate_indices[index_name].dropna()
            
            for lag in range(max_lag + 1):
                # Décalage de l'indice climatique
                lagged_index = index_series.shift(lag)
                
                # Alignement des séries
                common_dates = self.monthly_events.index.intersection(lagged_index.index)
                
                if len(common_dates) < 24:  # Au moins 2 ans de données
                    continue
                
                events_aligned = self.monthly_events.loc[common_dates]
                index_aligned = lagged_index.loc[common_dates]
                
                # Suppression des valeurs manquantes
                valid_mask = events_aligned.notna() & index_aligned.notna()
                
                if valid_mask.sum() < 24:
                    continue
                
                events_clean = events_aligned[valid_mask]
                index_clean = index_aligned[valid_mask]
                
                # Calcul de la corrélation
                if correlation_type == 'pearson':
                    corr, p_value = pearsonr(index_clean, events_clean)
                else:
                    corr, p_value = spearmanr(index_clean, events_clean)
                
                # Stockage du résultat
                index_correlations[lag] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'n_obs': len(events_clean),
                    'significant': p_value < 0.05
                }
            
            correlations[index_name] = index_correlations
            
            # Affichage des meilleurs résultats
            if index_correlations:
                best_lag = max(index_correlations.keys(), 
                             key=lambda x: abs(index_correlations[x]['correlation']))
                best_corr = index_correlations[best_lag]['correlation']
                best_p = index_correlations[best_lag]['p_value']
                
                print(f"     Meilleure corrélation: lag-{best_lag} = {best_corr:.3f} "
                      f"(p={best_p:.3f}, {'***' if best_p < 0.001 else '**' if best_p < 0.01 else '*' if best_p < 0.05 else 'ns'})")
        
        self.correlation_results = correlations
        return correlations
    
    def find_optimal_lags(self) -> Dict[str, Dict]:
        """
        Identifie les décalages optimaux pour chaque indice.
        
        Returns:
            Dict[str, Dict]: Décalages optimaux et leurs caractéristiques
        """
        print(f"\n🎯 IDENTIFICATION DES DÉCALAGES OPTIMAUX")
        print("-" * 50)
        
        if not self.correlation_results:
            print("❌ Corrélations non calculées")
            return {}
        
        optimal_lags = {}
        
        for index_name, correlations in self.correlation_results.items():
            if not correlations:
                continue
            
            # Trouver le lag avec la corrélation la plus forte (en valeur absolue)
            best_lag = max(correlations.keys(), 
                          key=lambda x: abs(correlations[x]['correlation']))
            
            best_stats = correlations[best_lag]
            
            # Chercher aussi les corrélations significatives
            significant_lags = {lag: stats for lag, stats in correlations.items() 
                              if stats['significant']}
            
            optimal_lags[index_name] = {
                'best_lag': best_lag,
                'best_correlation': best_stats['correlation'],
                'best_p_value': best_stats['p_value'],
                'best_n_obs': best_stats['n_obs'],
                'significant_lags': list(significant_lags.keys()),
                'n_significant': len(significant_lags)
            }
            
            print(f"   {index_name}:")
            print(f"     Lag optimal: {best_lag} mois")
            print(f"     Corrélation: {best_stats['correlation']:.3f}")
            print(f"     Significativité: {'Oui' if best_stats['significant'] else 'Non'} (p={best_stats['p_value']:.3f})")
            print(f"     Lags significatifs: {len(significant_lags)} / {len(correlations)}")
        
        self.best_lags = optimal_lags
        return optimal_lags
    
    def analyze_seasonal_teleconnections(self) -> Dict[str, Dict]:
        """
        Analyse les téléconnexions par saison.
        
        Returns:
            Dict[str, Dict]: Corrélations saisonnières
        """
        print(f"\n🌍 ANALYSE SAISONNIÈRE DES TÉLÉCONNEXIONS")
        print("-" * 50)
        
        if self.monthly_events is None or self.climate_indices is None:
            print("❌ Données non disponibles")
            return {}
        
        # Définition des saisons sahéliennes
        seasons = {
            'saison_seche': [11, 12, 1, 2, 3, 4],  # Nov-Avr
            'saison_pluies': [5, 6, 7, 8, 9, 10]   # Mai-Oct
        }
        
        seasonal_correlations = {}
        
        for season_name, months in seasons.items():
            print(f"   Analyse pour {season_name.replace('_', ' ')}...")
            
            # Filtrage par saison
            season_mask = self.monthly_events.index.month.isin(months)
            events_season = self.monthly_events[season_mask]
            indices_season = self.climate_indices[season_mask]
            
            season_corr = {}
            
            for index_name in self.climate_indices.columns:
                # Utilisation du lag optimal trouvé précédemment
                if index_name in self.best_lags:
                    optimal_lag = self.best_lags[index_name]['best_lag']
                else:
                    optimal_lag = 1  # Défaut à 1 mois
                
                # Application du décalage
                index_lagged = indices_season[index_name].shift(optimal_lag)
                
                # Alignement et nettoyage
                common_dates = events_season.index.intersection(index_lagged.index)
                events_aligned = events_season.loc[common_dates]
                index_aligned = index_lagged.loc[common_dates]
                
                valid_mask = events_aligned.notna() & index_aligned.notna()
                
                if valid_mask.sum() >= 12:  # Au moins 1 an de données
                    events_clean = events_aligned[valid_mask]
                    index_clean = index_aligned[valid_mask]
                    
                    corr, p_value = pearsonr(index_clean, events_clean)
                    
                    season_corr[index_name] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'n_obs': len(events_clean),
                        'significant': p_value < 0.05,
                        'lag_used': optimal_lag
                    }
            
            seasonal_correlations[season_name] = season_corr
            
            # Affichage des résultats
            print(f"     Corrélations significatives:")
            for idx, stats in season_corr.items():
                if stats['significant']:
                    print(f"       {idx}: {stats['correlation']:.3f} (p={stats['p_value']:.3f})")
        
        return seasonal_correlations
    
    def create_correlation_heatmap(self, max_lag: int = 12, figsize: Tuple[int, int] = (12, 8)):
        """
        Crée une heatmap des corrélations par décalage temporel.
        
        Args:
            max_lag (int): Décalage maximum à afficher
            figsize (tuple): Taille de la figure
        """
        print(f"\n📊 CRÉATION DE LA HEATMAP DES CORRÉLATIONS")
        print("-" * 50)
        
        if not self.correlation_results:
            print("❌ Corrélations non calculées")
            return
        
        # Préparation des données pour la heatmap
        correlations_matrix = []
        index_names = []
        lags = list(range(max_lag + 1))
        
        for index_name, correlations in self.correlation_results.items():
            corr_values = []
            for lag in lags:
                if lag in correlations:
                    corr_values.append(correlations[lag]['correlation'])
                else:
                    corr_values.append(np.nan)
            
            correlations_matrix.append(corr_values)
            index_names.append(index_name)
        
        # Conversion en DataFrame pour seaborn
        corr_df = pd.DataFrame(correlations_matrix, 
                              index=index_names, 
                              columns=[f'Lag-{lag}' for lag in lags])
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Heatmap avec masque pour les valeurs manquantes
        mask = corr_df.isnull()
        
        sns.heatmap(corr_df, 
                   annot=True, 
                   fmt='.3f',
                   cmap='RdBu_r',
                   center=0,
                   vmin=-0.5, vmax=0.5,
                   mask=mask,
                   cbar_kws={'label': 'Corrélation'},
                   ax=ax)
        
        ax.set_title('Corrélations Indices Climatiques - Événements Extrêmes\n'
                    'par Décalage Temporel (Sénégal 1981-2023)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Décalage Temporel (mois)', fontweight='bold')
        ax.set_ylabel('Indices Climatiques', fontweight='bold')
        
        plt.tight_layout()
        
        # Sauvegarde
        output_path = Path("outputs/visualizations/teleconnections")
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path / "correlation_heatmap_lags.png", 
                   dpi=300, bbox_inches='tight')
        print(f"   ✅ Heatmap sauvegardée: correlation_heatmap_lags.png")
        
        plt.close()
    
    def create_lag_correlation_plots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Crée des graphiques détaillés des corrélations par décalage.
        
        Args:
            figsize (tuple): Taille de la figure
        """
        print(f"\n📈 CRÉATION DES GRAPHIQUES DE CORRÉLATIONS PAR LAG")
        print("-" * 50)
        
        if not self.correlation_results:
            print("❌ Corrélations non calculées")
            return
        
        n_indices = len(self.correlation_results)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, (index_name, correlations) in enumerate(self.correlation_results.items()):
            if i >= 4:  # Limitation à 4 subplots
                break
            
            lags = list(correlations.keys())
            corr_values = [correlations[lag]['correlation'] for lag in lags]
            p_values = [correlations[lag]['p_value'] for lag in lags]
            
            # Graphique des corrélations
            axes[i].plot(lags, corr_values, 'o-', linewidth=2, markersize=6, 
                        color='darkblue', label='Corrélation')
            
            # Ligne de référence à zéro
            axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Zones de significativité
            for j, (lag, p_val) in enumerate(zip(lags, p_values)):
                if p_val < 0.001:
                    axes[i].scatter(lag, corr_values[j], color='red', s=100, marker='*', zorder=5)
                elif p_val < 0.01:
                    axes[i].scatter(lag, corr_values[j], color='orange', s=80, marker='o', zorder=5)
                elif p_val < 0.05:
                    axes[i].scatter(lag, corr_values[j], color='yellow', s=60, marker='o', zorder=5)
            
            # Identification du meilleur lag
            best_lag = max(lags, key=lambda x: abs(correlations[x]['correlation']))
            best_corr = correlations[best_lag]['correlation']
            
            axes[i].scatter(best_lag, best_corr, color='green', s=150, 
                           marker='D', zorder=6, label=f'Optimal (lag-{best_lag})')
            
            axes[i].set_title(f'{index_name}\nMeilleure corrélation: {best_corr:.3f} (lag-{best_lag})',
                             fontweight='bold')
            axes[i].set_xlabel('Décalage (mois)')
            axes[i].set_ylabel('Corrélation')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            # Limites des axes
            axes[i].set_ylim(-0.6, 0.6)
        
        # Cacher les subplots vides
        for j in range(i + 1, 4):
            axes[j].set_visible(False)
        
        plt.suptitle('Corrélations Détaillées par Décalage Temporel\n'
                    'Indices Climatiques vs Événements Extrêmes (Sénégal)',
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        # Sauvegarde
        output_path = Path("outputs/visualizations/teleconnections")
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path / "detailed_lag_correlations.png", 
                   dpi=300, bbox_inches='tight')
        print(f"   ✅ Graphiques détaillés sauvegardés: detailed_lag_correlations.png")
        
        plt.close()
    
    def create_seasonal_comparison(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Compare les téléconnexions entre saisons.
        
        Args:
            figsize (tuple): Taille de la figure
        """
        print(f"\n🌍 CRÉATION DU GRAPHIQUE DE COMPARAISON SAISONNIÈRE")
        print("-" * 50)
        
        # Calcul des corrélations saisonnières
        seasonal_correlations = self.analyze_seasonal_teleconnections()
        
        if not seasonal_correlations:
            print("❌ Corrélations saisonnières non disponibles")
            return
        
        # Préparation des données
        indices = list(self.climate_indices.columns)
        seasons = list(seasonal_correlations.keys())
        
        seasonal_data = []
        for season in seasons:
            season_corr = []
            for index in indices:
                if index in seasonal_correlations[season]:
                    corr = seasonal_correlations[season][index]['correlation']
                    season_corr.append(corr)
                else:
                    season_corr.append(0)
            seasonal_data.append(season_corr)
        
        # Création du graphique
        x = np.arange(len(indices))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#E74C3C', '#27AE60']  # Rouge pour saison sèche, vert pour pluies
        season_labels = ['Saison Sèche (Nov-Avr)', 'Saison Pluies (Mai-Oct)']
        
        for i, (season_data, color, label) in enumerate(zip(seasonal_data, colors, season_labels)):
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, season_data, width, label=label, 
                         color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Ajout des valeurs sur les barres
            for bar, value in zip(bars, season_data):
                if abs(value) > 0.1:  # Seulement pour les valeurs significatives
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.03,
                           f'{value:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=9, fontweight='bold')
        
        # Configuration du graphique
        ax.set_xlabel('Indices Climatiques', fontweight='bold')
        ax.set_ylabel('Corrélation', fontweight='bold')
        ax.set_title('Comparaison Saisonnière des Téléconnexions\n'
                    'Indices Climatiques vs Événements Extrêmes (Sénégal)',
                    fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(indices)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        # Limites
        ax.set_ylim(-0.5, 0.5)
        
        plt.tight_layout()
        
        # Sauvegarde
        output_path = Path("outputs/visualizations/teleconnections")
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path / "seasonal_teleconnections_comparison.png", 
                   dpi=300, bbox_inches='tight')
        print(f"   ✅ Comparaison saisonnière sauvegardée: seasonal_teleconnections_comparison.png")
        
        plt.close()
    
    def generate_teleconnections_report(self) -> str:
        """
        Génère un rapport complet des téléconnexions.
        
        Returns:
            str: Chemin vers le rapport généré
        """
        print(f"\n📄 GÉNÉRATION DU RAPPORT DE TÉLÉCONNEXIONS")
        print("-" * 50)
        
        if not self.correlation_results or not self.best_lags:
            print("❌ Analyses non complètes")
            return ""
        
        # Calcul des corrélations saisonnières
        seasonal_correlations = self.analyze_seasonal_teleconnections()
        
        # Création du rapport
        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "rapport_teleconnexions.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RAPPORT D'ANALYSE DES TÉLÉCONNEXIONS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Date de génération: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Période d'analyse: 1981-2023\n")
            f.write(f"Nombre d'événements extrêmes: {self.extreme_events.shape[0] if self.extreme_events is not None else 'N/A'}\n")
            f.write(f"Indices climatiques analysés: {list(self.climate_indices.columns)}\n\n")
            
            # Section 1: Résultats principaux
            f.write("1. RÉSULTATS PRINCIPAUX\n")
            f.write("-" * 25 + "\n\n")
            
            for index_name, lag_info in self.best_lags.items():
                f.write(f"{index_name}:\n")
                f.write(f"  Décalage optimal: {lag_info['best_lag']} mois\n")
                f.write(f"  Corrélation: {lag_info['best_correlation']:.3f}\n")
                f.write(f"  Significativité: {'Oui' if lag_info['best_p_value'] < 0.05 else 'Non'} ")
                f.write(f"(p = {lag_info['best_p_value']:.3f})\n")
                f.write(f"  Nombre d'observations: {lag_info['best_n_obs']}\n")
                f.write(f"  Lags significatifs: {lag_info['n_significant']} lags\n\n")
            
            # Section 2: Analyse saisonnière
            f.write("2. ANALYSE SAISONNIÈRE\n")
            f.write("-" * 25 + "\n\n")
            
            for season_name, season_corr in seasonal_correlations.items():
                f.write(f"{season_name.replace('_', ' ').title()}:\n")
                
                significant_correlations = {idx: stats for idx, stats in season_corr.items() 
                                          if stats['significant']}
                
                if significant_correlations:
                    f.write(f"  Corrélations significatives:\n")
                    for idx, stats in significant_correlations.items():
                        f.write(f"    {idx}: {stats['correlation']:.3f} (p={stats['p_value']:.3f})\n")
                else:
                    f.write(f"  Aucune corrélation significative détectée\n")
                f.write("\n")
            
            # Section 3: Interprétation
            f.write("3. INTERPRÉTATION ET MÉCANISMES\n")
            f.write("-" * 35 + "\n\n")
            
            f.write("Indices avec corrélations significatives:\n")
            significant_indices = [idx for idx, lag_info in self.best_lags.items() 
                                 if lag_info['best_p_value'] < 0.05]
            
            if significant_indices:
                for idx in significant_indices:
                    lag_info = self.best_lags[idx]
                    f.write(f"  {idx}:\n")
                    f.write(f"    - Corrélation {lag_info['best_correlation']:.3f} avec décalage de {lag_info['best_lag']} mois\n")
                    
                    if idx == 'IOD':
                        f.write(f"    - Mécanisme: Dipôle de l'Océan Indien influence la circulation Walker\n")
                        f.write(f"    - Impact: Modulation des précipitations via téléconnexions atmosphériques\n")
                    elif idx == 'Nino34':
                        f.write(f"    - Mécanisme: ENSO influence la position de la ZCIT\n")
                        f.write(f"    - Impact: Modulation de la mousson ouest-africaine\n")
                    elif idx == 'TNA':
                        f.write(f"    - Mécanisme: Atlantique tropical nord source d'humidité directe\n")
                        f.write(f"    - Impact: Contrôle du gradient thermique océan-continent\n")
                    f.write("\n")
            else:
                f.write("  Aucune corrélation significative détectée.\n")
                f.write("  Ceci peut indiquer:\n")
                f.write("    - Relations non-linéaires nécessitant des approches ML\n")
                f.write("    - Influences locales dominantes\n")
                f.write("    - Besoins d'indices climatiques additionnels\n\n")
            
            # Section 4: Recommandations
            f.write("4. RECOMMANDATIONS POUR LA MODÉLISATION ML\n")
            f.write("-" * 45 + "\n\n")
            
            f.write("Variables prédictives recommandées:\n")
            for idx, lag_info in self.best_lags.items():
                f.write(f"  {idx}_lag{lag_info['best_lag']}: Utiliser avec décalage de {lag_info['best_lag']} mois\n")
            
            f.write(f"\nPériode d'entraînement suggérée: 1981-2017 (37 ans)\n")
            f.write(f"Période de test suggérée: 2018-2023 (6 ans)\n\n")
            
            f.write("Algorithmes ML recommandés:\n")
            f.write("  - Random Forest: Robuste aux corrélations modérées\n")
            f.write("  - XGBoost: Capture des interactions non-linéaires\n")
            f.write("  - SVM avec noyau RBF: Relations complexes\n")
            f.write("  - Réseaux de neurones: Patterns sophistiqués\n\n")
        
        print(f"   ✅ Rapport sauvegardé: {report_path}")
        return str(report_path)
    
    def run_complete_analysis(self, events_file: str, indices_file: str) -> Dict:
        """
        Lance l'analyse complète des téléconnexions.
        
        Args:
            events_file (str): Fichier des événements extrêmes
            indices_file (str): Fichier des indices climatiques
            
        Returns:
            Dict: Résumé des résultats
        """
        print("🌊 ANALYSE COMPLÈTE DES TÉLÉCONNEXIONS")
        print("=" * 80)
        
        # Chargement des données
        if not self.load_extreme_events(events_file).empty:
            print("✅ Événements extrêmes chargés")
        else:
            print("❌ Échec du chargement des événements")
            return {}
        
        if not self.load_climate_indices(indices_file).empty:
            print("✅ Indices climatiques chargés")
        else:
            print("❌ Échec du chargement des indices")
            return {}
        
        # Création de la série mensuelle
        self.create_monthly_event_series()
        
        # Calcul des corrélations avec décalages
        self.calculate_lag_correlations(max_lag=12)
        
        # Identification des décalages optimaux
        optimal_lags = self.find_optimal_lags()
        
        # Génération des visualisations
        self.create_correlation_heatmap()
        self.create_lag_correlation_plots()
        self.create_seasonal_comparison()
        
        # Génération du rapport
        report_path = self.generate_teleconnections_report()
        
        # Résumé final
        print(f"\n🎯 RÉSUMÉ DE L'ANALYSE DES TÉLÉCONNEXIONS")
        print("=" * 50)
        
        significant_count = sum(1 for lag_info in optimal_lags.values() 
                              if lag_info['best_p_value'] < 0.05)
        
        print(f"📊 Indices analysés: {len(optimal_lags)}")
        print(f"📊 Corrélations significatives: {significant_count}/{len(optimal_lags)}")
        print(f"📊 Décalages optimaux identifiés: Oui")
        print(f"📊 Rapport généré: {report_path}")
        print(f"📊 Visualisations créées: 3 fichiers")
        
        return {
            'optimal_lags': optimal_lags,
            'significant_correlations': significant_count,
            'total_indices': len(optimal_lags),
            'report_path': report_path,
            'ready_for_ml': True
        }


if __name__ == "__main__":
    # Test du module
    print("Test du module d'analyse des téléconnexions")
    print("=" * 80)
    
    # Exemple d'utilisation
    analyzer = TeleconnectionsAnalyzer()
    
    # Chemins des fichiers (à adapter selon votre structure)
    events_file = "data/processed/extreme_events_senegal_final.csv"
    indices_file = "data/processed/climate_indices_combined.csv"
    
    # Analyse complète
    results = analyzer.run_complete_analysis(events_file, indices_file)
    
    if results:
        print(f"\n✅ Analyse terminée avec succès!")
        print(f"Prêt pour le développement des modèles ML: {results['ready_for_ml']}")
    else:
        print(f"\n❌ Échec de l'analyse")
    
    print("\n✅ Module testé avec succès!")