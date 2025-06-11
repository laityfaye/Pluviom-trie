# src/utils/season_classifier.py
"""
Module de classification saisonnière pour le climat sahélien du Sénégal.
"""

import pandas as pd
from typing import Tuple
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.settings import SEASONS_SENEGAL, get_season_from_month
except ImportError:
    # Valeurs par défaut
    SEASONS_SENEGAL = {
        'saison_seche': {'months': [11, 12, 1, 2, 3, 4]},
        'saison_des_pluies': {'months': [5, 6, 7, 8, 9, 10]}
    }
    
    def get_season_from_month(month: int) -> str:
        if month in [11, 12, 1, 2, 3, 4]:
            return 'Saison_seche'
        elif month in [5, 6, 7, 8, 9, 10]:
            return 'Saison_des_pluies'
        else:
            return 'Indetermine'


def classify_seasons_senegal_final(df_events: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Classification saisonnière pour le Sénégal.
    Saison des pluies: Mai-Octobre (6 mois)
    Saison sèche: Novembre-Avril (6 mois)
    
    Args:
        df_events (pd.DataFrame): DataFrame des événements
        
    Returns:
        Tuple[pd.DataFrame, str]: (DataFrame avec classification, statut_validation)
    """
    print("\n🔄 CLASSIFICATION SAISONNIÈRE")
    print("-" * 50)
    print("Définition des saisons pour le Sénégal:")
    print("• Saison des pluies: Mai à Octobre (6 mois)")
    print("• Saison sèche: Novembre à Avril (6 mois)")
    
    # Appliquer la classification
    df_events['saison'] = df_events['month'].apply(get_season_from_month)
    
    # Statistiques saisonnières
    season_counts = df_events['saison'].value_counts()
    total_events = len(df_events)
    
    print(f"\n📊 DISTRIBUTION SAISONNIÈRE:")
    for season, count in season_counts.items():
        percentage = count / total_events * 100
        print(f"   {season}: {count} événements ({percentage:.1f}%)")
    
    # Validation climatologique
    pluies_count = season_counts.get('Saison_des_pluies', 0)
    pluies_pct = pluies_count / total_events * 100 if total_events > 0 else 0
    
    print(f"\n🔍 VALIDATION CLIMATOLOGIQUE:")
    if pluies_pct >= 90:
        print(f"✅ Excellente cohérence: {pluies_pct:.1f}% des événements en saison des pluies")
        validation_status = "EXCELLENT"
    elif pluies_pct >= 80:
        print(f"✅ Très cohérent: {pluies_pct:.1f}% des événements en saison des pluies")
        validation_status = "TRES_COHERENT"
    elif pluies_pct >= 60:
        print(f"✅ Cohérent: {pluies_pct:.1f}% des événements en saison des pluies")
        validation_status = "COHERENT"
    elif pluies_pct >= 40:
        print(f"⚠️  Modéré: {pluies_pct:.1f}% des événements en saison des pluies")
        validation_status = "MODERE"
    else:
        print(f"❌ Incohérent: {pluies_pct:.1f}% des événements en saison des pluies")
        validation_status = "INCOHERENT"
    
    return df_events, validation_status


class SeasonClassifier:
    """
    Classe pour la classification saisonnière des événements.
    """
    
    def __init__(self):
        """Initialise le classificateur saisonnier."""
        self.seasons_info = SEASONS_SENEGAL
    
    def classify_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Classifie les événements par saison.
        
        Args:
            df_events (pd.DataFrame): DataFrame des événements
            
        Returns:
            pd.DataFrame: DataFrame avec colonne 'saison' ajoutée
        """
        df_events['saison'] = df_events['month'].apply(get_season_from_month)
        return df_events
    
    def validate_seasonal_distribution(self, df_events: pd.DataFrame) -> str:
        """
        Valide la distribution saisonnière des événements.
        
        Args:
            df_events (pd.DataFrame): DataFrame des événements avec classification
            
        Returns:
            str: Statut de validation
        """
        season_counts = df_events['saison'].value_counts()
        total_events = len(df_events)
        
        pluies_count = season_counts.get('Saison_des_pluies', 0)
        pluies_pct = pluies_count / total_events * 100 if total_events > 0 else 0
        
        if pluies_pct >= 90:
            return "EXCELLENT"
        elif pluies_pct >= 80:
            return "TRES_COHERENT"
        elif pluies_pct >= 60:
            return "COHERENT"
        elif pluies_pct >= 40:
            return "MODERE"
        else:
            return "INCOHERENT"
    
    def get_seasonal_statistics(self, df_events: pd.DataFrame) -> dict:
        """
        Calcule les statistiques saisonnières détaillées.
        
        Args:
            df_events (pd.DataFrame): DataFrame des événements
            
        Returns:
            dict: Statistiques saisonnières
        """
        stats = {}
        
        for saison in df_events['saison'].unique():
            saison_data = df_events[df_events['saison'] == saison]
            
            stats[saison] = {
                'count': len(saison_data),
                'percentage': len(saison_data) / len(df_events) * 100,
                'avg_coverage': saison_data['coverage_percent'].mean(),
                'avg_precipitation': saison_data['max_precip'].mean(),
                'avg_anomaly': saison_data['max_anomaly'].mean(),
                'median_precipitation': saison_data['max_precip'].median()
            }
        
        return stats
    
    def classify_and_validate(self, df_events: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Classifie les événements et valide la distribution.
        
        Args:
            df_events (pd.DataFrame): DataFrame des événements
            
        Returns:
            Tuple[pd.DataFrame, str]: (DataFrame classifié, statut_validation)
        """
        print("\n🔄 CLASSIFICATION SAISONNIÈRE")
        print("-" * 50)
        print("Définition des saisons pour le Sénégal:")
        print("• Saison des pluies: Mai à Octobre (6 mois)")
        print("• Saison sèche: Novembre à Avril (6 mois)")
        
        # Classifier
        df_classified = self.classify_events(df_events)
        
        # Statistiques
        stats = self.get_seasonal_statistics(df_classified)
        
        print(f"\n📊 DISTRIBUTION SAISONNIÈRE:")
        for saison, saison_stats in stats.items():
            print(f"   {saison}: {saison_stats['count']} événements ({saison_stats['percentage']:.1f}%)")
            print(f"     Précipitation moyenne: {saison_stats['avg_precipitation']:.2f} mm")
            print(f"     Couverture moyenne: {saison_stats['avg_coverage']:.2f}%")
        
        # Valider
        validation_status = self.validate_seasonal_distribution(df_classified)
        
        pluies_pct = stats.get('Saison_des_pluies', {}).get('percentage', 0)
        
        print(f"\n🔍 VALIDATION CLIMATOLOGIQUE:")
        if validation_status == "EXCELLENT":
            print(f"✅ Excellente cohérence: {pluies_pct:.1f}% des événements en saison des pluies")
        elif validation_status == "TRES_COHERENT":
            print(f"✅ Très cohérent: {pluies_pct:.1f}% des événements en saison des pluies")
        elif validation_status == "COHERENT":
            print(f"✅ Cohérent: {pluies_pct:.1f}% des événements en saison des pluies")
        elif validation_status == "MODERE":
            print(f"⚠️  Modéré: {pluies_pct:.1f}% des événements en saison des pluies")
        else:
            print(f"❌ Incohérent: {pluies_pct:.1f}% des événements en saison des pluies")
        
        return df_classified, validation_status


def get_month_name_fr(month: int) -> str:
    """
    Retourne le nom du mois en français.
    
    Args:
        month (int): Numéro du mois (1-12)
        
    Returns:
        str: Nom du mois en français
    """
    month_names = {
        1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril',
        5: 'Mai', 6: 'Juin', 7: 'Juillet', 8: 'Août',
        9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
    }
    return month_names.get(month, 'Inconnu')


def get_season_description(saison: str) -> str:
    """
    Retourne la description d'une saison.
    
    Args:
        saison (str): Nom de la saison
        
    Returns:
        str: Description de la saison
    """
    descriptions = {
        'Saison_des_pluies': 'Mai à Octobre - Période de précipitations importantes',
        'Saison_seche': 'Novembre à Avril - Période de faibles précipitations'
    }
    return descriptions.get(saison, 'Saison inconnue')


if __name__ == "__main__":
    print("Module de classification saisonnière")
    print("=" * 50)
    print("Ce module contient les outils pour:")
    print("• Classifier les événements par saison (Sénégal)")
    print("• Valider la cohérence climatologique")
    print("• Calculer les statistiques saisonnières")