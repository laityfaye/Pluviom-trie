#!/usr/bin/env python3
# quick_fix_columns.py
"""
Script rapide pour renommer les colonnes du dataset ML
"""

import pandas as pd
from pathlib import Path

def rename_columns():
    print("🔧 CORRECTION RAPIDE DES NOMS DE COLONNES")
    print("=" * 50)
    
    # Charger le dataset
    ml_file = Path("data/processed/ml_dataset_teleconnections.csv")
    df = pd.read_csv(ml_file, index_col=0, parse_dates=True)
    
    print(f"✅ Dataset chargé: {df.shape}")
    
    # Renommer les colonnes
    column_mapping = {
        'target_occurrence': 'occurrence',
        'target_count': 'count', 
        'target_intensity': 'intensity'
    }
    
    df = df.rename(columns=column_mapping)
    
    print(f"✅ Colonnes renommées:")
    for old, new in column_mapping.items():
        print(f"   {old} → {new}")
    
    # Sauvegarder
    df.to_csv(ml_file)
    print(f"💾 Dataset mis à jour: {ml_file}")
    
    # Vérification
    print(f"\n✅ VÉRIFICATION:")
    for col in ['occurrence', 'count', 'intensity']:
        if col in df.columns:
            print(f"   ✅ {col}: Trouvée")
        else:
            print(f"   ❌ {col}: Manquante")
    
    return True

if __name__ == "__main__":
    success = rename_columns()
    if success:
        print("\n🎉 CORRECTION RÉUSSIE!")
        print("Vous pouvez maintenant relancer le pipeline ML:")
        print("python main.py --only-ml")