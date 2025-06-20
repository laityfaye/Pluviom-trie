#!/usr/bin/env python3
# debug_database_structure.py
"""
Script de diagnostic pour vérifier la structure de la base de données
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv
from pathlib import Path

# Charger les variables d'environnement
PROJECT_ROOT = Path(__file__).parent
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)

def get_database_url():
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('POSTGRES_DB', 'climatsn_db')
    db_user = os.getenv('POSTGRES_USER', 'postgres')
    db_password = os.getenv('POSTGRES_PASSWORD', 'rout')
    
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

async def diagnose_database():
    """Diagnostic complet de la structure de la base."""
    print("🔍 DIAGNOSTIC DE LA BASE DE DONNÉES")
    print("=" * 50)
    
    DATABASE_URL = get_database_url()
    print(f"URL: {DATABASE_URL.replace(DATABASE_URL.split('@')[0].split('//')[1], '***')}")
    
    try:
        # Connexion
        conn = await asyncpg.connect(DATABASE_URL, timeout=10)
        print("✅ Connexion réussie")
        
        # 1. Vérifier l'existence de la table
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'ml_models'
            )
        """)
        print(f"📋 Table ml_models existe: {table_exists}")
        
        if not table_exists:
            print("❌ La table ml_models n'existe pas!")
            await conn.close()
            return
        
        # 2. Lister TOUTES les colonnes avec détails
        print("\n📊 STRUCTURE COMPLÈTE DE LA TABLE ml_models:")
        columns = await conn.fetch("""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default,
                ordinal_position
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = 'ml_models'
            ORDER BY ordinal_position
        """)
        
        print(f"Nombre total de colonnes: {len(columns)}")
        print("-" * 80)
        print(f"{'#':<3} {'Nom':<20} {'Type':<25} {'Null':<8} {'Défaut':<20}")
        print("-" * 80)
        
        column_names = []
        for col in columns:
            column_names.append(col['column_name'])
            print(f"{col['ordinal_position']:<3} {col['column_name']:<20} {col['data_type']:<25} {col['is_nullable']:<8} {str(col['column_default'])[:20]:<20}")
        
        # 3. Vérifier les colonnes attendues par le script
        expected_columns = [
            'id', 'name', 'version', 'model_type', 'target_variable', 
            'features', 'hyperparameters', 'performance_metrics', 
            'training_period', 'model_path', 'status', 'created_at', 'trained_at'
        ]
        
        print(f"\n🔍 VÉRIFICATION DES COLONNES ATTENDUES:")
        missing_columns = []
        present_columns = []
        
        for col in expected_columns:
            if col in column_names:
                present_columns.append(col)
                print(f"   ✅ {col}")
            else:
                missing_columns.append(col)
                print(f"   ❌ {col} - MANQUANTE")
        
        print(f"\n📊 RÉSUMÉ:")
        print(f"   • Colonnes présentes: {len(present_columns)}/{len(expected_columns)}")
        print(f"   • Colonnes manquantes: {len(missing_columns)}")
        
        if missing_columns:
            print(f"   • Manquantes: {', '.join(missing_columns)}")
        
        # 4. Tester la requête exacte du script original
        print(f"\n🧪 TEST DE LA REQUÊTE DU SCRIPT ORIGINAL:")
        try:
            script_columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'ml_models'
                ORDER BY ordinal_position
            """)
            
            script_column_names = [col['column_name'] for col in script_columns]
            print(f"   Colonnes détectées par la requête du script: {len(script_column_names)}")
            print(f"   Noms: {', '.join(script_column_names)}")
            
            required_by_script = ['hyperparameters', 'performance_metrics']
            missing_by_script = [col for col in required_by_script if col not in script_column_names]
            
            if missing_by_script:
                print(f"   ❌ Le script ne trouve pas: {missing_by_script}")
            else:
                print(f"   ✅ Le script devrait trouver toutes les colonnes requises")
                
        except Exception as e:
            print(f"   ❌ Erreur test requête script: {e}")
        
        # 5. Compter les enregistrements
        try:
            count = await conn.fetchval("SELECT COUNT(*) FROM ml_models")
            print(f"\n📈 Enregistrements dans ml_models: {count}")
        except Exception as e:
            print(f"\n❌ Erreur comptage: {e}")
        
        await conn.close()
        
        # Conclusion
        print(f"\n🎯 CONCLUSION:")
        if len(missing_columns) == 0:
            print(f"   ✅ La structure de la table est COMPLÈTE")
            print(f"   ✅ Le script de déploiement devrait fonctionner")
            print(f"   💡 Si le problème persiste, c'est un bug dans le script")
        else:
            print(f"   ❌ La structure de la table est INCOMPLÈTE")
            print(f"   💡 Ajoutez les colonnes manquantes avec ALTER TABLE")
        
    except Exception as e:
        print(f"❌ Erreur de diagnostic: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_database())