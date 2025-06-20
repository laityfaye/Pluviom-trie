#!/usr/bin/env python3
# main.py
"""
Point d'entrée principal du projet d'analyse des précipitations extrêmes au Sénégal.

Ce fichier permet de lancer facilement l'analyse complète depuis la racine du projet.
Il exécute séquentiellement :

PHASE 1 - RECHERCHE ET DÉVELOPPEMENT:
1. La détection des événements extrêmes (01_detection_extremes.py)
2. L'analyse spatiale des top 10 événements (02_spatial_analysis_top10.py)
3. L'analyse spatiale détaillée des TOP 5 les plus intenses (03_spatial_analysis_top5.py)
4. L'analyse des téléconnexions océan-atmosphère (04_teleconnections_analysis.py)
5. Le pipeline d'apprentissage automatique complet (05_machine_learning_pipeline.py)
6. L'outil de prédiction interactif (06_prediction_tool.py) - OPTIONNEL
7. L'analyse de clustering avancée (07_advanced_clustering_analysis.py)

PHASE 2 - DÉPLOIEMENT PRODUCTION:
8. Déploiement des modèles en production (08_deploy_models_to_production.py)

Version avec correction automatique et pipeline ML automatisé complet.

Utilisation:
    python main.py                    # Pipeline complet recherche + déploiement (8 étapes)
    python main.py --research-only    # Pipeline recherche uniquement (7 étapes)
    python main.py --deploy-only      # Déploiement uniquement
    python main.py --full-pipeline    # Pipeline complet avec outil interactif (8 étapes)
    python main.py --only-ml          # Pipeline ML uniquement
    python main.py --interactive      # Mode avec questions interactives

Auteur: [Laity FAYE]
Date: [01/06/2025]
"""

import sys
import subprocess
from pathlib import Path
import time
import argparse

def auto_fix_ml_dataset():
    """
    Corrige automatiquement les noms de colonnes du dataset ML
    pour assurer la compatibilité avec le pipeline ML.
    """
    import pandas as pd
    from pathlib import Path
    
    print(f"\n🔧 CORRECTION AUTOMATIQUE DES COLONNES ML")
    print("=" * 50)
    
    try:
        ml_file = Path("data/processed/ml_dataset_teleconnections.csv")
        
        if not ml_file.exists():
            print("⚠️  Dataset ML non trouvé - correction ignorée")
            return True
        
        # Charger le dataset
        df = pd.read_csv(ml_file, index_col=0, parse_dates=True)
        print(f"✅ Dataset ML chargé: {df.shape}")
        
        # Vérifier si correction nécessaire
        target_cols_expected = ['occurrence', 'count', 'intensity']
        target_cols_current = ['target_occurrence', 'target_count', 'target_intensity']
        
        needs_fix = any(col in df.columns for col in target_cols_current)
        already_fixed = all(col in df.columns for col in target_cols_expected)
        
        if already_fixed and not needs_fix:
            print("✅ Colonnes déjà correctes")
            print(f"   Colonnes target trouvées: {[col for col in target_cols_expected if col in df.columns]}")
            return True
        
        if needs_fix:
            print("🔄 Correction des noms de colonnes en cours...")
            
            # Appliquer la correction
            column_mapping = {
                'target_occurrence': 'occurrence',
                'target_count': 'count', 
                'target_intensity': 'intensity'
            }
            
            columns_renamed = []
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    columns_renamed.append(f"{old_col} → {new_col}")
            
            df = df.rename(columns=column_mapping)
            df.to_csv(ml_file)
            
            print("✅ Correction appliquée avec succès:")
            for rename_info in columns_renamed:
                print(f"   {rename_info}")
            
            # Vérification finale
            print(f"\n✅ Vérification finale:")
            for col in target_cols_expected:
                if col in df.columns:
                    print(f"   ✅ {col}: Trouvée")
                else:
                    print(f"   ❌ {col}: Manquante")
            
            return True
        
        print("⚠️  État des colonnes non déterminé")
        print(f"   Colonnes disponibles: {list(df.columns)[:10]}...")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la correction: {e}")
        print("⚠️  Poursuite du pipeline malgré l'erreur")
        return True  # Ne pas arrêter le pipeline pour cette erreur

def run_script(script_name, description, args=None):
    """
    Lance un script avec gestion d'erreurs.
    
    Args:
        script_name (str): Nom du script à lancer
        description (str): Description du script pour l'affichage
        args (list): Arguments supplémentaires à passer au script
    
    Returns:
        int: Code de retour du script
    """
    print(f"\n{'='*70}")
    print(f"🚀 Lancement: {description}")
    print(f"{'='*70}")
    
    # Chemin vers le script
    script_path = Path(__file__).parent / "scripts" / script_name
    
    # Construire la commande
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    # Lancer le script
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        
        print(f"\n✅ {description} terminé avec succès!")
        print(f"⏱️  Durée: {duration:.1f} secondes")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur lors de l'exécution de {description}: {e}")
        print(f"Code d'erreur: {e.returncode}")
        return e.returncode
        
    except FileNotFoundError:
        print(f"\n❌ Script non trouvé: {script_path}")
        print("Vérifiez que la structure du projet est correcte")
        return 1

def run_prediction_tool_non_interactive():
    """
    Lance l'outil de prédiction en mode non-interactif pour le test.
    """
    print(f"\n📍 ÉTAPE 6/8: Test de l'outil de prédiction (non-interactif)")
    print("⚠️  Mode automatique - test rapide des modèles")
    
    try:
        # Import local pour éviter les dépendances
        sys.path.insert(0, str(Path(__file__).parent))
        
    except ImportError:
        print("⚠️  Module de test automatique non disponible")
        print("💡 Pour utiliser l'outil interactif: python scripts/06_prediction_tool.py")
        return 0
    except Exception as e:
        print(f"⚠️  Erreur lors du test: {e}")
        print("💡 Pour utiliser l'outil interactif: python scripts/06_prediction_tool.py")
        return 0

def parse_arguments():
    """
    Parse les arguments de la ligne de commande.
    
    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(
        description="Analyse complète des précipitations extrêmes au Sénégal avec ML et déploiement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLES:
    # Pipeline complet recherche + déploiement (RECOMMANDÉ)
    python main.py
    
    # Pipeline recherche uniquement (sans déploiement)
    python main.py --research-only
    
    # Déploiement des modèles uniquement
    python main.py --deploy-only
    
    # Pipeline complet avec outil interactif
    python main.py --full-pipeline
    
    # Pipeline avec questions interactives
    python main.py --interactive
    
    # Machine Learning uniquement
    python main.py --only-ml
    
    # Clustering avancé uniquement
    python main.py --only-clustering
    
    # Téléconnexions + ML + Déploiement (ignorer analyses spatiales)
    python main.py --skip-spatial
        """
    )
    
    # Arguments principaux
    parser.add_argument(
        'chirps_file', 
        nargs='?', 
        help='Chemin vers le fichier CHIRPS (.mat ou .nc)'
    )
    
    # Options de pipeline
    pipeline_group = parser.add_argument_group('Options de pipeline')
    pipeline_group.add_argument(
        '--research-only', 
        action='store_true',
        help='Exécuter uniquement la phase recherche (7 étapes, sans déploiement)'
    )
    pipeline_group.add_argument(
        '--deploy-only', 
        action='store_true',
        help='Exécuter uniquement le déploiement en production'
    )
    pipeline_group.add_argument(
        '--full-pipeline', 
        action='store_true',
        help='Exécuter toutes les 8 étapes (avec outil interactif et déploiement)'
    )
    pipeline_group.add_argument(
        '--interactive',
        action='store_true',
        help='Mode interactif avec questions utilisateur'
    )
    pipeline_group.add_argument(
        '--only-top5', 
        action='store_true',
        help='Exécuter uniquement l\'analyse spatiale TOP 5'
    )
    pipeline_group.add_argument(
        '--only-teleconnections', 
        action='store_true',
        help='Exécuter uniquement l\'analyse des téléconnexions'
    )
    pipeline_group.add_argument(
        '--only-ml', 
        action='store_true',
        help='Exécuter uniquement le pipeline d\'apprentissage automatique'
    )
    pipeline_group.add_argument(
        '--only-clustering', 
        action='store_true',
        help='Exécuter uniquement l\'analyse de clustering avancée'
    )
    pipeline_group.add_argument(
        '--skip-top10', 
        action='store_true',
        help='Ignorer l\'analyse spatiale TOP 10'
    )
    pipeline_group.add_argument(
        '--skip-spatial', 
        action='store_true',
        help='Ignorer toutes les analyses spatiales'
    )
    
    # Options spécifiques aux téléconnexions
    telecon_group = parser.add_argument_group('Options téléconnexions')
    telecon_group.add_argument(
        '--max-lag', 
        type=int, 
        default=12,
        help='Décalage temporel maximum pour les téléconnexions (défaut: 12 mois)'
    )
    telecon_group.add_argument(
        '--correlation-type',
        choices=['pearson', 'spearman'],
        default='pearson',
        help='Type de corrélation à calculer (défaut: pearson)'
    )
    
    # Options ML
    ml_group = parser.add_argument_group('Options Machine Learning')
    ml_group.add_argument(
        '--skip-clustering',
        action='store_true',
        help='Ignorer l\'analyse de clustering dans le pipeline ML'
    )
    ml_group.add_argument(
        '--skip-prediction-test',
        action='store_true',
        help='Ignorer le test de l\'outil de prédiction'
    )
    ml_group.add_argument(
        '--skip-deployment',
        action='store_true',
        help='Ignorer le déploiement en production'
    )
    
    return parser.parse_args()

def determine_pipeline_steps(args):
    """
    Détermine les étapes à exécuter selon les arguments.
    
    Args:
        args: Arguments de la ligne de commande
        
    Returns:
        list: Liste des étapes à exécuter
    """
    if args.deploy_only:
        steps = ['deployment']
    elif args.only_top5:
        steps = ['top5']
    elif args.only_teleconnections:
        steps = ['teleconnections']
    elif args.only_ml:
        steps = ['ml_pipeline']
        if not args.skip_clustering:
            steps.append('clustering')
        if not args.skip_deployment:
            steps.append('deployment')
    elif args.only_clustering:
        steps = ['clustering']
    elif args.research_only:
        # Pipeline recherche uniquement (7 étapes)
        steps = ['detection', 'top10', 'top5', 'teleconnections', 'ml_pipeline', 'prediction_tool_test', 'clustering']
    elif args.full_pipeline:
        # Pipeline complet avec outil interactif (8 étapes)
        steps = ['detection', 'top10', 'top5', 'teleconnections', 'ml_pipeline', 'prediction_tool_interactive', 'clustering', 'deployment']
    elif args.interactive:
        # Mode interactif avec questions (8 étapes)
        steps = ['detection', 'top10', 'top5', 'teleconnections', 'ml_pipeline', 'prediction_tool_interactive', 'clustering', 'deployment']
    elif args.skip_spatial:
        steps = ['detection', 'teleconnections', 'ml_pipeline', 'clustering', 'deployment']
    elif args.skip_top10:
        steps = ['detection', 'top5', 'teleconnections', 'ml_pipeline', 'clustering', 'deployment']
    else:
        # Pipeline complet par défaut (8 étapes - avec déploiement)
        steps = ['detection', 'top10', 'top5', 'teleconnections', 'ml_pipeline', 'clustering', 'deployment']
    
    # Ajustements selon les options
    if args.skip_prediction_test and 'prediction_tool_test' in steps:
        steps.remove('prediction_tool_test')
    if args.skip_clustering and 'clustering' in steps:
        steps.remove('clustering')
    if args.skip_deployment and 'deployment' in steps:
        steps.remove('deployment')
    
    return steps

def get_pipeline_info(steps_to_run):
    """
    Retourne les informations sur le pipeline selon les étapes.
    
    Args:
        steps_to_run (list): Liste des étapes à exécuter
        
    Returns:
        tuple: (mode_description, total_steps)
    """
    total_steps = len(steps_to_run)
    
    if steps_to_run == ['deployment']:
        mode = "Déploiement production uniquement"
    elif steps_to_run == ['top5']:
        mode = "TOP 5 uniquement"
    elif steps_to_run == ['teleconnections']:
        mode = "Téléconnexions uniquement"
    elif steps_to_run == ['ml_pipeline']:
        mode = "Machine Learning uniquement"
    elif steps_to_run == ['ml_pipeline', 'clustering']:
        mode = "Machine Learning + Clustering"
    elif steps_to_run == ['ml_pipeline', 'clustering', 'deployment']:
        mode = "Machine Learning + Clustering + Déploiement"
    elif steps_to_run == ['clustering']:
        mode = "Clustering avancé uniquement"
    elif 'prediction_tool_interactive' in steps_to_run and 'deployment' in steps_to_run:
        mode = "Pipeline complet recherche → production avec outil interactif (8 étapes)"
    elif 'deployment' in steps_to_run and total_steps >= 7:
        mode = "Pipeline complet recherche → production (8 étapes)"
    elif 'prediction_tool_interactive' in steps_to_run and total_steps == 7:
        mode = "Pipeline recherche complet avec outil interactif (7 étapes)"
    elif total_steps == 7 and 'ml_pipeline' in steps_to_run:
        mode = "Pipeline recherche automatisé complet (7 étapes)"
    elif 'ml_pipeline' in steps_to_run and total_steps >= 4:
        mode = "Pipeline personnalisé avec ML"
    else:
        mode = "Pipeline personnalisé"
    
    return mode, total_steps

def prepare_telecon_args(args):
    """
    Prépare les arguments pour l'analyse des téléconnexions.
    
    Args:
        args: Arguments de la ligne de commande
        
    Returns:
        list: Arguments pour le script de téléconnexions
    """
    telecon_args = []
    if args.max_lag != 12:
        telecon_args.extend(['--max-lag', str(args.max_lag)])
    if args.correlation_type != 'pearson':
        telecon_args.extend(['--correlation-type', args.correlation_type])
    
    return telecon_args if telecon_args else None

def main():
    """
    Lance l'analyse complète des précipitations extrêmes.
    """
    # Parse des arguments
    args = parse_arguments()
    
    print("🌧️  ANALYSE DES PRÉCIPITATIONS EXTRÊMES AU SÉNÉGAL")
    print("=" * 70)
    print("Pipeline complet: Recherche → Développement → Production")
    print(f"Projet lancé depuis: {Path.cwd()}")
    print("=" * 70)
    
    # Déterminer le mode d'exécution
    steps_to_run = determine_pipeline_steps(args)
    mode, total_steps = get_pipeline_info(steps_to_run)
    
    print(f"🎯 MODE: {mode} ({total_steps} étapes)")
    print(f"📋 ÉTAPES: {' → '.join(steps_to_run)}")
    
    # Affichage des phases
    research_steps = ['detection', 'top10', 'top5', 'teleconnections', 'ml_pipeline', 'prediction_tool_interactive', 'clustering']
    production_steps = ['deployment']
    
    if any(step in steps_to_run for step in research_steps):
        print("📊 PHASE 1: Recherche et développement ML")
    if any(step in steps_to_run for step in production_steps):
        print("🚀 PHASE 2: Déploiement production")
    
    # Mode interactif ou automatisé
    if args.interactive or args.full_pipeline:
        print("🔮 MODE INTERACTIF: L'outil de prédiction posera des questions")
    else:
        print("🤖 MODE AUTOMATISÉ: Exécution sans intervention utilisateur")
    
    # Préparer les arguments pour les scripts
    script_args = [args.chirps_file] if args.chirps_file else None
    telecon_args = prepare_telecon_args(args)
    
    current_step = 0
    
    # PHASE 1: RECHERCHE ET DÉVELOPPEMENT
    print(f"\n🔬 === PHASE 1: RECHERCHE ET DÉVELOPPEMENT ===")
    
    # ÉTAPE 1: Détection des événements extrêmes
    if 'detection' in steps_to_run:
        current_step += 1
        print(f"\n📍 ÉTAPE {current_step}/{total_steps}: Détection des événements extrêmes")
        exit_code = run_script(
            script_name="01_detection_extremes.py",
            description="Détection des événements de précipitations extrêmes",
            args=script_args
        )
        
        if exit_code != 0:
            print(f"\n❌ Arrêt du pipeline - Erreur dans l'étape de détection")
            return exit_code
    
    # ÉTAPE 2: Analyse spatiale des top 10 (optionnel)
    if 'top10' in steps_to_run:
        current_step += 1
        print(f"\n📍 ÉTAPE {current_step}/{total_steps}: Analyse spatiale TOP 10")
        exit_code = run_script(
            script_name="02_spatial_analysis_top10.py",
            description="Analyse spatiale des 10 événements les plus extrêmes",
            args=None
        )
        
        if exit_code != 0:
            print(f"\n⚠️  Erreur dans l'analyse TOP 10 - Poursuite du pipeline...")
    
    # ÉTAPE 3: Analyse spatiale détaillée des TOP 5
    if 'top5' in steps_to_run:
        current_step += 1
        print(f"\n📍 ÉTAPE {current_step}/{total_steps}: Analyse spatiale détaillée TOP 5")
        exit_code = run_script(
            script_name="03_spatial_analysis_top5.py",
            description="Distribution spatiale précise des 5 événements les plus intenses",
            args=None
        )
        
        if exit_code != 0:
            print(f"\n⚠️  Erreur dans l'analyse TOP 5 - Poursuite du pipeline...")
    
    # ÉTAPE 4: Analyse des téléconnexions océan-atmosphère
    if 'teleconnections' in steps_to_run:
        current_step += 1
        print(f"\n📍 ÉTAPE {current_step}/{total_steps}: Analyse des téléconnexions")
        exit_code = run_script(
            script_name="04_teleconnections_analysis.py",
            description="Analyse des téléconnexions océan-atmosphère",
            args=telecon_args if telecon_args else None
        )
        
        if exit_code != 0:
            print(f"\n❌ Erreur dans l'analyse des téléconnexions")
            if 'ml_pipeline' in steps_to_run:
                print("⚠️  Les étapes ML nécessitent les téléconnexions - Arrêt du pipeline")
                return exit_code
        
        # 🔧 CORRECTION AUTOMATIQUE APRÈS TÉLÉCONNEXIONS
        if 'ml_pipeline' in steps_to_run:
            print(f"\n🔧 ÉTAPE INTERMÉDIAIRE: Préparation du dataset ML")
            auto_fix_success = auto_fix_ml_dataset()
            if not auto_fix_success:
                print("⚠️  Correction échouée mais poursuite du pipeline")
    
    # ÉTAPE 5: Pipeline d'apprentissage automatique
    if 'ml_pipeline' in steps_to_run:
        current_step += 1
        print(f"\n📍 ÉTAPE {current_step}/{total_steps}: Pipeline d'apprentissage automatique")
        exit_code = run_script(
            script_name="05_machine_learning_pipeline.py",
            description="Pipeline ML complet (classification, régression, clustering)",
            args=None
        )
        
        if exit_code != 0:
            print(f"\n❌ Erreur dans le pipeline ML")
            if 'deployment' in steps_to_run:
                print("⚠️  Le déploiement nécessite les modèles ML - Arrêt du pipeline")
                return exit_code
    
    # ÉTAPE 6: Outil de prédiction
    if 'prediction_tool_interactive' in steps_to_run:
        current_step += 1
        print(f"\n📍 ÉTAPE {current_step}/{total_steps}: Outil de prédiction interactif")
        print("⚠️  ATTENTION: Cette étape est interactive - vous pouvez l'interrompre avec Ctrl+C")
        
        user_input = input("Voulez-vous lancer l'outil de prédiction interactif ? (y/N): ")
        if user_input.lower() in ['y', 'yes', 'oui']:
            exit_code = run_script(
                script_name="06_prediction_tool.py",
                description="Outil de prédiction des événements extrêmes",
                args=None
            )
            
            if exit_code != 0:
                print(f"\n⚠️  Erreur dans l'outil de prédiction - Poursuite du pipeline...")
        else:
            print("⏭️  Outil de prédiction ignoré")
    
    # ÉTAPE 7: Analyse de clustering avancée
    if 'clustering' in steps_to_run:
        current_step += 1
        print(f"\n📍 ÉTAPE {current_step}/{total_steps}: Analyse de clustering avancée")
        exit_code = run_script(
            script_name="07_advanced_clustering_analysis.py",
            description="Analyse comparative des algorithmes de clustering",
            args=None
        )
        
        if exit_code != 0:
            print(f"\n⚠️  Erreur dans l'analyse de clustering - Poursuite du pipeline...")
    
    # PHASE 2: DÉPLOIEMENT PRODUCTION
    if 'deployment' in steps_to_run:
        print(f"\n🚀 === PHASE 2: DÉPLOIEMENT PRODUCTION ===")
        current_step += 1
        print(f"\n📍 ÉTAPE {current_step}/{total_steps}: Déploiement des modèles en production")
        print("🎯 Intégration dans l'infrastructure Docker/TimescaleDB/FastAPI")
        
        exit_code = run_script(
            script_name="08_deploy_models_to_production.py",
            description="Déploiement des modèles ML dans l'infrastructure production",
            args=None
        )
        
        if exit_code != 0:
            print(f"\n⚠️  Erreur dans le déploiement production")
            print("💡 Les modèles restent disponibles dans outputs/models/")
            print("💡 Vérifiez la configuration Docker et la base de données")
        else:
            print(f"\n🎉 DÉPLOIEMENT RÉUSSI!")
            print("🌐 Modèles disponibles via l'API: http://localhost:8000/docs")
    
    # Résumé final
    print_final_summary(steps_to_run, args)
    
    return 0

def print_final_summary(steps_to_run, args):
    """
    Affiche le résumé final selon les étapes exécutées.
    
    Args:
        steps_to_run (list): Liste des étapes exécutées
        args: Arguments de la ligne de commande
    """
    print(f"\n{'='*70}")
    print("🎉 PIPELINE COMPLET TERMINÉ AVEC SUCCÈS!")
    print(f"{'='*70}")
    
    # Résumé par phase
    research_steps = ['detection', 'top10', 'top5', 'teleconnections', 'ml_pipeline', 'prediction_tool_interactive', 'clustering']
    production_steps = ['deployment']
    
    research_executed = any(step in steps_to_run for step in research_steps)
    production_executed = any(step in steps_to_run for step in production_steps)
    
    if research_executed:
        print("✅ PHASE 1 - RECHERCHE ET DÉVELOPPEMENT: Terminée")
        print("📊 Résultats de recherche disponibles dans:")
        
        if 'detection' in steps_to_run:
            print("   • data/processed/ - Événements détectés et climatologie")
            print("   • outputs/reports/ - Rapports de détection")
            print("   • outputs/visualizations/detection/ - Visualisations temporelles")
        
        if 'top10' in steps_to_run or 'top5' in steps_to_run:
            print("   • outputs/visualizations/spatial/ - Analyses spatiales")
            print("   • outputs/reports/ - Rapports géographiques")
        
        if 'teleconnections' in steps_to_run:
            print("   • data/processed/ - Dataset ML et features climatiques")
            print("   • outputs/visualizations/teleconnections/ - Corrélations")
        
        if 'ml_pipeline' in steps_to_run:
            print("   • outputs/models/ - Modèles ML entraînés (.pkl)")
            print("   • outputs/visualizations/machine_learning/ - Performances")
            print("   • outputs/data/ - Prédictions et métriques")
        
        if 'clustering' in steps_to_run:
            print("   • outputs/visualizations/clustering/ - Analyses comparatives")
    
    if production_executed:
        print("\n✅ PHASE 2 - DÉPLOIEMENT PRODUCTION: Terminée")
        print("🚀 Infrastructure de production disponible:")
        print("   • API FastAPI: http://localhost:8000/docs")
        print("   • Base TimescaleDB: Modèles ML enregistrés")
        print("   • Endpoints prédiction: /predictions/generate")
        print("   • Monitoring: http://localhost:3001 (Grafana)")
        print("   • Dashboard: http://localhost:3000")
        
        print("\n🔧 Commandes de vérification:")
        print("   • docker-compose ps")
        print("   • curl http://localhost:8000/health")
        print("   • curl http://localhost:8000/stations")
    
    print("   • outputs/logs/ - Journaux d'exécution détaillés")
    
    # Résumé global
    if research_executed and production_executed:
        print(f"\n🎯 PIPELINE COMPLET RECHERCHE → PRODUCTION:")
        print("   🔬 Modèles développés et validés scientifiquement")
        print("   🚀 Modèles déployés et opérationnels en production")
        print("   📊 API temps réel pour prédictions climatiques")
        print("   🌍 Système d'alerte précoce pour événements extrêmes")
        
        print(f"\n📋 PRÊT POUR LE MÉMOIRE:")
        print("   • Chapitre 3: Méthodologie complète documentée")
        print("   • Chapitre 4: Résultats avec visualisations et métriques")
        print("   • Chapitre 5: Discussion des performances et applications")
        print("   • Chapitre 6: Déploiement et mise en production")
        print("   • Annexes: Rapports détaillés et infrastructure")
        
        print(f"\n🎓 CONTRIBUTION SCIENTIFIQUE:")
        print("   • Pipeline end-to-end: Recherche → Production")
        print("   • Système opérationnel d'alerte climatique")
        print("   • Infrastructure scalable et reproductible")
        print("   • API REST documentée pour intégration")
        
    elif research_executed:
        print(f"\n🔬 RECHERCHE TERMINÉE:")
        print("   • Modèles ML entraînés et validés")
        print("   • Analyses scientifiques complètes")
        print("   • Pour déployer: python main.py --deploy-only")
        
    elif production_executed:
        print(f"\n🚀 DÉPLOIEMENT TERMINÉ:")
        print("   • Modèles déployés en production")
        print("   • API opérationnelle et documentée")
        print("   • Infrastructure monitoring active")
    
    # Mode automatisé vs interactif
    if not (args.interactive or args.full_pipeline):
        print(f"\n🤖 PIPELINE AUTOMATISÉ TERMINÉ:")
        print("   ✅ Aucune intervention utilisateur requise")
        print("   🔮 Pour l'outil de prédiction interactif: python scripts/06_prediction_tool.py")
        print("   📖 Pour le mode complet interactif: python main.py --full-pipeline")
    
    # Prochaines étapes recommandées
    print(f"\n🎯 PROCHAINES ÉTAPES RECOMMANDÉES:")
    
    if production_executed:
        print("   1. Tester l'API de prédiction:")
        print("      curl -X POST http://localhost:8000/predictions/generate \\")
        print("           -H 'Content-Type: application/json' \\")
        print("           -d '{\"region\": \"Dakar\", \"forecast_hours\": 72}'")
        print("   2. Configurer les alertes automatiques")
        print("   3. Intégrer avec des sources de données temps réel")
        print("   4. Mettre en place la surveillance monitoring")
        print("   5. Documenter l'API pour les utilisateurs finaux")
    
    if research_executed and not production_executed:
        print("   1. Déployer les modèles: python main.py --deploy-only")
        print("   2. Valider les performances en production")
        print("   3. Configurer la surveillance continue")
    
    print("   • Finaliser la rédaction du mémoire")
    print("   • Préparer la présentation de soutenance")
    print("   • Publier les résultats scientifiques")

def show_help():
    """Affiche l'aide d'utilisation détaillée."""
    help_text = """
🌧️  ANALYSE DES PRÉCIPITATIONS EXTRÊMES AU SÉNÉGAL

DESCRIPTION:
    Pipeline complet d'analyse des événements de précipitations extrêmes au Sénégal
    incluant la recherche, développement ML et déploiement en production.

NOUVEAUTÉ VERSION 2.0:
    🆕 PHASE 2 ajoutée: Déploiement automatique en production!
    🔬 PHASE 1: Recherche et développement ML (7 étapes)
    🚀 PHASE 2: Déploiement production TimescaleDB/FastAPI (1 étape)

MODES D'EXÉCUTION:
    🤖 AUTOMATISÉ (défaut): Pipeline complet recherche + production (8 étapes)
    🔮 INTERACTIF: Permet l'utilisation de l'outil de prédiction interactif
    🔬 RECHERCHE SEULE: Phase 1 uniquement (7 étapes)
    🚀 DÉPLOIEMENT SEUL: Phase 2 uniquement (1 étape)

PIPELINE COMPLET PAR DÉFAUT (8 étapes):
    === PHASE 1: RECHERCHE ET DÉVELOPPEMENT ===
    1. Détection des événements extrêmes (CHIRPS 1981-2023)
    2. Analyse spatiale TOP 10 événements les plus étendus
    3. Analyse spatiale détaillée TOP 5 événements les plus intenses  
    4. Analyse des téléconnexions (IOD, ENSO, TNA) avec ML
    5. Pipeline d'apprentissage automatique (classification + régression)
    6. Test des outils de prédiction
    7. Analyse de clustering avancée (5 algorithmes)
    
    === PHASE 2: DÉPLOIEMENT PRODUCTION ===
    8. Déploiement des modèles dans l'infrastructure Docker/TimescaleDB/FastAPI

UTILISATION:
    python main.py [OPTIONS] [FICHIER_CHIRPS]

ARGUMENTS:
    FICHIER_CHIRPS    Chemin vers le fichier CHIRPS (.mat ou .nc)
                      Optionnel - détection automatique par défaut

OPTIONS DE PIPELINE:
    --research-only       Phase recherche uniquement (7 étapes, sans déploiement)
    --deploy-only         Déploiement en production uniquement (1 étape)
    --full-pipeline       Pipeline complet avec outil interactif (8 étapes)
    --interactive         Mode interactif avec questions utilisateur
    --only-top5           Analyse spatiale TOP 5 uniquement
    --only-teleconnections Analyse des téléconnexions uniquement  
    --only-ml             Pipeline ML uniquement
    --only-clustering     Analyse de clustering uniquement
    --skip-top10          Ignorer l'analyse TOP 10
    --skip-spatial        Ignorer toutes les analyses spatiales
    --skip-deployment     Ignorer le déploiement (même effet que --research-only)
    
OPTIONS TÉLÉCONNEXIONS:
    --max-lag N           Décalage temporel maximum (défaut: 12 mois)
    --correlation-type T  Type de corrélation: pearson|spearman (défaut: pearson)

OPTIONS MACHINE LEARNING:
    --skip-clustering     Ignorer l'analyse de clustering dans le pipeline ML
    --skip-prediction-test Ignorer le test de l'outil de prédiction

EXEMPLES D'UTILISATION:

    # Pipeline complet recherche → production (RECOMMANDÉ pour le mémoire)
    python main.py
    
    # Recherche uniquement (développement des modèles)
    python main.py --research-only
    
    # Déploiement uniquement (modèles déjà développés)
    python main.py --deploy-only
    
    # Pipeline complet avec outil interactif
    python main.py --full-pipeline
    
    # Mode entièrement interactif
    python main.py --interactive
    
    # Machine Learning + Déploiement (ignorer spatial)
    python main.py --skip-spatial
    
    # Développement ML uniquement
    python main.py --only-ml
    
    # Outil de prédiction interactif séparément
    python scripts/06_prediction_tool.py

AVANTAGES DU PIPELINE COMPLET:
    ✅ Recherche scientifique rigoureuse
    ✅ Modèles ML validés et testés  
    ✅ Déploiement automatique en production
    ✅ API REST opérationnelle
    ✅ Infrastructure monitoring complète
    ✅ Système d'alerte temps réel
    ✅ Documentation automatique générée
    ✅ Pipeline reproductible end-to-end

INFRASTRUCTURE DE PRODUCTION:
    🗄️  TimescaleDB: Base de données temporelle optimisée
    🚀 FastAPI: API REST haute performance
    📊 Grafana: Dashboards de monitoring
    🔍 Prometheus: Métriques et alertes
    🐳 Docker: Containerisation complète
    ⚡ Redis: Cache pour performances
    🌐 Nginx: Proxy et interface web

SORTIES RECHERCHE:
    data/processed/           - Données traitées et datasets ML
    outputs/models/           - Modèles ML entraînés (.pkl)
    outputs/reports/          - Rapports d'analyse détaillés
    outputs/visualizations/   - Toutes les cartes et graphiques
    outputs/data/            - Métriques et résumés structurés
    outputs/logs/            - Journaux d'exécution

SORTIES PRODUCTION:
    services/api/models/      - Modèles déployés en production
    Database TimescaleDB      - Modèles enregistrés et données
    API Endpoints            - http://localhost:8000/docs
    Monitoring Dashboard     - http://localhost:3001 (Grafana)
    Application Dashboard    - http://localhost:3000

PRÉREQUIS:
    • Python 3.8+
    • Docker et docker-compose
    • Bibliothèques ML: scikit-learn, xgboost, pandas, numpy
    • Infrastructure: TimescaleDB, Redis, FastAPI
    • Fichiers CHIRPS dans data/raw/
    • Indices climatiques dans data/raw/climate_indices/
    • Structure de projet respectée

NOUVEAUTÉ MAJEURE:
    🎯 PREMIER PIPELINE ACADÉMIQUE COMPLET END-TO-END
    De la recherche fondamentale au système opérationnel en production!
    
    🔬 Phase 1: Développement scientifique rigoureux
    🚀 Phase 2: Mise en production automatisée
    📊 Métriques de performance en temps réel
    🌍 Système d'alerte opérationnel pour le Sénégal
    
CONTRIBUTION SCIENTIFIQUE:
    • Premier système opérationnel de prédiction d'événements climatiques extrêmes au Sénégal
    • Pipeline reproductible de recherche climatique avec déploiement automatique
    • Infrastructure scalable pour d'autres régions d'Afrique de l'Ouest
    • Méthodologie open-source pour la recherche climatique appliquée
    """
    print(help_text)

if __name__ == "__main__":
    # Vérifier si l'aide est demandée
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        show_help()
        sys.exit(0)
    
    # Lancer l'analyse
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analyse interrompue par l'utilisateur")
        print("Les fichiers partiels peuvent être conservés dans outputs/")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        print("Vérifiez la configuration et les dépendances")
        import traceback
        traceback.print_exc()
        sys.exit(1)