#!/usr/bin/env python3
# main.py
"""
Point d'entrée principal du projet d'analyse des précipitations extrêmes au Sénégal.

Ce fichier permet de lancer facilement l'analyse complète depuis la racine du projet.
Il exécute séquentiellement :
1. La détection des événements extrêmes (01_detection_extremes.py)
2. L'analyse spatiale des top 10 événements (02_spatial_analysis_top10.py)
3. L'analyse spatiale détaillée des TOP 5 les plus intenses (03_spatial_analysis_top5.py)
4. L'analyse des téléconnexions océan-atmosphère (04_teleconnections_analysis.py)

Utilisation:
    python main.py
    python main.py /chemin/vers/chirps.mat
    python main.py --only-top5  # Analyse TOP 5 uniquement
    python main.py --only-teleconnections  # Téléconnexions uniquement
    python main.py --skip-spatial  # Détection + Téléconnexions

Auteur: [Laity FAYE]
Date: [01/06/2025]
"""

import sys
import subprocess
from pathlib import Path
import time
import argparse

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
    print(f"\n{'='*60}")
    print(f"🚀 Lancement: {description}")
    print(f"{'='*60}")
    
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

def parse_arguments():
    """
    Parse les arguments de la ligne de commande.
    
    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(
        description="Analyse complète des précipitations extrêmes au Sénégal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLES:
    # Pipeline complet (4 étapes)
    python main.py
    
    # Pipeline complet avec fichier CHIRPS spécifique
    python main.py data/mon_fichier_chirps.mat
    
    # Analyse TOP 5 uniquement
    python main.py --only-top5
    
    # Téléconnexions uniquement
    python main.py --only-teleconnections
    
    # Détection + Téléconnexions (ignorer analyses spatiales)
    python main.py --skip-spatial
    
    # Téléconnexions avec lag personnalisé
    python main.py --only-teleconnections --max-lag 18
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
    
    return parser.parse_args()

def main():
    """
    Lance l'analyse complète des précipitations extrêmes.
    """
    # Parse des arguments
    args = parse_arguments()
    
    print("🌧️  ANALYSE DES PRÉCIPITATIONS EXTRÊMES AU SÉNÉGAL")
    print("=" * 70)
    print("Analyse automatisée avec apprentissage automatique")
    print(f"Projet lancé depuis: {Path.cwd()}")
    print("=" * 70)
    
    # Déterminer le mode d'exécution
    if args.only_top5:
        mode = "TOP 5 uniquement"
        total_steps = 1
        steps_to_run = ['top5']
    elif args.only_teleconnections:
        mode = "Téléconnexions uniquement"
        total_steps = 1
        steps_to_run = ['teleconnections']
    elif args.skip_spatial:
        mode = "Détection + Téléconnexions"
        total_steps = 2
        steps_to_run = ['detection', 'teleconnections']
    elif args.skip_top10:
        mode = "Détection + TOP 5 + Téléconnexions"
        total_steps = 3
        steps_to_run = ['detection', 'top5', 'teleconnections']
    else:
        mode = "Pipeline complet"
        total_steps = 4
        steps_to_run = ['detection', 'top10', 'top5', 'teleconnections']
    
    print(f"🎯 MODE: {mode} ({total_steps} étapes)")
    
    # Préparer les arguments pour les scripts
    script_args = [args.chirps_file] if args.chirps_file else None
    telecon_args = []
    if args.max_lag != 12:
        telecon_args.extend(['--max-lag', str(args.max_lag)])
    if args.correlation_type != 'pearson':
        telecon_args.extend(['--correlation-type', args.correlation_type])
    
    current_step = 0
    
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
            return exit_code
    
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
    print("🎉 ANALYSE COMPLÈTE TERMINÉE AVEC SUCCÈS!")
    print(f"{'='*70}")
    print("📊 Résultats disponibles dans:")
    
    # Résultats selon les étapes exécutées
    if 'detection' in steps_to_run:
        print("   • data/processed/ - Événements détectés et climatologie")
        print("   • outputs/reports/ - Rapports de détection")
        print("   • outputs/visualizations/detection/ - Visualisations temporelles")
    
    if 'top10' in steps_to_run:
        print("   • outputs/visualizations/spatial/ - Analyses spatiales TOP 10")
        print("   • outputs/reports/ - Rapports géographiques TOP 10")
    
    if 'top5' in steps_to_run:
        print("   • outputs/visualizations/spatial_top5/ - Analyse détaillée TOP 5 ⭐")
        print("   • outputs/data/ - Données spatiales TOP 5")
    
    if 'teleconnections' in steps_to_run:
        print("   • data/processed/ - Dataset ML et features climatiques")
        print("   • outputs/visualizations/teleconnections/ - Corrélations et téléconnexions")
        print("   • outputs/reports/ - Rapport de téléconnexions")
    
    print("   • outputs/logs/ - Journaux d'exécution détaillés")
    
    # Résumé spécifique selon le mode
    if 'teleconnections' in steps_to_run:
        print(f"\n🌊 TÉLÉCONNEXIONS - FICHIERS GÉNÉRÉS:")
        print("   📊 climate_indices_combined.csv - Indices climatiques (IOD, Nino34, TNA)")
        print(f"   📊 climate_features_lag{args.max_lag}.csv - Features avec décalages")
        print("   📊 ml_dataset_teleconnections.csv - Dataset prêt pour ML")
        print("   📄 rapport_teleconnexions.txt - Analyse complète des corrélations")
        print("   🗺️  correlation_heatmap_lags.png - Heatmap des corrélations")
        print("   📈 detailed_lag_correlations.png - Corrélations détaillées")
        print("   🌍 seasonal_teleconnections_comparison.png - Comparaison saisonnière")
        
        print(f"\n🤖 PRÊT POUR LE MACHINE LEARNING:")
        print("   • Variables cibles: occurrence, count, intensity")
        print(f"   • Features climatiques: {3 * (args.max_lag + 1)} variables avec lags")
        print("   • Période d'analyse: 1981-2023 (42+ ans)")
        print("   • Téléconnexions quantifiées: IOD, ENSO (Nino34), TNA")
    
    if 'top5' in steps_to_run:
        print(f"\n🎯 TOP 5 - FICHIERS GÉNÉRÉS:")
        print("   📊 spatial_analysis_top5_intense.csv - Métriques détaillées")
        print("   📄 spatial_summary_top5_intense.json - Résumé structuré")
        print("   📋 rapport_spatial_top5.txt - Rapport géographique complet")
        print("   🗺️  5 cartes individuelles + analyses comparatives")
    
    # Recommandations selon les étapes
    if 'teleconnections' in steps_to_run:
        print(f"\n🚀 PROCHAINES ÉTAPES RECOMMANDÉES:")
        print("   • Développement des modèles ML (Random Forest, XGBoost, SVM)")
        print("   • Validation croisée temporelle")
        print("   • Optimisation des hyperparamètres")
        print("   • Tests de performance prédictive")
        print("   • Développement du système opérationnel de prévision")

def show_help():
    """Affiche l'aide d'utilisation détaillée."""
    help_text = """
🌧️  ANALYSE DES PRÉCIPITATIONS EXTRÊMES AU SÉNÉGAL

DESCRIPTION:
    Pipeline complet d'analyse des événements de précipitations extrêmes au Sénégal
    incluant la détection, l'analyse spatiale et l'étude des téléconnexions océan-atmosphère.

PIPELINE COMPLET (4 étapes):
    1. Détection des événements extrêmes (CHIRPS 1981-2023)
    2. Analyse spatiale TOP 10 événements les plus étendus
    3. Analyse spatiale détaillée TOP 5 événements les plus intenses  
    4. Analyse des téléconnexions (IOD, ENSO, TNA) avec ML

UTILISATION:
    python main.py [OPTIONS] [FICHIER_CHIRPS]

ARGUMENTS:
    FICHIER_CHIRPS    Chemin vers le fichier CHIRPS (.mat ou .nc)
                      Optionnel - détection automatique par défaut

OPTIONS DE PIPELINE:
    --only-top5           Analyse spatiale TOP 5 uniquement
    --only-teleconnections Analyse des téléconnexions uniquement  
    --skip-top10          Ignorer l'analyse TOP 10
    --skip-spatial        Ignorer toutes les analyses spatiales
    
OPTIONS TÉLÉCONNEXIONS:
    --max-lag N           Décalage temporel maximum (défaut: 12 mois)
    --correlation-type T  Type de corrélation: pearson|spearman (défaut: pearson)

EXEMPLES:
    # Pipeline complet (recommandé)
    python main.py
    
    # Pipeline avec fichier CHIRPS spécifique
    python main.py /chemin/vers/chirps_data.mat
    
    # Téléconnexions uniquement (après détection)
    python main.py --only-teleconnections
    
    # Téléconnexions avec lag étendu
    python main.py --only-teleconnections --max-lag 18
    
    # Détection + Téléconnexions (ignorer spatial)
    python main.py --skip-spatial
    
    # Analyse TOP 5 uniquement
    python main.py --only-top5

SORTIES:
    data/processed/           - Données traitées et datasets ML
    outputs/reports/          - Rapports d'analyse détaillés
    outputs/visualizations/   - Toutes les cartes et graphiques
    outputs/data/            - Métriques et résumés structurés
    outputs/logs/            - Journaux d'exécution

PRÉREQUIS:
    • Python 3.8+
    • Fichiers CHIRPS dans data/raw/
    • Indices climatiques dans data/raw/climate_indices/
    • Structure de projet respectée
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