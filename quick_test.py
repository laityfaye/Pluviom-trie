#!/usr/bin/env python3
# quick_test.py
"""
Test rapide pour vérifier que tous les modules sont correctement installés.
"""

import sys
from pathlib import Path

def test_imports():
    """Test tous les imports des modules."""
    print("Test des imports des modules...")
    print("="*50)
    
    # Ajouter src au path
    sys.path.insert(0, str(Path("src")))
    
    # Liste des modules à tester
    modules_tests = [
        ("config.settings", ["PROJECT_INFO", "DETECTION_CRITERIA", "SENEGAL_BOUNDS"]),
        ("data.loader", ["ChirpsDataLoader", "load_chirps_senegal"]),
        ("analysis.climatology", ["calculate_climatology_and_anomalies"]),
        ("analysis.detection", ["ExtremeEventDetector", "detect_extreme_precipitation_events_final"]),
        ("utils.season_classifier", ["SeasonClassifier", "classify_seasons_senegal_final"]),
        ("visualization.detection_plots", ["DetectionVisualizer"]),
        ("reports.detection_report", ["DetectionReportGenerator"])
    ]
    
    results = []
    
    for module_name, items in modules_tests:
        try:
            module = __import__(module_name, fromlist=items)
            
            # Vérifier que les éléments existent
            missing_items = []
            for item in items:
                if not hasattr(module, item):
                    missing_items.append(item)
            
            if missing_items:
                print(f"❌ {module_name}: Manque {missing_items}")
                results.append(False)
            else:
                print(f"✅ {module_name}: OK")
                results.append(True)
                
        except ImportError as e:
            print(f"❌ {module_name}: Erreur d'import - {e}")
            results.append(False)
        except Exception as e:
            print(f"❌ {module_name}: Erreur inattendue - {e}")
            results.append(False)
    
    return all(results)

def test_file_structure():
    """Vérifie la structure des fichiers."""
    print("\nTest de la structure des fichiers...")
    print("="*50)
    
    required_files = [
        "src/config/settings.py",
        "src/data/loader.py", 
        "src/analysis/climatology.py",
        "src/analysis/detection.py",
        "src/utils/season_classifier.py",
        "src/visualization/detection_plots.py",
        "src/reports/detection_report.py",
        "scripts/01_detection_extremes.py",
        "main.py",
        "requirements.txt"
    ]
    
    required_dirs = [
        "src", "scripts", "data/raw", "data/processed",
        "outputs/data", "outputs/visualizations", "outputs/reports"
    ]
    
    all_good = True
    
    print("Fichiers requis:")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MANQUANT")
            all_good = False
    
    print("\nDossiers requis:")
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MANQUANT")
            all_good = False
    
    return all_good

def test_dependencies():
    """Teste les dépendances Python."""
    print("\nTest des dépendances Python...")
    print("="*50)
    
    dependencies = [
        "numpy", "pandas", "matplotlib", "seaborn", 
        "scipy", "h5py", "tqdm", "pathlib"
    ]
    
    all_good = True
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - NON INSTALLÉ")
            all_good = False
    
    return all_good

def main():
    """Fonction principale du test."""
    print("TEST RAPIDE DU PROJET")
    print("="*80)
    
    # Tests
    structure_ok = test_file_structure()
    deps_ok = test_dependencies() 
    imports_ok = test_imports()
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)
    print(f"Structure des fichiers: {'✅ OK' if structure_ok else '❌ PROBLÈME'}")
    print(f"Dépendances Python: {'✅ OK' if deps_ok else '❌ PROBLÈME'}")
    print(f"Imports des modules: {'✅ OK' if imports_ok else '❌ PROBLÈME'}")
    
    if all([structure_ok, deps_ok, imports_ok]):
        print("\n🎉 TOUS LES TESTS PASSENT!")
        print("Vous pouvez lancer l'analyse avec: python main.py")
        return 0
    else:
        print("\n⚠️  CERTAINS TESTS ÉCHOUENT")
        
        if not structure_ok:
            print("- Créez les fichiers manquants avec le contenu des artifacts")
        if not deps_ok:
            print("- Installez les dépendances: pip install -r requirements.txt")
        if not imports_ok:
            print("- Vérifiez le contenu des modules et corrigez les erreurs")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)