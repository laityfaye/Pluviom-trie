#!/usr/bin/env python3
# fix_dashboard.py
"""
Script de diagnostic et correction pour les problèmes d'affichage des visualisations
dans le dashboard ClimaSen

Usage:
    python fix_dashboard.py [--check-only] [--fix-paths] [--test-api]
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import requests
from typing import List, Dict, Any

class DashboardFixer:
    """Classe pour diagnostiquer et corriger les problèmes du dashboard"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.project_root = self._find_project_root()
        self.outputs_dir = self.project_root / "outputs"
        self.visualizations_dir = self.outputs_dir / "visualizations"
        self.services_dir = self.project_root / "services"
        self.api_dir = self.services_dir / "api"
        
        print(f"🔧 DashboardFixer initialisé")
        print(f"   Project root: {self.project_root}")
        print(f"   Visualizations: {self.visualizations_dir}")
        print(f"   API directory: {self.api_dir}")
    
    def _find_project_root(self) -> Path:
        """Trouve la racine du projet en remontant depuis le script"""
        current = Path(__file__).parent
        
        # Chercher des marqueurs du projet
        markers = ["outputs", "services", "scripts", "src"]
        
        for _ in range(5):  # Remonter maximum 5 niveaux
            if any((current / marker).exists() for marker in markers):
                return current
            current = current.parent
        
        # Si pas trouvé, utiliser le dossier du script
        return Path(__file__).parent
    
    def check_directory_structure(self) -> Dict[str, Any]:
        """Vérifie la structure des dossiers"""
        print("\n📁 DIAGNOSTIC - Structure des dossiers")
        print("=" * 50)
        
        structure = {
            "project_root": {
                "path": str(self.project_root),
                "exists": self.project_root.exists()
            },
            "outputs": {
                "path": str(self.outputs_dir),
                "exists": self.outputs_dir.exists()
            },
            "visualizations": {
                "path": str(self.visualizations_dir),
                "exists": self.visualizations_dir.exists()
            },
            "services": {
                "path": str(self.services_dir),
                "exists": self.services_dir.exists()
            },
            "api": {
                "path": str(self.api_dir),
                "exists": self.api_dir.exists()
            }
        }
        
        for name, info in structure.items():
            status = "✅" if info["exists"] else "❌"
            print(f"  {status} {name:<15}: {info['path']}")
        
        return structure
    
    def scan_visualizations(self) -> Dict[str, Any]:
        """Scanne les visualisations disponibles"""
        print("\n📊 DIAGNOSTIC - Visualisations disponibles")
        print("=" * 50)
        
        scan_results = {
            "total_images": 0,
            "by_extension": {},
            "by_folder": {},
            "files_list": []
        }
        
        if not self.visualizations_dir.exists():
            print("❌ Dossier visualizations n'existe pas")
            return scan_results
        
        # Extensions d'images supportées
        image_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.pdf'}
        
        # Scanner récursif
        for file_path in self.visualizations_dir.rglob("*.*"):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # Statistiques par extension
                ext = file_path.suffix.lower()
                scan_results["by_extension"][ext] = scan_results["by_extension"].get(ext, 0) + 1
                
                # Statistiques par dossier
                relative_path = file_path.relative_to(self.visualizations_dir)
                folder = str(relative_path.parent) if relative_path.parent.name != '.' else 'root'
                scan_results["by_folder"][folder] = scan_results["by_folder"].get(folder, 0) + 1
                
                # Liste des fichiers
                scan_results["files_list"].append({
                    "name": file_path.name,
                    "path": str(relative_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
                
                scan_results["total_images"] += 1
        
        # Affichage des résultats
        print(f"📈 Total d'images trouvées: {scan_results['total_images']}")
        
        if scan_results["by_extension"]:
            print("\n📊 Par extension:")
            for ext, count in scan_results["by_extension"].items():
                print(f"  {ext}: {count} fichiers")
        
        if scan_results["by_folder"]:
            print("\n📂 Par dossier:")
            for folder, count in scan_results["by_folder"].items():
                print(f"  {folder}: {count} fichiers")
        
        # Afficher quelques exemples de fichiers
        if scan_results["files_list"]:
            print(f"\n📷 Exemples de fichiers (premiers 10):")
            for i, file_info in enumerate(scan_results["files_list"][:10]):
                print(f"  {i+1}. {file_info['name']} ({file_info['size']} bytes)")
        
        return scan_results
    
    def check_api_configuration(self) -> Dict[str, Any]:
        """Vérifie la configuration de l'API"""
        print("\n🔌 DIAGNOSTIC - Configuration API")
        print("=" * 50)
        
        api_status = {
            "main_py_exists": False,
            "visualizations_py_exists": False,
            "main_py_correct": False,
            "visualizations_py_correct": False,
            "paths_configured": False
        }
        
        # Vérifier main.py
        main_py_path = self.api_dir / "main.py"
        if main_py_path.exists():
            api_status["main_py_exists"] = True
            print("✅ main.py existe")
            
            # Vérifier le contenu
            content = main_py_path.read_text(encoding='utf-8')
            if "visualizations" in content.lower():
                api_status["main_py_correct"] = True
                print("✅ main.py contient des références aux visualisations")
            else:
                print("⚠️ main.py ne semble pas inclure le support des visualisations")
        else:
            print("❌ main.py n'existe pas")
        
        # Vérifier visualizations.py
        viz_py_path = self.api_dir / "visualizations.py"
        if viz_py_path.exists():
            api_status["visualizations_py_exists"] = True
            print("✅ visualizations.py existe")
            
            # Vérifier le contenu
            content = viz_py_path.read_text(encoding='utf-8')
            if "def scan_all_visualizations" in content:
                api_status["visualizations_py_correct"] = True
                print("✅ visualizations.py semble correct")
            else:
                print("⚠️ visualizations.py pourrait être incomplet")
        else:
            print("❌ visualizations.py n'existe pas")
        
        return api_status
    
    def test_api_connection(self, api_url: str = "http://localhost:8000") -> Dict[str, Any]:
        """Teste la connexion à l'API"""
        print(f"\n🌐 TEST - Connexion API ({api_url})")
        print("=" * 50)
        
        test_results = {
            "api_available": False,
            "health_check": False,
            "visualizations_endpoint": False,
            "can_list_visualizations": False,
            "visualization_count": 0
        }
        
        try:
            # Test de base
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                test_results["api_available"] = True
                test_results["health_check"] = True
                print("✅ API disponible et healthy")
            else:
                print(f"⚠️ API répond mais statut: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API non disponible: {e}")
            return test_results
        
        # Test endpoint visualisations
        try:
            response = requests.get(f"{api_url}/api/visualizations/list", timeout=5)
            if response.status_code == 200:
                test_results["visualizations_endpoint"] = True
                data = response.json()
                
                if "visualizations" in data:
                    test_results["can_list_visualizations"] = True
                    test_results["visualization_count"] = len(data["visualizations"])
                    print(f"✅ Endpoint visualisations fonctionne: {test_results['visualization_count']} visualisations")
                else:
                    print("⚠️ Endpoint visualisations répond mais format incorrect")
            else:
                print(f"❌ Endpoint visualisations erreur: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur test endpoint visualisations: {e}")
        
        return test_results
    
    def fix_api_files(self) -> bool:
        """Corrige les fichiers de l'API"""
        print("\n🔧 CORRECTION - Fichiers API")
        print("=" * 50)
        
        success = True
        
        # S'assurer que le dossier API existe
        self.api_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer/corriger visualizations.py
        try:
            viz_content = '''# services/api/visualizations.py - VERSION CORRIGÉE
"""
Endpoint API pour la gestion des visualisations
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
import mimetypes
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/visualizations", tags=["visualizations"])

# Configuration des chemins
API_DIR = Path(__file__).parent
PROJECT_ROOT = API_DIR.parent.parent
VISUALIZATIONS_DIR = PROJECT_ROOT / "outputs" / "visualizations"

@router.get("/list")
async def list_visualizations(
    category: Optional[str] = Query(None, description="Filtrer par catégorie"),
    limit: Optional[int] = Query(50, description="Nombre maximum")
):
    """Liste les visualisations disponibles"""
    try:
        if not VISUALIZATIONS_DIR.exists():
            return {"visualizations": [], "total": 0, "error": "Dossier non trouvé"}
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.pdf'}
        visualizations = []
        
        for img_file in VISUALIZATIONS_DIR.rglob("*.*"):
            if img_file.suffix.lower() in image_extensions:
                rel_path = img_file.relative_to(VISUALIZATIONS_DIR)
                visualizations.append({
                    "id": f"viz_{len(visualizations)}",
                    "title": img_file.stem.replace('_', ' ').title(),
                    "description": f"Visualisation: {img_file.stem}",
                    "category": "detection",
                    "image": f"/api/visualizations/serve/{rel_path.as_posix()}",
                    "file_name": img_file.name,
                    "date": datetime.fromtimestamp(img_file.stat().st_mtime).isoformat()
                })
        
        return {
            "visualizations": visualizations[:limit],
            "total": len(visualizations),
            "count": min(len(visualizations), limit)
        }
        
    except Exception as e:
        logger.error(f"Erreur list_visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/serve/{file_path:path}")
async def serve_file(file_path: str):
    """Sert un fichier de visualisation"""
    try:
        full_path = VISUALIZATIONS_DIR / file_path
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        mime_type, _ = mimetypes.guess_type(str(full_path))
        if not mime_type:
            mime_type = "application/octet-stream"
        
        return FileResponse(
            path=str(full_path),
            media_type=mime_type,
            filename=full_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur serve_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check des visualisations"""
    try:
        image_count = 0
        if VISUALIZATIONS_DIR.exists():
            image_count = len([f for f in VISUALIZATIONS_DIR.rglob("*.png")])
        
        return {
            "status": "healthy",
            "visualizations_dir": str(VISUALIZATIONS_DIR),
            "dir_exists": VISUALIZATIONS_DIR.exists(),
            "image_count": image_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}
'''
            
            viz_file = self.api_dir / "visualizations.py"
            viz_file.write_text(viz_content, encoding='utf-8')
            print("✅ visualizations.py créé/corrigé")
            
        except Exception as e:
            print(f"❌ Erreur création visualizations.py: {e}")
            success = False
        
        # Créer un main.py minimal s'il n'existe pas
        main_file = self.api_dir / "main.py"
        if not main_file.exists():
            try:
                main_content = '''# services/api/main.py - VERSION MINIMALE
"""
API FastAPI pour ClimaSen avec support des visualisations
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ClimaSen API",
    description="API pour visualisations climatiques",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check de l'API"""
    return {"status": "healthy", "message": "API ClimaSen opérationnelle"}

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {"message": "ClimaSen API - Visualisations climatiques"}

# Inclure le router des visualisations
try:
    from .visualizations import router as viz_router
    app.include_router(viz_router)
    logger.info("✅ Router visualisations inclus")
except ImportError as e:
    logger.warning(f"⚠️ Impossible d'importer visualizations: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
'''
                
                main_file.write_text(main_content, encoding='utf-8')
                print("✅ main.py créé")
                
            except Exception as e:
                print(f"❌ Erreur création main.py: {e}")
                success = False
        
        return success
    
    def fix_dashboard_config(self) -> bool:
        """Corrige la configuration du dashboard"""
        print("\n🔧 CORRECTION - Configuration Dashboard")
        print("=" * 50)
        
        success = True
        
        # Vérifier le fichier JavaScript du dashboard
        dashboard_js = self.services_dir / "dashboard" / "dist" / "js" / "dashboard.js"
        
        if dashboard_js.exists():
            try:
                content = dashboard_js.read_text(encoding='utf-8')
                
                # Vérifier si l'URL de l'API est correcte
                if "http://localhost:8000/api/visualizations" in content:
                    print("✅ URL API correcte dans dashboard.js")
                else:
                    print("⚠️ URL API pourrait être incorrecte dans dashboard.js")
                    
                    # Optionnel: corriger l'URL
                    # content = content.replace("ancien_url", "http://localhost:8000/api/visualizations")
                    # dashboard_js.write_text(content, encoding='utf-8')
                    
            except Exception as e:
                print(f"❌ Erreur lecture dashboard.js: {e}")
                success = False
        else:
            print("⚠️ dashboard.js non trouvé")
        
        return success
    
    def create_test_images(self) -> bool:
        """Crée des images de test si aucune n'existe"""
        print("\n🎨 CRÉATION - Images de test")
        print("=" * 50)
        
        try:
            # Créer le dossier de detection s'il n'existe pas
            detection_dir = self.visualizations_dir / "detection"
            detection_dir.mkdir(parents=True, exist_ok=True)
            
            # Créer une image de test simple (SVG)
            test_images = [
                "01_distribution_temporelle.svg",
                "02_intensite_couverture.svg", 
                "03_evolution_anomalies.svg",
                "04_distribution_spatiale.svg"
            ]
            
            svg_template = '''<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <circle cx="400" cy="300" r="100" fill="#3b82f6" opacity="0.7"/>
  <text x="400" y="320" text-anchor="middle" font-family="Arial" font-size="24" fill="#1e293b">
    {title}
  </text>
  <text x="400" y="350" text-anchor="middle" font-family="Arial" font-size="14" fill="#64748b">
    Visualisation générée automatiquement
  </text>
  <text x="400" y="380" text-anchor="middle" font-family="Arial" font-size="12" fill="#94a3b8">
    ClimaSen - {date}
  </text>
</svg>'''
            
            created_count = 0
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            for img_name in test_images:
                img_path = detection_dir / img_name
                
                if not img_path.exists():
                    title = img_name.replace('.svg', '').replace('_', ' ').title()
                    title = title.replace('01 ', '').replace('02 ', '').replace('03 ', '').replace('04 ', '')
                    
                    svg_content = svg_template.format(title=title, date=current_date)
                    img_path.write_text(svg_content, encoding='utf-8')
                    created_count += 1
                    print(f"  ✅ Créé: {img_name}")
            
            if created_count > 0:
                print(f"✅ {created_count} images de test créées")
            else:
                print("ℹ️ Images de test déjà présentes")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur création images de test: {e}")
            return False
    
    def run_full_diagnosis(self) -> Dict[str, Any]:
        """Lance un diagnostic complet"""
        print("🔍 DIAGNOSTIC COMPLET DU DASHBOARD CLIMASEN")
        print("=" * 60)
        
        results = {
            "structure": self.check_directory_structure(),
            "visualizations": self.scan_visualizations(),
            "api": self.check_api_configuration(),
            "timestamp": datetime.now().isoformat()
        }
        
        return results
    
    def run_full_fix(self) -> bool:
        """Lance une correction complète"""
        print("🔧 CORRECTION COMPLÈTE DU DASHBOARD CLIMASEN")
        print("=" * 60)
        
        success = True
        
        # 1. Créer la structure de base
        try:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            self.visualizations_dir.mkdir(parents=True, exist_ok=True)
            self.services_dir.mkdir(parents=True, exist_ok=True)
            self.api_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Structure de dossiers créée")
        except Exception as e:
            print(f"❌ Erreur création structure: {e}")
            success = False
        
        # 2. Corriger les fichiers API
        if not self.fix_api_files():
            success = False
        
        # 3. Créer des images de test si nécessaire
        if not self.create_test_images():
            success = False
        
        # 4. Corriger la configuration du dashboard
        if not self.fix_dashboard_config():
            print("⚠️ Avertissement: configuration dashboard non corrigée")
        
        return success

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Diagnostic et correction du dashboard ClimaSen")
    parser.add_argument("--check-only", action="store_true", help="Diagnostic seulement")
    parser.add_argument("--fix-paths", action="store_true", help="Corriger les chemins")
    parser.add_argument("--test-api", action="store_true", help="Tester l'API")
    parser.add_argument("--create-test-images", action="store_true", help="Créer des images de test")
    parser.add_argument("--full-fix", action="store_true", help="Correction complète")
    
    args = parser.parse_args()
    
    fixer = DashboardFixer()
    
    if args.check_only:
        results = fixer.run_full_diagnosis()
        print(f"\n📋 RÉSUMÉ DU DIAGNOSTIC")
        print("=" * 30)
        print(f"Total d'images: {results['visualizations']['total_images']}")
        print(f"API configurée: {results['api']['main_py_exists'] and results['api']['visualizations_py_exists']}")
        
    elif args.test_api:
        fixer.test_api_connection()
        
    elif args.create_test_images:
        fixer.create_test_images()
        
    elif args.full_fix:
        if fixer.run_full_fix():
            print("\n✅ CORRECTION TERMINÉE AVEC SUCCÈS")
            print("=" * 40)
            print("Prochaines étapes:")
            print("1. Démarrer l'API: cd services/api && python main.py")
            print("2. Ouvrir le dashboard: http://localhost:3000")
            print("3. Tester les visualisations")
        else:
            print("\n❌ ERREURS LORS DE LA CORRECTION")
            print("Vérifiez les messages d'erreur ci-dessus")
    
    elif args.fix_paths:
        fixer.fix_api_files()
        
    else:
        # Diagnostic par défaut
        fixer.run_full_diagnosis()
        
        print(f"\n🎯 ACTIONS RECOMMANDÉES")
        print("=" * 30)
        print("1. Pour une correction complète:")
        print("   python fix_dashboard.py --full-fix")
        print("2. Pour tester l'API uniquement:")
        print("   python fix_dashboard.py --test-api")
        print("3. Pour créer des images de test:")
        print("   python fix_dashboard.py --create-test-images")

if __name__ == "__main__":
    main()