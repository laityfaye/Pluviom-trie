# services/api/visualizations.py - VERSION CORRIGÉE AVEC URLs ABSOLUES
"""
Endpoint API pour la gestion des visualisations générées par les scripts ML
VERSION CORRIGÉE POUR URLs ABSOLUES ET CORS
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import FileResponse
from pathlib import Path
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import mimetypes
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/visualizations", tags=["visualizations"])

# ============================================================================
# CONFIGURATION DES CHEMINS - VERSION CORRIGÉE POUR VOTRE STRUCTURE
# ============================================================================

# Déterminer la racine du projet
API_DIR = Path(__file__).parent  # services/api/
PROJECT_ROOT = API_DIR.parent.parent  # Remonte à la racine du projet

# CHEMINS ADAPTÉS À VOTRE STRUCTURE RÉELLE
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Créer les dossiers s'ils n'existent pas
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

# DETECTION AUTOMATIQUE DU CHEMIN RÉEL (Fallback pour Windows)
def detect_real_visualization_path():
    """Détecte automatiquement le vrai chemin des visualisations"""
    possible_paths = [
        VISUALIZATIONS_DIR,  # Chemin relatif standard
        Path("outputs/visualizations"),  # Chemin relatif direct
        Path("./outputs/visualizations"),  # Chemin relatif avec ./
        Path("../outputs/visualizations"),  # Un niveau au-dessus
        Path("../../outputs/visualizations"),  # Deux niveaux au-dessus
    ]
    
    # Vérifier chaque chemin possible
    for path in possible_paths:
        if path.exists():
            logger.info(f"✅ Chemin visualizations trouvé: {path.absolute()}")
            return path.absolute()
    
    # Si aucun trouvé, créer le chemin standard
    logger.warning(f"⚠️ Aucun chemin trouvé, création de: {VISUALIZATIONS_DIR}")
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return VISUALIZATIONS_DIR

# Utiliser la détection automatique
REAL_VISUALIZATIONS_DIR = detect_real_visualization_path()

# Log de configuration
logger.info(f"🔧 Configuration des chemins (VERSION CORRIGÉE):")
logger.info(f"   API_DIR: {API_DIR}")
logger.info(f"   PROJECT_ROOT: {PROJECT_ROOT}")
logger.info(f"   VISUALIZATIONS_DIR detecté: {REAL_VISUALIZATIONS_DIR}")
logger.info(f"   Exists: {REAL_VISUALIZATIONS_DIR.exists()}")

# STRUCTURE DES CATÉGORIES ADAPTÉE À VOS IMAGES
CATEGORY_MAPPING = {
    "detection": {
        "path": "detection",
        "label": "Détection Événements",
        "icon": "🎯",
        "description": "Détection et analyse des événements de précipitations extrêmes",
        "scripts": ["01_detection_extremes.py"]
    },
    "spatial": {
        "path": "spatial",
        "label": "Analyse Spatiale",
        "icon": "🗺️",
        "description": "Analyses spatiales et cartographiques",
        "scripts": ["02_spatial_analysis_top10.py", "03_spatial_analysis_top5.py"]
    },
    "temporal": {
        "path": "temporal",
        "label": "Analyse Temporelle", 
        "icon": "📈",
        "description": "Évolution temporelle et tendances",
        "scripts": ["temporal_analysis.py"]
    },
    "machine-learning": {
        "path": "machine_learning",
        "label": "Machine Learning",
        "icon": "🤖", 
        "description": "Modèles d'apprentissage automatique et prédictions",
        "scripts": ["05_machine_learning_pipeline.py"]
    },
    "clustering": {
        "path": "clustering",
        "label": "Clustering",
        "icon": "🎭",
        "description": "Analyse de clustering et classification",
        "scripts": ["07_advanced_clustering_analysis.py"]
    },
    "teleconnections": {
        "path": "teleconnections", 
        "label": "Téléconnexions",
        "icon": "🌊",
        "description": "Corrélations et téléconnexions climatiques",
        "scripts": ["04_teleconnections_analysis.py"]
    }
}

class VisualizationScanner:
    """Scanner adapté pour votre structure de fichiers"""
    
    def __init__(self):
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.svg', '.pdf'}
        self.base_dir = REAL_VISUALIZATIONS_DIR
        
        # SCAN INITIAL POUR DEBUG
        self._initial_scan_debug()
        
    def _initial_scan_debug(self):
        """Scan initial pour diagnostiquer la structure"""
        logger.info(f"🔍 Scanner initialisé pour: {self.base_dir}")
        
        if not self.base_dir.exists():
            logger.error(f"❌ Dossier de base n'existe pas: {self.base_dir}")
            return
            
        # Scan récursif complet
        all_files = list(self.base_dir.rglob("*.*"))
        image_files = [f for f in all_files if f.suffix.lower() in self.supported_formats]
        
        logger.info(f"📊 Statistiques du scan:")
        logger.info(f"   Total fichiers: {len(all_files)}")
        logger.info(f"   Fichiers images: {len(image_files)}")
        
        # Lister les premières images trouvées
        for i, file in enumerate(image_files[:10]):
            rel_path = file.relative_to(self.base_dir)
            logger.info(f"   📷 {i+1}. {rel_path}")
            
        # Analyser la structure des dossiers
        self._analyze_folder_structure()
    
    def _analyze_folder_structure(self):
        """Analyse la structure des dossiers"""
        if not self.base_dir.exists():
            return
            
        subfolders = [d for d in self.base_dir.iterdir() if d.is_dir()]
        logger.info(f"📁 Sous-dossiers trouvés: {len(subfolders)}")
        
        for folder in subfolders:
            images_count = len([f for f in folder.rglob("*.*") if f.suffix.lower() in self.supported_formats])
            logger.info(f"   📂 {folder.name}: {images_count} images")
    
    def scan_all_visualizations(self, category_filter: str = None, base_url: str = "http://localhost:8000") -> List[Dict]:
        """Scan toutes les visualisations avec URLs ABSOLUES"""
        logger.info(f"🔍 Début scan complet - filtre: {category_filter}, base_url: {base_url}")
        visualizations = []
        
        if not self.base_dir.exists():
            logger.error(f"❌ Dossier de base n'existe pas: {self.base_dir}")
            return []
        
        # Scan récursif de tous les fichiers images
        for image_file in self.base_dir.rglob("*.*"):
            if not (image_file.is_file() and image_file.suffix.lower() in self.supported_formats):
                continue
                
            # Déterminer la catégorie
            detected_category = self._detect_category_from_path(image_file)
            
            # Appliquer le filtre si nécessaire
            if category_filter and category_filter != detected_category:
                continue
            
            # Créer les métadonnées avec URL ABSOLUE
            viz_data = self._create_visualization_metadata(detected_category, image_file, base_url)
            if viz_data:
                visualizations.append(viz_data)
                logger.info(f"   ✅ Ajouté: {image_file.name} ({detected_category}) -> {viz_data['image']}")
        
        # Trier par date de modification (plus récent d'abord)
        visualizations.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        logger.info(f"🎯 Scan terminé: {len(visualizations)} visualisations trouvées")
        return visualizations
    
    def _detect_category_from_path(self, file_path: Path) -> str:
        """Détecte la catégorie basée sur le chemin et le nom du fichier"""
        relative_path = file_path.relative_to(self.base_dir)
        path_str = str(relative_path).lower()
        filename = file_path.stem.lower()
        
        # 1. Vérifier si dans un sous-dossier de catégorie connue
        for category, info in CATEGORY_MAPPING.items():
            if info["path"] in path_str:
                return category
        
        # 2. Détection par mots-clés dans le nom de fichier
        keyword_mapping = {
            "detection": ["distribution", "temporelle", "intensite", "couverture", "evolution", "anomalies", "spatiale"],
            "spatial": ["spatial", "carte", "map", "synthesis", "comparative", "reference", "senegal"],
            "temporal": ["temporal", "evolution", "timeline", "series", "trend", "chronologique"],
            "machine-learning": ["model", "comparison", "learning", "feature", "importance", "confusion", "accuracy"],
            "clustering": ["clustering", "cluster", "pca", "hierarchical", "kmeans", "dendrogram"],
            "teleconnections": ["correlation", "heatmap", "teleconnection", "lag", "seasonal", "indices"]
        }
        
        for category, keywords in keyword_mapping.items():
            if any(keyword in filename for keyword in keywords):
                return category
        
        # 3. Catégorie par défaut basée sur votre structure
        if any(num in filename for num in ["01_", "02_", "03_", "04_"]):
            return "detection"
            
        return "detection"  # Catégorie par défaut
    
    def _create_visualization_metadata(self, category: str, file_path: Path, base_url: str) -> Optional[Dict]:
        """Crée les métadonnées complètes d'une visualisation avec URL ABSOLUE"""
        try:
            # Informations du fichier
            stat_info = file_path.stat()
            file_size = stat_info.st_size
            modified_time = datetime.fromtimestamp(stat_info.st_mtime)
            
            # Générer titre et description
            title = self._generate_title_from_filename(file_path.stem)
            description = self._generate_description_from_filename(file_path.stem, category)
            
            # URL ABSOLUE de l'image - CORRECTION CRUCIALE
            relative_path = file_path.relative_to(self.base_dir)
            image_url = f"{base_url}/api/visualizations/serve/{relative_path.as_posix()}"
            
            return {
                "id": f"{category}_{file_path.stem}",
                "title": title,
                "description": description,
                "category": category,
                "image": image_url,  # URL ABSOLUE
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_size,
                "file_format": file_path.suffix.lower().replace('.', ''),
                "date": modified_time.isoformat(),
                "date_formatted": modified_time.strftime("%Y-%m-%d %H:%M"),
                "script": self._identify_source_script(file_path.stem, category),
                "category_info": CATEGORY_MAPPING.get(category, CATEGORY_MAPPING["detection"])
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur création métadonnées pour {file_path}: {e}")
            return None
    
    def _generate_title_from_filename(self, filename: str) -> str:
        """Génère un titre basé sur le nom de fichier réel"""
        # Mapping spécifique pour vos fichiers
        title_mapping = {
            "01_distribution_temporelle": "Distribution Temporelle des Événements",
            "02_intensite_couverture": "Intensité et Couverture Spatiale", 
            "03_evolution_anomalies": "Évolution des Anomalies Climatiques",
            "04_distribution_spatiale": "Distribution Spatiale des Événements",
        }
        
        if filename in title_mapping:
            return title_mapping[filename]
        
        # Nettoyage générique du nom de fichier
        title = filename.replace('_', ' ').title()
        
        # Enlever les numéros de préfixe
        if title.startswith(('01 ', '02 ', '03 ', '04 ', '05 ', '06 ', '07 ', '08 ', '09 ')):
            title = title[3:]
            
        return title
    
    def _generate_description_from_filename(self, filename: str, category: str) -> str:
        """Génère une description basée sur le fichier et la catégorie"""
        descriptions = {
            "01_distribution_temporelle": "Analyse de la distribution temporelle des événements de précipitations extrêmes détectés au Sénégal par l'algorithme de détection.",
            "02_intensite_couverture": "Visualisation de l'intensité et de la couverture spatiale des événements extrêmes avec métriques quantitatives.",
            "03_evolution_anomalies": "Évolution temporelle des anomalies climatiques et identification des tendances à long terme.",
            "04_distribution_spatiale": "Distribution géographique et analyse spatiale des événements extrêmes sur le territoire sénégalais.",
        }
        
        if filename in descriptions:
            return descriptions[filename]
        
        # Description par défaut basée sur la catégorie
        category_info = CATEGORY_MAPPING.get(category, CATEGORY_MAPPING["detection"])
        return f"{category_info['description']} - Visualisation générée par analyse automatisée."
    
    def _identify_source_script(self, filename: str, category: str) -> str:
        """Identifie le script source"""
        # Scripts spécifiques basés sur vos noms de fichiers
        if filename.startswith(('01_', '02_', '03_', '04_')):
            return "01_detection_extremes.py"
        
        category_scripts = {
            "detection": "01_detection_extremes.py",
            "spatial": "02_spatial_analysis_top10.py",
            "temporal": "04_teleconnections_analysis.py",
            "machine-learning": "05_machine_learning_pipeline.py", 
            "clustering": "07_advanced_clustering_analysis.py",
            "teleconnections": "04_teleconnections_analysis.py"
        }
        
        return category_scripts.get(category, "unknown")

# Instance globale du scanner
scanner = VisualizationScanner()

# ============================================================================
# UTILITY FUNCTIONS POUR GÉNÉRER L'URL DE BASE
# ============================================================================

def get_base_url(request: Request) -> str:
    """Génère l'URL de base à partir de la requête"""
    # Prendre l'host de la requête ou utiliser localhost:8000 par défaut
    host = request.headers.get("host", "localhost:8000")
    scheme = "https" if "https" in str(request.url) else "http"
    
    # Si c'est une requête depuis le dashboard (port 3000), forcer localhost:8000
    if ":3000" in host:
        return "http://localhost:8000"
    
    return f"{scheme}://{host}"

# ============================================================================
# ENDPOINTS CORRIGÉS AVEC URLs ABSOLUES
# ============================================================================

@router.get("/list")
async def list_visualizations(
    request: Request,  # AJOUTÉ: Request pour récupérer l'URL de base
    category: Optional[str] = Query(None, description="Filtrer par catégorie"),
    limit: Optional[int] = Query(50, description="Nombre maximum de résultats"),
    offset: Optional[int] = Query(0, description="Décalage pour pagination")
):
    """Liste toutes les visualisations disponibles - VERSION CORRIGÉE AVEC URLs ABSOLUES"""
    try:
        # Générer l'URL de base
        base_url = get_base_url(request)
        logger.info(f"📋 Requête API: category={category}, limit={limit}, offset={offset}, base_url={base_url}")
        
        # Scanner toutes les visualisations avec URL de base
        all_visualizations = scanner.scan_all_visualizations(category, base_url)
        
        # Pagination
        total = len(all_visualizations)
        visualizations = all_visualizations[offset:offset + limit]
        
        logger.info(f"✅ API Response: {total} total, {len(visualizations)} returned")
        
        # Log des URLs pour debug
        if visualizations:
            logger.info(f"🖼️ Exemple URL image: {visualizations[0]['image']}")
        
        return {
            "visualizations": visualizations,
            "total": total,
            "count": len(visualizations),
            "offset": offset,
            "limit": limit,
            "categories": list(CATEGORY_MAPPING.keys()),
            "debug_info": {
                "base_url": base_url,
                "base_dir": str(REAL_VISUALIZATIONS_DIR),
                "dir_exists": REAL_VISUALIZATIONS_DIR.exists(),
                "scan_successful": True
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du scan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du scan: {str(e)}")

@router.get("/serve/{file_path:path}")
async def serve_visualization_file(file_path: str):
    """Sert un fichier de visualisation - ENDPOINT CORRIGÉ"""
    try:
        logger.info(f"📁 Demande fichier: {file_path}")
        
        # Construire le chemin complet avec nettoyage
        clean_path = file_path.replace('\\', '/').strip('/')
        full_path = REAL_VISUALIZATIONS_DIR / clean_path
        
        logger.info(f"   Chemin nettoyé: {clean_path}")
        logger.info(f"   Chemin complet: {full_path}")
        logger.info(f"   Exists: {full_path.exists()}")
        
        if not full_path.exists():
            # Essayer quelques variantes du chemin
            alternatives = [
                REAL_VISUALIZATIONS_DIR / file_path,
                REAL_VISUALIZATIONS_DIR / Path(file_path).name,  # Juste le nom de fichier
            ]
            
            for alt_path in alternatives:
                if alt_path.exists():
                    full_path = alt_path
                    logger.info(f"   ✅ Trouvé avec chemin alternatif: {full_path}")
                    break
            else:
                logger.error(f"❌ Fichier non trouvé: {file_path}")
                raise HTTPException(status_code=404, detail=f"Fichier non trouvé: {file_path}")
        
        if not full_path.is_file():
            raise HTTPException(status_code=404, detail="Le chemin ne pointe pas vers un fichier")
        
        # Déterminer le type MIME
        mime_type, _ = mimetypes.guess_type(str(full_path))
        if not mime_type:
            mime_type = "application/octet-stream"
        
        logger.info(f"✅ Serving file: {full_path} (type: {mime_type})")
        
        return FileResponse(
            path=str(full_path),
            media_type=mime_type,
            filename=full_path.name,
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache 1 heure
                "Access-Control-Allow-Origin": "*",  # CORS pour le dashboard
                "Access-Control-Allow-Methods": "GET",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur serve_file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lecture fichier: {str(e)}")

@router.get("/categories")
async def get_categories():
    """Retourne la liste des catégories disponibles"""
    return {
        "categories": CATEGORY_MAPPING,
        "total_categories": len(CATEGORY_MAPPING)
    }

@router.get("/health")
async def health_check():
    """Health check avec diagnostic complet"""
    try:
        # Scan rapide pour les statistiques
        stats = {}
        if REAL_VISUALIZATIONS_DIR.exists():
            for category, info in CATEGORY_MAPPING.items():
                cat_path = REAL_VISUALIZATIONS_DIR / info["path"]
                if cat_path.exists():
                    images = [f for f in cat_path.rglob("*.*") if f.suffix.lower() in scanner.supported_formats]
                    stats[category] = len(images)
                else:
                    stats[category] = 0
        
        # Images dans le dossier racine
        root_images = []
        if REAL_VISUALIZATIONS_DIR.exists():
            root_images = [f.name for f in REAL_VISUALIZATIONS_DIR.rglob("*.*") 
                          if f.suffix.lower() in scanner.supported_formats]
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "base_directory": str(REAL_VISUALIZATIONS_DIR),
                "directory_exists": REAL_VISUALIZATIONS_DIR.exists(),
                "supported_formats": list(scanner.supported_formats)
            },
            "statistics": {
                "categories": stats,
                "total_images": sum(stats.values()),
                "root_images_sample": root_images[:10]  # Premiers 10 fichiers
            },
            "categories_configured": list(CATEGORY_MAPPING.keys()),
            "cors_enabled": True,
            "absolute_urls": True
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }