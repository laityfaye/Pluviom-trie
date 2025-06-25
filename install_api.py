#!/usr/bin/env python3
"""
Script d'installation automatique pour l'API FastAPI
Résout les problèmes de dépendances et configure l'environnement
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Exécute une commande avec gestion d'erreur"""
    print(f"🔄 {description}")
    print(f"   Commande: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Succès")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Erreur")
        print(f"   Error: {e.stderr.strip() if e.stderr else str(e)}")
        return False

def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ requis")
        return False
    
    print("✅ Version Python compatible")
    return True

def install_dependencies():
    """Installe les dépendances étape par étape"""
    
    # 1. Mettre à jour pip
    if not run_command("python -m pip install --upgrade pip", "Mise à jour de pip"):
        return False
    
    # 2. Installer email-validator en premier (problème principal)
    if not run_command("pip install email-validator>=2.0.0", "Installation email-validator"):
        return False
    
    # 3. Installer FastAPI et uvicorn
    if not run_command("pip install fastapi==0.104.1 uvicorn[standard]==0.24.0", "Installation FastAPI"):
        return False
    
    # 4. Installer Pydantic
    if not run_command("pip install pydantic==2.5.0 pydantic-settings==2.1.0", "Installation Pydantic"):
        return False
    
    # 5. Bases de données
    if not run_command("pip install asyncpg==0.29.0 sqlalchemy[asyncio]==2.0.23", "Installation DB"):
        return False
    
    # 6. Redis
    if not run_command("pip install redis[hiredis]==5.0.1", "Installation Redis"):
        return False
    
    # 7. Outils scientifiques
    if not run_command("pip install numpy==1.24.4 pandas==2.0.3", "Installation outils scientifiques"):
        return False
    
    # 8. Autres dépendances
    if not run_command("pip install aiofiles==23.2.1 python-multipart==0.0.6 python-dotenv==1.0.0", "Installation utilitaires"):
        return False
    
    # 9. Optionnel: Pillow pour les images
    run_command("pip install Pillow==10.1.0", "Installation Pillow (optionnel)")
    
    print("✅ Toutes les dépendances installées")
    return True

def create_startup_script():
    """Crée un script de démarrage"""
    
    startup_content = """#!/usr/bin/env python3
# start_api.py - Script de démarrage de l'API

import sys
import os
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH pour les imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # Vérifier que les modules sont disponibles
    import fastapi
    import uvicorn
    print("✅ Modules FastAPI détectés")
    
    # Importer et lancer l'application
    from main import app
    print("✅ Application importée")
    
    if __name__ == "__main__":
        print("🚀 Démarrage de l'API...")
        print("📍 URL: http://localhost:8000")
        print("📖 Documentation: http://localhost:8000/docs")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Exécutez d'abord: python install_api.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)
"""
    
    with open("start_api.py", "w", encoding="utf-8") as f:
        f.write(startup_content)
    
    print("✅ Script de démarrage créé: start_api.py")

def create_env_file():
    """Crée un fichier .env par défaut"""
    
    env_content = """# Configuration API FastAPI
API_DEBUG=true
CACHE_TTL=300

# Base de données (PostgreSQL/TimescaleDB)
DATABASE_URL=postgresql://postgres:secure_password@localhost:5432/climatsn_db

# Cache Redis
REDIS_URL=redis://localhost:6379

# Fichiers statiques
STATIC_FILES_DIR=../../outputs

# Logging
LOG_LEVEL=INFO
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w", encoding="utf-8") as f:
            f.write(env_content)
        print("✅ Fichier .env créé")
    else:
        print("ℹ️ Fichier .env existant conservé")

def main():
    """Fonction principale d'installation"""
    
    print("🔧 Installation de l'API FastAPI Climat Sénégal")
    print("=" * 50)
    
    # 1. Vérifier Python
    if not check_python_version():
        sys.exit(1)
    
    # 2. Installer les dépendances
    print("\n📦 Installation des dépendances...")
    if not install_dependencies():
        print("\n❌ Échec de l'installation des dépendances")
        sys.exit(1)
    
    # 3. Créer les scripts utilitaires
    print("\n📝 Création des scripts...")
    create_startup_script()
    create_env_file()
    
    # 4. Test d'import
    print("\n🧪 Test de l'installation...")
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ Test d'import réussi")
    except ImportError as e:
        print(f"❌ Test d'import échoué: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ Installation terminée avec succès!")
    print("\n🚀 Pour démarrer l'API:")
    print("   python start_api.py")
    print("\n🌐 URLs importantes:")
    print("   • API: http://localhost:8000")
    print("   • Documentation: http://localhost:8000/docs")
    print("   • Santé: http://localhost:8000/health")
    print("   • Visualisations: http://localhost:8000/api/visualizations/list")
    print("\n📁 Fichiers créés:")
    print("   • start_api.py - Script de démarrage")
    print("   • .env - Configuration")
    print("\n💡 Note: Assurez-vous que PostgreSQL et Redis sont démarrés pour toutes les fonctionnalités")

if __name__ == "__main__":
    main()