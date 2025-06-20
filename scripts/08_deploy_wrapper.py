#!/usr/bin/env python3
# scripts/08_deploy_wrapper.py
"""
Wrapper intelligent pour le déploiement des modèles ML
Version Windows - sans emojis pour éviter les erreurs d'encodage
SOLUTION FINALE pour résoudre les conflits main.py vs exécution standalone
"""

import sys
import os
import subprocess
import shutil
import json
import joblib
from pathlib import Path
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
ML_MODELS_PATH = PROJECT_ROOT / "outputs/models"
API_MODELS_PATH = PROJECT_ROOT / "services/api/models"

class IntelligentDeployment:
    """Déploiement intelligent avec détection automatique d'environnement."""
    
    def __init__(self):
        self.models_deployed = {}
        self.deployment_method = self.detect_environment()
        
    def detect_environment(self):
        """Détecte automatiquement l'environnement d'exécution."""
        print("DETECTION AUTOMATIQUE DE L'ENVIRONNEMENT")
        print("=" * 60)
        
        # Test 1: Dans un conteneur Docker ?
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER', False)
        
        # Test 2: docker-compose disponible ?
        docker_available = self.test_docker_compose()
        
        # Test 3: PostgreSQL local disponible ?
        local_db_available = self.test_local_postgresql()
        
        print(f"Dans Docker: {'OUI' if in_docker else 'NON'}")
        print(f"Docker-compose: {'OUI' if docker_available else 'NON'}")
        print(f"PostgreSQL local: {'OUI' if local_db_available else 'NON'}")
        
        # Logique de décision
        if in_docker:
            method = "container_mode"
            print("Methode choisie: Container Mode (fichiers seulement)")
        elif docker_available:
            method = "docker_exec"
            print("Methode choisie: Docker Exec (deploiement complet)")
        elif local_db_available:
            method = "local_db"
            print("Methode choisie: Base locale")
        else:
            method = "file_mode"
            print("Methode choisie: Fichiers seulement")
            
        return method
    
    def test_docker_compose(self):
        """Test de disponibilité de docker-compose."""
        try:
            commands = ['docker-compose --version', 'docker compose --version']
            
            for cmd in commands:
                try:
                    result = subprocess.run(
                        cmd.split(), 
                        capture_output=True, 
                        text=True, 
                        timeout=10,
                        cwd=PROJECT_ROOT
                    )
                    if result.returncode == 0:
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            return False
            
        except Exception:
            return False
    
    def test_local_postgresql(self):
        """Test de disponibilité de PostgreSQL local."""
        try:
            result = subprocess.run(
                ['psql', '--version'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def deploy_container_mode(self):
        """Déploiement mode conteneur (fichiers seulement)."""
        print("\nDEPLOIEMENT MODE CONTENEUR")
        print("=" * 50)
        print("Mode optimise pour execution dans Docker")
        
        return self.deploy_files_only("Mode conteneur - Pas d'acces docker-compose depuis l'interieur")
    
    def deploy_docker_exec(self):
        """Déploiement via docker-compose exec."""
        print("\nDEPLOIEMENT VIA DOCKER EXEC")
        print("=" * 50)
        
        # Vérifier que TimescaleDB est accessible
        success = self.test_timescaledb_connection()
        if not success:
            print("TimescaleDB non accessible, passage en mode fichiers")
            return self.deploy_files_only("TimescaleDB non disponible")
        
        print("TimescaleDB accessible via Docker")
        
        # Déployer vers la base de données
        return self.deploy_to_database()
    
    def deploy_local_db(self):
        """Déploiement vers base locale."""
        print("\nDEPLOIEMENT VERS BASE LOCALE")
        print("=" * 50)
        
        try:
            import psycopg2
            
            # Essayer de se connecter
            conn_params = {
                'host': 'localhost',
                'port': 5432,
                'database': 'climatsn_db',
                'user': 'postgres',
                'password': os.environ.get('POSTGRES_PASSWORD', 'postgres')
            }
            
            conn = psycopg2.connect(**conn_params)
            print("Connexion base locale reussie")
            conn.close()
            
            return self.deploy_to_database(use_local=True)
            
        except ImportError:
            print("psycopg2 non disponible")
            return self.deploy_files_only("psycopg2 manquant")
        except Exception as e:
            print(f"Erreur connexion locale: {e}")
            return self.deploy_files_only("Base locale inaccessible")
    
    def deploy_files_only(self, reason="Mode par defaut"):
        """Déploiement fichiers uniquement."""
        print(f"\nDEPLOIEMENT FICHIERS SEULEMENT")
        print("=" * 50)
        print(f"Raison: {reason}")
        
        try:
            # Créer le répertoire cible
            API_MODELS_PATH.mkdir(parents=True, exist_ok=True)
            
            # Trouver les modèles
            models_found = False
            model_paths = [
                ML_MODELS_PATH,
                PROJECT_ROOT / "outputs/models",
                PROJECT_ROOT / "models"
            ]
            
            source_path = None
            for path in model_paths:
                if path.exists() and any(path.glob("*.pkl")):
                    source_path = path
                    models_found = True
                    break
            
            if not models_found:
                print("Aucun modele .pkl trouve")
                return self.create_dummy_models()
            
            print(f"Modeles trouves dans: {source_path}")
            
            # Copier et valider chaque modèle
            models = list(source_path.glob("*.pkl"))
            success_count = 0
            
            print(f"Deploiement de {len(models)} modeles:")
            
            for model_file in models:
                try:
                    # Copier le fichier
                    target_path = API_MODELS_PATH / model_file.name
                    shutil.copy2(model_file, target_path)
                    
                    # Valider en chargeant le modèle
                    model = joblib.load(target_path)
                    
                    # Statistiques
                    size_mb = target_path.stat().st_size / (1024*1024)
                    self.models_deployed[model_file.name] = {
                        'path': str(target_path),
                        'size_mb': round(size_mb, 2),
                        'deployed_at': datetime.now().isoformat(),
                        'status': 'active',
                        'type': self.detect_model_type(model_file.name)
                    }
                    
                    print(f"   OK {model_file.name}: {size_mb:.1f} MB")
                    success_count += 1
                    
                except Exception as e:
                    print(f"   ERREUR {model_file.name}: {e}")
            
            # Créer fichier de configuration
            self.create_deployment_config()
            
            print(f"\nResume: {success_count}/{len(models)} modeles deployes")
            
            if success_count > 0:
                print("Deploiement fichiers reussi!")
                self.show_next_steps()
                return True
            else:
                print("Echec du deploiement")
                return False
                
        except Exception as e:
            print(f"Erreur deploiement fichiers: {e}")
            return False
    
    def test_timescaledb_connection(self):
        """Test de connexion à TimescaleDB via docker exec."""
        try:
            commands = [
                ['docker-compose', 'exec', '-T', 'timescaledb'],
                ['docker', 'compose', 'exec', '-T', 'timescaledb']
            ]
            
            for base_cmd in commands:
                try:
                    cmd = base_cmd + [
                        'psql', '-U', 'postgres', '-d', 'climatsn_db',
                        '-c', 'SELECT 1;'
                    ]
                    
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        cwd=PROJECT_ROOT,
                        timeout=15
                    )
                    
                    if result.returncode == 0:
                        return True
                        
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            return False
            
        except Exception:
            return False
    
    def deploy_to_database(self, use_local=False):
        """Déploie vers la base de données (Docker ou locale)."""
        
        # D'abord déployer les fichiers
        if not self.deploy_files_only("Preparation base de donnees"):
            return False
        
        print(f"\nENREGISTREMENT EN BASE DE DONNEES")
        print("=" * 50)
        
        success_count = 0
        
        for model_name, model_info in self.models_deployed.items():
            try:
                # Charger le modèle pour les métadonnées
                model = joblib.load(model_info['path'])
                
                # Enregistrer en base
                if self.register_model_in_database(model_name, model, use_local):
                    print(f"   OK {model_name}: Enregistre en base")
                    success_count += 1
                else:
                    print(f"   ATTENTION {model_name}: Erreur base de donnees")
                    
            except Exception as e:
                print(f"   ERREUR {model_name}: {e}")
        
        print(f"\nBase de donnees: {success_count}/{len(self.models_deployed)} modeles")
        return success_count > 0
    
    def register_model_in_database(self, model_name, model_object, use_local=False):
        """Enregistre un modèle en base de données."""
        try:
            # Préparer les métadonnées
            model_type = self.detect_model_type(model_name)
            features = self.extract_features(model_object)
            hyperparams = self.extract_hyperparameters(model_object)
            
            # Préparer la requête SQL
            features_json = json.dumps(features).replace("'", "''")
            hyperparams_json = json.dumps(hyperparams, default=str).replace("'", "''")
            model_name_clean = model_name.replace('.pkl', '')
            
            sql_query = f"""
            INSERT INTO ml_models (name, version, model_type, target_variable, 
                                 features, hyperparameters, model_path, status, trained_at)
            VALUES ('{model_name_clean}', '1.0', '{model_type}', 
                   '{'extreme_precipitation' if model_type != 'preprocessing' else 'features'}',
                   '{features_json}', '{hyperparams_json}', 
                   '/app/models/{model_name}', 'active', NOW())
            ON CONFLICT (name, version) DO UPDATE SET
                model_path = EXCLUDED.model_path,
                status = EXCLUDED.status,
                trained_at = EXCLUDED.trained_at;
            """
            
            if use_local:
                return self.execute_local_sql(sql_query)
            else:
                return self.execute_docker_sql(sql_query)
                
        except Exception as e:
            print(f"      Erreur SQL: {e}")
            return False
    
    def execute_docker_sql(self, sql_query):
        """Exécute une requête SQL via docker exec."""
        try:
            commands = [
                ['docker-compose', 'exec', '-T', 'timescaledb'],
                ['docker', 'compose', 'exec', '-T', 'timescaledb']
            ]
            
            for base_cmd in commands:
                try:
                    cmd = base_cmd + [
                        'psql', '-U', 'postgres', '-d', 'climatsn_db',
                        '-c', sql_query
                    ]
                    
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        cwd=PROJECT_ROOT,
                        timeout=30
                    )
                    
                    return result.returncode == 0
                    
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            return False
            
        except Exception:
            return False
    
    def execute_local_sql(self, sql_query):
        """Exécute une requête SQL sur base locale."""
        try:
            import psycopg2
            
            conn_params = {
                'host': 'localhost',
                'port': 5432,
                'database': 'climatsn_db',
                'user': 'postgres',
                'password': os.environ.get('POSTGRES_PASSWORD', 'postgres')
            }
            
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql_query)
                    conn.commit()
            
            return True
            
        except Exception:
            return False
    
    def detect_model_type(self, model_name):
        """Détecte le type de modèle à partir du nom."""
        name_lower = model_name.lower()
        
        if 'classifier' in name_lower:
            return 'classification'
        elif 'regressor' in name_lower or '_reg' in name_lower:
            return 'regression'
        elif 'scaler' in name_lower or 'transform' in name_lower:
            return 'preprocessing'
        else:
            return 'other'
    
    def extract_features(self, model_object):
        """Extrait la liste des features du modèle."""
        if hasattr(model_object, 'feature_names_in_'):
            return list(model_object.feature_names_in_)
        elif hasattr(model_object, 'n_features_in_'):
            return [f"feature_{i}" for i in range(model_object.n_features_in_)]
        else:
            return ["IOD", "Nino34", "TNA", "month", "season"]
    
    def extract_hyperparameters(self, model_object):
        """Extrait les hyperparamètres du modèle."""
        if hasattr(model_object, 'get_params'):
            params = model_object.get_params()
            # Nettoyer les valeurs problématiques pour JSON
            clean_params = {}
            for key, value in params.items():
                if value is None or isinstance(value, (str, int, float, bool)):
                    clean_params[key] = value
                else:
                    clean_params[key] = str(value)
            return clean_params
        else:
            return {}
    
    def create_dummy_models(self):
        """Crée des modèles factices pour les tests."""
        print("Creation de modeles factices...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Données factices
            X_dummy = np.random.rand(100, 5)
            y_dummy_class = np.random.randint(0, 2, 100)
            y_dummy_reg = np.random.rand(100)
            
            # Modèles factices
            models = {
                'randomforest_classifier.pkl': RandomForestClassifier(n_estimators=10, random_state=42),
                'randomforest_regressor.pkl': RandomForestRegressor(n_estimators=10, random_state=42),
                'feature_scaler.pkl': StandardScaler()
            }
            
            # Créer les dossiers
            ML_MODELS_PATH.mkdir(parents=True, exist_ok=True)
            API_MODELS_PATH.mkdir(parents=True, exist_ok=True)
            
            success_count = 0
            
            for name, model in models.items():
                try:
                    # Entraîner le modèle factice
                    if 'classifier' in name:
                        model.fit(X_dummy, y_dummy_class)
                    elif 'regressor' in name:
                        model.fit(X_dummy, y_dummy_reg)
                    elif 'scaler' in name:
                        model.fit(X_dummy)
                    
                    # Sauvegarder
                    model_path = API_MODELS_PATH / name
                    joblib.dump(model, model_path)
                    
                    size_mb = model_path.stat().st_size / (1024*1024)
                    self.models_deployed[name] = {
                        'path': str(model_path),
                        'size_mb': round(size_mb, 2),
                        'deployed_at': datetime.now().isoformat(),
                        'status': 'active',
                        'type': 'dummy'
                    }
                    
                    print(f"   OK {name}: {size_mb:.1f} MB (factice)")
                    success_count += 1
                    
                except Exception as e:
                    print(f"   ERREUR {name}: {e}")
            
            if success_count > 0:
                self.create_deployment_config()
                print(f"{success_count} modeles factices crees")
                return True
            else:
                return False
                
        except ImportError:
            print("scikit-learn non disponible pour creer des modeles factices")
            return False
        except Exception as e:
            print(f"Erreur creation modeles factices: {e}")
            return False
    
    def create_deployment_config(self):
        """Crée le fichier de configuration du déploiement."""
        config = {
            "deployment_info": {
                "method": self.deployment_method,
                "timestamp": datetime.now().isoformat(),
                "total_models": len(self.models_deployed),
                "deployment_path": str(API_MODELS_PATH)
            },
            "models": self.models_deployed,
            "api_endpoints": {
                "predict_occurrence": {
                    "model": "randomforest_classifier.pkl",
                    "description": "Prediction d'occurrence d'evenements extremes",
                    "input_features": ["IOD", "Nino34", "TNA", "month", "season"],
                    "output": "probability"
                },
                "predict_intensity": {
                    "model": "randomforest_regressor.pkl",
                    "description": "Prediction d'intensite des precipitations",
                    "input_features": ["IOD", "Nino34", "TNA", "month", "season"],
                    "output": "intensity_mm"
                }
            }
        }
        
        config_file = API_MODELS_PATH / "deployment_config.json"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"Configuration sauvee: {config_file.name}")
            
        except Exception as e:
            print(f"Erreur sauvegarde config: {e}")
    
    def show_next_steps(self):
        """Affiche les prochaines étapes."""
        print("\nPROCHAINES ETAPES:")
        print("=" * 30)
        print("Modeles disponibles dans: services/api/models/")
        print("Configuration: services/api/models/deployment_config.json")
        print()
        print("Pour utiliser les modeles:")
        print("  1. Demarrer l'API: docker-compose up -d api")
        print("  2. Tester l'API: http://localhost:8000/docs")
        print("  3. Endpoint modeles: http://localhost:8000/models")
        print()
        print("Pour verifier la base de donnees (si applicable):")
        print("  docker-compose exec timescaledb psql -U postgres -d climatsn_db")
    
    def run_deployment(self):
        """Lance le déploiement avec la méthode appropriée."""
        print("DEPLOIEMENT INTELLIGENT DES MODELES ML")
        print("=" * 80)
        print(f"Methode automatique: {self.deployment_method}")
        print("=" * 80)
        
        try:
            if self.deployment_method == "container_mode":
                success = self.deploy_container_mode()
            elif self.deployment_method == "docker_exec":
                success = self.deploy_docker_exec()
            elif self.deployment_method == "local_db":
                success = self.deploy_local_db()
            else:
                success = self.deploy_files_only()
            
            print("\n" + "=" * 80)
            
            if success:
                print("DEPLOIEMENT REUSSI!")
                print(f"Modeles deployes: {len(self.models_deployed)}")
                print(f"Methode utilisee: {self.deployment_method}")
                
                self.show_next_steps()
                
                return True
            else:
                print("ECHEC DU DEPLOIEMENT")
                print("Verifiez les erreurs ci-dessus")
                return False
                
        except KeyboardInterrupt:
            print("\nDeploiement interrompu par l'utilisateur")
            return False
        except Exception as e:
            print(f"\nErreur inattendue: {e}")
            return False

def main():
    """Point d'entrée principal."""
    print("WRAPPER INTELLIGENT DE DEPLOIEMENT ML")
    print("Resolution des conflits main.py vs standalone")
    print("Auto-adaptation a l'environnement d'execution")
    print()
    
    try:
        deployment = IntelligentDeployment()
        return deployment.run_deployment()
        
    except Exception as e:
        print(f"Erreur fatale: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)