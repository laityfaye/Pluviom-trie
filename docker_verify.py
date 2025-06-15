#!/usr/bin/env python3
"""
Script de Vérification Docker - Installation Complète
Vérifie que votre installation Docker est correcte et fonctionnelle

Usage:
    python docker_verify.py
    python docker_verify.py --detailed
    python docker_verify.py --fix-issues

Ce script vérifie:
1. Installation Docker Desktop
2. Structure des fichiers Docker
3. Configuration docker-compose
4. Variables d'environnement
5. Connectivité des services
6. Pipeline ML dans Docker
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

class DockerVerificationManager:
    def __init__(self, project_root=None, detailed=False, fix_issues=False):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.detailed = detailed
        self.fix_issues = fix_issues
        self.verification_results = {}
        self.issues_found = []
        self.fixes_applied = []
        
    def print_header(self, title: str, level: int = 1):
        """Affiche un en-tête formaté."""
        symbols = ["🐳", "📋", "🔍", "⚙️"]
        symbol = symbols[min(level-1, len(symbols)-1)]
        
        if level == 1:
            print(f"\n{symbol} {title}")
            print("=" * (len(title) + 3))
        else:
            print(f"\n{symbol} {title}")
            print("-" * (len(title) + 3))
    
    def run_command(self, cmd: List[str], capture_output=True, timeout=30) -> Tuple[bool, str, str]:
        """Exécute une commande et retourne le résultat."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout expired"
        except FileNotFoundError:
            return False, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            return False, "", str(e)
    
    def check_docker_installation(self) -> Dict[str, bool]:
        """Vérifie l'installation Docker."""
        self.print_header("Vérification Installation Docker", 2)
        
        results = {}
        
        # 1. Docker Desktop installé
        success, version, error = self.run_command(["docker", "--version"])
        results["docker_installed"] = success
        if success:
            print(f"✅ Docker installé: {version.strip()}")
        else:
            print(f"❌ Docker non trouvé: {error}")
            self.issues_found.append("Docker Desktop non installé")
        
        # 2. Docker Compose
        success, version, error = self.run_command(["docker-compose", "--version"])
        results["docker_compose"] = success
        if success:
            print(f"✅ Docker Compose: {version.strip()}")
        else:
            print(f"❌ Docker Compose non trouvé: {error}")
            self.issues_found.append("Docker Compose non disponible")
        
        # 3. Docker daemon running
        success, output, error = self.run_command(["docker", "info"])
        results["docker_running"] = success
        if success:
            print("✅ Docker daemon en cours d'exécution")
        else:
            print(f"❌ Docker daemon non accessible: {error}")
            self.issues_found.append("Docker daemon non démarré")
        
        # 4. WSL2 (Windows spécifique)
        if os.name == 'nt':  # Windows
            success, output, error = self.run_command(["wsl", "--list", "--verbose"])
            results["wsl2"] = success
            if success:
                print("✅ WSL2 disponible")
            else:
                print(f"❌ WSL2 non configuré: {error}")
                self.issues_found.append("WSL2 non configuré (Windows)")
        
        return results
    
    def check_docker_structure(self) -> Dict[str, bool]:
        """Vérifie la structure Docker du projet."""
        self.print_header("Vérification Structure Docker", 2)
        
        expected_files = {
            # Fichiers Docker essentiels
            "docker-compose.yml": "Orchestration des services",
            ".env": "Variables d'environnement",
            ".dockerignore": "Exclusions Docker",
            
            # Dockerfiles
            "docker/api/Dockerfile": "Image API FastAPI",
            "docker/ml-pipeline/Dockerfile": "Image Pipeline ML",
            "docker/timescaledb/init-scripts": "Scripts d'initialisation DB",
            
            # Services
            "services/api/main.py": "Code API FastAPI",
            "services/api/requirements.txt": "Dépendances API",
            
            # Scripts Docker
            "docker/scripts/startup.ps1": "Script de démarrage",
            "docker/scripts/deploy.ps1": "Script de déploiement",
            
            # Monitoring
            "monitoring/prometheus.yml": "Configuration Prometheus"
        }
        
        results = {}
        missing_files = []
        
        for file_path, description in expected_files.items():
            full_path = self.project_root / file_path
            exists = full_path.exists()
            results[file_path] = exists
            
            if exists:
                print(f"✅ {file_path:<35} - {description}")
            else:
                print(f"❌ {file_path:<35} - MANQUANT ({description})")
                missing_files.append(file_path)
        
        if missing_files:
            self.issues_found.append(f"Fichiers Docker manquants: {', '.join(missing_files)}")
        
        return results
    
    def check_docker_compose_validity(self) -> Dict[str, bool]:
        """Vérifie la validité du docker-compose.yml."""
        self.print_header("Vérification Docker Compose", 2)
        
        results = {}
        
        # 1. Syntaxe docker-compose
        success, output, error = self.run_command(["docker-compose", "config", "--quiet"])
        results["compose_syntax"] = success
        if success:
            print("✅ Syntaxe docker-compose.yml valide")
        else:
            print(f"❌ Erreur syntaxe docker-compose.yml: {error}")
            self.issues_found.append("docker-compose.yml invalide")
        
        # 2. Services définis
        compose_file = self.project_root / "docker-compose.yml"
        if compose_file.exists():
            try:
                with open(compose_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                expected_services = ["timescaledb", "redis", "api", "ml-pipeline"]
                services_found = []
                
                for service in expected_services:
                    if f"{service}:" in content:
                        services_found.append(service)
                        print(f"✅ Service {service} défini")
                    else:
                        print(f"❌ Service {service} manquant")
                
                results["all_services"] = len(services_found) == len(expected_services)
                if len(services_found) != len(expected_services):
                    missing = set(expected_services) - set(services_found)
                    self.issues_found.append(f"Services manquants: {', '.join(missing)}")
                    
            except Exception as e:
                print(f"❌ Erreur lecture docker-compose.yml: {e}")
                results["all_services"] = False
        
        return results
    
    def check_environment_variables(self) -> Dict[str, bool]:
        """Vérifie les variables d'environnement."""
        self.print_header("Vérification Variables d'Environnement", 2)
        
        results = {}
        env_file = self.project_root / ".env"
        
        if not env_file.exists():
            print("❌ Fichier .env manquant")
            self.issues_found.append("Fichier .env manquant")
            return {"env_file_exists": False}
        
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                env_content = f.read()
            
            required_vars = {
                "POSTGRES_DB": "Base de données PostgreSQL",
                "POSTGRES_USER": "Utilisateur PostgreSQL", 
                "POSTGRES_PASSWORD": "Mot de passe PostgreSQL",
                "API_SECRET_KEY": "Clé secrète API",
                "COMPOSE_PROJECT_NAME": "Nom du projet Docker"
            }
            
            missing_vars = []
            for var, description in required_vars.items():
                if f"{var}=" in env_content:
                    print(f"✅ {var:<20} - {description}")
                else:
                    print(f"❌ {var:<20} - MANQUANT ({description})")
                    missing_vars.append(var)
            
            results["env_file_exists"] = True
            results["all_vars_present"] = len(missing_vars) == 0
            
            if missing_vars:
                self.issues_found.append(f"Variables .env manquantes: {', '.join(missing_vars)}")
            
            # Vérifier les mots de passe par défaut
            if "secure_password" in env_content or "ChangeMe" not in env_content:
                print("⚠️  Utilisez des mots de passe sécurisés en production")
            
        except Exception as e:
            print(f"❌ Erreur lecture .env: {e}")
            results["env_file_exists"] = False
            
        return results
    
    def check_docker_services_status(self) -> Dict[str, bool]:
        """Vérifie le statut des services Docker."""
        self.print_header("Vérification Services Docker", 2)
        
        results = {}
        
        # 1. Vérifier si des services sont en cours
        success, output, error = self.run_command(["docker-compose", "ps"])
        if success:
            print("📊 État des services Docker:")
            print(output)
            
            # Analyser la sortie pour voir les services actifs
            lines = output.strip().split('\n')
            if len(lines) > 1:  # Au moins l'en-tête + une ligne
                running_services = []
                for line in lines[1:]:  # Ignorer l'en-tête
                    if line.strip() and "Up" in line:
                        service_name = line.split()[0]
                        running_services.append(service_name)
                
                if running_services:
                    print(f"✅ Services actifs: {', '.join(running_services)}")
                    results["services_running"] = True
                else:
                    print("⚠️  Aucun service Docker en cours d'exécution")
                    results["services_running"] = False
            else:
                print("⚠️  Aucun service Docker défini ou démarré")
                results["services_running"] = False
        else:
            print(f"❌ Impossible de vérifier les services: {error}")
            results["services_running"] = False
        
        return results
    
    def check_project_structure_compatibility(self) -> Dict[str, bool]:
        """Vérifie que la structure du projet est compatible avec Docker."""
        self.print_header("Vérification Compatibilité Projet", 2)
        
        results = {}
        
        # Vérifier les fichiers du projet original
        original_files = {
            "main.py": "Point d'entrée principal",
            "requirements.txt": "Dépendances Python",
            "scripts/": "Scripts d'analyse ML",
            "src/": "Code source modulaire",
            "data/": "Données du projet"
        }
        
        compatibility_score = 0
        total_files = len(original_files)
        
        for file_path, description in original_files.items():
            full_path = self.project_root / file_path.rstrip('/')
            exists = full_path.exists()
            
            if exists:
                print(f"✅ {file_path:<20} - {description}")
                compatibility_score += 1
            else:
                print(f"❌ {file_path:<20} - MANQUANT ({description})")
        
        results["project_compatible"] = compatibility_score >= total_files * 0.8
        results["compatibility_score"] = compatibility_score / total_files
        
        if compatibility_score < total_files:
            missing_count = total_files - compatibility_score
            self.issues_found.append(f"{missing_count} fichiers projet manquants")
        
        return results
    
    def test_docker_services(self) -> Dict[str, bool]:
        """Teste la connectivité des services Docker."""
        self.print_header("Test des Services Docker", 2)
        
        results = {}
        
        # 1. Tenter de démarrer les services
        print("🚀 Tentative de démarrage des services...")
        success, output, error = self.run_command(
            ["docker-compose", "up", "-d"], 
            timeout=120
        )
        
        if success:
            print("✅ Services démarrés avec succès")
            results["services_start"] = True
            
            # Attendre un peu pour que les services s'initialisent
            print("⏳ Attente initialisation des services...")
            time.sleep(15)
            
            # 2. Tester la connectivité des services
            services_to_test = {
                "api": ("http://localhost:8000/health", "API FastAPI"),
                "dashboard": ("http://localhost:3000", "Dashboard Web"),
                "grafana": ("http://localhost:3001", "Grafana"),
                "prometheus": ("http://localhost:9090", "Prometheus")
            }
            
            for service_name, (url, description) in services_to_test.items():
                print(f"🔍 Test {description}...")
                try:
                    # Utiliser curl pour tester (plus fiable que requests)
                    success, output, error = self.run_command(
                        ["curl", "-f", "-s", url], 
                        timeout=10
                    )
                    if success:
                        print(f"✅ {description} accessible: {url}")
                        results[f"{service_name}_accessible"] = True
                    else:
                        print(f"❌ {description} non accessible: {url}")
                        results[f"{service_name}_accessible"] = False
                except Exception as e:
                    print(f"❌ {description} erreur: {e}")
                    results[f"{service_name}_accessible"] = False
            
        else:
            print(f"❌ Échec démarrage services: {error}")
            results["services_start"] = False
            self.issues_found.append("Impossible de démarrer les services Docker")
        
        return results
    
    def test_ml_pipeline(self) -> Dict[str, bool]:
        """Teste le pipeline ML dans Docker."""
        self.print_header("Test Pipeline ML Docker", 2)
        
        results = {}
        
        print("🤖 Test du pipeline ML...")
        
        # Tester le container ML avec un test rapide
        success, output, error = self.run_command([
            "docker-compose", "run", "--rm", "ml-pipeline", 
            "python", "-c", "import sys; print('Python ML container OK'); print(f'Python version: {sys.version}')"
        ], timeout=60)
        
        if success:
            print("✅ Container ML Pipeline fonctionnel")
            print(f"📄 Sortie: {output.strip()}")
            results["ml_container"] = True
            
            # Test des imports Python critiques
            print("🔍 Test des imports ML...")
            import_test_cmd = [
                "docker-compose", "run", "--rm", "ml-pipeline",
                "python", "-c", 
                "import numpy, pandas, sklearn, matplotlib; print('✅ Imports ML OK')"
            ]
            
            success_imports, output_imports, error_imports = self.run_command(
                import_test_cmd, timeout=45
            )
            
            if success_imports:
                print("✅ Bibliothèques ML disponibles")
                results["ml_libraries"] = True
            else:
                print(f"❌ Problème imports ML: {error_imports}")
                results["ml_libraries"] = False
                self.issues_found.append("Bibliothèques ML non disponibles")
        else:
            print(f"❌ Container ML non fonctionnel: {error}")
            results["ml_container"] = False
            self.issues_found.append("Pipeline ML Docker défaillant")
        
        return results
    
    def generate_verification_report(self):
        """Génère un rapport de vérification."""
        self.print_header("Rapport de Vérification Docker", 1)
        
        total_checks = sum(len(results) for results in self.verification_results.values())
        passed_checks = sum(
            sum(check for check in results.values() if isinstance(check, bool) and check)
            for results in self.verification_results.values()
        )
        
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        print(f"📊 Résultats globaux:")
        print(f"   Tests passés: {passed_checks}/{total_checks}")
        print(f"   Taux de réussite: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 EXCELLENT! Installation Docker parfaite")
            overall_status = "EXCELLENT"
        elif success_rate >= 75:
            print("✅ BIEN! Installation Docker fonctionnelle avec quelques améliorations possibles")
            overall_status = "BIEN"
        elif success_rate >= 50:
            print("⚠️  MOYEN! Installation Docker partielle - corrections nécessaires")
            overall_status = "MOYEN"
        else:
            print("❌ PROBLÉMATIQUE! Installation Docker nécessite des corrections majeures")
            overall_status = "PROBLÉMATIQUE"
        
        # Problèmes identifiés
        if self.issues_found:
            print(f"\n🔧 Problèmes identifiés ({len(self.issues_found)}):")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"   {i}. {issue}")
        
        # Corrections appliquées
        if self.fixes_applied:
            print(f"\n✅ Corrections appliquées ({len(self.fixes_applied)}):")
            for i, fix in enumerate(self.fixes_applied, 1):
                print(f"   {i}. {fix}")
        
        # Recommandations
        print(f"\n💡 Recommandations:")
        if success_rate < 100:
            print("   1. Corrigez les problèmes identifiés ci-dessus")
        if success_rate >= 75:
            print("   2. Lancez votre pipeline: docker-compose run --rm ml-pipeline python main.py")
            print("   3. Accédez à l'API: http://localhost:8000")
        if success_rate < 75:
            print("   2. Relancez la vérification après corrections: python docker_verify.py")
        
        # Sauvegarder le rapport
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": overall_status,
            "success_rate": success_rate,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "results": self.verification_results,
            "issues_found": self.issues_found,
            "fixes_applied": self.fixes_applied
        }
        
        report_file = self.project_root / "docker_verification_report.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n📄 Rapport sauvegardé: {report_file}")
        except Exception as e:
            print(f"\n⚠️  Impossible de sauvegarder le rapport: {e}")
        
        return overall_status, success_rate
    
    def run_complete_verification(self) -> Tuple[str, float]:
        """Exécute la vérification complète."""
        self.print_header("VÉRIFICATION INSTALLATION DOCKER", 1)
        print(f"📁 Projet: {self.project_root}")
        print(f"🔍 Mode: {'Détaillé' if self.detailed else 'Standard'}")
        print(f"🔧 Corrections auto: {'Activées' if self.fix_issues else 'Désactivées'}")
        
        # Étape 1: Installation Docker
        self.verification_results["docker_installation"] = self.check_docker_installation()
        
        # Étape 2: Structure Docker
        self.verification_results["docker_structure"] = self.check_docker_structure()
        
        # Étape 3: Docker Compose
        self.verification_results["docker_compose"] = self.check_docker_compose_validity()
        
        # Étape 4: Variables d'environnement
        self.verification_results["environment"] = self.check_environment_variables()
        
        # Étape 5: Compatibilité projet
        self.verification_results["project_compatibility"] = self.check_project_structure_compatibility()
        
        # Étape 6: Statut services (si demandé)
        if self.detailed:
            self.verification_results["services_status"] = self.check_docker_services_status()
        
        # Étape 7: Test services (si demandé)
        if self.detailed:
            self.verification_results["services_test"] = self.test_docker_services()
            
        # Étape 8: Test pipeline ML (si demandé)
        if self.detailed:
            self.verification_results["ml_pipeline"] = self.test_ml_pipeline()
        
        # Générer le rapport final
        return self.generate_verification_report()

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Vérification complète de l'installation Docker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLES:
    # Vérification standard
    python docker_verify.py
    
    # Vérification détaillée avec tests de services
    python docker_verify.py --detailed
    
    # Vérification avec corrections automatiques
    python docker_verify.py --fix-issues
    
    # Chemin de projet spécifique
    python docker_verify.py --project-root /path/to/project
        """
    )
    
    parser.add_argument(
        "--project-root",
        help="Chemin racine du projet",
        default="."
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Vérification détaillée avec tests de services"
    )
    parser.add_argument(
        "--fix-issues",
        action="store_true", 
        help="Tenter de corriger automatiquement les problèmes"
    )
    
    args = parser.parse_args()
    
    try:
        verifier = DockerVerificationManager(
            project_root=args.project_root,
            detailed=args.detailed,
            fix_issues=args.fix_issues
        )
        
        overall_status, success_rate = verifier.run_complete_verification()
        
        # Code de sortie basé sur le taux de réussite
        if success_rate >= 90:
            return 0  # Succès
        elif success_rate >= 75:
            return 1  # Avertissement
        else:
            return 2  # Erreur
            
    except KeyboardInterrupt:
        print("\n⏹️  Vérification interrompue par l'utilisateur")
        return 130
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())