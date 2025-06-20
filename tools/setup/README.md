# Outils de Configuration - Projet Climat Sénégal

Ce dossier contient les outils de configuration et d'installation automatisés pour le projet.

## Scripts Disponibles

### `project_setup.py` (Ancien `setup.py`)
**Configuration initiale du projet Python**

**Usage :**
```bash
cd ../..  # Retourner à la racine du projet
python tools/setup/project_setup.py
```

**Ce qu'il fait :**
- ✅ Crée la structure complète des dossiers
- ✅ Génère les fichiers `__init__.py`
- ✅ Crée `requirements.txt` et installe les dépendances
- ✅ Vérifie la présence des données CHIRPS
- ✅ Crée des modules stub si nécessaire
- ✅ Lance les tests de validation
- ✅ Guide pour les prochaines étapes

**Quand l'utiliser :**
- Premier setup du projet
- Réinstallation sur nouveau système
- Réparation d'environnement corrompu
- Onboarding de nouveaux collaborateurs

---

### `docker_infrastructure_setup.py` (Ancien `docker_setup.py`)
**Configuration de l'infrastructure Docker complète**

**Usage :**
```bash
cd ../..  # Retourner à la racine du projet
python tools/setup/docker_infrastructure_setup.py
```

**Ce qu'il fait :**
- ✅ Analyse la structure existante du projet
- ✅ Crée l'architecture Docker complète
- ✅ Génère les Dockerfiles (API, ML Pipeline)
- ✅ Crée `docker-compose.yml` avec tous les services
- ✅ Configure l'API FastAPI automatiquement
- ✅ Met en place le monitoring (Prometheus/Grafana)
- ✅ Génère les scripts de gestion (PowerShell)

**Quand l'utiliser :**
- Dockerisation d'un projet existant
- Création de l'infrastructure de production
- Mise en place du monitoring
- Déploiement automatisé

---

## Ordre d'Utilisation Recommandé

### Nouveau Projet (Partir de zéro)
```bash
# 1. Configuration du projet Python
python tools/setup/project_setup.py

# 2. Développer le code du projet
# (créer les modules, analyser les données, etc.)

# 3. Configuration Docker
python tools/setup/docker_infrastructure_setup.py

# 4. Validation et démarrage
python docker_verify.py
.\docker-start.ps1
```

### Projet Existant (Ajout Docker)
```bash
# Si le projet Python existe déjà, passer directement à Docker
python tools/setup/docker_infrastructure_setup.py

# Puis validation
python docker_verify.py
.\docker-start.ps1
```

### Réinstallation Complète
```bash
# 1. Cleanup (optionnel)
docker-compose down -v
rm -rf src/ docker/ services/

# 2. Reconfiguration complète
python tools/setup/project_setup.py
python tools/setup/docker_infrastructure_setup.py

# 3. Validation
python quick_test.py
python docker_verify.py
.\docker-start.ps1
```

---

## Scripts de Gestion (Générés)

Après exécution des outils de setup, vous aurez :

### Scripts Principaux
- `quick_test.py` - Test rapide du projet
- `docker_verify.py` - Vérification Docker
- `docker-start.ps1` - Démarrage rapide

### Scripts Docker (Générés)
- `docker/scripts/startup.ps1` - Démarrage avancé
- `docker/scripts/deploy.ps1` - Déploiement
- `docker/scripts/backup.ps1` - Sauvegarde

---

## Dépannage

### Le projet ne se configure pas
```bash
# Vérifier Python
python --version  # Doit être 3.8+

# Réexécuter project_setup
python tools/setup/project_setup.py
```

### Docker ne fonctionne pas
```bash
# Vérifier Docker
docker --version
docker-compose --version

# Réexécuter docker_infrastructure_setup
python tools/setup/docker_infrastructure_setup.py
```

### Modules manquants
```bash
# Vérifier les imports
python quick_test.py

# Si échec, reconfigurer
python tools/setup/project_setup.py
```

---

## Maintenance

### Mise à jour des outils
Ces scripts sont versionnés avec le projet. Pour les mettre à jour :
1. Modifier le script concerné
2. Tester sur un environnement de développement
3. Documenter les changements

### Ajout de nouveaux outils
Pour ajouter un nouvel outil de setup :
1. Créer le script dans `tools/setup/`
2. Documenter dans ce README
3. Ajouter dans l'ordre d'utilisation si nécessaire

---

## Contact et Support

Pour toute question sur ces outils de configuration :
- Vérifier d'abord ce README
- Exécuter les tests de validation
- Consulter les logs des scripts en cas d'erreur

**Version :** 1.0  
**Dernière mise à jour :** Juin 2025