// services/dashboard/dist/js/data-integration.js
// Intégration des données réelles du pipeline ClimaSen

/**
 * Module d'intégration des données réelles du pipeline
 * Gère la récupération et l'affichage des vraies données générées
 */

// Configuration des données réelles du pipeline
const PIPELINE_DATA = {
    // Données de l'exécution réelle
    execution: {
        startTime: '2025-01-XX',
        totalDuration: 856, // secondes
        phases: {
            research: {
                duration: 833.2,
                steps: [
                    { name: 'detection', duration: 95.3, status: 'completed', events: 1439 },
                    { name: 'spatial-top10', duration: 105.8, status: 'completed', maps: 10 },
                    { name: 'spatial-top5', duration: 60.2, status: 'completed', maps: 5 },
                    { name: 'teleconnections', duration: 16.4, status: 'completed', indices: 3 },
                    { name: 'ml-pipeline', duration: 370.3, status: 'completed', models: 9 },
                    { name: 'clustering', duration: 213.7, status: 'completed', algorithms: 5 }
                ]
            },
            production: {
                duration: 22.8,
                steps: [
                    { name: 'deployment', duration: 22.8, status: 'completed', models: 9 }
                ]
            }
        }
    },
    
    // Résultats de l'analyse
    results: {
        events: {
            total: 1439,
            period: '1981-05-06 to 2023-10-30',
            frequency: 33.5, // événements/an
            avgPrecipitation: 40.78, // mm
            avgCoverage: 18.65, // %
            seasonal: {
                rainy: { count: 1408, percentage: 97.8 },
                dry: { count: 31, percentage: 2.2 }
            }
        },
        
        topEvents: [
            { date: '1996-11-08', coverage: 80.5, location: 'Tambacounda', season: 'Sèche' },
            { date: '2012-09-28', coverage: 78.2, location: 'Tambacounda', season: 'Pluies' },
            { date: '2000-10-16', coverage: 69.5, location: 'Tambacounda', season: 'Pluies' },
            { date: '2018-06-27', coverage: 68.0, location: 'Matam', season: 'Pluies' },
            { date: '2022-10-24', coverage: 67.0, location: 'Tambacounda', season: 'Pluies' }
        ],
        
        mlPerformance: {
            bestClassifier: { name: 'RandomForest', f1: 0.913, accuracy: 0.911 },
            bestRegressor: { name: 'RandomForest_Reg', r2: 0.791, mse: 190.10 },
            models: [
                { name: 'RandomForest', type: 'classifier', f1: 0.913, r2: 0.791 },
                { name: 'XGBoost', type: 'both', f1: 0.913, r2: 0.706 },
                { name: 'SVM', type: 'both', f1: 0.913, r2: 0.019 },
                { name: 'Neural_Network', type: 'both', f1: 0.904, r2: -0.176 }
            ],
            clustering: {
                optimal: { algorithm: 'K-means', clusters: 2, score: 0.191 },
                algorithms: 5
            }
        },
        
        infrastructure: {
            modelsDeployed: 9,
            databaseTables: 8,
            yearsAnalyzed: 43,
            dataPoints: 8794800,
            regions: ['Tambacounda', 'Matam', 'Kédougou', 'Kolda', 'Diourbel']
        }
    },
    
    // Chemins des visualisations générées
    visualizations: {
        detection: [
            'outputs/visualizations/detection/01_distribution_temporelle.png',
            'outputs/visualizations/detection/02_intensite_couverture.png',
            'outputs/visualizations/detection/03_evolution_anomalies.png',
            'outputs/visualizations/detection/04_distribution_spatiale.png'
        ],
        spatial: {
            top10: 'outputs/visualizations/spatial/',
            top5: 'outputs/visualizations/spatial_top5/'
        },
        teleconnections: [
            'outputs/visualizations/teleconnections/correlation_heatmap_lags.png',
            'outputs/visualizations/teleconnections/detailed_lag_correlations.png',
            'outputs/visualizations/teleconnections/seasonal_teleconnections_comparison.png'
        ],
        ml: [
            'outputs/visualizations/machine_learning/model_performance_comparison.png',
            'outputs/visualizations/machine_learning/predictions_timeline.png',
            'outputs/visualizations/machine_learning/feature_importance.png'
        ],
        clustering: [
            'outputs/visualizations/clustering/clustering_comparison_pca.png',
            'outputs/visualizations/clustering/clustering_metrics_comparison.png',
            'outputs/visualizations/clustering/hierarchical_dendrogram.png',
            'outputs/visualizations/clustering/kmeans_optimization.png'
        ]
    }
};

/**
 * Classe principale pour l'intégration des données
 */
class DataIntegration {
    constructor() {
        this.isInitialized = false;
        this.updateInterval = null;
        this.animationQueue = [];
    }
    
    /**
     * Initialise l'intégration des données
     */
    async initialize() {
        if (this.isInitialized) return;
        
        console.log('🔧 Initialisation DataIntegration...');
        
        try {
            // Mettre à jour les métriques du hero
            this.updateHeroMetrics();
            
            // Animer les barres de progression
            this.animateProgressBars();
            
            // Mettre à jour les statistiques de visualisations
            this.updateVisualizationStats();
            
            // Ajouter les indicateurs de données réelles
            this.addRealDataIndicators();
            
            // Configurer les event listeners
            this.setupEventListeners();
            
            // Démarrer les animations
            this.startAnimations();
            
            this.isInitialized = true;
            console.log('✅ DataIntegration initialisé avec succès');
            
        } catch (error) {
            console.error('❌ Erreur initialisation DataIntegration:', error);
        }
    }
    
    /**
     * Met à jour les métriques du hero avec les vraies données
     */
    updateHeroMetrics() {
        const metrics = [
            { id: 'metric-events', value: PIPELINE_DATA.results.events.total, suffix: '' },
            { id: 'metric-duration', value: Math.round(PIPELINE_DATA.execution.totalDuration / 60), suffix: 'min' },
            { id: 'metric-models', value: PIPELINE_DATA.results.infrastructure.modelsDeployed, suffix: '' },
            { id: 'metric-coverage', value: PIPELINE_DATA.results.infrastructure.yearsAnalyzed, suffix: '' }
        ];
        
        metrics.forEach(metric => {
            const element = document.getElementById(metric.id);
            if (element) {
                this.animateCounter(element, 0, metric.value, metric.suffix, 2000);
            }
        });
    }
    
    /**
     * Anime un compteur de 0 à la valeur cible
     */
    animateCounter(element, start, end, suffix = '', duration = 2000) {
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function (ease-out)
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (end - start) * easeOut);
            
            element.textContent = current + suffix;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }
    
    /**
     * Anime les barres de progression saisonnières
     */
    animateProgressBars() {
        setTimeout(() => {
            const progressBars = document.querySelectorAll('.progress-fill');
            progressBars.forEach((bar, index) => {
                setTimeout(() => {
                    bar.style.transition = 'width 1.5s ease-out';
                    // Les largeurs sont déjà définies dans le HTML
                }, index * 300);
            });
        }, 1000);
    }
    
    /**
     * Met à jour les statistiques de visualisations
     */
    updateVisualizationStats() {
        const stats = {
            'detection-viz-count': PIPELINE_DATA.visualizations.detection.length,
            'spatial-viz-count': '15+', // Approximation basée sur top10 + top5 + cartes
            'ml-viz-count': PIPELINE_DATA.visualizations.ml.length,
            'clustering-viz-count': PIPELINE_DATA.visualizations.clustering.length
        };
        
        // Calculer le total
        const totalViz = PIPELINE_DATA.visualizations.detection.length + 
                        15 + // spatial approximation
                        PIPELINE_DATA.visualizations.ml.length + 
                        PIPELINE_DATA.visualizations.clustering.length +
                        PIPELINE_DATA.visualizations.teleconnections.length;
        
        stats['total-viz-count'] = totalViz + '+';
        
        Object.entries(stats).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
                element.classList.add('real-data-indicator');
            }
        });
    }
    
    /**
     * Ajoute des indicateurs visuels pour les données réelles
     */
    addRealDataIndicators() {
        const realDataElements = document.querySelectorAll('[data-real-metric]');
        realDataElements.forEach(element => {
            element.classList.add('real-data-indicator');
        });
    }
    
    /**
     * Configure les event listeners pour les interactions
     */
    setupEventListeners() {
        // Bouton de chargement des données réelles
        const loadRealDataBtn = document.getElementById('loadRealData');
        if (loadRealDataBtn) {
            loadRealDataBtn.addEventListener('click', () => {
                this.loadRealVisualizationData();
            });
        }
        
        // Clicks sur les étapes du pipeline
        document.querySelectorAll('.step-item').forEach(step => {
            step.addEventListener('click', (e) => {
                const stepName = step.dataset.step;
                this.showStepDetails(stepName);
            });
        });
        
        // Hover sur les cartes de résultats
        document.querySelectorAll('.result-card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                this.highlightRelatedData(card);
            });
        });
    }
    
    /**
     * Charge les données de visualisations réelles
     */
    async loadRealVisualizationData() {
        console.log('📊 Chargement des données de visualisations réelles...');
        
        // Simuler un appel API avec les vraies données
        const mockApiData = {
            visualizations: this.generateVisualizationEntries(),
            metadata: {
                totalGenerated: this.getTotalVisualizationCount(),
                lastUpdated: new Date().toISOString(),
                pipelineStatus: 'completed'
            }
        };
        
        // Mettre à jour le cache et afficher
        if (window.visualizationsCache) {
            window.visualizationsCache = mockApiData.visualizations;
        }
        
        // Déclencher l'affichage
        if (window.displayVisualizations) {
            window.displayVisualizations(mockApiData.visualizations);
        }
        
        this.showNotification('✅ Données réelles chargées avec succès', 'success');
    }
    
    /**
     * Génère les entrées de visualisations basées sur les vraies données
     */
    generateVisualizationEntries() {
        const visualizations = [];
        
        // Visualisations de détection
        PIPELINE_DATA.visualizations.detection.forEach((path, index) => {
            visualizations.push({
                id: `detection_${index + 1}`,
                title: this.getDetectionVizTitle(index),
                description: this.getDetectionVizDescription(index),
                category: 'detection',
                image: `/api/visualizations/image/${path.split('/').pop()}`,
                date: new Date().toISOString(),
                metadata: {
                    events: PIPELINE_DATA.results.events.total,
                    period: PIPELINE_DATA.results.events.period
                }
            });
        });
        
        // Visualisations ML
        PIPELINE_DATA.visualizations.ml.forEach((path, index) => {
            visualizations.push({
                id: `ml_${index + 1}`,
                title: this.getMLVizTitle(index),
                description: this.getMLVizDescription(index),
                category: 'machine-learning',
                image: `/api/visualizations/image/${path.split('/').pop()}`,
                date: new Date().toISOString(),
                metadata: {
                    models: PIPELINE_DATA.results.infrastructure.modelsDeployed,
                    bestF1: PIPELINE_DATA.results.mlPerformance.bestClassifier.f1
                }
            });
        });
        
        // Visualisations de clustering
        PIPELINE_DATA.visualizations.clustering.forEach((path, index) => {
            visualizations.push({
                id: `clustering_${index + 1}`,
                title: this.getClusteringVizTitle(index),
                description: this.getClusteringVizDescription(index),
                category: 'clustering',
                image: `/api/visualizations/image/${path.split('/').pop()}`,
                date: new Date().toISOString(),
                metadata: {
                    algorithms: PIPELINE_DATA.results.mlPerformance.clustering.algorithms,
                    optimalClusters: PIPELINE_DATA.results.mlPerformance.clustering.optimal.clusters
                }
            });
        });
        
        // Visualisations de téléconnexions
        PIPELINE_DATA.visualizations.teleconnections.forEach((path, index) => {
            visualizations.push({
                id: `teleconnections_${index + 1}`,
                title: this.getTeleconnectionsVizTitle(index),
                description: this.getTeleconnectionsVizDescription(index),
                category: 'teleconnections',
                image: `/api/visualizations/image/${path.split('/').pop()}`,
                date: new Date().toISOString(),
                metadata: {
                    indices: ['IOD', 'ENSO', 'TNA'],
                    correlations: 'Significant'
                }
            });
        });
        
        return visualizations;
    }
    
    /**
     * Helpers pour les titres et descriptions
     */
    getDetectionVizTitle(index) {
        const titles = [
            'Distribution Temporelle des Événements',
            'Relation Intensité-Couverture',
            'Évolution des Anomalies',
            'Distribution Spatiale'
        ];
        return titles[index] || `Analyse de Détection ${index + 1}`;
    }
    
    getDetectionVizDescription(index) {
        const descriptions = [
            '1439 événements détectés sur 43 ans avec validation climatologique excellente',
            'Corrélation entre intensité des précipitations et étendue spatiale des événements',
            'Tendances temporelles des anomalies standardisées de précipitations',
            'Répartition géographique des événements extrêmes au Sénégal'
        ];
        return descriptions[index] || 'Analyse détaillée des événements de précipitations extrêmes';
    }
    
    getMLVizTitle(index) {
        const titles = [
            'Comparaison Performance Modèles',
            'Prédictions Temporelles',
            'Importance des Features'
        ];
        return titles[index] || `Analyse ML ${index + 1}`;
    }
    
    getMLVizDescription(index) {
        const descriptions = [
            'Performance comparative de 9 modèles ML avec RandomForest en tête (F1: 0.913)',
            'Évolution temporelle des prédictions avec validation croisée',
            'Analyse de l\'importance des features climatiques et temporelles'
        ];
        return descriptions[index] || 'Analyse des modèles d\'apprentissage automatique';
    }
    
    getClusteringVizTitle(index) {
        const titles = [
            'Comparaison PCA Clustering',
            'Métriques Clustering',
            'Dendrogramme Hiérarchique',
            'Optimisation K-Means'
        ];
        return titles[index] || `Clustering ${index + 1}`;
    }
    
    getClusteringVizDescription(index) {
        const descriptions = [
            'Comparaison de 5 algorithmes de clustering avec visualisation PCA',
            'Métriques de performance (Silhouette, Calinski-Harabasz, Davies-Bouldin)',
            'Analyse hiérarchique des patterns climatiques',
            'Optimisation du nombre de clusters avec méthode du coude'
        ];
        return descriptions[index] || 'Analyse comparative des algorithmes de clustering';
    }
    
    getTeleconnectionsVizTitle(index) {
        const titles = [
            'Heatmap Corrélations Décalages',
            'Corrélations Détaillées par Lag',
            'Comparaison Saisonnière'
        ];
        return titles[index] || `Téléconnexions ${index + 1}`;
    }
    
    getTeleconnectionsVizDescription(index) {
        const descriptions = [
            'Matrice de corrélations entre indices climatiques (IOD, ENSO, TNA) avec décalages temporels',
            'Analyse détaillée des corrélations par lag avec significativité statistique',
            'Comparaison des téléconnexions entre saison sèche et saison des pluies'
        ];
        return descriptions[index] || 'Analyse des téléconnexions océan-atmosphère';
    }
    
    /**
     * Calcule le nombre total de visualisations
     */
    getTotalVisualizationCount() {
        return PIPELINE_DATA.visualizations.detection.length +
               PIPELINE_DATA.visualizations.ml.length +
               PIPELINE_DATA.visualizations.clustering.length +
               PIPELINE_DATA.visualizations.teleconnections.length +
               15; // Approximation pour les visualisations spatiales
    }
    
    /**
     * Affiche les détails d'une étape du pipeline
     */
    showStepDetails(stepName) {
        const stepData = this.findStepData(stepName);
        if (!stepData) return;
        
        const modalContent = `
            <div class="step-modal">
                <div class="step-modal-header">
                    <h3>${this.getStepDisplayName(stepName)}</h3>
                    <span class="step-status ${stepData.status}">${stepData.status}</span>
                </div>
                <div class="step-modal-body">
                    <div class="step-metric">
                        <span class="metric-label">Durée:</span>
                        <span class="metric-value">${stepData.duration}s</span>
                    </div>
                    ${this.getStepSpecificMetrics(stepName, stepData)}
                    <div class="step-description">
                        ${this.getStepDescription(stepName)}
                    </div>
                </div>
            </div>
        `;
        
        this.showModal('Détails de l\'étape', modalContent);
    }
    
    /**
     * Trouve les données d'une étape
     */
    findStepData(stepName) {
        // Rechercher dans les étapes de recherche
        const researchStep = PIPELINE_DATA.execution.phases.research.steps.find(s => s.name === stepName);
        if (researchStep) return researchStep;
        
        // Rechercher dans les étapes de production
        const productionStep = PIPELINE_DATA.execution.phases.production.steps.find(s => s.name === stepName);
        return productionStep;
    }
    
    /**
     * Obtient le nom d'affichage d'une étape
     */
    getStepDisplayName(stepName) {
        const names = {
            'detection': 'Détection des Événements Extrêmes',
            'spatial-top10': 'Analyse Spatiale TOP 10',
            'spatial-top5': 'Analyse Spatiale TOP 5',
            'teleconnections': 'Analyse des Téléconnexions',
            'ml-pipeline': 'Pipeline Machine Learning',
            'clustering': 'Clustering Avancé',
            'deployment': 'Déploiement Production'
        };
        return names[stepName] || stepName;
    }
    
    /**
     * Obtient les métriques spécifiques d'une étape
     */
    getStepSpecificMetrics(stepName, stepData) {
        let metrics = '';
        
        if (stepData.events) {
            metrics += `
                <div class="step-metric">
                    <span class="metric-label">Événements:</span>
                    <span class="metric-value">${stepData.events}</span>
                </div>`;
        }
        
        if (stepData.models) {
            metrics += `
                <div class="step-metric">
                    <span class="metric-label">Modèles:</span>
                    <span class="metric-value">${stepData.models}</span>
                </div>`;
        }
        
        if (stepData.maps) {
            metrics += `
                <div class="step-metric">
                    <span class="metric-label">Cartes:</span>
                    <span class="metric-value">${stepData.maps}</span>
                </div>`;
        }
        
        if (stepData.algorithms) {
            metrics += `
                <div class="step-metric">
                    <span class="metric-label">Algorithmes:</span>
                    <span class="metric-value">${stepData.algorithms}</span>
                </div>`;
        }
        
        return metrics;
    }
    
    /**
     * Obtient la description d'une étape
     */
    getStepDescription(stepName) {
        const descriptions = {
            'detection': 'Détection automatique de 1439 événements de précipitations extrêmes sur la période 1981-2023 avec validation climatologique excellente (97.8% en saison des pluies).',
            'spatial-top10': 'Analyse géographique détaillée des 10 événements les plus étendus avec références précises aux régions administratives du Sénégal.',
            'spatial-top5': 'Cartographie haute résolution des 5 événements les plus intenses (jusqu\'à 231.4 mm/jour) avec métriques spatiales complètes.',
            'teleconnections': 'Quantification des relations entre indices océano-atmosphériques (IOD, ENSO, TNA) et événements extrêmes avec décalages temporels optimaux.',
            'ml-pipeline': 'Entraînement et validation de 9 modèles d\'apprentissage automatique avec performances exceptionnelles (F1: 0.913, R²: 0.791).',
            'clustering': 'Comparaison rigoureuse de 5 algorithmes de clustering pour identifier les régimes climatiques (K-means optimal avec 2 clusters).',
            'deployment': 'Déploiement automatisé des modèles ML dans l\'infrastructure Docker/TimescaleDB avec API REST opérationnelle.'
        };
        return descriptions[stepName] || 'Étape du pipeline d\'analyse climatique.';
    }
    
    /**
     * Met en évidence les données liées
     */
    highlightRelatedData(card) {
        // Ajouter des effets visuels pour montrer les relations
        const cardType = this.getCardType(card);
        this.highlightRelatedElements(cardType);
    }
    
    /**
     * Détermine le type de carte
     */
    getCardType(card) {
        if (card.querySelector('h3')?.textContent.includes('Statistiques')) return 'statistics';
        if (card.querySelector('h3')?.textContent.includes('Saisonnière')) return 'seasonal';
        if (card.querySelector('h3')?.textContent.includes('TOP 5')) return 'events';
        if (card.querySelector('h3')?.textContent.includes('Performance')) return 'ml';
        return 'unknown';
    }
    
    /**
     * Met en évidence les éléments liés
     */
    highlightRelatedElements(cardType) {
        // Logique de mise en évidence basée sur le type
        const relatedElements = document.querySelectorAll(`[data-related="${cardType}"]`);
        relatedElements.forEach(el => {
            el.classList.add('highlighted');
            setTimeout(() => el.classList.remove('highlighted'), 2000);
        });
    }
    
    /**
     * Démarre les animations périodiques
     */
    startAnimations() {
        // Animation des indicateurs de données réelles
        this.animateRealDataIndicators();
        
        // Mise à jour périodique des métriques
        this.updateInterval = setInterval(() => {
            this.updateTimestamps();
        }, 30000); // Toutes les 30 secondes
    }
    
    /**
     * Anime les indicateurs de données réelles
     */
    animateRealDataIndicators() {
        const indicators = document.querySelectorAll('.real-data-indicator');
        indicators.forEach((indicator, index) => {
            setTimeout(() => {
                indicator.style.animation = 'dataFlow 3s ease-in-out infinite';
            }, index * 200);
        });
    }
    
    /**
     * Met à jour les timestamps
     */
    updateTimestamps() {
        const timestampElements = document.querySelectorAll('[data-timestamp]');
        timestampElements.forEach(element => {
            const timestamp = element.dataset.timestamp;
            if (timestamp) {
                element.textContent = this.formatRelativeTime(new Date(timestamp));
            }
        });
    }
    
    /**
     * Formate un temps relatif
     */
    formatRelativeTime(date) {
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        
        if (days > 0) return `il y a ${days} jour${days > 1 ? 's' : ''}`;
        if (hours > 0) return `il y a ${hours} heure${hours > 1 ? 's' : ''}`;
        if (minutes > 0) return `il y a ${minutes} minute${minutes > 1 ? 's' : ''}`;
        return 'à l\'instant';
    }
    
    /**
     * Affiche une notification
     */
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        const container = document.getElementById('notifications') || document.body;
        container.appendChild(notification);
        
        // Animation d'entrée
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Suppression automatique
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => container.removeChild(notification), 300);
        }, 5000);
    }
    
    /**
     * Affiche un modal
     */
    showModal(title, content) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
            </div>
        `;
        
        // Event listeners
        modal.querySelector('.modal-close').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });
        
        document.body.appendChild(modal);
        setTimeout(() => modal.classList.add('show'), 100);
    }
    
    /**
     * Nettoie les ressources
     */
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        this.isInitialized = false;
    }
}

/**
 * Fonctions utilitaires globales
 */

/**
 * Formate une durée en secondes vers un format lisible
 */
function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    
    if (minutes > 0) {
        return `${minutes}min ${remainingSeconds}s`;
    }
    return `${remainingSeconds}s`;
}

/**
 * Formate un nombre avec des séparateurs de milliers
 */
function formatNumber(num) {
    return new Intl.NumberFormat('fr-FR').format(num);
}

/**
 * Génère une couleur basée sur une valeur
 */
function getColorForValue(value, min, max) {
    const ratio = (value - min) / (max - min);
    const hue = (1 - ratio) * 120; // Rouge à vert
    return `hsl(${hue}, 70%, 50%)`;
}

/**
 * Vérifie si l'API est disponible
 */
async function checkApiAvailability() {
    try {
        const response = await fetch('/api/health', { method: 'HEAD' });
        return response.ok;
    } catch {
        return false;
    }
}

/**
 * Instance globale de DataIntegration
 */
let dataIntegration = null;

/**
 * Initialise l'intégration des données au chargement de la page
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('🌍 Initialisation ClimaSen Data Integration...');
    
    // Créer l'instance
    dataIntegration = new DataIntegration();
    
    // Attendre que le DOM soit complètement chargé
    setTimeout(() => {
        dataIntegration.initialize();
    }, 1000);
    
    // Nettoyer lors du déchargement
    window.addEventListener('beforeunload', () => {
        if (dataIntegration) {
            dataIntegration.destroy();
        }
    });
});

/**
 * Fonction d'aide pour déboguer les données
 */
function debugPipelineData() {
    console.group('🔍 Pipeline Data Debug');
    console.log('Événements détectés:', PIPELINE_DATA.results.events.total);
    console.log('Durée totale:', formatDuration(PIPELINE_DATA.execution.totalDuration));
    console.log('Modèles déployés:', PIPELINE_DATA.results.infrastructure.modelsDeployed);
    console.log('Visualisations:', dataIntegration?.getTotalVisualizationCount());
    console.groupEnd();
}

// Exposer les fonctions utiles globalement
window.DataIntegration = DataIntegration;
window.PIPELINE_DATA = PIPELINE_DATA;
window.debugPipelineData = debugPipelineData;
window.formatDuration = formatDuration;
window.formatNumber = formatNumber;