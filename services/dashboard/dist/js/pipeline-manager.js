// services/dashboard/dist/js/pipeline-manager.js
// Gestionnaire des données du pipeline scientifique

/**
 * Données réelles du pipeline basées sur l'exécution
 */
const REAL_PIPELINE_DATA = {
    metadata: {
        executionDate: '2025-06-27',
        totalDuration: 1108.4, // secondes (18.5 minutes)
        version: '3.0',
        dataSource: 'CHIRPS',
        coverage: {
            temporal: '1981-2023',
            spatial: 'Sénégal (12°N-17°N, 18°W-11°W)',
            gridPoints: 560,
            years: 43
        }
    },
    
    phases: {
        research: {
            status: 'completed',
            duration: 1096.4, // secondes
            steps: {
                detection: {
                    name: 'Détection Événements Extrêmes',
                    duration: 375.4,
                    status: 'completed',
                    results: {
                        eventsDetected: 1439,
                        period: '1981-05-06 à 2023-10-30',
                        validation: 'EXCELLENT',
                        rainySeason: { events: 1408, percentage: 97.8 },
                        drySeason: { events: 31, percentage: 2.2 },
                        avgPrecipitation: 40.78, // mm
                        avgCoverage: 18.65, // %
                        avgAnomaly: 6.47 // σ
                    }
                },
                spatialTop10: {
                    name: 'Analyse Spatiale TOP 10',
                    duration: 77.2,
                    status: 'completed',
                    results: {
                        eventsAnalyzed: 10,
                        mainRegion: 'Tambacounda',
                        regionalCount: { 'Tambacounda': 10 },
                        avgCoverage: 67.3,
                        maxCoverage: 80.5
                    }
                },
                spatialTop5: {
                    name: 'Analyse Spatiale TOP 5',
                    duration: 115.1,
                    status: 'completed',
                    results: {
                        eventsAnalyzed: 5,
                        maxIntensity: 231.4, // mm
                        regions: ['Kédougou', 'Kolda', 'Tambacounda', 'Diourbel'],
                        avgCoverage: 20.86,
                        totalSurface: 439469 // km²
                    }
                },
                teleconnections: {
                    name: 'Téléconnexions Océan-Atmosphère',
                    duration: 69.4,
                    status: 'completed',
                    results: {
                        indicesAnalyzed: 3,
                        significantCorrelations: 2,
                        bestCorrelation: { index: 'TNA', value: 0.219, pValue: '<0.001' },
                        featuresGenerated: 39,
                        mlDatasetSize: 507
                    }
                },
                mlPipeline: {
                    name: 'Machine Learning Pipeline',
                    duration: 444.0,
                    status: 'completed',
                    results: {
                        modelsCompared: 9,
                        bestClassifier: { name: 'RandomForest', f1Score: 0.913, accuracy: 0.911 },
                        bestRegressor: { name: 'RandomForest_Reg', r2Score: 0.791, mse: 190.10 },
                        clustering: { algorithm: 'K-Means', clusters: 2, silhouette: 0.191 },
                        dataset: { observations: 507, features: 41 }
                    }
                },
                clustering: {
                    name: 'Analyse Clustering Avancée',
                    duration: 143.3,
                    status: 'completed',
                    results: {
                        algorithmsCompared: 5,
                        bestAlgorithm: 'DBSCAN',
                        bestSilhouette: 0.256,
                        optimalClusters: 2,
                        outliers: 16
                    }
                }
            }
        },
        production: {
            status: 'completed',
            duration: 12.0,
            steps: {
                deployment: {
                    name: 'Déploiement Production',
                    duration: 12.0,
                    status: 'completed',
                    results: {
                        modelsDeployed: 9,
                        apiEndpoint: 'http://localhost:8000',
                        database: 'TimescaleDB',
                        infrastructure: 'Docker',
                        monitoring: 'Grafana (port 3001)'
                    }
                }
            }
        }
    },
    
    // Métriques globales pour le dashboard
    globalMetrics: {
        eventsDetected: 1439,
        mlAccuracy: 91.3, // %
        gridPoints: 560,
        pipelineDurationMinutes: 18.5,
        dataYears: 43,
        modelsDeployed: 9,
        regionsAnalyzed: 14,
        visualizationsGenerated: 25
    },
    
    // Distribution mensuelle des événements
    monthlyDistribution: {
        'Janvier': 0, 'Février': 0, 'Mars': 0,
        'Avril': 11, 'Mai': 109, 'Juin': 226,
        'Juillet': 280, 'Août': 291, 'Septembre': 267,
        'Octobre': 235, 'Novembre': 20, 'Décembre': 0
    },
    
    // TOP 10 événements les plus extrêmes
    top10Events: [
        { date: '1996-11-08', region: 'Tambacounda', coverage: 80.5, intensity: 8.1, season: 'Sèche' },
        { date: '2012-09-28', region: 'Tambacounda', coverage: 78.2, intensity: 44.3, season: 'Pluies' },
        { date: '2000-10-16', region: 'Tambacounda', coverage: 69.5, intensity: 30.7, season: 'Pluies' },
        { date: '2018-06-27', region: 'Matam', coverage: 68.0, intensity: 37.5, season: 'Pluies' },
        { date: '2022-10-24', region: 'Tambacounda', coverage: 67.0, intensity: 56.4, season: 'Pluies' },
        { date: '2022-05-27', region: 'Tambacounda', coverage: 65.2, intensity: 37.5, season: 'Pluies' },
        { date: '2022-06-15', region: 'Tambacounda', coverage: 63.9, intensity: 46.9, season: 'Pluies' },
        { date: '1985-06-27', region: 'Tambacounda', coverage: 61.1, intensity: 49.9, season: 'Pluies' },
        { date: '1992-06-08', region: 'Tambacounda', coverage: 60.4, intensity: 42.4, season: 'Pluies' },
        { date: '2021-08-18', region: 'Tambacounda', coverage: 59.3, intensity: 79.5, season: 'Pluies' }
    ],
    
    // TOP 5 événements les plus intenses
    top5Intense: [
        { date: '1985-08-06', region: 'Kédougou', intensity: 231.4, coverage: 15.2, surface: 64177 },
        { date: '2014-08-14', region: 'Kolda', intensity: 193.9, coverage: 18.6, surface: 78321 },
        { date: '1996-08-04', region: 'Tambacounda', intensity: 136.9, coverage: 24.8, surface: 104850 },
        { date: '2009-08-28', region: 'Diourbel', intensity: 132.6, coverage: 27.1, surface: 113693 },
        { date: '2011-08-21', region: 'Tambacounda', intensity: 129.8, coverage: 18.6, surface: 78429 }
    ]
};

/**
 * Gestionnaire du pipeline scientifique
 */
class PipelineManager {
    constructor() {
        this.data = REAL_PIPELINE_DATA;
        this.isInitialized = false;
        this.animationQueues = new Map();
    }
    
    /**
     * Initialise le gestionnaire du pipeline
     */
    initialize() {
        console.log('🔬 Initialisation Pipeline Manager...');
        this.setupMetricsAnimations();
        this.setupPhaseIndicators();
        this.setupStepInteractions();
        this.isInitialized = true;
        console.log('✅ Pipeline Manager initialisé');
    }
    
    /**
     * Configure les animations des métriques
     */
    setupMetricsAnimations() {
        const metrics = [
            { id: 'events-count', target: this.data.globalMetrics.eventsDetected, suffix: '' },
            { id: 'ml-f1', target: this.data.globalMetrics.mlAccuracy, suffix: '%' },
            { id: 'coverage-area', target: this.data.globalMetrics.gridPoints, suffix: '' },
            { id: 'pipeline-duration', target: this.data.globalMetrics.pipelineDurationMinutes, suffix: '' }
        ];
        
        // Démarrer les animations après un court délai
        setTimeout(() => {
            metrics.forEach(metric => this.animateCounter(metric));
        }, 1000);
    }
    
    /**
     * Anime un compteur vers sa valeur cible
     */
    animateCounter({ id, target, suffix = '', duration = 2000 }) {
        const element = document.getElementById(id);
        if (!element) return;
        
        const start = 0;
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Fonction d'easing pour une animation fluide
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (target - start) * easeOut);
            
            element.textContent = this.formatNumber(current) + suffix;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }
    
    /**
     * Formate un nombre avec des séparateurs
     */
    formatNumber(num) {
        return new Intl.NumberFormat('fr-FR').format(num);
    }
    
    /**
     * Configure les indicateurs de phase
     */
    setupPhaseIndicators() {
        const phases = document.querySelectorAll('.pipeline-phase');
        phases.forEach((phase, index) => {
            // Animation d'apparition décalée
            setTimeout(() => {
                phase.classList.add('animate-in');
            }, index * 300);
            
            // Données contextuelles au hover
            this.addPhaseTooltip(phase);
        });
    }
    
    /**
     * Ajoute un tooltip contextuel à une phase
     */
    addPhaseTooltip(phaseElement) {
        const phaseType = phaseElement.dataset.phase;
        if (!phaseType || !this.data.phases[phaseType]) return;
        
        const phaseData = this.data.phases[phaseType];
        
        phaseElement.addEventListener('mouseenter', (e) => {
            this.showTooltip(e, {
                title: `Phase ${phaseType === 'research' ? 'Recherche' : 'Production'}`,
                content: [
                    `Durée: ${this.formatDuration(phaseData.duration)}`,
                    `Status: ${phaseData.status === 'completed' ? 'Terminée' : 'En cours'}`,
                    `Étapes: ${Object.keys(phaseData.steps).length}`
                ]
            });
        });
        
        phaseElement.addEventListener('mouseleave', () => {
            this.hideTooltip();
        });
    }
    
    /**
     * Configure les interactions avec les étapes
     */
    setupStepInteractions() {
        const steps = document.querySelectorAll('.step-item');
        steps.forEach(step => {
            step.addEventListener('click', (e) => {
                this.showStepDetails(step);
            });
            
            // Animation au hover
            step.addEventListener('mouseenter', () => {
                step.style.transform = 'translateX(8px) scale(1.02)';
            });
            
            step.addEventListener('mouseleave', () => {
                step.style.transform = 'translateX(0) scale(1)';
            });
        });
    }
    
    /**
     * Affiche les détails d'une étape
     */
    showStepDetails(stepElement) {
        const stepType = stepElement.dataset.step;
        const stepData = this.findStepData(stepType);
        
        if (!stepData) return;
        
        // Créer le modal de détails
        const modal = this.createStepModal(stepData);
        document.body.appendChild(modal);
        
        // Animer l'apparition
        setTimeout(() => modal.classList.add('show'), 100);
    }
    
    /**
     * Trouve les données d'une étape
     */
    findStepData(stepType) {
        for (const phase of Object.values(this.data.phases)) {
            if (phase.steps && phase.steps[stepType]) {
                return { ...phase.steps[stepType], type: stepType };
            }
        }
        return null;
    }
    
    /**
     * Crée le modal de détails d'étape
     */
    createStepModal(stepData) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay step-modal';
        
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${stepData.name}</h3>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="step-details">
                        <div class="step-metrics-grid">
                            <div class="step-metric">
                                <span class="metric-label">Durée</span>
                                <span class="metric-value">${this.formatDuration(stepData.duration)}</span>
                            </div>
                            <div class="step-metric">
                                <span class="metric-label">Status</span>
                                <span class="metric-value status-${stepData.status}">
                                    ${stepData.status === 'completed' ? 'Terminée' : 'En cours'}
                                </span>
                            </div>
                        </div>
                        <div class="step-results">
                            <h4>Résultats Principaux</h4>
                            ${this.formatStepResults(stepData.results)}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        return modal;
    }
    
    /**
     * Formate les résultats d'une étape
     */
    formatStepResults(results) {
        if (!results) return '<p>Aucun résultat disponible</p>';
        
        let html = '<div class="results-list">';
        
        for (const [key, value] of Object.entries(results)) {
            const label = this.formatResultLabel(key);
            const formattedValue = this.formatResultValue(value);
            
            html += `
                <div class="result-item">
                    <span class="result-label">${label}</span>
                    <span class="result-value">${formattedValue}</span>
                </div>
            `;
        }
        
        html += '</div>';
        return html;
    }
    
    /**
     * Formate un label de résultat
     */
    formatResultLabel(key) {
        const labels = {
            eventsDetected: 'Événements détectés',
            validation: 'Validation',
            avgPrecipitation: 'Précipitation moyenne',
            avgCoverage: 'Couverture moyenne',
            maxIntensity: 'Intensité maximale',
            modelsCompared: 'Modèles comparés',
            bestClassifier: 'Meilleur classificateur',
            f1Score: 'F1-Score',
            r2Score: 'R² Score',
            algorithmsCompared: 'Algorithmes comparés',
            bestAlgorithm: 'Meilleur algorithme',
            modelsDeployed: 'Modèles déployés'
        };
        
        return labels[key] || key.replace(/([A-Z])/g, ' $1').toLowerCase();
    }
    
    /**
     * Formate une valeur de résultat
     */
    formatResultValue(value) {
        if (typeof value === 'object' && value !== null) {
            if (value.name && value.f1Score) {
                return `${value.name} (F1: ${(value.f1Score * 100).toFixed(1)}%)`;
            }
            if (value.name && value.r2Score) {
                return `${value.name} (R²: ${(value.r2Score * 100).toFixed(1)}%)`;
            }
            return JSON.stringify(value);
        }
        
        if (typeof value === 'number') {
            if (value < 1 && value > 0) {
                return (value * 100).toFixed(1) + '%';
            }
            return this.formatNumber(value);
        }
        
        return String(value);
    }
    
    /**
     * Formate une durée en secondes
     */
    formatDuration(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.round(seconds % 60);
        
        if (minutes > 0) {
            return `${minutes}min ${remainingSeconds}s`;
        }
        return `${remainingSeconds}s`;
    }
    
    /**
     * Affiche un tooltip
     */
    showTooltip(event, { title, content }) {
        this.hideTooltip(); // Cacher le tooltip existant
        
        const tooltip = document.createElement('div');
        tooltip.className = 'pipeline-tooltip';
        tooltip.innerHTML = `
            <div class="tooltip-title">${title}</div>
            <div class="tooltip-content">
                ${content.map(item => `<div class="tooltip-item">${item}</div>`).join('')}
            </div>
        `;
        
        document.body.appendChild(tooltip);
        
        // Positionner le tooltip
        const rect = event.currentTarget.getBoundingClientRect();
        tooltip.style.left = (rect.left + rect.width / 2) + 'px';
        tooltip.style.top = (rect.bottom + 10) + 'px';
        
        // Animer l'apparition
        setTimeout(() => tooltip.classList.add('show'), 50);
        
        this.currentTooltip = tooltip;
    }
    
    /**
     * Cache le tooltip
     */
    hideTooltip() {
        if (this.currentTooltip) {
            this.currentTooltip.remove();
            this.currentTooltip = null;
        }
    }
    
    /**
     * Anime les barres de progression saisonnières
     */
    animateSeasonalBars() {
        setTimeout(() => {
            const rainyBar = document.querySelector('.season-fill.rainy');
            const dryBar = document.querySelector('.season-fill.dry');
            
            if (rainyBar) {
                rainyBar.style.width = '97.8%';
            }
            if (dryBar) {
                dryBar.style.width = '2.2%';
            }
        }, 1500);
    }
    
    /**
     * Anime les barres régionales
     */
    animateRegionalBars() {
        const regionBars = document.querySelectorAll('.region-fill');
        regionBars.forEach((bar, index) => {
            setTimeout(() => {
                const width = bar.parentElement.dataset.width || '50%';
                bar.style.width = width;
            }, 2000 + (index * 200));
        });
    }
    
    /**
     * Retourne les données du pipeline
     */
    getData() {
        return this.data;
    }
    
    /**
     * Retourne les métriques globales
     */
    getGlobalMetrics() {
        return this.data.globalMetrics;
    }
    
    /**
     * Retourne les TOP événements
     */
    getTopEvents(type = 'coverage') {
        return type === 'intensity' ? this.data.top5Intense : this.data.top10Events;
    }
    
    /**
     * Retourne la distribution mensuelle
     */
    getMonthlyDistribution() {
        return this.data.monthlyDistribution;
    }
    
    /**
     * Démarre toutes les animations
     */
    startAnimations() {
        if (!this.isInitialized) {
            console.warn('PipelineManager non initialisé');
            return;
        }
        
        this.setupMetricsAnimations();
        this.animateSeasonalBars();
        this.animateRegionalBars();
    }
    
    /**
     * Nettoie les ressources
     */
    destroy() {
        this.hideTooltip();
        this.animationQueues.clear();
        this.isInitialized = false;
    }
}

// Styles CSS pour les tooltips et modals (injectés dynamiquement)
const PIPELINE_STYLES = `
    .pipeline-tooltip {
        position: absolute;
        background: var(--bg-card);
        border: 2px solid var(--glass-border);
        border-radius: var(--radius-lg);
        padding: var(--space-md);
        box-shadow: var(--shadow-xl);
        z-index: 10000;
        opacity: 0;
        transform: translateX(-50%) translateY(-10px);
        transition: all 0.3s ease;
        max-width: 300px;
        backdrop-filter: blur(8px);
    }
    
    .pipeline-tooltip.show {
        opacity: 1;
        transform: translateX(-50%) translateY(0);
    }
    
    .tooltip-title {
        font-family: var(--font-heading);
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--space-sm);
        font-size: 0.9rem;
    }
    
    .tooltip-content {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    
    .tooltip-item {
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-family: var(--font-mono);
    }
    
    .step-modal .modal-content {
        max-width: 600px;
        width: 90vw;
    }
    
    .step-details {
        padding: var(--space-lg);
    }
    
    .step-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: var(--space-lg);
        margin-bottom: var(--space-xl);
    }
    
    .step-metric {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: var(--space-md);
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        border: 1px solid var(--glass-border);
    }
    
    .step-metric .metric-label {
        font-size: 0.8rem;
        color: var(--text-tertiary);
        margin-bottom: var(--space-sm);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .step-metric .metric-value {
        font-family: var(--font-heading);
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--primary-blue);
    }
    
    .step-metric .metric-value.status-completed {
        color: var(--primary-emerald);
    }
    
    .step-results h4 {
        font-family: var(--font-heading);
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 var(--space-lg) 0;
        border-bottom: 2px solid var(--glass-border);
        padding-bottom: var(--space-sm);
    }
    
    .results-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-md);
    }
    
    .result-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--space-sm) var(--space-md);
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        border: 1px solid var(--glass-border);
    }
    
    .result-label {
        font-weight: 500;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    
    .result-value {
        font-family: var(--font-mono);
        font-weight: 600;
        color: var(--primary-blue);
        font-size: 0.9rem;
    }
    
    .animate-in {
        animation: slideInUp 0.6s ease-out forwards;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;

// Injecter les styles
function injectPipelineStyles() {
    if (!document.getElementById('pipeline-styles')) {
        const style = document.createElement('style');
        style.id = 'pipeline-styles';
        style.textContent = PIPELINE_STYLES;
        document.head.appendChild(style);
    }
}

// Initialisation automatique
document.addEventListener('DOMContentLoaded', () => {
    injectPipelineStyles();
});

// Export pour utilisation globale
window.PipelineManager = PipelineManager;
window.REAL_PIPELINE_DATA = REAL_PIPELINE_DATA;