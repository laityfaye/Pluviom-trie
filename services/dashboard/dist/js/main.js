// services/dashboard/dist/js/main.js
// Orchestrateur principal du dashboard ClimaSen

/**
 * Gestionnaire principal du dashboard
 */
class ClimaSenDashboard {
    constructor() {
        this.components = {
            pipelineManager: null,
            vizManager: null,
            reportsViewer: null,
            dataIntegration: null
        };
        
        this.isInitialized = false;
        this.initializationPromise = null;
    }
    
    /**
     * Initialise le dashboard complet
     */
    async initialize() {
        if (this.isInitialized || this.initializationPromise) {
            return this.initializationPromise;
        }
        
        console.log('🌍 Initialisation ClimaSen Dashboard...');
        
        this.initializationPromise = this._doInitialize();
        return this.initializationPromise;
    }
    
    /**
     * Processus d'initialisation interne
     */
    async _doInitialize() {
        try {
            // 1. Initialiser les composants de base
            await this.initializeBaseComponents();
            
            // 2. Configurer la navigation
            this.setupNavigation();
            
            // 3. Configurer les animations d'entrée
            this.setupEntryAnimations();
            
            // 4. Initialiser les gestionnaires d'événements
            this.setupEventHandlers();
            
            // 5. Démarrer les animations et chargements
            this.startInitialAnimations();
            
            // 6. Configuration finale
            this.finalizeSetup();
            
            this.isInitialized = true;
            console.log('✅ ClimaSen Dashboard initialisé avec succès');
            
            // Notification de succès
            this.showWelcomeMessage();
            
        } catch (error) {
            console.error('❌ Erreur initialisation dashboard:', error);
            this.showErrorMessage('Erreur lors de l\'initialisation du dashboard');
        }
    }
    
    /**
     * Initialise les composants de base
     */
    async initializeBaseComponents() {
        console.log('🔧 Initialisation des composants...');
        
        // Pipeline Manager
        if (window.PipelineManager) {
            this.components.pipelineManager = new window.PipelineManager();
            this.components.pipelineManager.initialize();
        }
        
        // Visualization Manager
        if (window.VisualizationManager) {
            this.components.vizManager = new window.VisualizationManager();
            this.components.vizManager.initialize();
        }
        
        // Reports Viewer
        if (window.ReportsViewer) {
            this.components.reportsViewer = new window.ReportsViewer();
            this.components.reportsViewer.initialize();
        }
        
        // Data Integration
        if (window.DataIntegration) {
            this.components.dataIntegration = new window.DataIntegration();
            this.components.dataIntegration.initialize();
        }
    }
    
    /**
     * Configure la navigation fluide
     */
    setupNavigation() {
        console.log('🧭 Configuration de la navigation...');
        
        // Navigation par sections
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href').substring(1);
                this.scrollToSection(targetId);
            });
        });
        
        // Navigation par ancres internes
        const internalLinks = document.querySelectorAll('a[href^="#"]');
        internalLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const href = link.getAttribute('href');
                if (href !== '#') {
                    e.preventDefault();
                    const targetId = href.substring(1);
                    this.scrollToSection(targetId);
                }
            });
        });
        
        // Mise à jour de la navigation active au scroll
        this.setupScrollSpy();
    }
    
    /**
     * Navigation fluide vers une section
     */
    scrollToSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            const headerHeight = document.querySelector('.header')?.offsetHeight || 80;
            const targetPosition = section.offsetTop - headerHeight - 20;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
            
            // Mettre à jour l'URL sans recharger
            history.replaceState(null, '', `#${sectionId}`);
        }
    }
    
    /**
     * Configuration du scroll spy pour la navigation active
     */
    setupScrollSpy() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-link');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Retirer la classe active de tous les liens
                    navLinks.forEach(link => link.classList.remove('active'));
                    
                    // Ajouter la classe active au lien correspondant
                    const activeLink = document.querySelector(`.nav-link[href="#${entry.target.id}"]`);
                    if (activeLink) {
                        activeLink.classList.add('active');
                    }
                }
            });
        }, {
            rootMargin: '-20% 0px -70% 0px'
        });
        
        sections.forEach(section => observer.observe(section));
    }
    
    /**
     * Configure les animations d'entrée
     */
    setupEntryAnimations() {
        console.log('✨ Configuration des animations...');
        
        // Observer pour les animations au scroll
        const animationObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                    animationObserver.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -100px 0px'
        });
        
        // Éléments à animer
        const animatableElements = document.querySelectorAll(`
            .metric-card,
            .pipeline-phase,
            .result-card,
            .model-card,
            .index-card,
            .report-card,
            .viz-card
        `);
        
        animatableElements.forEach(element => {
            animationObserver.observe(element);
        });
    }
    
    /**
     * Configure les gestionnaires d'événements globaux
     */
    setupEventHandlers() {
        console.log('⚡ Configuration des événements...');
        
        // Gestion des erreurs globales
        window.addEventListener('error', (e) => {
            console.error('Erreur JavaScript:', e.error);
            this.showErrorMessage('Une erreur est survenue dans l\'application');
        });
        
        // Gestion des promesses rejetées
        window.addEventListener('unhandledrejection', (e) => {
            console.error('Promise rejetée:', e.reason);
            this.showErrorMessage('Erreur de traitement des données');
        });
        
        // Gestion du redimensionnement
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.handleResize();
            }, 250);
        });
        
        // Gestion de la visibilité de la page
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseAnimations();
            } else {
                this.resumeAnimations();
            }
        });
        
        // Raccourcis clavier
        this.setupKeyboardShortcuts();
    }
    
    /**
     * Configure les raccourcis clavier
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K : Recherche globale
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.openGlobalSearch();
            }
            
            // Escape : Fermer les modals
            if (e.key === 'Escape') {
                this.closeAllModals();
            }
            
            // Touches de navigation (1-6)
            if (e.key >= '1' && e.key <= '6' && !e.ctrlKey && !e.metaKey) {
                const sections = ['hero', 'pipeline', 'results', 'ml-section', 'teleconnections', 'reports'];
                const sectionIndex = parseInt(e.key) - 1;
                if (sections[sectionIndex]) {
                    this.scrollToSection(sections[sectionIndex]);
                }
            }
        });
    }
    
    /**
     * Démarre les animations initiales
     */
    startInitialAnimations() {
        console.log('🎬 Démarrage des animations...');
        
        // Animation des métriques du héro
        if (this.components.pipelineManager) {
            setTimeout(() => {
                this.components.pipelineManager.startAnimations();
            }, 1000);
        }
        
        // Animation des visualisations si le gestionnaire existe
        if (this.components.vizManager) {
            setTimeout(() => {
                this.components.vizManager.loadVisualizations();
            }, 2000);
        }
        
        // Animation des sections
        this.animateSectionsSequentially();
    }
    
    /**
     * Anime les sections de manière séquentielle
     */
    animateSectionsSequentially() {
        const sections = document.querySelectorAll('section');
        sections.forEach((section, index) => {
            setTimeout(() => {
                section.classList.add('section-ready');
            }, index * 200);
        });
    }
    
    /**
     * Finalise la configuration
     */
    finalizeSetup() {
        console.log('🏁 Finalisation de la configuration...');
        
        // Ajouter les indicateurs de performance
        this.addPerformanceIndicators();
        
        // Configurer le lazy loading des images
        this.setupLazyLoading();
        
        // Initialiser les tooltips
        this.initializeTooltips();
        
        // Configurer les interactions avancées
        this.setupAdvancedInteractions();
    }
    
    /**
     * Ajoute les indicateurs de performance
     */
    addPerformanceIndicators() {
        if ('performance' in window) {
            const loadTime = performance.now();
            console.log(`⚡ Dashboard chargé en ${Math.round(loadTime)}ms`);
            
            // Ajouter un indicateur visuel discret
            const indicator = document.createElement('div');
            indicator.className = 'performance-indicator';
            indicator.textContent = `Chargé en ${Math.round(loadTime)}ms`;
            indicator.style.cssText = `
                position: fixed;
                bottom: 10px;
                right: 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 10px;
                z-index: 1000;
                opacity: 0.7;
            `;
            
            document.body.appendChild(indicator);
            
            // Masquer après 3 secondes
            setTimeout(() => {
                indicator.style.opacity = '0';
                setTimeout(() => indicator.remove(), 300);
            }, 3000);
        }
    }
    
    /**
     * Configure le lazy loading des images
     */
    setupLazyLoading() {
        const images = document.querySelectorAll('img[data-src]');
        
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.classList.remove('lazy');
                        imageObserver.unobserve(img);
                    }
                });
            });
            
            images.forEach(img => imageObserver.observe(img));
        } else {
            // Fallback pour les navigateurs anciens
            images.forEach(img => {
                img.src = img.dataset.src;
                img.classList.remove('lazy');
            });
        }
    }
    
    /**
     * Initialise les tooltips
     */
    initializeTooltips() {
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        tooltipElements.forEach(element => {
            element.addEventListener('mouseenter', this.showTooltip.bind(this));
            element.addEventListener('mouseleave', this.hideTooltip.bind(this));
        });
    }
    
    /**
     * Configure les interactions avancées
     */
    setupAdvancedInteractions() {
        // Effet parallax léger pour le héro
        this.setupParallaxEffect();
        
        // Animations au scroll
        this.setupScrollAnimations();
        
        // Effets de particules (optionnel)
        this.setupParticleEffects();
    }
    
    /**
     * Configure l'effet parallax
     */
    setupParallaxEffect() {
        const heroSection = document.querySelector('.hero');
        if (heroSection) {
            window.addEventListener('scroll', () => {
                const scrolled = window.pageYOffset;
                const rate = scrolled * -0.5;
                heroSection.style.transform = `translateY(${rate}px)`;
            });
        }
    }
    
    /**
     * Configure les animations au scroll
     */
    setupScrollAnimations() {
        let ticking = false;
        
        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    this.updateScrollAnimations();
                    ticking = false;
                });
                ticking = true;
            }
        });
    }
    
    /**
     * Met à jour les animations basées sur le scroll
     */
    updateScrollAnimations() {
        const scrollTop = window.pageYOffset;
        const windowHeight = window.innerHeight;
        
        // Animer les éléments visibles
        const animatedElements = document.querySelectorAll('.scroll-animate');
        animatedElements.forEach(element => {
            const elementTop = element.offsetTop;
            const elementHeight = element.offsetHeight;
            
            if (scrollTop + windowHeight > elementTop && scrollTop < elementTop + elementHeight) {
                element.classList.add('in-view');
            }
        });
    }
    
    /**
     * Configure les effets de particules (optionnel)
     */
    setupParticleEffects() {
        // Implémentation simple d'effets de particules pour le héro
        const hero = document.querySelector('.hero');
        if (hero && window.innerWidth > 1024) {
            this.createParticleSystem(hero);
        }
    }
    
    /**
     * Crée un système de particules simple
     */
    createParticleSystem(container) {
        const canvas = document.createElement('canvas');
        canvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            opacity: 0.1;
            z-index: 1;
        `;
        
        container.style.position = 'relative';
        container.appendChild(canvas);
        
        // Animation simple des particules
        const ctx = canvas.getContext('2d');
        const particles = [];
        
        const resizeCanvas = () => {
            canvas.width = container.offsetWidth;
            canvas.height = container.offsetHeight;
        };
        
        const createParticle = () => ({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            size: Math.random() * 2 + 1
        });
        
        // Initialiser les particules
        for (let i = 0; i < 50; i++) {
            particles.push(createParticle());
        }
        
        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#2563EB';
            
            particles.forEach(particle => {
                particle.x += particle.vx;
                particle.y += particle.vy;
                
                // Rebond sur les bords
                if (particle.x < 0 || particle.x > canvas.width) particle.vx *= -1;
                if (particle.y < 0 || particle.y > canvas.height) particle.vy *= -1;
                
                ctx.beginPath();
                ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                ctx.fill();
            });
            
            requestAnimationFrame(animate);
        };
        
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        animate();
    }
    
    /**
     * Gère le redimensionnement
     */
    handleResize() {
        // Recalculer les layouts si nécessaire
        if (this.components.vizManager) {
            this.components.vizManager.handleResize();
        }
        
        // Mettre à jour les animations
        this.updateScrollAnimations();
    }
    
    /**
     * Met en pause les animations
     */
    pauseAnimations() {
        document.body.classList.add('animations-paused');
    }
    
    /**
     * Reprend les animations
     */
    resumeAnimations() {
        document.body.classList.remove('animations-paused');
    }
    
    /**
     * Ouvre la recherche globale
     */
    openGlobalSearch() {
        // Créer un modal de recherche simple
        const searchModal = document.createElement('div');
        searchModal.className = 'search-modal-overlay';
        searchModal.innerHTML = `
            <div class="search-modal">
                <div class="search-header">
                    <h3>🔍 Recherche Globale</h3>
                    <button class="search-close" onclick="this.closest('.search-modal-overlay').remove()">×</button>
                </div>
                <div class="search-content">
                    <input type="text" class="search-input" placeholder="Rechercher dans les rapports, visualisations..." autofocus>
                    <div class="search-results"></div>
                </div>
            </div>
        `;
        
        document.body.appendChild(searchModal);
        setTimeout(() => searchModal.classList.add('show'), 100);
        
        // Configurer la recherche
        const searchInput = searchModal.querySelector('.search-input');
        const searchResults = searchModal.querySelector('.search-results');
        
        searchInput.addEventListener('input', (e) => {
            this.performGlobalSearch(e.target.value, searchResults);
        });
    }
    
    /**
     * Effectue une recherche globale
     */
    performGlobalSearch(query, resultsContainer) {
        if (query.length < 2) {
            resultsContainer.innerHTML = '';
            return;
        }
        
        const results = [];
        
        // Rechercher dans les rapports
        Object.entries(REPORTS_CONFIG).forEach(([key, config]) => {
            if (config.title.toLowerCase().includes(query.toLowerCase()) ||
                config.description.toLowerCase().includes(query.toLowerCase())) {
                results.push({
                    type: 'rapport',
                    title: config.title,
                    description: config.description,
                    action: () => this.components.reportsViewer?.viewReport(key)
                });
            }
        });
        
        // Rechercher dans les données du pipeline
        if (REAL_PIPELINE_DATA) {
            const searchInPipeline = (obj, path = '') => {
                Object.entries(obj).forEach(([key, value]) => {
                    if (typeof value === 'string' && value.toLowerCase().includes(query.toLowerCase())) {
                        results.push({
                            type: 'données',
                            title: `${path} > ${key}`,
                            description: value,
                            action: () => this.scrollToSection('pipeline')
                        });
                    } else if (typeof value === 'object' && value !== null) {
                        searchInPipeline(value, path ? `${path} > ${key}` : key);
                    }
                });
            };
            
            searchInPipeline(REAL_PIPELINE_DATA);
        }
        
        // Afficher les résultats
        this.displaySearchResults(results.slice(0, 10), resultsContainer);
    }
    
    /**
     * Affiche les résultats de recherche
     */
    displaySearchResults(results, container) {
        if (results.length === 0) {
            container.innerHTML = '<div class="no-results">Aucun résultat trouvé</div>';
            return;
        }
        
        container.innerHTML = results.map(result => `
            <div class="search-result" onclick="this.dispatchEvent(new CustomEvent('select'))">
                <div class="result-type">${result.type}</div>
                <div class="result-title">${result.title}</div>
                <div class="result-description">${result.description}</div>
            </div>
        `).join('');
        
        // Ajouter les événements
        container.querySelectorAll('.search-result').forEach((element, index) => {
            element.addEventListener('select', () => {
                results[index].action();
                document.querySelector('.search-modal-overlay')?.remove();
            });
        });
    }
    
    /**
     * Ferme tous les modals ouverts
     */
    closeAllModals() {
        const modals = document.querySelectorAll('.modal-overlay, .search-modal-overlay');
        modals.forEach(modal => {
            modal.classList.remove('show');
            setTimeout(() => modal.remove(), 300);
        });
    }
    
    /**
     * Affiche un tooltip
     */
    showTooltip(event) {
        const element = event.currentTarget;
        const text = element.dataset.tooltip;
        
        if (!text) return;
        
        const tooltip = document.createElement('div');
        tooltip.className = 'global-tooltip';
        tooltip.textContent = text;
        
        document.body.appendChild(tooltip);
        
        // Positionner le tooltip
        const rect = element.getBoundingClientRect();
        tooltip.style.left = (rect.left + rect.width / 2) + 'px';
        tooltip.style.top = (rect.bottom + 10) + 'px';
        
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
     * Affiche le message de bienvenue
     */
    showWelcomeMessage() {
        const message = document.createElement('div');
        message.className = 'welcome-message';
        message.innerHTML = `
            <div class="welcome-content">
                <h3>🎉 Bienvenue dans ClimaSen!</h3>
                <p>Dashboard d'intelligence climatique pour le Sénégal</p>
                <div class="welcome-stats">
                    <span>1,439 événements analysés</span>
                    <span>91.3% précision ML</span>
                    <span>43 années de données</span>
                </div>
            </div>
        `;
        
        message.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: linear-gradient(135deg, #2563EB 0%, #8B5CF6 100%);
            color: white;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(37, 99, 235, 0.3);
            z-index: 10000;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.5s ease;
            max-width: 300px;
        `;
        
        document.body.appendChild(message);
        
        setTimeout(() => {
            message.style.opacity = '1';
            message.style.transform = 'translateX(0)';
        }, 1000);
        
        setTimeout(() => {
            message.style.opacity = '0';
            message.style.transform = 'translateX(100%)';
            setTimeout(() => message.remove(), 500);
        }, 5000);
    }
    
    /**
     * Affiche un message d'erreur
     */
    showErrorMessage(text) {
        const message = document.createElement('div');
        message.className = 'error-message';
        message.innerHTML = `
            <div class="error-content">
                <span class="error-icon">⚠️</span>
                <span class="error-text">${text}</span>
            </div>
        `;
        
        message.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #EF4444;
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            z-index: 10001;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;
        
        document.body.appendChild(message);
        setTimeout(() => message.style.opacity = '1', 100);
        
        setTimeout(() => {
            message.style.opacity = '0';
            setTimeout(() => message.remove(), 300);
        }, 5000);
    }
    
    /**
     * Retourne les composants initialisés
     */
    getComponents() {
        return this.components;
    }
    
    /**
     * Retourne l'état d'initialisation
     */
    isReady() {
        return this.isInitialized;
    }
    
    /**
     * Nettoie les ressources
     */
    destroy() {
        // Nettoyer tous les composants
        Object.values(this.components).forEach(component => {
            if (component && typeof component.destroy === 'function') {
                component.destroy();
            }
        });
        
        // Supprimer les event listeners
        this.removeEventListeners();
        
        // Nettoyer l'état
        this.isInitialized = false;
        this.initializationPromise = null;
        
        console.log('🧹 Dashboard nettoyé');
    }
    
    /**
     * Supprime les event listeners
     */
    removeEventListeners() {
        // Supprimer les listeners de navigation
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.replaceWith(link.cloneNode(true));
        });
        
        // Supprimer les listeners globaux
        window.removeEventListener('scroll', this.updateScrollAnimations);
        window.removeEventListener('resize', this.handleResize);
    }
}

// Styles pour les nouveaux composants
const DASHBOARD_STYLES = `
    .search-modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(8px);
        z-index: 10000;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding-top: 10vh;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
    }
    
    .search-modal-overlay.show {
        opacity: 1;
        visibility: visible;
    }
    
    .search-modal {
        background: var(--bg-card);
        border-radius: var(--radius-xl);
        border: 2px solid var(--glass-border);
        box-shadow: var(--shadow-2xl);
        width: 600px;
        max-width: 90vw;
        max-height: 80vh;
        overflow: hidden;
        transform: scale(0.9) translateY(-20px);
        transition: transform 0.3s ease;
    }
    
    .search-modal-overlay.show .search-modal {
        transform: scale(1) translateY(0);
    }
    
    .search-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--space-lg) var(--space-xl);
        background: var(--gradient-primary);
        color: white;
    }
    
    .search-header h3 {
        margin: 0;
        font-family: var(--font-heading);
    }
    
    .search-close {
        background: none;
        border: none;
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        padding: 0.25rem;
        border-radius: 50%;
        transition: background 0.3s ease;
    }
    
    .search-close:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    .search-content {
        padding: var(--space-xl);
    }
    
    .search-input {
        width: 100%;
        padding: var(--space-md);
        border: 2px solid var(--glass-border);
        border-radius: var(--radius-lg);
        font-size: 1rem;
        background: var(--bg-secondary);
        color: var(--text-primary);
        margin-bottom: var(--space-lg);
    }
    
    .search-input:focus {
        outline: none;
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
    }
    
    .search-results {
        max-height: 400px;
        overflow-y: auto;
    }
    
    .search-result {
        padding: var(--space-md);
        border-radius: var(--radius-md);
        border: 1px solid var(--glass-border);
        margin-bottom: var(--space-sm);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .search-result:hover {
        background: var(--glass-hover);
        border-color: var(--primary-blue);
        transform: translateY(-2px);
    }
    
    .result-type {
        font-size: 0.8rem;
        color: var(--primary-blue);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    
    .result-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .result-description {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.4;
    }
    
    .no-results {
        text-align: center;
        color: var(--text-tertiary);
        font-style: italic;
        padding: var(--space-xl);
    }
    
    .global-tooltip {
        position: absolute;
        background: var(--bg-card);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-md);
        padding: var(--space-sm) var(--space-md);
        font-size: 0.8rem;
        color: var(--text-primary);
        box-shadow: var(--shadow-lg);
        z-index: 10000;
        opacity: 0;
        transform: translateX(-50%) translateY(-5px);
        transition: all 0.3s ease;
        max-width: 200px;
        backdrop-filter: blur(8px);
    }
    
    .global-tooltip.show {
        opacity: 1;
        transform: translateX(-50%) translateY(0);
    }
    
    .animations-paused * {
        animation-play-state: paused !important;
    }
    
    .section-ready {
        opacity: 1;
        transform: translateY(0);
    }
    
    .animate-in {
        animation: slideInUp 0.6s ease-out forwards;
    }
    
    .in-view {
        transform: translateY(0);
        opacity: 1;
    }
    
    .welcome-content h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
    }
    
    .welcome-content p {
        margin: 0 0 1rem 0;
        opacity: 0.9;
    }
    
    .welcome-stats {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        font-size: 0.8rem;
        opacity: 0.8;
    }
    
    .error-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .error-icon {
        font-size: 1.2rem;
    }
`;

// Injecter les styles
function injectDashboardStyles() {
    if (!document.getElementById('dashboard-main-styles')) {
        const style = document.createElement('style');
        style.id = 'dashboard-main-styles';
        style.textContent = DASHBOARD_STYLES;
        document.head.appendChild(style);
    }
}

// Variables globales
let climaSenDashboard;

// Initialisation automatique
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🌍 ClimaSen Dashboard - Initialisation...');
    
    // Injecter les styles
    injectDashboardStyles();
    
    // Créer et initialiser le dashboard
    climaSenDashboard = new ClimaSenDashboard();
    await climaSenDashboard.initialize();
    
    // Exposer globalement
    window.climaSen = climaSenDashboard;
    
    console.log('✅ ClimaSen Dashboard prêt!');
});

// Export pour utilisation
window.ClimaSenDashboard = ClimaSenDashboard;