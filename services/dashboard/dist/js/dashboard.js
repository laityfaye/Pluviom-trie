// services/dashboard/dist/js/dashboard.js - VERSION THÈME BLANC PREMIUM
// ============================================================================
// FONCTION UTILITAIRE POUR FORCER LES URLs ABSOLUES
// ============================================================================

function ensureAbsoluteUrl(url) {
    if (url.startsWith('http://') || url.startsWith('https://')) {
        return url;
    }
    
    if (url.startsWith('/api/')) {
        return `http://localhost:8000${url}`;
    }
    
    if (!url.startsWith('/')) {
        url = '/' + url;
    }
    
    return `http://localhost:8000${url}`;
}

// ============================================================================
// GESTIONNAIRE DE VISUALISATIONS - VERSION THÈME BLANC PREMIUM
// ============================================================================

class VisualizationManager {
    constructor() {
        this.currentFilter = 'all';
        this.visualizations = [];
        this.visualizationsByCategory = {};
        this.apiBaseUrl = 'http://localhost:8000/api/visualizations';
        this.isLoading = false;
        this.currentMainImage = null;
        this.isFullScreenMode = false; // Nouveau flag pour mode plein écran
        
        // OPTIMISATIONS: Cache et performance
        this.cachedVisualizations = null;
        this.cacheTimestamp = 0;
        this.imagePreloadCache = new Map();
        this.renderQueue = [];
        this.isRendering = false;
        
        this.setupControls();
        this.setupAdvancedEffects();
        this.setupPerformanceOptimizations();
    }

    // NOUVELLE MÉTHODE: Optimisations de performance
    setupPerformanceOptimizations() {
        // Debounce pour les actions répétées
        this.debouncedRender = this.debounce(() => {
            this.processRenderQueue();
        }, 100);
        
        // Intersection Observer pour lazy loading
        this.setupLazyLoading();
        
        // Préchargement intelligent des images
        this.setupImagePreloading();
    }

    // NOUVELLE MÉTHODE: Debounce utility
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // NOUVELLE MÉTHODE: Lazy loading des images
    setupLazyLoading() {
        this.imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.removeAttribute('data-src');
                        this.imageObserver.unobserve(img);
                    }
                }
            });
        }, {
            rootMargin: '50px'
        });
    }

    // NOUVELLE MÉTHODE: Préchargement intelligent
    setupImagePreloading() {
        this.preloadedImages = new Set();
    }

    // NOUVELLE MÉTHODE: Précharger une image
    preloadImage(src) {
        if (this.preloadedImages.has(src)) return Promise.resolve();
        
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.preloadedImages.add(src);
                resolve();
            };
            img.onerror = reject;
            img.src = src;
        });
    }

    // NOUVELLE MÉTHODE: Queue de rendu pour éviter les blocages
    addToRenderQueue(item) {
        this.renderQueue.push(item);
        this.debouncedRender();
    }

    processRenderQueue() {
        if (this.isRendering || this.renderQueue.length === 0) return;
        
        this.isRendering = true;
        
        const batchSize = 5;
        const batch = this.renderQueue.splice(0, batchSize);
        
        // Traitement du batch
        batch.forEach(item => {
            // Traitement optimisé selon le type d'item
            this.processRenderItem(item);
        });
        
        this.isRendering = false;
        
        // Continuer le traitement s'il reste des éléments
        if (this.renderQueue.length > 0) {
            setTimeout(() => this.processRenderQueue(), 16); // ~60fps
        }
    }

    processRenderItem(item) {
        // Méthode pour traiter un élément de la queue
        // Implémentation selon le type d'item
    }

    setupControls() {
        // Filtres par onglets avec effets premium thème blanc
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                if (e.target.dataset.filter && !this.isLoading) {
                    this.setFilter(e.target.dataset.filter);
                    this.createRippleEffect(e.target, e, 'light');
                }
            });
            
            // Effets de survol adaptés au thème blanc
            btn.addEventListener('mouseenter', (e) => {
                this.createHoverGlow(e.target, 'light');
            });
            
            btn.addEventListener('mouseleave', (e) => {
                this.removeHoverGlow(e.target);
            });
        });

        // Bouton actualiser avec effet
        document.getElementById('refreshViz')?.addEventListener('click', (e) => {
            this.createRippleEffect(e.target, e, 'light');
            this.loadVisualizations();
        });
    }

    setupAdvancedEffects() {
        // Effet parallaxe subtil sur scroll
        window.addEventListener('scroll', () => {
            this.handleParallaxEffect();
        }, { passive: true });

        // Effets de particules adaptés au thème blanc
        this.createParticleBackground();
    }

    // Nouvelle méthode pour basculer le mode plein écran
    toggleFullScreenMode(enable = true) {
        this.isFullScreenMode = enable;
        const sidebar = document.querySelector('.sidebar, .nav-sidebar, .left-panel'); // Adapter selon votre structure HTML
        const mainContent = document.querySelector('.main-content, .content-area, .viz-container');
        const body = document.body;

        if (enable) {
            // Masquer la sidebar
            if (sidebar) {
                sidebar.style.display = 'none';
                sidebar.classList.add('hidden-for-album');
            }
            
            // Étendre le contenu principal sur toute la largeur
            if (mainContent) {
                mainContent.style.marginLeft = '0';
                mainContent.style.width = '100%';
                mainContent.classList.add('fullscreen-album');
            }
            
            // Ajouter une classe au body pour les styles globaux
            body.classList.add('album-fullscreen-mode');
            
        } else {
            // Restaurer la sidebar
            if (sidebar) {
                sidebar.style.display = '';
                sidebar.classList.remove('hidden-for-album');
            }
            
            // Restaurer la largeur normale du contenu
            if (mainContent) {
                mainContent.style.marginLeft = '';
                mainContent.style.width = '';
                mainContent.classList.remove('fullscreen-album');
            }
            
            // Retirer la classe du body
            body.classList.remove('album-fullscreen-mode');
        }
    }

    // Modifier la méthode openAlbum existante
    openAlbum(category) {
        const currentCard = document.querySelector(`[data-category="${category}"]`);
        if (currentCard) {
            currentCard.style.transform = 'scale(1.05)';
            currentCard.style.opacity = '0.8';
            
            setTimeout(() => {
                // Activer le mode plein écran AVANT d'ouvrir l'album
                this.toggleFullScreenMode(true);
                
                this.setFilter(category);
                document.getElementById('vizGrid').scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
            }, 200);
        } else {
            this.toggleFullScreenMode(true);
            this.setFilter(category);
        }
    }

    // Nouvelle méthode pour retourner à la vue albums
    returnToAlbumView() {
        // Désactiver le mode plein écran et retourner à la vue "all"
        this.setFilter('all');
        
        // Notification optionnelle
        window.climaSen?.showNotification('← Retour à la collection d\'albums', 'info');
    }

    createRippleEffect(element, event, theme = 'light') {
        const ripple = document.createElement('span');
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;
        
        const rippleColor = theme === 'light' 
            ? 'rgba(37, 99, 235, 0.2)' 
            : 'rgba(255, 255, 255, 0.6)';
        
        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
            background: radial-gradient(circle, ${rippleColor} 0%, transparent 70%);
            border-radius: 50%;
            transform: scale(0);
            animation: ripple 0.6s ease-out;
            pointer-events: none;
            z-index: 1000;
        `;
        
        element.style.position = 'relative';
        element.style.overflow = 'hidden';
        element.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
    }

    createHoverGlow(element, theme = 'light') {
        const glowColor = theme === 'light' 
            ? '0 0 20px rgba(37, 99, 235, 0.3), 0 0 40px rgba(139, 92, 246, 0.15)'
            : '0 0 30px rgba(37, 99, 235, 0.4), 0 0 60px rgba(139, 92, 246, 0.2)';
            
        element.style.boxShadow = glowColor;
        element.style.filter = 'brightness(1.05)';
        element.style.transition = 'all 0.3s ease';
    }

    removeHoverGlow(element) {
        element.style.boxShadow = '';
        element.style.filter = '';
        element.style.transition = 'all 0.3s ease';
    }

    handleParallaxEffect() {
        const scrolled = window.pageYOffset;
        const albums = document.querySelectorAll('.album-card');
        
        albums.forEach((album, index) => {
            const speed = 0.05 + (index % 3) * 0.02; // Effet plus subtil pour thème blanc
            const yPos = -(scrolled * speed);
            if (!album.style.transform.includes('scale') && !album.style.transform.includes('rotate')) {
                album.style.transform = `translateY(${yPos}px)`;
            }
        });
    }

    createParticleBackground() {
        const particleContainer = document.createElement('div');
        particleContainer.className = 'particle-container';
        particleContainer.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
            overflow: hidden;
        `;
        
        // Particules adaptées au thème blanc
        for (let i = 0; i < 15; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.cssText = `
                position: absolute;
                width: ${Math.random() * 3 + 1}px;
                height: ${Math.random() * 3 + 1}px;
                background: linear-gradient(45deg, 
                    rgba(37, 99, 235, 0.3), 
                    rgba(139, 92, 246, 0.2));
                border-radius: 50%;
                left: ${Math.random() * 100}%;
                top: ${Math.random() * 100}%;
                animation: float ${Math.random() * 25 + 15}s linear infinite;
            `;
            particleContainer.appendChild(particle);
        }
        
        document.body.appendChild(particleContainer);
    }

    // Modifier la méthode setFilter pour gérer le mode plein écran
    setFilter(filter) {
        this.currentFilter = filter;
        
        // Animation de transition des boutons
        document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.filter === filter) {
                btn.classList.add('active');
                this.animateFilterActivation(btn);
            }
        });
        
        // Gérer le mode plein écran
        if (filter === 'all') {
            // Retour à la vue normale - désactiver le mode plein écran
            this.toggleFullScreenMode(false);
        } else {
            // Vue d'un album spécifique - activer le mode plein écran
            this.toggleFullScreenMode(true);
        }
        
        this.renderVisualizationsAsAlbums();
    }

    animateFilterActivation(button) {
        // Effet d'onde adapté au thème blanc
        const shockwave = document.createElement('div');
        shockwave.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border: 2px solid rgba(37, 99, 235, 0.4);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            animation: shockwave 0.6s ease-out;
            pointer-events: none;
            z-index: 1000;
        `;
        
        button.style.position = 'relative';
        button.appendChild(shockwave);
        
        setTimeout(() => {
            shockwave.remove();
        }, 600);
    }

    async loadVisualizations() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        const grid = document.getElementById('vizGrid');
        
        // Animation de chargement adaptée au thème blanc
        grid.innerHTML = `
            <div class="loading premium-loading light-theme">
                <div class="loading-content">
                    <div class="spinner-container">
                        <div class="premium-spinner light"></div>
                        <div class="loading-particles light"></div>
                    </div>
                    <h3>Chargement des albums premium...</h3>
                    <p>Préparation de l'expérience visuelle</p>
                    <div class="loading-progress">
                        <div class="progress-bar light"></div>
                    </div>
                </div>
            </div>
        `;

        this.addPremiumLoadingStyles();

        try {
            console.log('🔍 Chargement premium des visualisations...');
            
            // OPTIMISATION: Éviter les appels multiples avec cache
            if (this.cachedVisualizations && (Date.now() - this.cacheTimestamp) < 60000) {
                console.log('📦 Utilisation du cache des visualisations');
                this.visualizations = this.cachedVisualizations;
                this.organizeByCategory();
                this.renderVisualizationsAsAlbums();
                this.animateAlbumsEntrance();
                this.isLoading = false;
                return;
            }
            
            const response = await fetch(`${this.apiBaseUrl}/list`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                },
                // OPTIMISATION: Timeout et cache HTTP
                signal: AbortSignal.timeout(10000) // 10 secondes timeout
            });
            
            if (!response.ok) {
                throw new Error(`Erreur API: ${response.status} - ${response.statusText}`);
            }
            
            const data = await response.json();
            this.visualizations = data.visualizations || [];
            
            // OPTIMISATION: Mise en cache
            this.cachedVisualizations = this.visualizations;
            this.cacheTimestamp = Date.now();
            
            console.log(`✅ ${this.visualizations.length} visualisations premium chargées`);
            
            // OPTIMISATION: Traitement par batch
            this.processVisualizationsInBatches();
            
            this.organizeByCategory();
            
            if (this.visualizations.length === 0) {
                this.showEmptyState();
                return;
            }
            
            // OPTIMISATION: Délai réduit
            setTimeout(() => {
                this.renderVisualizationsAsAlbums();
                this.animateAlbumsEntrance();
                window.climaSen?.showNotification(
                    `📊 ${this.visualizations.length} visualisations organisées en albums premium!`, 
                    'success'
                );
            }, 300); // Réduit de 800ms à 300ms
            
        } catch (error) {
            if (error.name === 'TimeoutError') {
                console.error('⏰ Timeout lors du chargement des visualisations');
                this.showTimeoutState();
            } else {
                console.error('❌ Erreur chargement visualisations:', error);
                this.showErrorState(error);
            }
        } finally {
            this.isLoading = false;
        }
    }

    // NOUVELLE MÉTHODE: Traitement optimisé des visualisations
    processVisualizationsInBatches() {
        const batchSize = 10;
        let processedCount = 0;
        
        // Traitement par batch pour éviter de bloquer l'UI
        const processBatch = () => {
            const endIndex = Math.min(processedCount + batchSize, this.visualizations.length);
            
            for (let i = processedCount; i < endIndex; i++) {
                const viz = this.visualizations[i];
                
                // Forcer les URLs absolues avec validation
                const originalUrl = viz.image;
                viz.image = ensureAbsoluteUrl(viz.image);
                viz.isValid = true;
                
                if (i < 5) { // Log seulement les 5 premiers
                    console.log(`🔧 URL ${i + 1}: ${originalUrl} → ${viz.image}`);
                }
            }
            
            processedCount = endIndex;
            
            // Continuer le traitement si nécessaire
            if (processedCount < this.visualizations.length) {
                setTimeout(processBatch, 0); // Yield to browser
            }
        };
        
        processBatch();
    }

    // NOUVELLE MÉTHODE: État de timeout
    showTimeoutState() {
        const grid = document.getElementById('vizGrid');
        grid.innerHTML = `
            <div class="loading timeout-state light-theme">
                <div class="loading-content">
                    <div style="font-size: 4rem; margin-bottom: 1rem; color: #F59E0B;">⏰</div>
                    <h3 style="color: #F59E0B;">Timeout de Connexion</h3>
                    <p style="color: var(--text-secondary);">
                        Le chargement des visualisations prend plus de temps que prévu.<br>
                        Vérifiez votre connexion et l'état de l'API.
                    </p>
                    <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 2rem;">
                        <button class="btn btn-primary" onclick="vizManager.loadVisualizations()">
                            🔄 Réessayer
                        </button>
                        <button class="btn btn-secondary" onclick="vizManager.showEmptyState()">
                            📊 Mode Hors Ligne
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    addPremiumLoadingStyles() {
        if (document.getElementById('premium-loading-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'premium-loading-styles';
        style.textContent = `
            .premium-loading.light-theme {
                background: linear-gradient(135deg, 
                    rgba(37, 99, 235, 0.03) 0%, 
                    rgba(139, 92, 246, 0.02) 50%, 
                    rgba(16, 185, 129, 0.02) 100%);
                border: 2px solid rgba(226, 232, 240, 0.8);
                backdrop-filter: blur(20px);
                border-radius: 2rem;
                padding: 4rem;
                text-align: center;
                position: relative;
                overflow: hidden;
                box-shadow: 0 10px 40px rgba(15, 23, 42, 0.08);
            }
            
            .premium-loading.light-theme::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 2px;
                background: linear-gradient(90deg, 
                    transparent, 
                    var(--primary-blue), 
                    transparent);
                animation: shimmer 2s ease-in-out infinite;
            }
            
            .premium-spinner.light {
                width: 60px;
                height: 60px;
                border: 4px solid rgba(226, 232, 240, 0.3);
                border-top: 4px solid var(--primary-blue);
                border-right: 4px solid var(--primary-purple);
                border-radius: 50%;
                animation: premiumSpin 1.5s linear infinite;
                box-shadow: 0 4px 20px rgba(37, 99, 235, 0.1);
            }
            
            .progress-bar.light {
                background: var(--gradient-primary);
                box-shadow: 0 2px 10px rgba(37, 99, 235, 0.2);
            }
        `;
        document.head.appendChild(style);
    }

    // MODIFICATION: Supprimer complètement le header d'album
    renderSingleAlbum(category) {
        const grid = document.getElementById('vizGrid');
        const visualizations = this.visualizationsByCategory[category] || [];
        const categoryInfo = this.getCategoryInfo(category);
        
        if (visualizations.length === 0) {
            grid.innerHTML = `
                <div class="loading light-theme">
                    <div class="loading-content">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                        <h3 style="margin-bottom: 0.5rem;">Aucune visualisation dans cet album</h3>
                        <p style="color: var(--text-secondary);">L'album "${categoryInfo.label}" est vide</p>
                        <button class="btn btn-secondary" onclick="vizManager.returnToAlbumView()">← Retour aux albums</button>
                    </div>
                </div>
            `;
            return;
        }

        // SUPPRESSION COMPLÈTE DU HEADER - Affichage direct de la galerie
        const galleryHtml = this.createGalleryView(visualizations);

        // Afficher UNIQUEMENT la galerie sans header
        grid.innerHTML = galleryHtml;
        
        this.addGalleryStyles();
        this.addFullscreenStyles(); // Nouvelle méthode pour les styles plein écran
        this.setupGalleryInteractions(visualizations);
        
        // Réactiver les animations
        setTimeout(() => {
            document.querySelectorAll('.fade-in:not(.visible)').forEach((el, index) => {
                setTimeout(() => el.classList.add('visible'), index * 100);
            });
        }, 100);
    }

    // Ajouter un bouton retour flottant discret en haut à gauche
    createGalleryView(visualizations) {
        const mainImage = visualizations[0]; // Première image comme principale
        this.currentMainImage = mainImage;
        
        return `
            <!-- Bouton retour flottant discret -->
            <div class="floating-back-button" onclick="vizManager.returnToAlbumView()">
                <span class="back-arrow">←</span>
            </div>
            
            <div class="gallery-container fade-in">
                <!-- Image principale -->
                <div class="main-image-container">
                    <div class="main-image-wrapper">
                        <img id="mainImage" 
                             src="${ensureAbsoluteUrl(mainImage.image)}" 
                             alt="${mainImage.title || 'Visualisation'}"
                             onload="this.style.opacity='1';"
                             onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjYwMCIgdmlld0JveD0iMCAwIDgwMCA2MDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSI4MDAiIGhlaWdodD0iNjAwIiBmaWxsPSIjRjhGQUZDIi8+CjxjaXJjbGUgY3g9IjQwMCIgY3k9IjMwMCIgcj0iNjAiIGZpbGw9IiMyNTYzRUIiLz4KPHR4dCB4PSI0MDAiIHk9IjQwMCIgZm9udC1mYW1pbHk9IkludGVyIiBmb250LXNpemU9IjE4IiBmaWxsPSIjMzM0MTU1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5WaXN1YWxpc2F0aW9uIFByaW5jaXBhbGU8L3R4dD4KPC9zdmc+';"
                             style="opacity: 0; transition: opacity 0.5s ease; cursor: pointer; position: relative; z-index: 10;"
                             onclick="openVisualization('${ensureAbsoluteUrl(mainImage.image)}', '${mainImage.title || 'Visualisation'}', 0)">
                        
                        <div class="image-overlay">
                            <div class="overlay-content">
                                <div class="overlay-icon">🔍</div>
                                <div class="overlay-text">Cliquer pour agrandir</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="main-image-info">
                        <h3 id="mainImageTitle">${mainImage.title || 'Sans titre'}</h3>
                        <p id="mainImageDescription">${mainImage.description || 'Aucune description disponible'}</p>
                        <div class="image-meta">
                            <span id="mainImageDate">${mainImage.date_formatted || mainImage.date || 'Date inconnue'}</span>
                            ${mainImage.script ? `<span>📄 ${mainImage.script}</span>` : ''}
                        </div>
                        <div class="image-actions">
                            <button class="btn btn-primary" onclick="openVisualization(document.getElementById('mainImage').src, '${mainImage.title || 'Visualisation'}', 0)">
                                🔍 Voir en grand
                            </button>
                            <button class="btn btn-secondary" onclick="downloadVisualization(document.getElementById('mainImage').src, '${mainImage.title || 'visualization'}')">
                                📥 Télécharger
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- Miniatures -->
                <div class="thumbnails-container">
                    <h4>Toutes les visualisations (${visualizations.length})</h4>
                    <div class="thumbnails-grid">
                        ${visualizations.map((viz, index) => `
                            <div class="thumbnail-item ${index === 0 ? 'active' : ''}" 
                                 data-index="${index}"
                                 onclick="vizManager.selectMainImage(${index})">
                                <img src="${ensureAbsoluteUrl(viz.image)}" 
                                     alt="${viz.title || 'Visualisation'}"
                                     onload="this.style.opacity='1';"
                                     onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgdmlld0JveD0iMCAwIDIwMCAxNTAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMTUwIiBmaWxsPSIjRjFGNUY5Ii8+CjxjaXJjbGUgY3g9IjEwMCIgY3k9Ijc1IiByPSIyMCIgZmlsbD0iIzY0NzQ4QiIvPgo8dGV4dCB4PSIxMDAiIHk9IjEyMCIgZm9udC1mYW1pbHk9IkludGVyIiBmb250LXNpemU9IjEwIiBmaWxsPSIjNjQ3NDhCIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5NaW5pYXR1cmU8L3RleHQ+Cjwvc3ZnPgo=';"
                                     style="opacity: 0; transition: opacity 0.3s ease;">
                                <div class="thumbnail-overlay">
                                    <span class="thumbnail-number">${index + 1}</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    selectMainImage(index) {
        const visualizations = this.visualizationsByCategory[this.currentFilter] || [];
        const selectedViz = visualizations[index];
        
        if (!selectedViz) return;
        
        // Animer la transition
        const mainImage = document.getElementById('mainImage');
        const mainImageTitle = document.getElementById('mainImageTitle');
        const mainImageDescription = document.getElementById('mainImageDescription');
        const mainImageDate = document.getElementById('mainImageDate');
        
        // Effet de transition
        mainImage.style.opacity = '0';
        mainImage.style.transform = 'scale(0.95)';
        
        setTimeout(() => {
            mainImage.src = ensureAbsoluteUrl(selectedViz.image);
            mainImageTitle.textContent = selectedViz.title || 'Sans titre';
            mainImageDescription.textContent = selectedViz.description || 'Aucune description disponible';
            mainImageDate.textContent = selectedViz.date_formatted || selectedViz.date || 'Date inconnue';
            
            // Actualiser les boutons d'action avec l'index correct
            const viewBtn = document.querySelector('.image-actions .btn-primary');
            const downloadBtn = document.querySelector('.image-actions .btn-secondary');
            
            if (viewBtn) {
                viewBtn.onclick = () => {
                    console.log(`Opening modal from selectMainImage with index: ${index}`);
                    openVisualization(mainImage.src, selectedViz.title || 'Visualisation', index);
                };
            }
            if (downloadBtn) {
                downloadBtn.onclick = () => downloadVisualization(mainImage.src, selectedViz.title || 'visualization');
            }
            
            mainImage.style.opacity = '1';
            mainImage.style.transform = 'scale(1)';
        }, 200);
        
        // Mettre à jour les miniatures actives
        document.querySelectorAll('.thumbnail-item').forEach((thumb, i) => {
            thumb.classList.toggle('active', i === index);
        });
        
        this.currentMainImage = selectedViz;
    }

    setupGalleryInteractions(visualizations) {
        // Navigation au clavier
        document.addEventListener('keydown', (e) => {
            if (this.currentFilter === 'all') return;
            
            const currentIndex = visualizations.findIndex(viz => viz === this.currentMainImage);
            
            if (e.key === 'ArrowLeft' && currentIndex > 0) {
                this.selectMainImage(currentIndex - 1);
            } else if (e.key === 'ArrowRight' && currentIndex < visualizations.length - 1) {
                this.selectMainImage(currentIndex + 1);
            }
        });
        
        // Swipe pour mobile (simplifié)
        let startX = 0;
        const mainImageContainer = document.querySelector('.main-image-container');
        
        if (mainImageContainer) {
            mainImageContainer.addEventListener('touchstart', (e) => {
                startX = e.touches[0].clientX;
            });
            
            mainImageContainer.addEventListener('touchend', (e) => {
                const endX = e.changedTouches[0].clientX;
                const diff = startX - endX;
                
                if (Math.abs(diff) > 50) { // Seuil de swipe
                    const currentIndex = visualizations.findIndex(viz => viz === this.currentMainImage);
                    
                    if (diff > 0 && currentIndex < visualizations.length - 1) {
                        this.selectMainImage(currentIndex + 1);
                    } else if (diff < 0 && currentIndex > 0) {
                        this.selectMainImage(currentIndex - 1);
                    }
                }
            });
        }
    }

    addGalleryStyles() {
        if (document.getElementById('gallery-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'gallery-styles';
        style.textContent = `
            .gallery-container {
                margin-top: 2rem;
            }
            
            .main-image-container {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 2rem;
                margin-bottom: 3rem;
                background: var(--bg-card);
                border: 2px solid var(--glass-border);
                border-radius: 2rem;
                padding: 2rem;
                box-shadow: var(--shadow-lg);
            }
            
            .main-image-wrapper {
                position: relative;
                border-radius: 1.5rem;
                overflow: hidden;
                cursor: pointer;
                transition: all 0.3s ease;
                background: var(--bg-secondary);
            }
            
            .main-image-wrapper:hover {
                transform: scale(1.02);
                box-shadow: var(--shadow-xl);
            }
            
            .main-image-wrapper img {
                width: 100%;
                height: auto;
                min-height: 400px;
                object-fit: cover;
                transition: all 0.3s ease;
                display: block !important;
                pointer-events: auto !important;
                cursor: pointer !important;
            }
            
            .image-overlay {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(135deg, 
                    rgba(37, 99, 235, 0.8) 0%, 
                    rgba(139, 92, 246, 0.8) 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                opacity: 0;
                transition: all 0.3s ease;
                pointer-events: none;
                z-index: 5;
            }
            
            .main-image-wrapper:hover .image-overlay {
                opacity: 1;
                pointer-events: auto;
                cursor: pointer;
            }
            
            .overlay-content {
                text-align: center;
                color: white;
                font-weight: 600;
                pointer-events: none;
            }
            
            .overlay-icon {
                font-size: 3rem;
                margin-bottom: 0.5rem;
            }
            
            .overlay-text {
                font-size: 1.1rem;
            }
            
            .main-image-info {
                padding: 1rem 0;
            }
            
            .main-image-info h3 {
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 1rem;
                line-height: 1.3;
            }
            
            .main-image-info p {
                color: var(--text-secondary);
                line-height: 1.6;
                margin-bottom: 1.5rem;
            }
            
            .image-meta {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
                margin-bottom: 2rem;
                font-size: 0.875rem;
                color: var(--text-tertiary);
            }
            
            .image-actions {
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
            }
            
            .image-actions .btn {
                flex: 1;
                min-width: 120px;
            }
            
            .thumbnails-container {
                margin-top: 2rem;
            }
            
            .thumbnails-container h4 {
                color: var(--text-primary);
                margin-bottom: 1.5rem;
                font-size: 1.2rem;
                font-weight: 600;
            }
            
            .thumbnails-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 1rem;
            }
            
            .thumbnail-item {
                position: relative;
                border-radius: 1rem;
                overflow: hidden;
                cursor: pointer;
                transition: all 0.3s ease;
                border: 3px solid transparent;
                background: var(--bg-secondary);
            }
            
            .thumbnail-item.active {
                border-color: var(--primary-blue);
                box-shadow: 0 0 20px rgba(37, 99, 235, 0.3);
            }
            
            .thumbnail-item:hover {
                transform: scale(1.05);
                box-shadow: var(--shadow-md);
            }
            
            .thumbnail-item img {
                width: 100%;
                height: 120px;
                object-fit: cover;
                transition: all 0.3s ease;
            }
            
            .thumbnail-overlay {
                position: absolute;
                top: 0.5rem;
                right: 0.5rem;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 0.25rem 0.5rem;
                border-radius: 0.5rem;
                font-size: 0.75rem;
                font-weight: 600;
            }
            
            .thumbnail-item.active .thumbnail-overlay {
                background: var(--primary-blue);
            }
            
            /* Responsive */
            @media (max-width: 768px) {
                .main-image-container {
                    grid-template-columns: 1fr;
                    gap: 1.5rem;
                    padding: 1.5rem;
                }
                
                .thumbnails-grid {
                    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                    gap: 0.75rem;
                }
                
                .thumbnail-item img {
                    height: 100px;
                }
                
                .image-actions {
                    flex-direction: column;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Nouvelle méthode pour ajouter les styles du mode plein écran
    addFullscreenStyles() {
        if (document.getElementById('fullscreen-album-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'fullscreen-album-styles';
        style.textContent = `
            /* Styles pour le mode plein écran des albums */
            .album-fullscreen-mode {
                overflow-x: hidden;
            }
            
            .album-fullscreen-mode .sidebar,
            .album-fullscreen-mode .nav-sidebar,
            .album-fullscreen-mode .left-panel {
                display: none !important;
            }
            
            .fullscreen-album {
                margin-left: 0 !important;
                width: 100% !important;
                max-width: 100vw !important;
                padding-left: 2rem !important;
                padding-right: 2rem !important;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            /* Bouton retour flottant discret */
            .floating-back-button {
                position: fixed;
                top: 2rem;
                left: 2rem;
                z-index: 10000;
                background: var(--bg-card);
                border: 2px solid var(--glass-border);
                border-radius: 50%;
                width: 50px;
                height: 50px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: var(--shadow-lg);
                backdrop-filter: blur(10px);
                opacity: 0.8;
            }
            
            .floating-back-button:hover {
                opacity: 1;
                transform: scale(1.1);
                box-shadow: var(--shadow-xl);
                background: var(--primary-blue);
                color: white;
                border-color: var(--primary-blue);
            }
            
            .back-arrow {
                font-size: 1.5rem;
                font-weight: bold;
                transition: transform 0.3s ease;
            }
            
            .floating-back-button:hover .back-arrow {
                transform: translateX(-2px);
            }
            
            /* Amélioration de la galerie en mode plein écran */
            .album-fullscreen-mode .gallery-container {
                max-width: none;
                width: 100%;
                margin-top: 1rem; /* Réduire l'espace en haut */
            }
            
            .album-fullscreen-mode .main-image-container {
                grid-template-columns: 2.5fr 1fr;
                gap: 3rem;
            }
            
            .album-fullscreen-mode .thumbnails-grid {
                grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                gap: 1rem;
            }
            
            /* Responsive pour le mode plein écran */
            @media (max-width: 1024px) {
                .floating-back-button {
                    top: 1.5rem;
                    left: 1.5rem;
                    width: 45px;
                    height: 45px;
                }
                
                .back-arrow {
                    font-size: 1.3rem;
                }
            }
            
            @media (max-width: 768px) {
                .fullscreen-album {
                    padding-left: 1rem !important;
                    padding-right: 1rem !important;
                }
                
                .floating-back-button {
                    top: 1rem;
                    left: 1rem;
                    width: 40px;
                    height: 40px;
                }
                
                .back-arrow {
                    font-size: 1.2rem;
                }
                
                .album-fullscreen-mode .main-image-container {
                    grid-template-columns: 1fr;
                    gap: 2rem;
                }
            }
            
            /* Animation d'entrée pour le mode plein écran */
            @keyframes fullscreenSlideIn {
                from {
                    opacity: 0;
                    transform: translateX(50px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            .fullscreen-album {
                animation: fullscreenSlideIn 0.5s ease-out;
            }
            
            /* Animation pour le bouton flottant */
            @keyframes floatingButtonEntrance {
                from {
                    opacity: 0;
                    transform: translateX(-100px) scale(0.5);
                }
                to {
                    opacity: 0.8;
                    transform: translateX(0) scale(1);
                }
            }
            
            .floating-back-button {
                animation: floatingButtonEntrance 0.6s ease-out 0.3s both;
            }
            
            /* Améliorer l'accessibilité du bouton flottant */
            .floating-back-button:focus {
                outline: 3px solid var(--primary-blue);
                outline-offset: 2px;
            }
            
            /* Indication visuelle pour les utilisateurs */
            .floating-back-button::after {
                content: '';
                position: absolute;
                top: -2px;
                left: -2px;
                right: -2px;
                bottom: -2px;
                border: 2px solid var(--primary-blue);
                border-radius: 50%;
                opacity: 0;
                animation: pulseRing 2s infinite;
            }
            
            @keyframes pulseRing {
                0% {
                    opacity: 0;
                    transform: scale(0.8);
                }
                50% {
                    opacity: 0.3;
                    transform: scale(1.2);
                }
                100% {
                    opacity: 0;
                    transform: scale(1.4);
                }
            }
            
            /* Styles pour les raccourcis clavier */
            .keyboard-hint {
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                background: var(--bg-card);
                border: 2px solid var(--glass-border);
                border-radius: 1rem;
                padding: 1rem 1.5rem;
                font-size: 0.8rem;
                color: var(--text-secondary);
                box-shadow: var(--shadow-md);
                z-index: 1000;
                opacity: 0.7;
                transition: opacity 0.3s ease;
            }
            
            .keyboard-hint:hover {
                opacity: 1;
            }
            
            .album-fullscreen-mode .keyboard-hint {
                display: block;
            }
            
            .keyboard-hint kbd {
                background: var(--text-primary);
                color: var(--bg-primary);
                padding: 0.2rem 0.4rem;
                border-radius: 0.3rem;
                font-size: 0.7rem;
                margin: 0 0.2rem;
            }
        `;
        document.head.appendChild(style);
    }

    organizeByCategory() {
        this.visualizationsByCategory = {};
        
        this.visualizations.forEach(viz => {
            const category = viz.category || 'uncategorized';
            if (!this.visualizationsByCategory[category]) {
                this.visualizationsByCategory[category] = [];
            }
            this.visualizationsByCategory[category].push(viz);
        });
        
        // Trier chaque catégorie par date et qualité
        Object.keys(this.visualizationsByCategory).forEach(category => {
            this.visualizationsByCategory[category].sort((a, b) => {
                if (a.isValid !== b.isValid) {
                    return b.isValid ? 1 : -1;
                }
                return new Date(b.date || 0) - new Date(a.date || 0);
            });
        });
        
        console.log('📁 Visualisations organisées (thème blanc):', this.visualizationsByCategory);
    }

    showEmptyState() {
        const grid = document.getElementById('vizGrid');
        grid.innerHTML = `
            <div class="loading empty-state light-theme">
                <div class="loading-content">
                    <div style="font-size: 4rem; margin-bottom: 1rem; color: var(--text-tertiary);">📊</div>
                    <h3 style="color: var(--text-primary);">Collection Premium Vide</h3>
                    <p style="color: var(--text-secondary);">
                        Aucune visualisation premium n'est disponible actuellement.<br>
                        Les albums seront automatiquement mis à jour dès que du contenu sera ajouté.
                    </p>
                    <button class="btn btn-primary" onclick="vizManager.loadVisualizations()">
                        🔄 Actualiser la Collection
                    </button>
                </div>
            </div>
        `;
    }

    showErrorState(error) {
        const grid = document.getElementById('vizGrid');
        grid.innerHTML = `
            <div class="loading error-state light-theme">
                <div class="loading-content">
                    <div style="font-size: 4rem; margin-bottom: 1rem; color: #EF4444;">⚠️</div>
                    <h3 style="color: #EF4444;">Erreur de Chargement Premium</h3>
                    <p style="color: var(--text-secondary);">
                        Impossible de charger les visualisations premium.<br>
                        ${error.message}
                    </p>
                    <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 2rem;">
                        <button class="btn btn-primary" onclick="vizManager.loadVisualizations()">
                            🔄 Réessayer
                        </button>
                        <button class="btn btn-secondary" onclick="vizManager.showEmptyState()">
                            📊 Mode Démo
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    renderVisualizationsAsAlbums() {
        const grid = document.getElementById('vizGrid');
        
        if (this.currentFilter === 'all') {
            this.renderAllAlbums();
        } else {
            this.renderSingleAlbum(this.currentFilter);
        }
    }

    renderAllAlbums() {
        const grid = document.getElementById('vizGrid');
        
        const albumsHtml = Object.keys(this.visualizationsByCategory).map(category => {
            const visualizations = this.visualizationsByCategory[category];
            const categoryInfo = this.getCategoryInfo(category);
            
            const previewImages = visualizations.filter(viz => viz.isValid).slice(0, 4);
            
            return `
                <div class="album-card fade-in light-theme" data-category="${category}">
                    <div class="album-header">
                        <div class="album-icon light">${categoryInfo.icon}</div>
                        <div class="album-info">
                            <h3 class="album-title">${categoryInfo.label}</h3>
                            <p class="album-count">${visualizations.length} visualisation${visualizations.length > 1 ? 's' : ''}</p>
                        </div>
                    </div>
                    
                    <div class="album-preview light">
                        ${previewImages.map((viz, index) => `
                            <div class="preview-image" style="z-index: ${4 - index};">
                                <img src="${viz.image}" alt="${viz.title}" 
                                     onload="this.style.opacity='1'; this.parentElement.style.transform='scale(1)';"
                                     onerror="this.parentElement.style.display='none';"
                                     style="opacity: 0; transition: opacity 0.5s ease;">
                            </div>
                        `).join('')}
                        
                        ${visualizations.length > 4 ? `
                            <div class="preview-overlay light">
                                <span>+${visualizations.length - 4}</span>
                            </div>
                        ` : ''}
                    </div>
                    
                    <div class="album-description">
                        <p>${categoryInfo.description}</p>
                    </div>
                    
                    <div class="album-actions">
                        <button class="btn btn-primary" onclick="vizManager.openAlbum('${category}')">
                            Explorer l'Album →
                        </button>
                    </div>
                </div>
            `;
        }).join('');
        
        grid.innerHTML = `
            <div class="albums-grid">
                ${albumsHtml}
            </div>
        `;
        
        this.addAlbumStyles();
        this.setupPremiumAlbumInteractions();
        
        setTimeout(() => {
            document.querySelectorAll('.fade-in:not(.visible)').forEach((el, index) => {
                setTimeout(() => el.classList.add('visible'), index * 200);
            });
        }, 100);
    }

    // MÉTHODE OPTIMISÉE: Interactions avec throttling
    setupPremiumAlbumInteractions() {
        // Throttle pour les événements mousemove
        const throttle = (func, limit) => {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    func.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            }
        };

        document.querySelectorAll('.album-card').forEach(card => {
            // OPTIMISATION: Throttle les événements mousemove
            const throttledMouseMove = throttle((e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                
                const rotateX = (y - centerY) / 40; // Moins intense
                const rotateY = (centerX - x) / 40;
                
                // Utiliser transform3d pour l'accélération GPU
                card.style.transform = `
                    translate3d(0, -8px, 0) 
                    scale3d(1.02, 1.02, 1) 
                    rotateX(${rotateX}deg) 
                    rotateY(${rotateY}deg)
                `;
            }, 16); // ~60fps
            
            card.addEventListener('mousemove', throttledMouseMove);
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translate3d(0, 0, 0) scale3d(1, 1, 1) rotateX(0) rotateY(0)';
            });
            
            // OPTIMISATION: Délégation d'événements pour les clics
            card.addEventListener('click', (e) => {
                if (!e.target.closest('button')) {
                    const category = card.dataset.category;
                    
                    // Feedback visuel immédiat
                    card.style.transition = 'transform 0.1s ease';
                    card.style.transform = 'scale(0.98)';
                    
                    setTimeout(() => {
                        card.style.transition = '';
                        card.style.transform = '';
                        this.openAlbum(category);
                    }, 100);
                }
            });
        });
    }

    getCategoryInfo(category) {
        const categoryInfos = {
            detection: { 
                icon: '🎯', 
                label: 'Détection Événements', 
                description: 'Identification et analyse des événements de précipitations extrêmes avec algorithmes IA avancés' 
            },
            spatial: { 
                icon: '🗺️', 
                label: 'Analyse Spatiale', 
                description: 'Cartographie géospatiale et analyse des patterns géographiques des événements climatiques' 
            },
            temporal: { 
                icon: '📈', 
                label: 'Analyse Temporelle', 
                description: 'Évolution temporelle, tendances et prédictions basées sur les séries chronologiques' 
            },
            'machine-learning': { 
                icon: '🤖', 
                label: 'Intelligence Artificielle', 
                description: 'Modèles d\'apprentissage automatique et réseaux de neurones pour prédictions climatiques' 
            },
            clustering: { 
                icon: '🎭', 
                label: 'Classification Avancée', 
                description: 'Algorithmes de clustering et classification pour identifier les patterns cachés' 
            },
            teleconnections: { 
                icon: '🌊', 
                label: 'Téléconnexions Climatiques', 
                description: 'Corrélations et influences à distance des phénomènes climatiques globaux' 
            },
            'spatial_top5': { 
                icon: '🥇', 
                label: 'TOP 5 Premium', 
                description: 'Analyse exclusive des 5 événements les plus intenses et significatifs' 
            },
            uncategorized: { 
                icon: '📊', 
                label: 'Collection Diverse', 
                description: 'Visualisations diverses et analyses exploratoires complémentaires' 
            }
        };
        
        return categoryInfos[category] || categoryInfos.uncategorized;
    }

    addAlbumStyles() {
        if (document.getElementById('album-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'album-styles';
        style.textContent = `
            .albums-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 2rem;
                margin-top: 2rem;
            }
            
            .album-card.light-theme {
                background: var(--bg-card);
                border: 2px solid var(--glass-border);
                border-radius: 2rem;
                padding: 1.5rem;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
                position: relative;
                overflow: hidden;
                box-shadow: var(--shadow-lg);
            }
            
            .album-card.light-theme::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(135deg, 
                    rgba(37, 99, 235, 0.02) 0%, 
                    rgba(139, 92, 246, 0.02) 100%);
                opacity: 0;
                transition: opacity 0.3s ease;
                z-index: 1;
            }
            
            .album-card.light-theme:hover::before {
                opacity: 1;
            }
            
            .album-card.light-theme:hover {
                transform: translateY(-12px) scale(1.02);
                border-color: rgba(37, 99, 235, 0.4);
                box-shadow: var(--shadow-xl);
            }
            
            .album-header {
                display: flex;
                align-items: center;
                gap: 1rem;
                margin-bottom: 1.5rem;
                z-index: 2;
                position: relative;
            }
            
            .album-icon.light {
                font-size: 2.5rem;
                background: var(--gradient-primary);
                color: white;
                padding: 0.8rem;
                border-radius: 1.5rem;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: var(--shadow-colored);
                transition: transform 0.3s ease;
            }
            
            .album-card:hover .album-icon.light {
                transform: scale(1.1) rotate(5deg);
            }
            
            .album-title {
                font-size: 1.25rem;
                font-weight: 700;
                color: var(--text-primary);
                margin: 0 0 0.25rem 0;
                line-height: 1.3;
            }
            
            .album-count {
                color: var(--text-tertiary);
                margin: 0;
                font-size: 0.875rem;
                font-weight: 500;
            }
            
            .album-preview.light {
                position: relative;
                height: 200px;
                border-radius: 1.5rem;
                overflow: hidden;
                margin-bottom: 1rem;
                background: linear-gradient(135deg, 
                    rgba(37, 99, 235, 0.05) 0%, 
                    rgba(139, 92, 246, 0.03) 50%, 
                    rgba(16, 185, 129, 0.02) 100%);
                border: 2px solid var(--glass-border);
            }
            
            .preview-image {
                position: absolute;
                width: 70%;
                height: 70%;
                border-radius: 1rem;
                overflow: hidden;
                box-shadow: var(--shadow-lg);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                transform: scale(0.9);
                border: 2px solid rgba(255, 255, 255, 0.8);
            }
            
            .preview-image:nth-child(1) { 
                top: 10%; left: 15%; 
                transform: rotate(-5deg) scale(0.9); 
                z-index: 4;
            }
            .preview-image:nth-child(2) { 
                top: 15%; left: 20%; 
                transform: rotate(3deg) scale(0.9); 
                z-index: 3;
            }
            .preview-image:nth-child(3) { 
                top: 20%; left: 25%; 
                transform: rotate(-2deg) scale(0.9); 
                z-index: 2;
            }
            .preview-image:nth-child(4) { 
                top: 25%; left: 30%; 
                transform: rotate(4deg) scale(0.9); 
                z-index: 1;
            }
            
            .album-card:hover .preview-image {
                transform: rotate(0deg) scale(1.05);
            }
            
            .album-card:hover .preview-image:nth-child(1) { 
                transform: rotate(-2deg) scale(1.08); 
            }
            
            .preview-image img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                filter: brightness(0.95) contrast(1.05);
                transition: filter 0.3s ease;
            }
            
            .album-card:hover .preview-image img {
                filter: brightness(1) contrast(1.1);
            }
            
            .preview-overlay.light {
                position: absolute;
                bottom: 10px;
                right: 10px;
                background: var(--gradient-primary);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 1.5rem;
                font-weight: 600;
                font-size: 0.875rem;
                box-shadow: var(--shadow-colored);
                z-index: 10;
            }
            
            .album-description {
                margin-bottom: 1.5rem;
                z-index: 2;
                position: relative;
            }
            
            .album-description p {
                color: var(--text-secondary);
                line-height: 1.6;
                margin: 0;
                font-size: 0.9rem;
            }
            
            .album-actions {
                margin-top: auto;
                z-index: 2;
                position: relative;
            }
            
            .album-actions .btn {
                background: var(--gradient-primary);
                border: none;
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 1.2rem;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: var(--shadow-colored);
                position: relative;
                overflow: hidden;
                width: 100%;
            }
            
            .album-actions .btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, 
                    transparent, 
                    rgba(255, 255, 255, 0.2), 
                    transparent);
                transition: left 0.6s ease;
            }
            
            .album-actions .btn:hover::before {
                left: 100%;
            }
            
            .album-actions .btn:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-glow);
            }
        `;
        document.head.appendChild(style);
    }

    animateAlbumsEntrance() {
        const albums = document.querySelectorAll('.album-card');
        albums.forEach((album, index) => {
            album.style.opacity = '0';
            album.style.transform = 'translateY(50px) scale(0.9)';
            
            setTimeout(() => {
                album.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
                album.style.opacity = '1';
                album.style.transform = 'translateY(0) scale(1)';
            }, index * 100);
        });
    }
}

// ============================================================================
// SYSTÈME CLIMASEN - VERSION THÈME BLANC PREMIUM
// ============================================================================

class ClimaSenSystem {
    constructor() {
        this.services = [
            { name: 'Neural API', url: 'http://localhost:8000/health', port: '8000', status: 'online' },
            { name: 'Grafana Analytics', url: 'http://localhost:3001', port: '3001', status: 'online' },
            { name: 'Prometheus Core', url: 'http://localhost:9090', port: '9090', status: 'online' },
            { name: 'TimescaleDB', url: 'http://localhost:5432', port: '5432', status: 'online' },
            { name: 'Redis Cache', url: 'http://localhost:6379', port: '6379', status: 'online' },
            { name: 'Neural Engine', url: 'internal://ml-engine', port: 'internal', status: 'training' }
        ];
        this.notifications = document.getElementById('notifications');
        this.isMonitoring = false;
        this.init();
    }

    init() {
        console.log('🌟 ClimaSen Climate AI Platform Premium (Thème Blanc)');
        console.log('🚀 Initialisation du système premium...');
        
        this.setupScrollEffects();
        this.setupAnimations();
        this.setupInteractions();
        this.startMonitoring();
        this.setupAdvancedFeatures();
        
        setTimeout(() => {
            this.showNotification('🌟 ClimaSen Premium Platform Initialisée!', 'success');
        }, 1500);
        
        console.log('✅ Système premium opérationnel (thème blanc)');
    }

    setupAdvancedFeatures() {
        this.setupTypingEffect();
        this.setupKeyboardShortcuts();
        this.setupConnectionQuality();
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                window.vizManager?.loadVisualizations();
                this.showNotification('🔄 Actualisation des visualisations', 'info');
            }
            
            if (e.key === 'Escape' && window.vizManager?.currentFilter !== 'all') {
                window.vizManager?.setFilter('all');
                this.showNotification('← Retour aux albums', 'info');
            }
        });
    }

    showNotification(message, type = 'info', duration = 4000) {
        if (!this.notifications) return;
        
        const notification = document.createElement('div');
        notification.className = `notification ${type} premium-notification light-theme`;
        
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: '💡'
        };
        
        const icon = icons[type] || icons.info;
        
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.2rem; animation: bounce 0.6s ease-out;">${icon}</span>
                <span style="font-weight: 500; color: var(--text-primary);">${message}</span>
                <button class="notification-close light" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        // Styles adaptés au thème blanc
        notification.style.cssText = `
            background: var(--bg-card);
            border: 2px solid var(--glass-border);
            border-radius: 1.2rem;
            padding: 1rem 1.5rem;
            margin-bottom: 0.5rem;
            box-shadow: var(--shadow-lg);
            transform: translateX(100%);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(20px);
        `;
        
        this.notifications.appendChild(notification);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            notification.style.opacity = '0';
            
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 400);
        }, duration);
        
        this.addNotificationStyles();
    }

    addNotificationStyles() {
        if (document.getElementById('notification-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            .notification-close.light {
                background: none;
                border: none;
                color: var(--text-tertiary);
                font-size: 1.2rem;
                cursor: pointer;
                padding: 0;
                margin-left: auto;
                transition: color 0.3s ease;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .notification-close.light:hover {
                color: var(--text-primary);
                background: rgba(239, 68, 68, 0.1);
            }
            
            @keyframes bounce {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.2); }
            }
        `;
        document.head.appendChild(style);
    }

    setupScrollEffects() {
        const nav = document.getElementById('nav');
        let lastScrollY = window.scrollY;
        
        window.addEventListener('scroll', () => {
            const currentScrollY = window.scrollY;
            
            if (currentScrollY > 100) {
                nav.classList.add('scrolled');
                
                if (currentScrollY > lastScrollY) {
                    nav.style.transform = 'translateY(-100%)';
                } else {
                    nav.style.transform = 'translateY(0)';
                }
            } else {
                nav.classList.remove('scrolled');
                nav.style.transform = 'translateY(0)';
            }
            
            lastScrollY = currentScrollY;
        }, { passive: true });
    }

    setupAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    setTimeout(() => {
                        entry.target.classList.add('visible');
                    }, index * 100);
                }
            });
        }, observerOptions);

        document.querySelectorAll('.fade-in, .slide-in').forEach(el => {
            observer.observe(el);
        });
    }

    setupInteractions() {
        // Copie des commandes adaptée au thème blanc
        document.querySelectorAll('.command-block').forEach(block => {
            block.addEventListener('click', async () => {
                try {
                    await navigator.clipboard.writeText(block.textContent.trim());
                    
                    const originalBg = block.style.background;
                    const originalColor = block.style.color;
                    
                    block.style.background = 'var(--primary-emerald)';
                    block.style.color = 'white';
                    block.style.transform = 'scale(1.02)';
                    block.style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.4)';
                    
                    setTimeout(() => {
                        block.style.background = originalBg;
                        block.style.color = originalColor;
                        block.style.transform = 'scale(1)';
                        block.style.boxShadow = '';
                    }, 1200);
                    
                    this.showNotification('✨ Commande copiée avec succès!', 'success');
                } catch (err) {
                    this.showNotification('❌ Erreur lors de la copie', 'error');
                }
            });
        });
    }

    async startMonitoring() {
        if (this.isMonitoring) return;
        this.isMonitoring = true;
        
        console.log('🔍 Démarrage de la surveillance premium (thème blanc)...');
        await this.performHealthCheck();
        setInterval(() => this.performHealthCheck(), 30000);
    }

    async performHealthCheck() {
        console.log(`🔮 Vérification système à ${new Date().toLocaleTimeString()}`);
        
        const statusElements = document.querySelectorAll('.status-pulse');
        statusElements.forEach((element, index) => {
            setTimeout(() => {
                element.style.background = 'var(--primary-emerald)';
                element.style.boxShadow = '0 0 15px rgba(16, 185, 129, 0.5)';
                element.style.transform = 'scale(1.1)';
                
                setTimeout(() => {
                    element.style.transform = 'scale(1)';
                }, 200);
            }, index * 100);
        });
    }

    setupTypingEffect() {
        const typingElements = document.querySelectorAll('[data-typing]');
        typingElements.forEach(element => {
            const text = element.textContent;
            element.textContent = '';
            let i = 0;
            
            const typeInterval = setInterval(() => {
                element.textContent += text[i];
                i++;
                if (i >= text.length) {
                    clearInterval(typeInterval);
                }
            }, 50);
        });
    }

    setupConnectionQuality() {
        const updateConnectionStatus = () => {
            const quality = Math.random() > 0.1 ? 'excellent' : 'poor';
            const statusElement = document.querySelector('.connection-status');
            
            if (statusElement) {
                statusElement.className = `connection-status ${quality}`;
                statusElement.textContent = quality === 'excellent' ? '🟢 Excellent' : '🟡 Instable';
            }
        };
        
        updateConnectionStatus();
        setInterval(updateConnectionStatus, 30000);
    }
}

// ============================================================================
// FONCTIONS GLOBALES PREMIUM
// ============================================================================

// Fonction pour ouvrir les visualisations en modal avec navigation
function openVisualization(imageUrl, title, currentIndex = 0) {
    // Récupérer toutes les visualisations de la catégorie actuelle
    const visualizations = window.vizManager?.visualizationsByCategory[window.vizManager?.currentFilter] || [];
    let currentVizIndex = currentIndex || visualizations.findIndex(viz => ensureAbsoluteUrl(viz.image) === imageUrl) || 0;
    
    const modal = document.createElement('div');
    modal.className = 'image-modal light-theme';
    modal.id = 'imageModal';
    
    const updateModalContent = (index) => {
        currentVizIndex = index; // Mettre à jour l'index courant
        const viz = visualizations[index];
        const vizImageUrl = ensureAbsoluteUrl(viz.image);
        const vizTitle = viz.title || 'Visualisation';
        
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <button class="modal-close" onclick="closeModal()">×</button>
                    <div class="modal-counter">${index + 1} / ${visualizations.length}</div>
                </div>
                
                <div class="modal-image-container">
                    ${index > 0 ? '<button class="modal-nav modal-prev" onclick="navigateModal(-1)">‹</button>' : ''}
                    <img id="modalImage" src="${vizImageUrl}" alt="${vizTitle}" 
                         style="max-width: 90vw; max-height: 85vh; border-radius: 1rem; box-shadow: var(--shadow-2xl); cursor: grab;"
                         onload="this.style.opacity='1';" 
                         onerror="this.style.opacity='1';"
                         style="opacity: 0; transition: opacity 0.3s ease;">
                    ${index < visualizations.length - 1 ? '<button class="modal-nav modal-next" onclick="navigateModal(1)">›</button>' : ''}
                </div>
                
                <div class="modal-info">
                    <div class="modal-title">${vizTitle}</div>
                    <div class="modal-description">${viz.description || 'Aucune description disponible'}</div>
                    <div class="modal-meta">
                        <span>${viz.date_formatted || viz.date || 'Date inconnue'}</span>
                        ${viz.script ? `<span>📄 ${viz.script}</span>` : ''}
                    </div>
                </div>
                
                <div class="modal-actions">
                    <button class="btn btn-secondary" onclick="downloadVisualization('${vizImageUrl}', '${vizTitle}')">
                        📥 Télécharger
                    </button>
                </div>
            </div>
        `;
    };
    
    // Fonction globale pour naviguer dans le modal
    window.navigateModal = (direction) => {
        console.log(`Navigation: direction=${direction}, currentIndex=${currentVizIndex}, total=${visualizations.length}`);
        
        const newIndex = currentVizIndex + direction;
        
        // Vérifier les limites
        if (newIndex >= 0 && newIndex < visualizations.length) {
            console.log(`Navigating to index: ${newIndex}`);
            updateModalContent(newIndex);
            
            // Mettre à jour l'image principale en arrière-plan
            if (window.vizManager) {
                window.vizManager.selectMainImage(newIndex);
            }
        } else {
            console.log(`Navigation blocked: newIndex=${newIndex} out of bounds [0, ${visualizations.length - 1}]`);
        }
    };
    
    // Fonction globale pour fermer le modal
    window.closeModal = () => {
        const modal = document.getElementById('imageModal');
        if (modal) {
            modal.style.opacity = '0';
            modal.style.transform = 'scale(0.95)';
            setTimeout(() => modal.remove(), 300);
        }
        // Nettoyer les fonctions globales
        delete window.navigateModal;
        delete window.closeModal;
    };
    
    updateModalContent(currentVizIndex);
    
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(15px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        animation: modalFadeIn 0.3s ease;
        opacity: 0;
        transform: scale(0.95);
        transition: all 0.3s ease;
    `;
    
    // Événements
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            window.closeModal();
        }
    });
    
    // Navigation au clavier améliorée
    const handleKeydown = (e) => {
        console.log(`Key pressed: ${e.key}, currentIndex: ${currentVizIndex}`);
        
        switch(e.key) {
            case 'Escape':
                window.closeModal();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                if (currentVizIndex > 0) {
                    window.navigateModal(-1);
                }
                break;
            case 'ArrowRight':
                e.preventDefault();
                if (currentVizIndex < visualizations.length - 1) {
                    window.navigateModal(1);
                }
                break;
        }
    };
    
    document.addEventListener('keydown', handleKeydown);
    
    // Nettoyer l'événement lors de la fermeture
    const originalClose = window.closeModal;
    window.closeModal = () => {
        document.removeEventListener('keydown', handleKeydown);
        originalClose();
    };
    
    document.body.appendChild(modal);
    
    // Animer l'ouverture
    setTimeout(() => {
        modal.style.opacity = '1';
        modal.style.transform = 'scale(1)';
    }, 10);

    // Debug: afficher les informations de navigation
    console.log(`Modal opened: currentIndex=${currentVizIndex}, totalImages=${visualizations.length}`);
    console.log('Available visualizations:', visualizations.map((v, i) => `${i}: ${v.title}`));
    
    // Ajouter styles pour le modal
    addEnhancedModalStyles();
}

function addEnhancedModalStyles() {
    if (document.getElementById('enhanced-modal-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'enhanced-modal-styles';
    style.textContent = `
        @keyframes modalFadeIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
        
        .modal-content {
            position: relative;
            text-align: center;
            animation: modalImageZoom 0.4s ease;
            max-width: 95vw;
            max-height: 95vh;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        @keyframes modalImageZoom {
            from { transform: scale(0.8); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 1rem;
            margin-bottom: 0.5rem;
        }
        
        .modal-close {
            background: var(--bg-card);
            border: 2px solid var(--glass-border);
            color: var(--text-primary);
            width: 40px;
            height: 40px;
            border-radius: 50%;
            font-size: 1.5rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-md);
        }
        
        .modal-close:hover {
            background: #EF4444;
            color: white;
            transform: scale(1.1);
        }
        
        .modal-counter {
            background: var(--bg-card);
            border: 2px solid var(--glass-border);
            padding: 0.5rem 1rem;
            border-radius: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            font-size: 0.9rem;
        }
        
        .modal-image-container {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 2rem;
        }
        
        .modal-nav {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: var(--bg-card);
            border: 2px solid var(--glass-border);
            color: var(--text-primary);
            width: 50px;
            height: 50px;
            border-radius: 50%;
            font-size: 2rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-lg);
            backdrop-filter: blur(10px);
            z-index: 10001;
        }
        
        .modal-prev {
            left: -70px;
        }
        
        .modal-next {
            right: -70px;
        }
        
        .modal-nav:hover {
            background: var(--primary-blue);
            color: white;
            transform: translateY(-50%) scale(1.1);
            box-shadow: var(--shadow-xl);
        }
        
        .modal-info {
            text-align: center;
            padding: 1rem;
            background: var(--bg-card);
            border: 2px solid var(--glass-border);
            border-radius: 1.5rem;
            box-shadow: var(--shadow-md);
            backdrop-filter: blur(10px);
        }
        
        .modal-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }
        
        .modal-description {
            color: var(--text-secondary);
            margin-bottom: 1rem;
            line-height: 1.5;
        }
        
        .modal-meta {
            display: flex;
            justify-content: center;
            gap: 1rem;
            font-size: 0.875rem;
            color: var(--text-tertiary);
            flex-wrap: wrap;
        }
        
        .modal-actions {
            display: flex;
            justify-content: center;
            gap: 1rem;
        }
        
        .modal-actions .btn {
            padding: 0.75rem 1.5rem;
            border-radius: 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
        }
        
        .modal-actions .btn-secondary {
            background: var(--gradient-primary);
            color: white;
            box-shadow: var(--shadow-colored);
        }
        
        .modal-actions .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-glow);
        }
        
        /* Responsive pour le modal */
        @media (max-width: 768px) {
            .modal-nav {
                width: 40px;
                height: 40px;
                font-size: 1.5rem;
            }
            
            .modal-prev {
                left: -50px;
            }
            
            .modal-next {
                right: -50px;
            }
            
            .modal-info {
                padding: 0.75rem;
            }
            
            .modal-title {
                font-size: 1.1rem;
            }
            
            .modal-meta {
                flex-direction: column;
                gap: 0.5rem;
            }
        }
        
        @media (max-width: 480px) {
            .modal-nav {
                position: fixed;
                top: auto;
                bottom: 2rem;
                transform: none;
                width: 45px;
                height: 45px;
            }
            
            .modal-prev {
                left: 2rem;
                bottom: 2rem;
            }
            
            .modal-next {
                right: 2rem;
                bottom: 2rem;
            }
            
            .modal-header {
                padding: 0 0.5rem;
            }
            
            .modal-close {
                width: 35px;
                height: 35px;
                font-size: 1.2rem;
            }
        }
    `;
    document.head.appendChild(style);
}

// Fonction pour télécharger les visualisations
function downloadVisualization(imageUrl, filename) {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = `${filename.replace(/[^a-z0-9]/gi, '_')}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    window.climaSen?.showNotification('📥 Téléchargement démarré!', 'success');
}

// Konami Code Easter Egg
function setupKonamiCode() {
    let konamiCode = [];
    const konamiSequence = [
        'ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown',
        'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight',
        'KeyB', 'KeyA'
    ];
    
    document.addEventListener('keydown', (e) => {
        konamiCode.push(e.code);
        
        if (konamiCode.length > konamiSequence.length) {
            konamiCode.shift();
        }
        
        if (konamiCode.join(',') === konamiSequence.join(',')) {
            activateEasterEgg();
            konamiCode = [];
        }
    });
}

function activateEasterEgg() {
    // Effet spécial Konami Code adapté au thème blanc
    const rainbow = document.createElement('div');
    rainbow.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, 
            rgba(255, 0, 0, 0.1),
            rgba(255, 165, 0, 0.1),
            rgba(255, 255, 0, 0.1),
            rgba(0, 128, 0, 0.1),
            rgba(0, 0, 255, 0.1),
            rgba(75, 0, 130, 0.1),
            rgba(238, 130, 238, 0.1));
        background-size: 400% 400%;
        animation: rainbowFlow 3s ease-in-out;
        pointer-events: none;
        z-index: 9999;
    `;
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes rainbowFlow {
            0% { background-position: 0% 50%; opacity: 0; }
            50% { background-position: 100% 50%; opacity: 1; }
            100% { background-position: 0% 50%; opacity: 0; }
        }
    `;
    document.head.appendChild(style);
    
    document.body.appendChild(rainbow);
    
    setTimeout(() => {
        rainbow.remove();
        style.remove();
    }, 3000);
    
    window.climaSen?.showNotification('🎉 Konami Code activé! Easter Egg débloqué!', 'success', 3000);
    
    // Activer le mode fête temporaire
    document.body.classList.add('party-mode');
    setTimeout(() => {
        document.body.classList.remove('party-mode');
    }, 10000);
}

// Ajouter un raccourci clavier pour revenir aux albums
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && window.vizManager?.currentFilter !== 'all') {
        window.vizManager?.returnToAlbumView();
    }
});

// Ajouter une indication visuelle des raccourcis clavier
function addKeyboardHints() {
    const hint = document.createElement('div');
    hint.className = 'keyboard-hint';
    hint.innerHTML = `
        <div>💡 <strong>Raccourcis:</strong></div>
        <div><kbd>Esc</kbd> Retour aux albums</div>
        <div><kbd>←</kbd> <kbd>→</kbd> Navigation images</div>
    `;
    
    // N'ajouter que si pas déjà présent
    if (!document.querySelector('.keyboard-hint')) {
        document.body.appendChild(hint);
    }
}

// Variables globales
let climaSen;
let vizManager;

// Initialisation premium thème blanc OPTIMISÉE
document.addEventListener('DOMContentLoaded', () => {
    console.log('🌟 Initialisation ClimaSen Premium (Thème Blanc)...');
    
    // OPTIMISATION: Initialisation asynchrone
    const initializeAsync = async () => {
        try {
            // Initialiser les systèmes de base
            climaSen = new ClimaSenSystem();
            vizManager = new VisualizationManager();
            
            // Exposition globale
            window.climaSen = climaSen;
            window.vizManager = vizManager;
            
            // OPTIMISATION: Chargement différé des visualisations
            setTimeout(() => {
                // Vérifier si l'API est disponible avant de charger
                fetch('http://localhost:8000/health', { 
                    method: 'GET',
                    timeout: 5000 
                })
                .then(response => {
                    if (response.ok) {
                        console.log('✅ API disponible, chargement des visualisations...');
                        vizManager.loadVisualizations();
                    } else {
                        console.warn('⚠️ API non disponible, mode dégradé');
                        vizManager.showEmptyState();
                    }
                })
                .catch(error => {
                    console.warn('⚠️ Erreur connexion API:', error);
                    vizManager.showTimeoutState();
                });
            }, 1000); // Réduit de 2000ms à 1000ms
            
            // Configuration du Konami Code
            setupKonamiCode();
            
            // Ajout des styles globaux pour le thème blanc
            addGlobalWhiteThemeStyles();
            
            // Ajouter les indices de raccourcis clavier
            addKeyboardHints();
            
            console.log('✨ ClimaSen Premium (Thème Blanc) prêt!');
            
        } catch (error) {
            console.error('❌ Erreur lors de l\'initialisation:', error);
            
            // Mode de secours
            document.getElementById('vizGrid').innerHTML = `
                <div class="loading error-state light-theme">
                    <div class="loading-content">
                        <div style="font-size: 4rem; margin-bottom: 1rem; color: #EF4444;">⚠️</div>
                        <h3 style="color: #EF4444;">Erreur d'Initialisation</h3>
                        <p style="color: var(--text-secondary);">
                            Une erreur est survenue lors du démarrage.<br>
                            Rechargez la page ou vérifiez la console.
                        </p>
                        <button class="btn btn-primary" onclick="location.reload()">
                            🔄 Recharger la Page
                        </button>
                    </div>
                </div>
            `;
        }
    };
    
    // Démarrer l'initialisation
    initializeAsync();
});

// Styles globaux additionnels pour le thème blanc OPTIMISÉS
function addGlobalWhiteThemeStyles() {
    if (document.getElementById('global-white-theme-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'global-white-theme-styles';
    style.textContent = `
        /* Animations et transitions globales pour thème blanc OPTIMISÉES */
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        @keyframes premiumSpin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes shockwave {
            to {
                width: 100px;
                height: 100px;
                opacity: 0;
            }
        }
        
        @keyframes ripple {
            to {
                transform: scale(2);
                opacity: 0;
            }
        }
        
        @keyframes float {
            0% {
                transform: translateY(100vh) rotate(0deg);
                opacity: 0;
            }
            10% {
                opacity: 1;
            }
            90% {
                opacity: 1;
            }
            100% {
                transform: translateY(-100px) rotate(360deg);
                opacity: 0;
            }
        }
        
        /* OPTIMISATION: Lazy loading styles */
        .lazy-image {
            transition: opacity 0.3s ease;
        }
        
        .lazy-image[data-src] {
            background: linear-gradient(90deg, 
                rgba(226, 232, 240, 0.2) 25%, 
                rgba(226, 232, 240, 0.4) 50%, 
                rgba(226, 232, 240, 0.2) 75%);
            background-size: 200% 100%;
            animation: shimmerLoading 1.5s infinite;
        }
        
        @keyframes shimmerLoading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        
        /* Mode fête pour Easter Egg OPTIMISÉ */
        .party-mode * {
            animation: partyPulse 0.5s ease-in-out infinite alternate !important;
        }
        
        @keyframes partyPulse {
            0% { filter: hue-rotate(0deg) brightness(1); }
            100% { filter: hue-rotate(360deg) brightness(1.2); }
        }
        
        /* États de chargement améliorés */
        .loading.light-theme {
            background: var(--bg-card);
            border: 2px solid var(--glass-border);
            box-shadow: var(--shadow-xl);
        }
        
        .empty-state.light-theme {
            background: linear-gradient(135deg, 
                rgba(37, 99, 235, 0.02) 0%, 
                rgba(139, 92, 246, 0.01) 100%);
        }
        
        .error-state.light-theme {
            background: linear-gradient(135deg, 
                rgba(239, 68, 68, 0.02) 0%, 
                rgba(239, 68, 68, 0.01) 100%);
            border-color: rgba(239, 68, 68, 0.2);
        }
        
        .timeout-state.light-theme {
            background: linear-gradient(135deg, 
                rgba(245, 158, 11, 0.02) 0%, 
                rgba(245, 158, 11, 0.01) 100%);
            border-color: rgba(245, 158, 11, 0.2);
        }
        
        /* OPTIMISATION: Utiliser transform3d pour l'accélération GPU */
        .album-card {
            transform: translate3d(0, 0, 0);
            backface-visibility: hidden;
            perspective: 1000px;
        }
        
        .preview-image {
            transform: translate3d(0, 0, 0);
            backface-visibility: hidden;
        }
        
        /* Responsive amélioré pour thème blanc */
        @media (max-width: 768px) {
            .main-image-container {
                grid-template-columns: 1fr;
                gap: 1.5rem;
                padding: 1.5rem;
            }
            
            .thumbnails-grid {
                grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
                gap: 0.5rem;
            }
            
            .thumbnail-item img {
                height: 80px;
            }
            
            .image-actions {
                flex-direction: column;
            }
            
            .album-card.light-theme {
                margin: 0 0.5rem;
                padding: 1rem;
            }
            
            /* OPTIMISATION: Désactiver les animations 3D sur mobile */
            .album-card {
                transform: none !important;
            }
        }
        
        @media (max-width: 480px) {
            .albums-grid {
                grid-template-columns: 1fr;
                gap: 1rem;
                padding: 0.5rem;
            }
            
            .main-image-wrapper img {
                min-height: 250px;
            }
            
            .thumbnails-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        
        /* OPTIMISATION: Accessibilité renforcée pour le thème blanc */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
            
            .album-card {
                transform: none !important;
            }
        }
        
        /* Focus visible amélioré */
        *:focus-visible {
            outline: 2px solid var(--primary-blue);
            outline-offset: 2px;
            border-radius: 4px;
        }
        
        /* Sélection de texte personnalisée pour thème blanc */
        ::selection {
            background: rgba(37, 99, 235, 0.2);
            color: var(--text-primary);
        }
        
        ::-moz-selection {
            background: rgba(37, 99, 235, 0.2);
            color: var(--text-primary);
        }
        
        /* Scrollbar personnalisée pour thème blanc */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--gradient-primary);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-blue);
        }
        
        /* OPTIMISATION: Amélioration des performances de rendu */
        .albums-grid {
            contain: layout style paint;
        }
        
        .album-card {
            contain: layout style paint;
            will-change: transform;
        }
        
        .preview-image {
            contain: layout style paint;
        }
        
        /* OPTIMISATION: Préchargement des états hover */
        .album-card:hover {
            will-change: auto;
        }
    `;
    document.head.appendChild(style);
}

// OPTIMISATION: Fonction utilitaire pour déboguer les performances
function debugVisualization(viz) {
    console.group(`🔍 Debug Visualisation: ${viz.title || 'Sans titre'}`);
    console.log('📊 Données:', viz);
    console.log('🖼️ URL Image:', viz.image);
    console.log('📅 Date:', viz.date || 'Non définie');
    console.log('📁 Catégorie:', viz.category || 'Non catégorisée');
    console.log('✅ Valide:', viz.isValid ? 'Oui' : 'Non');
    console.groupEnd();
}

// OPTIMISATION: Monitoring des performances
function monitorPerformance() {
    if ('performance' in window) {
        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.entryType === 'navigation') {
                    console.log(`⚡ Navigation: ${Math.round(entry.loadEventEnd - entry.fetchStart)}ms`);
                }
                if (entry.entryType === 'resource' && entry.name.includes('/api/visualizations/')) {
                    console.log(`📡 API Call: ${Math.round(entry.responseEnd - entry.startTime)}ms`);
                }
            }
        });
        
        observer.observe({entryTypes: ['navigation', 'resource']});
    }
}

// Démarrer le monitoring
monitorPerformance();

// Export pour utilisation externe si nécessaire
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        VisualizationManager,
        ClimaSenSystem,
        ensureAbsoluteUrl,
        openVisualization,
        downloadVisualization
    };
}

// Gestion des erreurs globales OPTIMISÉE
window.addEventListener('error', (e) => {
    console.error('🚨 Erreur globale détectée:', e.error);
    if (!e.error.message.includes('Script error')) { // Éviter les erreurs CORS
        window.climaSen?.showNotification('⚠️ Une erreur est survenue', 'error');
    }
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('🚨 Promise rejetée non gérée:', e.reason);
    if (e.reason && e.reason.name !== 'AbortError') { // Éviter les timeouts normaux
        window.climaSen?.showNotification('⚠️ Erreur de traitement', 'error');
    }
});

console.log('📝 Dashboard JavaScript (Thème Blanc Premium) chargé avec optimisations de performance !');ale
    window.climaSen = climaSen;
    window.vizManager = vizManager;
    
    // Auto-chargement des visualisations avec délai premium
    setTimeout(() => {
        vizManager.loadVisualizations();
    }, 2000);
    
    // Configuration du Konami Code
    setupKonamiCode();
    
    // Ajout des styles globaux pour le thème blanc
    addGlobalWhiteThemeStyles();
    
    // Ajouter les indices de raccourcis clavier
    addKeyboardHints();
    
    console.log('✨ ClimaSen Premium (Thème Blanc) prêt!');
;

// Styles globaux additionnels pour le thème blanc
function addGlobalWhiteThemeStyles() {
    if (document.getElementById('global-white-theme-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'global-white-theme-styles';
    style.textContent = `
        /* Animations et transitions globales pour thème blanc */
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        @keyframes premiumSpin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes shockwave {
            to {
                width: 100px;
                height: 100px;
                opacity: 0;
            }
        }
        
        @keyframes ripple {
            to {
                transform: scale(2);
                opacity: 0;
            }
        }
        
        @keyframes float {
            0% {
                transform: translateY(100vh) rotate(0deg);
                opacity: 0;
            }
            10% {
                opacity: 1;
            }
            90% {
                opacity: 1;
            }
            100% {
                transform: translateY(-100px) rotate(360deg);
                opacity: 0;
            }
        }
        
        /* Mode fête pour Easter Egg */
        .party-mode * {
            animation: partyPulse 0.5s ease-in-out infinite alternate !important;
        }
        
        @keyframes partyPulse {
            0% { filter: hue-rotate(0deg) brightness(1); }
            100% { filter: hue-rotate(360deg) brightness(1.2); }
        }
        
        /* États de chargement améliorés */
        .loading.light-theme {
            background: var(--bg-card);
            border: 2px solid var(--glass-border);
            box-shadow: var(--shadow-xl);
        }
        
        .empty-state.light-theme {
            background: linear-gradient(135deg, 
                rgba(37, 99, 235, 0.02) 0%, 
                rgba(139, 92, 246, 0.01) 100%);
        }
        
        .error-state.light-theme {
            background: linear-gradient(135deg, 
                rgba(239, 68, 68, 0.02) 0%, 
                rgba(239, 68, 68, 0.01) 100%);
            border-color: rgba(239, 68, 68, 0.2);
        }
        
        /* Responsive amélioré pour thème blanc */
        @media (max-width: 768px) {
            .main-image-container {
                grid-template-columns: 1fr;
                gap: 1.5rem;
                padding: 1.5rem;
            }
            
            .thumbnails-grid {
                grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
                gap: 0.5rem;
            }
            
            .thumbnail-item img {
                height: 80px;
            }
            
            .image-actions {
                flex-direction: column;
            }
            
            .album-card.light-theme {
                margin: 0 0.5rem;
                padding: 1rem;
            }
        }
        
        @media (max-width: 480px) {
            .albums-grid {
                grid-template-columns: 1fr;
                gap: 1rem;
                padding: 0.5rem;
            }
            
            .main-image-wrapper img {
                min-height: 250px;
            }
            
            .thumbnails-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        
        /* Accessibilité renforcée pour le thème blanc */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
        
        /* Focus visible amélioré */
        *:focus-visible {
            outline: 2px solid var(--primary-blue);
            outline-offset: 2px;
            border-radius: 4px;
        }
        
        /* Sélection de texte personnalisée pour thème blanc */
        ::selection {
            background: rgba(37, 99, 235, 0.2);
            color: var(--text-primary);
        }
        
        ::-moz-selection {
            background: rgba(37, 99, 235, 0.2);
            color: var(--text-primary);
        }
        
        /* Scrollbar personnalisée pour thème blanc */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--gradient-primary);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-blue);
        }
    `;
    document.head.appendChild(style);
}

// Fonction utilitaire pour déboguer les visualisations
function debugVisualization(viz) {
    console.group(`🔍 Debug Visualisation: ${viz.title || 'Sans titre'}`);
    console.log('📊 Données:', viz);
    console.log('🖼️ URL Image:', viz.image);
    console.log('📅 Date:', viz.date || 'Non définie');
    console.log('📁 Catégorie:', viz.category || 'Non catégorisée');
    console.log('✅ Valide:', viz.isValid ? 'Oui' : 'Non');
    console.groupEnd();
}

// Export pour utilisation externe si nécessaire
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        VisualizationManager,
        ClimaSenSystem,
        ensureAbsoluteUrl,
        openVisualization,
        downloadVisualization
    };
}

// Gestion des erreurs globales
window.addEventListener('error', (e) => {
    console.error('🚨 Erreur globale détectée:', e.error);
    window.climaSen?.showNotification('⚠️ Une erreur est survenue', 'error');
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('🚨 Promise rejetée non gérée:', e.reason);
    window.climaSen?.showNotification('⚠️ Erreur de traitement', 'error');
});

console.log('📝 Dashboard JavaScript (Thème Blanc Premium) chargé avec succès!');