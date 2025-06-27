// services/dashboard/dist/js/header-hero.js
// Animations et interactions pour le header et la section hero

/**
 * Gestionnaire du header et de la section hero
 */
class HeaderHeroManager {
    constructor() {
        this.header = null;
        this.progressBar = null;
        this.isInitialized = false;
        this.scrollListeners = [];
        this.animationObservers = [];
        this.particleContainer = null;
        this.particles = [];
        this.animationFrame = null;
    }
    
    /**
     * Initialise le gestionnaire
     */
    initialize() {
        console.log('🎯 Initialisation Header & Hero Manager...');
        
        this.header = document.querySelector('.header');
        this.progressBar = document.querySelector('.scroll-progress-bar');
        
        if (this.header && this.progressBar) {
            this.setupScrollEffects();
            this.setupHeroAnimations();
            this.setupNavigationInteractions();
            this.setupMobileMenu();
            this.animateCounters();
            this.enhanceMetricCards();
            this.addParticleEffects();
            this.setupKeyboardShortcuts();
            this.isInitialized = true;
            console.log('✅ Header & Hero Manager initialisé');
        } else {
            console.warn('⚠️ Éléments header/hero non trouvés');
        }
    }
    
    /**
     * Configure les effets de scroll
     */
    setupScrollEffects() {
        let ticking = false;
        
        const updateHeader = () => {
            const scrollY = window.scrollY;
            const scrollProgress = Math.min(scrollY / (document.documentElement.scrollHeight - window.innerHeight), 1);
            
            // Effet header au scroll
            if (scrollY > 50) {
                this.header.classList.add('scrolled');
            } else {
                this.header.classList.remove('scrolled');
            }
            
            // Barre de progression
            this.progressBar.style.transform = `scaleX(${scrollProgress})`;
            
            // Effet parallax sur les particules
            if (this.particleContainer) {
                this.updateParticleParallax(scrollY);
            }
            
            ticking = false;
        };
        
        const onScroll = () => {
            if (!ticking) {
                requestAnimationFrame(updateHeader);
                ticking = true;
            }
        };
        
        window.addEventListener('scroll', onScroll, { passive: true });
        this.scrollListeners.push({ element: window, event: 'scroll', handler: onScroll });
    }
    
    /**
     * Configure les animations de la section hero
     */
    setupHeroAnimations() {
        // Observer pour déclencher les animations au bon moment
        const heroObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.triggerHeroAnimations();
                    heroObserver.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.2
        });
        
        const heroSection = document.querySelector('.hero-section');
        if (heroSection) {
            heroObserver.observe(heroSection);
            this.animationObservers.push(heroObserver);
        }
        
        // Animation de l'indicateur de scroll
        this.setupScrollIndicator();
    }
    
    /**
     * Déclenche toutes les animations de la section hero
     */
    triggerHeroAnimations() {
        // Animation des métriques avec délai échelonné
        const metricCards = document.querySelectorAll('.metric-card');
        metricCards.forEach((card, index) => {
            setTimeout(() => {
                card.classList.add('animate-in');
                card.style.animation = `fadeInUp 0.8s ease-out ${index * 0.15}s both`;
            }, index * 150);
        });
        
        // Animation de la barre de progression
        setTimeout(() => {
            const progressFill = document.querySelector('.progress-fill');
            if (progressFill) {
                progressFill.style.width = '100%';
            }
        }, 1000);
        
        // Animation des phases du pipeline
        this.animatePipelinePhases();
    }
    
    /**
     * Anime les phases du pipeline
     */
    animatePipelinePhases() {
        const phases = document.querySelectorAll('.phase-item');
        phases.forEach((phase, index) => {
            setTimeout(() => {
                phase.classList.add('completed');
                phase.style.animation = `phaseComplete 0.6s ease-out both`;
            }, 1500 + (index * 300));
        });
    }
    
    /**
     * Anime les compteurs de statistiques
     */
    animateCounters() {
        const counters = document.querySelectorAll('.stat-number[data-target], .value-number');
        
        // Observer pour déclencher l'animation quand les compteurs sont visibles
        const counterObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateCounter(entry.target);
                    counterObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });
        
        counters.forEach(counter => {
            counterObserver.observe(counter);
        });
        
        this.animationObservers.push(counterObserver);
    }
    
    /**
     * Anime un compteur individuel
     */
    animateCounter(element) {
        const target = parseFloat(element.dataset.target || element.textContent || 0);
        const duration = 2000;
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Fonction d'easing pour une animation fluide
            const easeOut = 1 - Math.pow(1 - progress, 3);
            let current = target * easeOut;
            
            // Formatage selon le type de nombre
            if (target >= 1000) {
                current = Math.round(current);
                element.textContent = this.formatNumber(current);
            } else if (target < 100 && target % 1 !== 0) {
                // Nombre décimal
                current = Math.round(current * 10) / 10;
                element.textContent = current.toFixed(1);
            } else {
                current = Math.round(current);
                element.textContent = current;
            }
            
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
     * Configure les interactions de navigation
     */
    setupNavigationInteractions() {
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Retirer la classe active de tous les liens
                navLinks.forEach(l => l.classList.remove('active'));
                
                // Ajouter la classe active au lien cliqué
                link.classList.add('active');
                
                // Effet ripple
                this.createRippleEffect(link, e);
                
                // Navigation fluide
                const targetId = link.getAttribute('href').substring(1);
                this.smoothScrollTo(targetId);
            });
            
            // Effets de hover
            link.addEventListener('mouseenter', () => {
                link.style.transform = 'translateY(-2px)';
            });
            
            link.addEventListener('mouseleave', () => {
                if (!link.classList.contains('active')) {
                    link.style.transform = 'translateY(0)';
                }
            });
        });
        
        // Mise à jour de la navigation active au scroll
        this.setupNavigationScrollSpy(navLinks);
    }
    
    /**
     * Crée un effet ripple sur un élément
     */
    createRippleEffect(element, event) {
        const ripple = document.createElement('span');
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;
        
        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
            background: radial-gradient(circle, rgba(37, 99, 235, 0.3) 0%, transparent 70%);
            border-radius: 50%;
            transform: scale(0);
            animation: ripple 0.6s ease-out;
            pointer-events: none;
            z-index: 1;
        `;
        
        // Assurer que l'élément parent a position relative
        if (getComputedStyle(element).position === 'static') {
            element.style.position = 'relative';
        }
        
        element.appendChild(ripple);
        
        // Supprimer le ripple après l'animation
        setTimeout(() => {
            if (ripple.parentNode) {
                ripple.parentNode.removeChild(ripple);
            }
        }, 600);
    }
    
    /**
     * Configure le scroll spy pour la navigation
     */
    setupNavigationScrollSpy(navLinks) {
        const sections = document.querySelectorAll('section[id]');
        
        const updateActiveNav = () => {
            const scrollY = window.scrollY + 100; // Offset pour le header
            
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.offsetHeight;
                const sectionId = section.getAttribute('id');
                
                if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
                    navLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${sectionId}`) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        };
        
        window.addEventListener('scroll', updateActiveNav, { passive: true });
        this.scrollListeners.push({ element: window, event: 'scroll', handler: updateActiveNav });
    }
    
    /**
     * Navigation fluide vers une section
     */
    smoothScrollTo(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            const headerHeight = this.header?.offsetHeight || 80;
            const targetPosition = section.offsetTop - headerHeight - 20;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
            
            // Mettre à jour l'URL
            history.replaceState(null, '', `#${sectionId}`);
        }
    }
    
    /**
     * Configure l'indicateur de scroll
     */
    setupScrollIndicator() {
        const scrollIndicator = document.querySelector('.scroll-indicator');
        if (scrollIndicator) {
            scrollIndicator.addEventListener('click', () => {
                this.smoothScrollTo('pipeline');
            });
            
            // Masquer l'indicateur après le premier scroll
            let hasScrolled = false;
            const hideIndicator = () => {
                if (!hasScrolled && window.scrollY > 100) {
                    hasScrolled = true;
                    scrollIndicator.style.transition = 'all 0.5s ease';
                    scrollIndicator.style.opacity = '0';
                    scrollIndicator.style.transform = 'translateX(-50%) translateY(20px)';
                }
            };
            
            window.addEventListener('scroll', hideIndicator, { passive: true });
            this.scrollListeners.push({ element: window, event: 'scroll', handler: hideIndicator });
        }
    }
    
    /**
     * Configure le menu mobile
     */
    setupMobileMenu() {
        const mobileToggle = document.querySelector('.mobile-menu-toggle');
        const navMenu = document.querySelector('.nav-menu');
        
        if (mobileToggle && navMenu) {
            mobileToggle.addEventListener('click', () => {
                const isOpen = navMenu.classList.contains('mobile-open');
                
                if (isOpen) {
                    this.closeMobileMenu(navMenu, mobileToggle);
                } else {
                    this.openMobileMenu(navMenu, mobileToggle);
                }
            });
            
            // Fermer le menu au clic sur un lien
            const mobileNavLinks = navMenu.querySelectorAll('.nav-link');
            mobileNavLinks.forEach(link => {
                link.addEventListener('click', () => {
                    this.closeMobileMenu(navMenu, mobileToggle);
                });
            });
            
            // Fermer le menu en cliquant à l'extérieur
            document.addEventListener('click', (e) => {
                if (!navMenu.contains(e.target) && !mobileToggle.contains(e.target)) {
                    this.closeMobileMenu(navMenu, mobileToggle);
                }
            });
        }
    }
    
    /**
     * Ouvre le menu mobile
     */
    openMobileMenu(navMenu, mobileToggle) {
        navMenu.classList.add('mobile-open');
        mobileToggle.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        // Animation des liens du menu
        const links = navMenu.querySelectorAll('.nav-link');
        links.forEach((link, index) => {
            link.style.animation = `slideInRight 0.3s ease-out ${index * 0.1}s both`;
        });
    }
    
    /**
     * Ferme le menu mobile
     */
    closeMobileMenu(navMenu, mobileToggle) {
        navMenu.classList.remove('mobile-open');
        mobileToggle.classList.remove('active');
        document.body.style.overflow = '';
    }
    
    /**
     * Configure les raccourcis clavier
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K pour ouvrir la recherche
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.openGlobalSearch();
            }
            
            // Echap pour fermer les modals
            if (e.key === 'Escape') {
                this.closeAllModals();
            }
        });
    }
    
    /**
     * Ouvre la recherche globale
     */
    openGlobalSearch() {
        // Implémentation de la recherche globale
        console.log('🔍 Ouverture de la recherche globale');
        // TODO: Implémenter l'interface de recherche
    }
    
    /**
     * Ferme toutes les modals ouvertes
     */
    closeAllModals() {
        const modals = document.querySelectorAll('.modal-overlay');
        modals.forEach(modal => {
            if (modal.style.display !== 'none') {
                modal.style.display = 'none';
            }
        });
    }
    
    /**
     * Met à jour les valeurs des métriques
     */
    updateMetrics(data) {
        const updates = [
            { id: 'events-count', value: data.eventsDetected || 1439 },
            { id: 'ml-f1', value: data.mlAccuracy || 91.3 },
            { id: 'coverage-area', value: data.gridPoints || 560 },
            { id: 'pipeline-duration', value: data.pipelineDurationMinutes || 18.5 }
        ];
        
        updates.forEach(update => {
            const element = document.getElementById(update.id);
            if (element) {
                element.dataset.target = update.value;
                this.animateCounter(element);
            }
        });
        
        // Mettre à jour les tendances
        this.updateTrends(data);
    }
    
    /**
     * Met à jour les indicateurs de tendance
     */
    updateTrends(data) {
        const trends = data.trends || {};
        
        Object.keys(trends).forEach(key => {
            const trendElement = document.querySelector(`[data-trend="${key}"]`);
            if (trendElement) {
                const trend = trends[key];
                trendElement.className = `value-trend ${trend.direction}`;
                trendElement.querySelector('span').textContent = trend.value;
            }
        });
    }
    
    /**
     * Ajoute des effets visuels aux cartes métriques
     */
    enhanceMetricCards() {
        const metricCards = document.querySelectorAll('.metric-card');
        
        metricCards.forEach(card => {
            // Effet de parallax léger au hover
            card.addEventListener('mouseenter', () => {
                const icon = card.querySelector('.metric-icon');
                if (icon) {
                    icon.style.transition = 'transform 0.3s ease';
                    icon.style.transform = 'scale(1.1) rotate(5deg)';
                }
                
                // Effet de lueur
                card.style.boxShadow = '0 20px 40px rgba(15, 23, 42, 0.15)';
            });
            
            card.addEventListener('mouseleave', () => {
                const icon = card.querySelector('.metric-icon');
                if (icon) {
                    icon.style.transform = 'scale(1) rotate(0deg)';
                }
                
                card.style.boxShadow = '';
            });
            
            // Effet de clic
            card.addEventListener('click', () => {
                this.pulseCard(card);
            });
        });
    }
    
    /**
     * Effet de pulsation sur une carte
     */
    pulseCard(card) {
        card.style.animation = 'pulse 0.3s ease-out';
        setTimeout(() => {
            card.style.animation = '';
        }, 300);
    }
    
    /**
     * Ajoute des effets de particules au hero
     */
    addParticleEffects() {
        const heroParticles = document.querySelector('.hero-particles');
        if (heroParticles && window.innerWidth > 1024) {
            this.createFloatingParticles();
            this.startParticleAnimation();
        }
    }
    
    /**
     * Crée les particules flottantes
     */
    createFloatingParticles() {
        const heroSection = document.querySelector('.hero-section');
        if (!heroSection) return;
        
        this.particleContainer = document.createElement('div');
        this.particleContainer.className = 'floating-particles';
        this.particleContainer.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        `;
        
        heroSection.appendChild(this.particleContainer);
        
        // Créer des particules individuelles
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.className = 'floating-particle';
            
            const size = Math.random() * 4 + 2;
            const color = ['37, 99, 235', '139, 92, 246', '16, 185, 129'][Math.floor(Math.random() * 3)];
            
            particle.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                background: radial-gradient(circle, rgba(${color}, 0.6) 0%, rgba(${color}, 0) 70%);
                border-radius: 50%;
                animation: float ${8 + Math.random() * 4}s ease-in-out infinite;
                animation-delay: ${Math.random() * 2}s;
                left: ${Math.random() * 100}%;
                top: ${Math.random() * 100}%;
            `;
            
            this.particleContainer.appendChild(particle);
            this.particles.push({
                element: particle,
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5
            });
        }
    }
    
    /**
     * Démarre l'animation des particules
     */
    startParticleAnimation() {
        const animate = () => {
            this.particles.forEach(particle => {
                particle.x += particle.vx;
                particle.y += particle.vy;
                
                // Rebond sur les bords
                if (particle.x <= 0 || particle.x >= window.innerWidth) {
                    particle.vx *= -1;
                }
                if (particle.y <= 0 || particle.y >= window.innerHeight) {
                    particle.vy *= -1;
                }
                
                particle.element.style.transform = `translate(${particle.x}px, ${particle.y}px)`;
            });
            
            this.animationFrame = requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    /**
     * Met à jour l'effet parallax des particules
     */
    updateParticleParallax(scrollY) {
        if (this.particleContainer) {
            const speed = scrollY * 0.3;
            this.particleContainer.style.transform = `translateY(${speed}px)`;
        }
    }
    
    /**
     * Nettoie les resources
     */
    cleanup() {
        // Arrêter l'animation des particules
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        
        // Supprimer les event listeners
        this.scrollListeners.forEach(listener => {
            listener.element.removeEventListener(listener.event, listener.handler);
        });
        this.scrollListeners = [];
        
        // Nettoyer les observers
        this.animationObservers.forEach(observer => {
            observer.disconnect();
        });
        this.animationObservers = [];
        
        // Supprimer le container de particules
        if (this.particleContainer && this.particleContainer.parentNode) {
            this.particleContainer.parentNode.removeChild(this.particleContainer);
        }
        
        this.isInitialized = false;
        console.log('🧹 Header & Hero Manager nettoyé');
    }
}

// Ajouter les styles CSS pour les animations
const addAnimationStyles = () => {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            0% { transform: scale(0); opacity: 1; }
            100% { transform: scale(4); opacity: 0; }
        }
        
        @keyframes fadeInUp {
            0% { opacity: 0; transform: translateY(30px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes phaseComplete {
            0% { opacity: 0; transform: translateX(-20px); }
            100% { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes slideInRight {
            0% { opacity: 0; transform: translateX(20px); }
            100% { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            33% { transform: translateY(-10px) rotate(120deg); }
            66% { transform: translateY(-5px) rotate(240deg); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .mobile-menu-toggle.active span:nth-child(1) {
            transform: rotate(45deg) translate(6px, 6px);
        }
        
        .mobile-menu-toggle.active span:nth-child(2) {
            opacity: 0;
        }
        
        .mobile-menu-toggle.active span:nth-child(3) {
            transform: rotate(-45deg) translate(6px, -6px);
        }
        
        .nav-menu.mobile-open {
            transform: translateX(0);
            opacity: 1;
            visibility: visible;
        }
        
        @media (max-width: 768px) {
            .nav-menu {
                position: fixed;
                top: 70px;
                left: 0;
                right: 0;
                background: rgba(255, 255, 255, 0.98);
                backdrop-filter: blur(20px);
                padding: 2rem;
                border-radius: 0 0 1rem 1rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.1);
                transform: translateX(100%);
                opacity: 0;
                visibility: hidden;
                transition: all 0.3s ease;
                flex-direction: column;
                gap: 1rem;
            }
        }
    `;
    document.head.appendChild(style);
};

// Initialiser automatiquement le gestionnaire
document.addEventListener('DOMContentLoaded', () => {
    addAnimationStyles();
    
    const headerHeroManager = new HeaderHeroManager();
    headerHeroManager.initialize();
    
    // Exposer l'instance globalement pour les autres scripts
    window.headerHeroManager = headerHeroManager;
});

// Fonctions globales pour la compatibilité
window.scrollToSection = (sectionId) => {
    if (window.headerHeroManager && window.headerHeroManager.isInitialized) {
        window.headerHeroManager.smoothScrollTo(sectionId);
    }
};

window.openGlobalSearch = () => {
    if (window.headerHeroManager && window.headerHeroManager.isInitialized) {
        window.headerHeroManager.openGlobalSearch();
    }
};

export default HeaderHeroManager;