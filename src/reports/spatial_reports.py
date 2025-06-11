"""
Générateur de rapports spatiaux centralisé.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

class SpatialReportGenerator:
    """Générateur centralisé pour tous les rapports spatiaux."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, spatial_results: List[Dict[str, Any]], 
                                    analysis_type: str,
                                    title: str) -> str:
        """Génère un rapport spatial complet et standardisé."""
        
        df = pd.DataFrame(spatial_results)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Nom du fichier basé sur le type d'analyse
        filename = f"rapport_spatial_{analysis_type.lower()}.txt"
        report_path = self.output_dir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # En-tête standardisé
            f.write(f"RAPPORT SPATIAL - {title.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Date de génération: {timestamp}\n")
            f.write(f"Type d'analyse: {analysis_type}\n")
            f.write(f"Nombre d'événements: {len(spatial_results)}\n")
            f.write(f"Période: 1981-2023\n\n")
            
            # Résumé exécutif
            self._write_executive_summary(f, df)
            
            # Détails par événement
            self._write_event_details(f, spatial_results)
            
            # Analyse comparative
            self._write_comparative_analysis(f, df)
            
            # Recommandations
            self._write_recommendations(f, df, analysis_type)
        
        return str(report_path)
    
    def _write_executive_summary(self, f, df: pd.DataFrame):
        """Écrit le résumé exécutif."""
        f.write("RÉSUMÉ EXÉCUTIF\n")
        f.write("-" * 20 + "\n")
        f.write(f"• Intensité maximale: {df['max_intensity_mm'].max():.1f} mm/jour\n")
        f.write(f"• Surface totale affectée: {df['total_area_km2'].sum():.0f} km²\n")
        f.write(f"• Couverture moyenne: {df['coverage_percent'].mean():.2f}% du territoire\n")
        f.write(f"• Dispersion géographique: {df['centroid_lat'].std():.3f}° (lat), {df['centroid_lon'].std():.3f}° (lon)\n\n")
    
    def _write_event_details(self, f, spatial_results: List[Dict]):
        """Écrit les détails de chaque événement."""
        f.write("ANALYSE DÉTAILLÉE PAR ÉVÉNEMENT\n")
        f.write("-" * 40 + "\n\n")
        
        for result in spatial_results:
            f.write(f"RANG #{result['rank']}: {result['date']}\n")
            f.write(f"{'='*30}\n")
            f.write(f"Localisation: {result['region']}\n")
            f.write(f"Centroïde: {result['centroid_lat']:.4f}°N, {abs(result['centroid_lon']):.4f}°W\n")
            f.write(f"Surface: {result['total_area_km2']:.0f} km²\n")
            f.write(f"Intensité max: {result['max_intensity_mm']:.1f} mm/jour\n")
            f.write(f"Intensité moyenne: {result['intensity_stats']['mean']:.1f} mm/jour\n\n")
    
    def _write_comparative_analysis(self, f, df: pd.DataFrame):
        """Écrit l'analyse comparative."""
        f.write("ANALYSE COMPARATIVE\n")
        f.write("-" * 25 + "\n\n")
        
        # Distribution régionale
        region_counts = df['region'].value_counts()
        f.write("Distribution régionale:\n")
        for region, count in region_counts.items():
            pct = count / len(df) * 100
            f.write(f"  • {region}: {count} événement(s) ({pct:.1f}%)\n")
        
        # Corrélations
        if 'max_intensity_mm' in df.columns and 'total_area_km2' in df.columns:
            corr = df['max_intensity_mm'].corr(df['total_area_km2'])
            f.write(f"\nCorrélation intensité-surface: {corr:.3f}\n")
    
    def _write_recommendations(self, f, df: pd.DataFrame, analysis_type: str):
        """Écrit les recommandations."""
        f.write("\nRECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        
        most_affected_region = df['region'].value_counts().index[0]
        
        f.write(f"1. Surveillance prioritaire: {most_affected_region}\n")
        f.write(f"2. Infrastructure adaptée aux intensités jusqu'à {df['max_intensity_mm'].max():.0f} mm/jour\n")
        f.write(f"3. Systèmes d'alerte précoce pour les zones de {df['total_area_km2'].min():.0f}-{df['total_area_km2'].max():.0f} km²\n")