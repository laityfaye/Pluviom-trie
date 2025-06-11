# src/analysis/__init__.py
"""
Module d'analyse climatologique et de détection.
"""

from .climatology import (
    calculate_daily_climatology_robust,
    calculate_standardized_anomalies_robust,
    calculate_climatology_and_anomalies
)
from .detection import (
    detect_extreme_precipitation_events_final,
    analyze_spatial_distribution,
    ExtremeEventDetector
)

__all__ = [
    'calculate_daily_climatology_robust',
    'calculate_standardized_anomalies_robust', 
    'calculate_climatology_and_anomalies',
    'detect_extreme_precipitation_events_final',
    'analyze_spatial_distribution',
    'ExtremeEventDetector'
]