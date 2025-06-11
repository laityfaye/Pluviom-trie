# src/reports/__init__.py
"""
Module de génération de rapports.
"""

from .detection_report import (
    DetectionReportGenerator,
    analyze_extreme_events_final,
    generate_detection_report,
    generate_summary_statistics,
    print_summary_statistics
)

__all__ = [
    'DetectionReportGenerator',
    'analyze_extreme_events_final',
    'generate_detection_report',
    'generate_summary_statistics',
    'print_summary_statistics'
]