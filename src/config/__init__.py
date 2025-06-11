# src/config/__init__.py
"""
Module de configuration pour le projet.
"""

from .settings import (
    SENEGAL_BOUNDS,
    DETECTION_CRITERIA,
    CLIMATOLOGY_PARAMS,
    SEASONS_SENEGAL,
    PROJECT_INFO,
    create_output_directories,
    print_project_info,
    get_output_path,
    get_season_from_month
)

__all__ = [
    'SENEGAL_BOUNDS',
    'DETECTION_CRITERIA', 
    'CLIMATOLOGY_PARAMS',
    'SEASONS_SENEGAL',
    'PROJECT_INFO',
    'create_output_directories',
    'print_project_info',
    'get_output_path',
    'get_season_from_month'
]