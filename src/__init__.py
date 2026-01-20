"""
Source package initialization for UIDAI Analysis modules.
"""

from .anomaly_detection import generate_anomaly_report
from .forecasting import generate_forecast_report
from .clustering import generate_clustering_report
from .ml_models import generate_ml_report
from .operational_insights import generate_operational_report
from .advanced_visualizations import generate_advanced_visualizations

__all__ = [
    'generate_anomaly_report',
    'generate_forecast_report',
    'generate_clustering_report',
    'generate_ml_report',
    'generate_operational_report',
    'generate_advanced_visualizations',
]
