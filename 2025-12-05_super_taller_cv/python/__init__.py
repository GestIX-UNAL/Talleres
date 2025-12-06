"""
Subsystem 5: Model Training & Comparison
Advanced Computer Vision Workshop 2025

Package initialization
"""

__version__ = "1.0.0"
__author__ = "Computer Vision Workshop"
__description__ = "Deep Learning Model Training and Comparison Subsystem"

from .training.cnn_trainer import CustomCNNTrainer, create_sample_dataset
from .training.finetuning_trainer import FineTuningTrainer
from .training.model_comparison import ModelComparator
from .dashboards.performance_dashboard import PerformanceDashboard
from .utils.visualization_utils import (
    DataAugmentation,
    ResultsExporter,
    VisualizationUtils,
    PerformanceLogger
)

__all__ = [
    'CustomCNNTrainer',
    'FineTuningTrainer',
    'ModelComparator',
    'PerformanceDashboard',
    'DataAugmentation',
    'ResultsExporter',
    'VisualizationUtils',
    'PerformanceLogger',
    'create_sample_dataset'
]
