"""
Training module initialization
"""

from .cnn_trainer import CustomCNNTrainer
from .finetuning_trainer import FineTuningTrainer
from .model_comparison import ModelComparator

__all__ = ['CustomCNNTrainer', 'FineTuningTrainer', 'ModelComparator']
