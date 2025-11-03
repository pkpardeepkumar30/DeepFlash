"""
DeepFlash: Deep Learning for Thermodynamic Phase Split Calculations

This package provides tools and models for performing thermodynamic phase split
calculations using deep learning techniques.
"""

__version__ = "0.1.0"
__author__ = "DeepFlash Team"

from .phase_split import PhaseSplitCalculator
from .models import DeepFlashModel

__all__ = ["PhaseSplitCalculator", "DeepFlashModel", "__version__"]
