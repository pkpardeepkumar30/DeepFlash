"""
Deep learning models for thermodynamic property prediction.
"""

from .deep_flash_model import DeepFlashModel
from .neural_network import ThermodynamicNN

__all__ = ["DeepFlashModel", "ThermodynamicNN"]
