"""
Data processing utilities for thermodynamic calculations.
"""

import numpy as np
from typing import Tuple, List, Optional


def normalize_composition(composition: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Normalize composition to ensure mole fractions sum to 1.
    
    Args:
        composition: Array of mole fractions
        eps: Small value to prevent division by zero
        
    Returns:
        Normalized composition
    """
    total = np.sum(composition)
    if total < eps:
        raise ValueError("Composition sum is too small to normalize")
    return composition / total


def denormalize_composition(composition: np.ndarray, 
                           original_sum: float = 1.0) -> np.ndarray:
    """
    Denormalize composition back to original scale.
    
    Args:
        composition: Normalized composition
        original_sum: Original sum of composition
        
    Returns:
        Denormalized composition
    """
    return composition * original_sum


def generate_sample_compositions(n_components: int, n_samples: int, 
                                 random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate random sample compositions for testing.
    
    Args:
        n_components: Number of components
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        Array of shape (n_samples, n_components) with normalized compositions
    """
    # Use numpy's recommended random generator for better state management
    rng = np.random.default_rng(random_state)
    
    # Generate random compositions using Dirichlet distribution
    # This ensures compositions sum to 1
    alpha = np.ones(n_components)
    compositions = rng.dirichlet(alpha, size=n_samples)
    
    return compositions


def create_training_data(n_components: int, n_samples: int,
                        temperature_range: Tuple[float, float] = (250.0, 450.0),
                        pressure_range: Tuple[float, float] = (1e5, 1e7),
                        random_state: Optional[int] = None) -> dict:
    """
    Create synthetic training data for model development.
    
    Args:
        n_components: Number of components
        n_samples: Number of samples
        temperature_range: Min and max temperature in Kelvin
        pressure_range: Min and max pressure in Pascal
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with 'inputs' and 'targets' arrays
    """
    # Use numpy's recommended random generator for better state management
    rng = np.random.default_rng(random_state)
    
    # Generate compositions
    compositions = generate_sample_compositions(n_components, n_samples, random_state)
    
    # Generate temperatures and pressures
    temperatures = rng.uniform(temperature_range[0], temperature_range[1], n_samples)
    pressures = rng.uniform(pressure_range[0], pressure_range[1], n_samples)
    
    # Normalize for neural network input
    T_norm = temperatures / 500.0
    P_norm = pressures / 1e6
    
    # Combine into input array
    inputs = np.column_stack([compositions, T_norm, P_norm])
    
    # Generate synthetic K-values (for demonstration)
    # In practice, these would come from thermodynamic simulations or experiments
    targets = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        # Simple model: K-values depend on temperature and pressure
        base_k = np.exp((temperatures[i] - 298.15) / 100.0) * (101325.0 / pressures[i])
        for j in range(n_components):
            targets[i, j] = base_k * (2.0 - 0.3 * j)
    
    return {
        'inputs': inputs,
        'targets': targets,
        'compositions': compositions,
        'temperatures': temperatures,
        'pressures': pressures
    }


def validate_composition(composition: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Validate that composition array is physical.
    
    Args:
        composition: Array of mole fractions
        tolerance: Tolerance for sum check
        
    Returns:
        True if valid, False otherwise
    """
    # Check all values are non-negative
    if np.any(composition < 0):
        return False
    
    # Check sum is approximately 1
    if abs(np.sum(composition) - 1.0) > tolerance:
        return False
    
    return True


def interpolate_properties(x1: float, y1: float, x2: float, y2: float, 
                          x: float) -> float:
    """
    Linear interpolation between two points.
    
    Args:
        x1, y1: First point
        x2, y2: Second point
        x: Point to interpolate at
        
    Returns:
        Interpolated y value
    """
    if abs(x2 - x1) < 1e-10:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
