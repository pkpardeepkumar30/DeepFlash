"""
Phase Split Calculator for thermodynamic equilibrium calculations.

This module implements the core functionality for performing phase split
calculations using both traditional methods and deep learning approaches.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class PhaseSplitCalculator:
    """
    Main class for performing thermodynamic phase split calculations.
    
    This calculator can use both traditional thermodynamic methods and
    deep learning models to predict phase equilibria for multicomponent
    mixtures.
    
    Attributes:
        n_components (int): Number of components in the mixture
        temperature (float): Temperature in Kelvin
        pressure (float): Pressure in Pascal
        model: Optional deep learning model for predictions
    """
    
    def __init__(self, n_components: int, temperature: float = 298.15, 
                 pressure: float = 101325.0, model=None):
        """
        Initialize the phase split calculator.
        
        Args:
            n_components: Number of components in the mixture
            temperature: Temperature in Kelvin (default: 298.15 K)
            pressure: Pressure in Pascal (default: 101325 Pa)
            model: Optional deep learning model for predictions
        """
        self.n_components = n_components
        self.temperature = temperature
        self.pressure = pressure
        self.model = model
        
    def flash_calculation(self, feed_composition: np.ndarray, 
                         k_values: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Perform a flash calculation to determine phase split.
        
        This implements the Rachford-Rice algorithm for vapor-liquid equilibrium.
        
        Args:
            feed_composition: Array of feed mole fractions (length n_components)
            k_values: Optional equilibrium K-values. If None, uses model prediction
            
        Returns:
            Dictionary containing:
                - 'vapor_fraction': Fraction of vapor phase (beta)
                - 'liquid_composition': Liquid phase mole fractions
                - 'vapor_composition': Vapor phase mole fractions
                - 'k_values': Equilibrium K-values used
        """
        if k_values is None:
            if self.model is not None:
                k_values = self.model.predict_k_values(feed_composition, 
                                                      self.temperature, 
                                                      self.pressure)
            else:
                # Use simple Raoult's law approximation if no model
                k_values = self._estimate_k_values_raoult(feed_composition)
        
        # Solve Rachford-Rice equation
        vapor_fraction = self._solve_rachford_rice(feed_composition, k_values)
        
        # Calculate phase compositions
        liquid_composition = feed_composition / (1 + vapor_fraction * (k_values - 1))
        vapor_composition = k_values * liquid_composition
        
        # Normalize compositions
        liquid_composition /= np.sum(liquid_composition)
        vapor_composition /= np.sum(vapor_composition)
        
        return {
            'vapor_fraction': vapor_fraction,
            'liquid_composition': liquid_composition,
            'vapor_composition': vapor_composition,
            'k_values': k_values
        }
    
    def _solve_rachford_rice(self, z: np.ndarray, k: np.ndarray, 
                            tol: float = 1e-8, max_iter: int = 100) -> float:
        """
        Solve the Rachford-Rice equation for vapor fraction.
        
        The Rachford-Rice equation: sum(z_i * (K_i - 1) / (1 + beta * (K_i - 1))) = 0
        
        Args:
            z: Feed composition
            k: K-values
            tol: Convergence tolerance
            max_iter: Maximum number of iterations
            
        Returns:
            Vapor fraction (beta)
        """
        def rachford_rice(beta):
            return np.sum(z * (k - 1) / (1 + beta * (k - 1)))
        
        def rachford_rice_derivative(beta):
            return -np.sum(z * (k - 1)**2 / (1 + beta * (k - 1))**2)
        
        # Calculate valid bounds for beta to avoid singularities
        # beta must satisfy: -1/(K_i - 1) for all i
        eps = 1e-10
        beta_min = 0.0
        beta_max = 1.0
        
        for i in range(len(k)):
            if abs(k[i] - 1.0) > eps:
                limit = -1.0 / (k[i] - 1.0)
                if k[i] > 1.0:
                    beta_min = max(beta_min, limit + eps)
                else:
                    beta_max = min(beta_max, limit - eps)
        
        # Check if bounds are valid
        if beta_min >= beta_max:
            # Single phase - return appropriate boundary
            return 0.0 if np.mean(k) < 1.0 else 1.0
        
        # Initial guess
        beta = (beta_min + beta_max) / 2.0
        
        # Newton-Raphson iteration with bisection fallback
        for _ in range(max_iter):
            f = rachford_rice(beta)
            if abs(f) < tol:
                break
            
            df = rachford_rice_derivative(beta)
            if abs(df) < eps:
                break
                
            beta_new = beta - f / df
            
            # Constrain beta to valid bounds
            beta = max(beta_min, min(beta_max, beta_new))
        
        return beta
    
    def _estimate_k_values_raoult(self, composition: np.ndarray) -> np.ndarray:
        """
        Estimate K-values using Raoult's law approximation.
        
        This is a simple approximation: K_i = P_sat_i / P
        For demonstration, we use a simplified model.
        
        Args:
            composition: Feed composition
            
        Returns:
            Estimated K-values
        """
        # Simple approximation: K-values vary by component
        # In practice, these would be calculated from vapor pressure correlations
        k_values = np.ones(self.n_components)
        
        # Create variation in K-values (lighter components have higher K)
        for i in range(self.n_components):
            k_values[i] = 2.0 - 0.3 * i  # Decreasing trend
        
        return np.maximum(k_values, 0.1)  # Ensure positive K-values
    
    def stability_analysis(self, composition: np.ndarray) -> bool:
        """
        Perform stability analysis to check if mixture is in single or two-phase region.
        
        Args:
            composition: Mixture composition
            
        Returns:
            True if mixture is stable (single phase), False if unstable (two-phase)
        """
        if self.model is not None:
            return self.model.predict_stability(composition, self.temperature, 
                                               self.pressure)
        
        # Simple heuristic for demonstration
        # In practice, this would use Gibbs energy minimization
        return bool(np.all(composition > 0.1))
    
    def equilibrium_calculation(self, feed_composition: np.ndarray, 
                               feed_rate: float = 1.0) -> Dict[str, Union[float, np.ndarray]]:
        """
        Perform complete equilibrium calculation.
        
        Args:
            feed_composition: Feed mole fractions
            feed_rate: Total feed flow rate (mol/s)
            
        Returns:
            Dictionary containing phase split results and flow rates
        """
        # Check stability
        is_stable = self.stability_analysis(feed_composition)
        
        if is_stable:
            return {
                'status': 'single_phase',
                'vapor_rate': 0.0,
                'liquid_rate': feed_rate,
                'liquid_composition': feed_composition,
                'vapor_composition': None
            }
        
        # Perform flash calculation
        results = self.flash_calculation(feed_composition)
        
        vapor_rate = results['vapor_fraction'] * feed_rate
        liquid_rate = (1 - results['vapor_fraction']) * feed_rate
        
        return {
            'status': 'two_phase',
            'vapor_rate': vapor_rate,
            'liquid_rate': liquid_rate,
            'vapor_fraction': results['vapor_fraction'],
            'liquid_composition': results['liquid_composition'],
            'vapor_composition': results['vapor_composition'],
            'k_values': results['k_values']
        }
