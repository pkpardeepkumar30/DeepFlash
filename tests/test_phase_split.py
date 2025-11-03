"""
Unit tests for phase split calculations.
"""

import unittest
import numpy as np
from deepflash.phase_split import PhaseSplitCalculator


class TestPhaseSplitCalculator(unittest.TestCase):
    """Test cases for PhaseSplitCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_components = 3
        self.calculator = PhaseSplitCalculator(
            n_components=self.n_components,
            temperature=350.0,
            pressure=5e5
        )
    
    def test_initialization(self):
        """Test calculator initialization."""
        self.assertEqual(self.calculator.n_components, self.n_components)
        self.assertEqual(self.calculator.temperature, 350.0)
        self.assertEqual(self.calculator.pressure, 5e5)
    
    def test_flash_calculation(self):
        """Test basic flash calculation."""
        feed = np.array([0.3, 0.5, 0.2])
        results = self.calculator.flash_calculation(feed)
        
        # Check that all required keys are present
        self.assertIn('vapor_fraction', results)
        self.assertIn('liquid_composition', results)
        self.assertIn('vapor_composition', results)
        self.assertIn('k_values', results)
        
        # Check vapor fraction is between 0 and 1
        self.assertGreaterEqual(results['vapor_fraction'], 0.0)
        self.assertLessEqual(results['vapor_fraction'], 1.0)
        
        # Check compositions sum to 1
        np.testing.assert_almost_equal(
            np.sum(results['liquid_composition']), 1.0, decimal=6
        )
        np.testing.assert_almost_equal(
            np.sum(results['vapor_composition']), 1.0, decimal=6
        )
        
        # Check K-values are positive
        self.assertTrue(np.all(results['k_values'] > 0))
    
    def test_flash_calculation_with_k_values(self):
        """Test flash calculation with provided K-values."""
        feed = np.array([0.3, 0.5, 0.2])
        k_values = np.array([2.0, 1.5, 0.8])
        
        results = self.calculator.flash_calculation(feed, k_values)
        
        # Check that provided K-values are used
        np.testing.assert_array_equal(results['k_values'], k_values)
    
    def test_rachford_rice_solver(self):
        """Test Rachford-Rice equation solver."""
        z = np.array([0.3, 0.5, 0.2])
        # Use K-values that allow a valid two-phase flash
        # Must have f(0) and f(1) with opposite signs for a solution to exist
        k = np.array([2.5, 1.0, 0.4])
        
        beta = self.calculator._solve_rachford_rice(z, k)
        
        # Check beta is between 0 and 1
        self.assertGreaterEqual(beta, 0.0)
        self.assertLessEqual(beta, 1.0)
        
        # Check Rachford-Rice equation is satisfied
        rachford_rice = np.sum(z * (k - 1) / (1 + beta * (k - 1)))
        self.assertAlmostEqual(rachford_rice, 0.0, places=6)
    
    def test_stability_analysis(self):
        """Test stability analysis."""
        # Test stable composition
        stable_comp = np.array([0.33, 0.33, 0.34])
        is_stable = self.calculator.stability_analysis(stable_comp)
        self.assertIsInstance(is_stable, bool)
        
        # Test unstable composition (one component dominant)
        unstable_comp = np.array([0.95, 0.03, 0.02])
        is_stable = self.calculator.stability_analysis(unstable_comp)
        self.assertIsInstance(is_stable, bool)
    
    def test_equilibrium_calculation(self):
        """Test complete equilibrium calculation."""
        feed = np.array([0.3, 0.5, 0.2])
        feed_rate = 100.0
        
        results = self.calculator.equilibrium_calculation(feed, feed_rate)
        
        # Check required keys
        self.assertIn('status', results)
        self.assertIn('vapor_rate', results)
        self.assertIn('liquid_rate', results)
        
        # Check flow rates
        self.assertGreaterEqual(results['vapor_rate'], 0.0)
        self.assertGreaterEqual(results['liquid_rate'], 0.0)
        
        # Check total flow rate is conserved
        total_rate = results['vapor_rate'] + results['liquid_rate']
        self.assertAlmostEqual(total_rate, feed_rate, places=6)
    
    def test_estimate_k_values(self):
        """Test K-value estimation."""
        composition = np.array([0.3, 0.5, 0.2])
        k_values = self.calculator._estimate_k_values_raoult(composition)
        
        # Check shape
        self.assertEqual(len(k_values), self.n_components)
        
        # Check K-values are positive
        self.assertTrue(np.all(k_values > 0))


if __name__ == '__main__':
    unittest.main()
