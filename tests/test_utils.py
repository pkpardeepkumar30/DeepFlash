"""
Unit tests for utility functions.
"""

import unittest
import numpy as np
from deepflash.utils.data_processing import (
    normalize_composition,
    denormalize_composition,
    generate_sample_compositions,
    validate_composition,
    create_training_data
)


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing utilities."""
    
    def test_normalize_composition(self):
        """Test composition normalization."""
        composition = np.array([1.0, 2.0, 3.0])
        normalized = normalize_composition(composition)
        
        # Check sum is 1
        self.assertAlmostEqual(np.sum(normalized), 1.0, places=10)
        
        # Check proportions are maintained
        expected = composition / np.sum(composition)
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_normalize_composition_error(self):
        """Test normalization with invalid input."""
        composition = np.array([0.0, 0.0, 0.0])
        
        with self.assertRaises(ValueError):
            normalize_composition(composition)
    
    def test_denormalize_composition(self):
        """Test composition denormalization."""
        normalized = np.array([0.2, 0.3, 0.5])
        original_sum = 10.0
        
        denormalized = denormalize_composition(normalized, original_sum)
        
        # Check sum is original_sum
        self.assertAlmostEqual(np.sum(denormalized), original_sum, places=10)
    
    def test_generate_sample_compositions(self):
        """Test sample composition generation."""
        n_components = 3
        n_samples = 10
        
        compositions = generate_sample_compositions(n_components, n_samples, 
                                                   random_state=42)
        
        # Check shape
        self.assertEqual(compositions.shape, (n_samples, n_components))
        
        # Check all compositions sum to 1
        sums = np.sum(compositions, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(n_samples))
        
        # Check all values are non-negative
        self.assertTrue(np.all(compositions >= 0))
    
    def test_generate_sample_compositions_reproducibility(self):
        """Test that random seed produces reproducible results."""
        n_components = 3
        n_samples = 10
        random_state = 42
        
        comp1 = generate_sample_compositions(n_components, n_samples, random_state)
        comp2 = generate_sample_compositions(n_components, n_samples, random_state)
        
        np.testing.assert_array_equal(comp1, comp2)
    
    def test_validate_composition_valid(self):
        """Test validation of valid composition."""
        composition = np.array([0.3, 0.5, 0.2])
        self.assertTrue(validate_composition(composition))
    
    def test_validate_composition_invalid_negative(self):
        """Test validation rejects negative values."""
        composition = np.array([0.3, -0.1, 0.8])
        self.assertFalse(validate_composition(composition))
    
    def test_validate_composition_invalid_sum(self):
        """Test validation rejects incorrect sum."""
        composition = np.array([0.3, 0.3, 0.3])
        self.assertFalse(validate_composition(composition))
    
    def test_create_training_data(self):
        """Test training data generation."""
        n_components = 3
        n_samples = 100
        
        data = create_training_data(n_components, n_samples, random_state=42)
        
        # Check all required keys
        self.assertIn('inputs', data)
        self.assertIn('targets', data)
        self.assertIn('compositions', data)
        self.assertIn('temperatures', data)
        self.assertIn('pressures', data)
        
        # Check shapes
        # Input: composition (n_components) + T_norm (1) + P_norm (1)
        self.assertEqual(data['inputs'].shape, (n_samples, n_components + 2))
        self.assertEqual(data['targets'].shape, (n_samples, n_components))
        self.assertEqual(data['compositions'].shape, (n_samples, n_components))
        
        # Check temperature and pressure ranges
        self.assertTrue(np.all(data['temperatures'] >= 250.0))
        self.assertTrue(np.all(data['temperatures'] <= 450.0))
        self.assertTrue(np.all(data['pressures'] >= 1e5))
        self.assertTrue(np.all(data['pressures'] <= 1e7))
        
        # Check targets are positive (K-values)
        self.assertTrue(np.all(data['targets'] > 0))


if __name__ == '__main__':
    unittest.main()
