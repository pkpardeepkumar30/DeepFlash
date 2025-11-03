"""
Complete workflow example demonstrating all DeepFlash capabilities.

This example shows how to:
1. Generate training data
2. Train a model
3. Use the model for predictions
4. Perform flash calculations
"""

import numpy as np
from deepflash import PhaseSplitCalculator, DeepFlashModel
from deepflash.utils.data_processing import create_training_data


def main():
    """Run complete DeepFlash workflow."""
    
    print("=" * 80)
    print("DeepFlash: Complete Workflow Example")
    print("=" * 80)
    print()
    
    # Configuration
    n_components = 3
    n_samples = 500
    
    print("Step 1: Generate Training Data")
    print("-" * 80)
    train_data = create_training_data(
        n_components=n_components,
        n_samples=n_samples,
        random_state=42
    )
    print(f"Generated {n_samples} training samples")
    print()
    
    # Train model
    print("Step 2: Train Deep Learning Model")
    print("-" * 80)
    model = DeepFlashModel(n_components=n_components)
    
    history = model.train_k_value_predictor(
        train_data=train_data,
        epochs=30,
        learning_rate=0.001,
        batch_size=32
    )
    print(f"Training completed. Final loss: {history['loss'][-1]:.6f}")
    print()
    
    # Use model for predictions
    print("Step 3: Phase Split Calculations with ML Model")
    print("-" * 80)
    
    # Create calculator with trained model
    calculator = PhaseSplitCalculator(
        n_components=n_components,
        temperature=350.0,
        pressure=5e5,
        model=model
    )
    
    # Test case 1: Light mixture
    feed1 = np.array([0.5, 0.3, 0.2])
    print(f"Test Case 1 - Feed: {feed1}")
    results1 = calculator.flash_calculation(feed1)
    print(f"  Vapor fraction: {results1['vapor_fraction']:.4f}")
    print(f"  K-values (ML predicted): {results1['k_values']}")
    print()
    
    # Test case 2: Different composition
    feed2 = np.array([0.2, 0.5, 0.3])
    print(f"Test Case 2 - Feed: {feed2}")
    results2 = calculator.flash_calculation(feed2)
    print(f"  Vapor fraction: {results2['vapor_fraction']:.4f}")
    print(f"  K-values (ML predicted): {results2['k_values']}")
    print()
    
    # Complete equilibrium with flow rates
    print("Step 4: Complete Equilibrium Calculation")
    print("-" * 80)
    feed_rate = 100.0
    eq_results = calculator.equilibrium_calculation(feed1, feed_rate)
    
    print(f"Feed rate: {feed_rate} mol/s")
    print(f"Status: {eq_results['status']}")
    print(f"Vapor rate: {eq_results['vapor_rate']:.2f} mol/s")
    print(f"Liquid rate: {eq_results['liquid_rate']:.2f} mol/s")
    
    if eq_results['status'] == 'two_phase':
        print(f"\nLiquid composition: {eq_results['liquid_composition']}")
        print(f"Vapor composition: {eq_results['vapor_composition']}")
    print()
    
    # Compare with traditional method (no ML)
    print("Step 5: Comparison with Traditional Method")
    print("-" * 80)
    
    calculator_traditional = PhaseSplitCalculator(
        n_components=n_components,
        temperature=350.0,
        pressure=5e5,
        model=None  # No ML model
    )
    
    results_traditional = calculator_traditional.flash_calculation(feed1)
    
    print("ML Model K-values:         ", results1['k_values'])
    print("Traditional K-values:      ", results_traditional['k_values'])
    print()
    
    print("=" * 80)
    print("Workflow Complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print("- Training data generated successfully")
    print("- Deep learning model trained")
    print("- Phase split calculations performed")
    print("- Results compared between ML and traditional methods")


if __name__ == "__main__":
    main()
