"""
Basic example of performing flash calculations with DeepFlash.

This example demonstrates how to use the PhaseSplitCalculator to perform
vapor-liquid equilibrium calculations for a multicomponent mixture.
"""

import numpy as np
from deepflash import PhaseSplitCalculator


def main():
    """Run basic flash calculation example."""
    
    print("=" * 70)
    print("DeepFlash: Basic Flash Calculation Example")
    print("=" * 70)
    print()
    
    # Define system parameters
    n_components = 3
    temperature = 350.0  # Kelvin
    pressure = 5e5  # Pascal (5 bar)
    
    print(f"System Configuration:")
    print(f"  Number of components: {n_components}")
    print(f"  Temperature: {temperature} K")
    print(f"  Pressure: {pressure/1e5:.1f} bar")
    print()
    
    # Create calculator
    calculator = PhaseSplitCalculator(
        n_components=n_components,
        temperature=temperature,
        pressure=pressure
    )
    
    # Define feed composition
    feed_composition = np.array([0.3, 0.5, 0.2])
    print(f"Feed Composition:")
    for i, comp in enumerate(feed_composition):
        print(f"  Component {i+1}: {comp:.3f}")
    print()
    
    # Perform flash calculation
    print("Performing flash calculation...")
    results = calculator.flash_calculation(feed_composition)
    
    # Display results
    print("\nResults:")
    print("-" * 70)
    print(f"Vapor Fraction (β): {results['vapor_fraction']:.4f}")
    print()
    
    print("K-values:")
    for i, k in enumerate(results['k_values']):
        print(f"  Component {i+1}: {k:.4f}")
    print()
    
    print("Liquid Phase Composition:")
    for i, comp in enumerate(results['liquid_composition']):
        print(f"  Component {i+1}: {comp:.4f}")
    print(f"  Sum: {np.sum(results['liquid_composition']):.4f}")
    print()
    
    print("Vapor Phase Composition:")
    for i, comp in enumerate(results['vapor_composition']):
        print(f"  Component {i+1}: {comp:.4f}")
    print(f"  Sum: {np.sum(results['vapor_composition']):.4f}")
    print()
    
    # Perform complete equilibrium calculation with flow rates
    feed_rate = 100.0  # mol/s
    print(f"Complete Equilibrium Calculation (Feed rate: {feed_rate} mol/s):")
    print("-" * 70)
    
    eq_results = calculator.equilibrium_calculation(feed_composition, feed_rate)
    
    print(f"Status: {eq_results['status']}")
    print(f"Liquid rate: {eq_results['liquid_rate']:.2f} mol/s")
    print(f"Vapor rate: {eq_results['vapor_rate']:.2f} mol/s")
    print()
    
    print("=" * 70)
    print("Calculation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
