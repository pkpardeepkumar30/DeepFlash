"""
Example of training a DeepFlash model on synthetic data.

This example demonstrates how to generate training data and train
a deep learning model for K-value prediction.
"""

import numpy as np
from deepflash import DeepFlashModel
from deepflash.utils.data_processing import create_training_data


def main():
    """Run model training example."""
    
    print("=" * 70)
    print("DeepFlash: Model Training Example")
    print("=" * 70)
    print()
    
    # Configuration
    n_components = 3
    n_samples = 1000
    random_state = 42
    
    print(f"Configuration:")
    print(f"  Number of components: {n_components}")
    print(f"  Number of training samples: {n_samples}")
    print()
    
    # Generate synthetic training data
    print("Generating synthetic training data...")
    train_data = create_training_data(
        n_components=n_components,
        n_samples=n_samples,
        temperature_range=(250.0, 450.0),
        pressure_range=(1e5, 1e7),
        random_state=random_state
    )
    
    print(f"  Input shape: {train_data['inputs'].shape}")
    print(f"  Target shape: {train_data['targets'].shape}")
    print()
    
    # Create model
    print("Initializing DeepFlash model...")
    model = DeepFlashModel(n_components=n_components, device='cpu')
    print(f"  Device: {model.device}")
    print()
    
    # Train the model
    print("Training K-value predictor...")
    print("-" * 70)
    
    history = model.train_k_value_predictor(
        train_data=train_data,
        epochs=50,
        learning_rate=0.001,
        batch_size=32
    )
    
    print()
    print("Training complete!")
    print(f"Final loss: {history['loss'][-1]:.6f}")
    print()
    
    # Test prediction on a sample
    print("Testing prediction on sample data:")
    print("-" * 70)
    
    sample_idx = 0
    sample_composition = train_data['compositions'][sample_idx]
    sample_temperature = train_data['temperatures'][sample_idx]
    sample_pressure = train_data['pressures'][sample_idx]
    
    print(f"Input:")
    print(f"  Composition: {sample_composition}")
    print(f"  Temperature: {sample_temperature:.2f} K")
    print(f"  Pressure: {sample_pressure/1e5:.2f} bar")
    print()
    
    # Predict K-values
    predicted_k = model.predict_k_values(
        sample_composition, 
        sample_temperature, 
        sample_pressure
    )
    
    actual_k = train_data['targets'][sample_idx]
    
    print("Predicted K-values:")
    for i, k in enumerate(predicted_k):
        print(f"  Component {i+1}: {k:.4f}")
    print()
    
    print("Actual K-values:")
    for i, k in enumerate(actual_k):
        print(f"  Component {i+1}: {k:.4f}")
    print()
    
    # Calculate error
    error = np.abs(predicted_k - actual_k)
    print("Absolute Error:")
    for i, e in enumerate(error):
        print(f"  Component {i+1}: {e:.4f}")
    print()
    
    print("=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
