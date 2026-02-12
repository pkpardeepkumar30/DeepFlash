# DeepFlash

A Python library for thermodynamic phase split calculations using Deep Learning.

## Overview

DeepFlash provides tools for performing vapor-liquid equilibrium (VLE) calculations and phase split predictions using both traditional thermodynamic methods and deep learning approaches. The library is designed for researchers and engineers working on separation processes, petroleum engineering, and chemical engineering applications.

## Features

- **Phase Split Calculations**: Perform flash calculations using the Rachford-Rice algorithm
- **Deep Learning Models**: Neural network models for predicting K-values and phase stability
- **Equilibrium Calculations**: Complete equilibrium calculations with flow rate predictions
- **Data Processing**: Utilities for composition normalization and synthetic data generation
- **Visualization**: Tools for plotting phase diagrams and results
- **Extensible Architecture**: Easy to extend with custom models and thermodynamic correlations

## Installation

### From Source

```bash
git clone https://github.com/pkpardeepkumar30/DeepFlash.git
cd DeepFlash
pip install -e .
```

### Requirements

- Python >= 3.7
- NumPy >= 1.21.0
- PyTorch >= 1.9.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- scikit-learn >= 0.24.0
- pandas >= 1.3.0
- seaborn >= 0.11.0

## Quick Start

### Basic Flash Calculation

```python
import numpy as np
from deepflash import PhaseSplitCalculator

# Create calculator
calculator = PhaseSplitCalculator(
    n_components=3,
    temperature=350.0,  # Kelvin
    pressure=5e5        # Pascal
)

# Define feed composition
feed = np.array([0.3, 0.5, 0.2])

# Perform flash calculation
results = calculator.flash_calculation(feed)

print(f"Vapor fraction: {results['vapor_fraction']:.3f}")
print(f"Liquid composition: {results['liquid_composition']}")
print(f"Vapor composition: {results['vapor_composition']}")
```

### Using Deep Learning Model

```python
from deepflash import DeepFlashModel, PhaseSplitCalculator

# Create model
model = DeepFlashModel(n_components=3)

# Create calculator with model
calculator = PhaseSplitCalculator(
    n_components=3,
    temperature=350.0,
    pressure=5e5,
    model=model
)

# Perform flash calculation using ML predictions
results = calculator.flash_calculation(feed)
```

### Training a Model

```python
from deepflash import DeepFlashModel
from deepflash.utils.data_processing import create_training_data

# Generate training data
train_data = create_training_data(
    n_components=3,
    n_samples=1000,
    temperature_range=(250.0, 450.0),
    pressure_range=(1e5, 1e7)
)

# Create and train model
model = DeepFlashModel(n_components=3)
history = model.train_k_value_predictor(
    train_data=train_data,
    epochs=100,
    learning_rate=0.001
)
```

## Examples

The `examples/` directory contains several demonstration scripts:

- `basic_flash_calculation.py`: Basic phase split calculation
- `train_model_example.py`: Training a deep learning model
- `complete_workflow.py`: Complete workflow from data generation to predictions

Run an example:

```bash
python examples/basic_flash_calculation.py
```

## Core Components

### PhaseSplitCalculator

The main class for performing thermodynamic calculations:

- `flash_calculation()`: Perform flash calculation using Rachford-Rice algorithm
- `equilibrium_calculation()`: Complete equilibrium with flow rates
- `stability_analysis()`: Check phase stability

### DeepFlashModel

Deep learning model for property prediction:

- `predict_k_values()`: Predict equilibrium K-values
- `predict_stability()`: Predict phase stability
- `train_k_value_predictor()`: Train the model on data
- `save_model()` / `load_model()`: Model persistence

### Neural Network Architectures

- `ThermodynamicNN`: Base feedforward neural network
- `KValuePredictor`: Specialized network for K-value prediction
- `PhaseStabilityClassifier`: Binary classifier for phase stability

## Methodology

### Thermodynamic Foundation

DeepFlash implements the Rachford-Rice algorithm for isothermal flash calculations:

$$\sum_{i=1}^{n} \frac{z_i(K_i - 1)}{1 + \beta(K_i - 1)} = 0$$

Where:
- $z_i$ = feed mole fraction of component i
- $K_i$ = equilibrium K-value for component i
- $\beta$ = vapor fraction

### Deep Learning Approach

The library uses neural networks to predict:

1. **K-values**: Direct prediction of equilibrium constants from composition, temperature, and pressure
2. **Phase Stability**: Classification of single-phase vs two-phase regions

This hybrid approach combines:
- Speed and efficiency of neural networks
- Thermodynamic rigor of traditional methods

## Project Structure

```
DeepFlash/
├── deepflash/           # Main package
│   ├── __init__.py
│   ├── phase_split.py   # Phase split calculations
│   ├── models/          # Deep learning models
│   │   ├── __init__.py
│   │   ├── deep_flash_model.py
│   │   └── neural_network.py
│   ├── utils/           # Utility functions
│   │   ├── __init__.py
│   │   ├── data_processing.py
│   │   └── visualization.py
│   └── data/            # Data handling
├── examples/            # Example scripts
├── tests/               # Unit tests
├── setup.py            # Package setup
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use DeepFlash in your research, please cite:

```
@software{deepflash2024,
  title={DeepFlash: Deep Learning for Thermodynamic Phase Split Calculations},
  author={DeepFlash Team},
  year={2024},
  url={https://github.com/pkpardeepkumar30/DeepFlash}
}
```

## References

- Rachford, H.H., Rice, J.D. (1952). "Procedure for Use of Electronic Digital Computers in Calculating Flash Vaporization Hydrocarbon Equilibrium"
- Thermodynamic modeling and deep learning approaches for phase equilibrium calculations

## Contact

For questions and support, please open an issue on GitHub.