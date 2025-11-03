"""
Neural network architectures for thermodynamic property prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class ThermodynamicNN(nn.Module):
    """
    Feedforward neural network for predicting thermodynamic properties.
    
    This network takes composition, temperature, and pressure as inputs
    and predicts equilibrium K-values or other thermodynamic properties.
    
    Attributes:
        input_size (int): Size of input layer
        hidden_sizes (List[int]): Sizes of hidden layers
        output_size (int): Size of output layer
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], 
                 output_size: int, dropout_rate: float = 0.1):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output features
            dropout_rate: Dropout rate for regularization
        """
        super(ThermodynamicNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        return self.network(x)


class KValuePredictor(ThermodynamicNN):
    """
    Neural network specifically for predicting K-values.
    
    This network predicts equilibrium K-values for vapor-liquid equilibrium
    given composition, temperature, and pressure.
    """
    
    def __init__(self, n_components: int, hidden_sizes: Optional[List[int]] = None):
        """
        Initialize K-value predictor.
        
        Args:
            n_components: Number of components in the mixture
            hidden_sizes: List of hidden layer sizes (default: [128, 64, 32])
        """
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        # Input: composition (n_components) + temperature (1) + pressure (1)
        input_size = n_components + 2
        
        # Output: K-values for each component
        output_size = n_components
        
        super(KValuePredictor, self).__init__(input_size, hidden_sizes, output_size)
        
        # Add softplus activation to ensure positive K-values
        self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with constraint for positive K-values.
        
        Args:
            x: Input tensor containing [composition, temperature, pressure]
            
        Returns:
            K-values (positive)
        """
        output = super().forward(x)
        # Ensure K-values are positive using softplus + small constant
        return self.softplus(output) + 0.1


class PhaseStabilityClassifier(nn.Module):
    """
    Neural network for classifying phase stability.
    
    This network predicts whether a mixture is in single-phase or two-phase region.
    """
    
    def __init__(self, n_components: int, hidden_sizes: Optional[List[int]] = None):
        """
        Initialize phase stability classifier.
        
        Args:
            n_components: Number of components in the mixture
            hidden_sizes: List of hidden layer sizes (default: [64, 32])
        """
        super(PhaseStabilityClassifier, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        
        # Input: composition (n_components) + temperature (1) + pressure (1)
        input_size = n_components + 2
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Binary classification output
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for stability classification.
        
        Args:
            x: Input tensor containing [composition, temperature, pressure]
            
        Returns:
            Probability of being stable (single-phase)
        """
        return self.network(x)
