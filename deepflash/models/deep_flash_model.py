"""
DeepFlash model for thermodynamic phase split calculations.

This module provides a high-level interface for using deep learning models
to predict phase equilibria and perform flash calculations.
"""

import torch
import numpy as np
from typing import Optional, Dict, Union
from .neural_network import KValuePredictor, PhaseStabilityClassifier


class DeepFlashModel:
    """
    Main model class for deep learning-based phase split calculations.
    
    This class wraps the neural network models and provides convenient
    methods for prediction and training.
    
    Attributes:
        n_components (int): Number of components in the mixture
        k_value_predictor: Neural network for K-value prediction
        stability_classifier: Neural network for phase stability classification
        device: PyTorch device (cpu or cuda)
    """
    
    def __init__(self, n_components: int, device: Optional[str] = None):
        """
        Initialize the DeepFlash model.
        
        Args:
            n_components: Number of components in the mixture
            device: Device to run the model on ('cpu' or 'cuda'). If None, auto-detect.
        """
        self.n_components = n_components
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize models
        self.k_value_predictor = KValuePredictor(n_components).to(self.device)
        self.stability_classifier = PhaseStabilityClassifier(n_components).to(self.device)
        
        # Set to evaluation mode by default
        self.k_value_predictor.eval()
        self.stability_classifier.eval()
    
    def predict_k_values(self, composition: np.ndarray, temperature: float, 
                        pressure: float) -> np.ndarray:
        """
        Predict equilibrium K-values for the given conditions.
        
        Args:
            composition: Mole fractions (length n_components)
            temperature: Temperature in Kelvin
            pressure: Pressure in Pascal
            
        Returns:
            Predicted K-values
        """
        # Normalize inputs
        T_norm = temperature / 500.0  # Normalize to typical range
        P_norm = pressure / 1e6  # Normalize to MPa
        
        # Prepare input tensor
        input_data = np.concatenate([composition, [T_norm, P_norm]])
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            k_values = self.k_value_predictor(input_tensor)
        
        return k_values.cpu().numpy().squeeze()
    
    def predict_stability(self, composition: np.ndarray, temperature: float, 
                         pressure: float, threshold: float = 0.5) -> bool:
        """
        Predict phase stability of the mixture.
        
        Args:
            composition: Mole fractions (length n_components)
            temperature: Temperature in Kelvin
            pressure: Pressure in Pascal
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            True if stable (single-phase), False if unstable (two-phase)
        """
        # Normalize inputs
        T_norm = temperature / 500.0
        P_norm = pressure / 1e6
        
        # Prepare input tensor
        input_data = np.concatenate([composition, [T_norm, P_norm]])
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            stability_prob = self.stability_classifier(input_tensor)
        
        return stability_prob.item() > threshold
    
    def train_k_value_predictor(self, train_data: Dict[str, np.ndarray], 
                               epochs: int = 100, learning_rate: float = 0.001,
                               batch_size: int = 32) -> Dict[str, list]:
        """
        Train the K-value predictor on provided data.
        
        Args:
            train_data: Dictionary with 'inputs' and 'targets' arrays
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training history
        """
        self.k_value_predictor.train()
        
        # Prepare data
        X = torch.FloatTensor(train_data['inputs']).to(self.device)
        y = torch.FloatTensor(train_data['targets']).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                 shuffle=True)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.k_value_predictor.parameters(), 
                                    lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        history = {'loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.k_value_predictor(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.k_value_predictor.eval()
        return history
    
    def save_model(self, filepath: str):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'k_value_predictor': self.k_value_predictor.state_dict(),
            'stability_classifier': self.stability_classifier.state_dict(),
            'n_components': self.n_components
        }, filepath)
    
    def load_model(self, filepath: str):
        """
        Load the model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.n_components = checkpoint['n_components']
        self.k_value_predictor.load_state_dict(checkpoint['k_value_predictor'])
        self.stability_classifier.load_state_dict(checkpoint['stability_classifier'])
        
        self.k_value_predictor.eval()
        self.stability_classifier.eval()
