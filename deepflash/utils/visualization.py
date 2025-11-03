"""
Visualization utilities for phase diagrams and results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import warnings


def plot_phase_diagram(vapor_fraction: np.ndarray, 
                      variable: np.ndarray,
                      variable_name: str = "Temperature (K)",
                      title: str = "Phase Diagram",
                      save_path: Optional[str] = None):
    """
    Plot a phase diagram showing vapor fraction vs a variable.
    
    Args:
        vapor_fraction: Array of vapor fractions
        variable: Array of the independent variable (e.g., temperature)
        variable_name: Name of the independent variable
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(variable, vapor_fraction, 'b-', linewidth=2)
    plt.xlabel(variable_name, fontsize=12)
    plt.ylabel('Vapor Fraction', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_ternary_diagram(compositions: np.ndarray, 
                        labels: Optional[List[str]] = None,
                        title: str = "Ternary Phase Diagram",
                        save_path: Optional[str] = None):
    """
    Plot a ternary diagram for 3-component systems.
    
    Note: This is a simplified implementation. For production use,
    consider using specialized libraries like python-ternary.
    
    Args:
        compositions: Array of shape (n_points, 3) with mole fractions
        labels: Optional component labels
        title: Plot title
        save_path: Optional path to save the figure
    """
    if compositions.shape[1] != 3:
        raise ValueError("Ternary diagrams require exactly 3 components")
    
    if labels is None:
        labels = ['Component 1', 'Component 2', 'Component 3']
    
    # Note: This is a placeholder. A full ternary plot implementation
    # would require more complex coordinate transformations
    warnings.warn("Ternary diagram plotting requires additional dependencies. "
                 "Consider installing python-ternary for full functionality.")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(compositions[:, 0], compositions[:, 1], c=compositions[:, 2], 
               cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(label=labels[2])
    plt.xlabel(labels[0], fontsize=12)
    plt.ylabel(labels[1], fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_k_values(k_values: np.ndarray, 
                 component_names: Optional[List[str]] = None,
                 title: str = "Equilibrium K-values",
                 save_path: Optional[str] = None):
    """
    Plot K-values for each component.
    
    Args:
        k_values: Array of K-values for each component
        component_names: Optional component names
        title: Plot title
        save_path: Optional path to save the figure
    """
    n_components = len(k_values)
    
    if component_names is None:
        component_names = [f'Component {i+1}' for i in range(n_components)]
    
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(n_components)
    plt.bar(x_pos, k_values, alpha=0.7, color='steelblue')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='K = 1')
    plt.xlabel('Component', fontsize=12)
    plt.ylabel('K-value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(x_pos, component_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_composition_comparison(liquid_comp: np.ndarray, 
                               vapor_comp: np.ndarray,
                               component_names: Optional[List[str]] = None,
                               title: str = "Phase Compositions",
                               save_path: Optional[str] = None):
    """
    Compare liquid and vapor phase compositions.
    
    Args:
        liquid_comp: Liquid phase mole fractions
        vapor_comp: Vapor phase mole fractions
        component_names: Optional component names
        title: Plot title
        save_path: Optional path to save the figure
    """
    n_components = len(liquid_comp)
    
    if component_names is None:
        component_names = [f'Comp {i+1}' for i in range(n_components)]
    
    x = np.arange(n_components)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, liquid_comp, width, label='Liquid', alpha=0.8)
    rects2 = ax.bar(x + width/2, vapor_comp, width, label='Vapor', alpha=0.8)
    
    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Mole Fraction', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(component_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history: dict, 
                         metrics: Optional[List[str]] = None,
                         title: str = "Training History",
                         save_path: Optional[str] = None):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary with training metrics
        metrics: List of metrics to plot (default: all in history)
        title: Plot title
        save_path: Optional path to save the figure
    """
    if metrics is None:
        metrics = list(history.keys())
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in history:
            axes[i].plot(history[metric], linewidth=2)
            axes[i].set_xlabel('Epoch', fontsize=12)
            axes[i].set_ylabel(metric.capitalize(), fontsize=12)
            axes[i].set_title(f'{metric.capitalize()} vs Epoch', fontsize=12)
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
