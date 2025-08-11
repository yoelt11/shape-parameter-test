"""
Basic usage example for the Shape Parameter Test Project.

This script demonstrates how to use the basic RBF model and run a simple comparison.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.model import RBFModel, StandardModel, precompute_params, fn_evaluate

def create_sample_data():
    """Create sample data for demonstration."""
    x = np.linspace(-2, 2, 100).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * x) * np.exp(-x**2)
    return x, y_true

def run_basic_comparison():
    """Run a basic comparison between RBF and Standard models."""
    print("Running basic model comparison...")
    
    # Create sample data
    x, y_true = create_sample_data()
    
    # For RBF model, we need to create parameters
    # Create a simple 2D example with 5 centers
    K = 5  # number of centers
    mus = np.random.randn(K, 2) * 0.5  # centers
    epsilons = np.ones(K) * 1.0  # shape parameters
    weights = np.random.randn(K) * 0.1  # weights
    
    # Precompute RBF parameters
    mus_comp, weights_comp, inv_covs = precompute_params(mus, epsilons, weights)
    
    # Create evaluation points (2D for this example)
    X_eval = np.column_stack([x.flatten(), np.zeros_like(x.flatten())])
    
    # Evaluate RBF model
    y_rbf = fn_evaluate(X_eval, mus_comp, weights_comp, inv_covs)
    
    # For standard model, we'll just show the structure
    print(f"RBF Model centers: {K}")
    print(f"RBF Model parameters: {mus.size + epsilons.size + weights.size}")
    print(f"Standard Model would have similar parameter count for comparison")
    
    # Plot the sample data and RBF approximation
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Original data
    plt.subplot(2, 1, 1)
    plt.plot(x, y_true, 'b-', label='True Function', linewidth=2)
    plt.plot(x, y_rbf, 'r--', label='RBF Approximation', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sample Data and RBF Approximation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Centers and evaluation
    plt.subplot(2, 1, 2)
    plt.scatter(mus[:, 0], mus[:, 1], c='red', s=100, label='RBF Centers', alpha=0.7)
    plt.scatter(X_eval[:, 0], X_eval[:, 1], c='blue', s=20, label='Evaluation Points', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RBF Centers and Evaluation Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'basic_usage_example.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Basic comparison completed!")
    print(f"Plot saved to: {os.path.join(output_dir, 'basic_usage_example.png')}")

if __name__ == "__main__":
    run_basic_comparison()
