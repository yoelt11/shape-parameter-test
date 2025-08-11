"""
Basic usage example for the Shape Parameter Test Project.

This script demonstrates how to use the basic RBF model and run a simple comparison.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.model import RBFModel, StandardModel

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
    
    # Initialize models
    rbf_model = RBFModel(
        input_dim=1,
        output_dim=1,
        num_centers=20,
        shape_parameter=1.0
    )
    
    standard_model = StandardModel(
        input_dim=1,
        output_dim=1,
        hidden_layers=[50, 50]
    )
    
    # Simple forward pass (without training for this example)
    print(f"RBF Model parameters: {rbf_model.num_parameters}")
    print(f"Standard Model parameters: {standard_model.num_parameters}")
    
    # Plot the sample data
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, 'b-', label='True Function', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sample Data for Model Comparison')
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
