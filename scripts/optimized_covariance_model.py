#!/usr/bin/env python3
"""
Optimized covariance model implementation for faster convergence.
This module provides simplified covariance parameterizations that are easier to optimize.
"""

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Callable, List
import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)

class OptimizedRBFModel:
    """Optimized RBF model with simplified covariance parameterization for faster convergence."""
    
    def __init__(self, model_type: str = 'isotropic', n_kernels: int = 25):
        """
        Initialize optimized RBF model.
        
        Args:
            model_type: 'isotropic', 'scaled_diagonal', or 'direct_inverse'
            n_kernels: Number of RBF kernels
        """
        self.model_type = model_type
        self.n_kernels = n_kernels
        
        if model_type == 'isotropic':
            self.param_dim = 4  # [mu_x, mu_y, log_sigma, weight]
        elif model_type == 'scaled_diagonal':
            self.param_dim = 6  # [mu_x, mu_y, log_sigma, scale_x, scale_y, weight]
        elif model_type == 'direct_inverse':
            self.param_dim = 5  # [mu_x, mu_y, inv_cov_11, inv_cov_12, inv_cov_22, weight]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def initialize_parameters(self, key: jax.random.PRNGKey = None) -> jnp.ndarray:
        """Initialize parameters for the optimized RBF model."""
        if key is None:
            key = jax.random.PRNGKey(42)
        
        # Create a grid of centers
        grid_size = int(jnp.sqrt(self.n_kernels))
        x_centers = jnp.linspace(-0.8, 0.8, grid_size)
        y_centers = jnp.linspace(-0.8, 0.8, grid_size)
        
        xx, yy = jnp.meshgrid(x_centers, y_centers)
        centers = jnp.stack([xx.flatten(), yy.flatten()], axis=1)
        
        # Initialize parameters
        params = jnp.zeros((self.n_kernels, self.param_dim))
        
        # Set means (centers)
        params = params.at[:, 0:2].set(centers)
        
        if self.model_type == 'isotropic':
            # Set log_sigma (isotropic)
            params = params.at[:, 2].set(jnp.log(0.1) * jnp.ones(self.n_kernels))
            # Set weights (random initialization)
            key, subkey = jax.random.split(key)
            params = params.at[:, 3].set(jax.random.normal(subkey, (self.n_kernels,)) * 0.1)
            
        elif self.model_type == 'scaled_diagonal':
            # Set log_sigma (base sigma)
            params = params.at[:, 2].set(jnp.log(0.1) * jnp.ones(self.n_kernels))
            # Set scale factors (close to 1.0 initially)
            params = params.at[:, 3:5].set(jnp.ones((self.n_kernels, 2)))
            # Set weights (random initialization)
            key, subkey = jax.random.split(key)
            params = params.at[:, 5].set(jax.random.normal(subkey, (self.n_kernels,)) * 0.1)
            
        elif self.model_type == 'direct_inverse':
            # Set direct inverse covariance parameters
            params = params.at[:, 2].set(100.0 * jnp.ones(self.n_kernels))  # inv_cov_11
            params = params.at[:, 3].set(0.0 * jnp.ones(self.n_kernels))    # inv_cov_12
            params = params.at[:, 4].set(100.0 * jnp.ones(self.n_kernels))  # inv_cov_22
            # Set weights (random initialization)
            key, subkey = jax.random.split(key)
            params = params.at[:, 5].set(jax.random.normal(subkey, (self.n_kernels,)) * 0.1)
        
        return params
    
    def precompute_parameters(self, params: jnp.ndarray, epsilon: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Precompute parameters for the optimized RBF model."""
        mus = params[:, 0:2]  # (K, 2)
        
        if self.model_type == 'isotropic':
            log_sigmas = params[:, 2]  # (K,)
            weights = params[:, 3]  # (K,)
            
            # Compute isotropic sigmas
            sigmas = jnp.exp(log_sigmas)  # (K,)
            
            # Create diagonal inverse covariance matrices
            inv_covs = jnp.zeros((params.shape[0], 2, 2))
            inv_covs = inv_covs.at[:, 0, 0].set(1.0 / (sigmas**2 + epsilon))
            inv_covs = inv_covs.at[:, 1, 1].set(1.0 / (sigmas**2 + epsilon))
            
        elif self.model_type == 'scaled_diagonal':
            log_sigma = params[:, 2]  # (K,)
            scale_factors = params[:, 3:5]  # (K, 2)
            weights = params[:, 5]  # (K,)
            
            # Compute base sigma
            sigma = jnp.exp(log_sigma)  # (K,)
            
            # Create scaled diagonal inverse covariance matrices
            inv_covs = jnp.zeros((params.shape[0], 2, 2))
            inv_covs = inv_covs.at[:, 0, 0].set(scale_factors[:, 0] / (sigma**2 + epsilon))
            inv_covs = inv_covs.at[:, 1, 1].set(scale_factors[:, 1] / (sigma**2 + epsilon))
            
        elif self.model_type == 'direct_inverse':
            inv_cov_11 = params[:, 2]  # (K,)
            inv_cov_12 = params[:, 3]  # (K,)
            inv_cov_22 = params[:, 4]  # (K,)
            weights = params[:, 5]  # (K,)
            
            # Create inverse covariance matrices directly
            inv_covs = jnp.zeros((params.shape[0], 2, 2))
            inv_covs = inv_covs.at[:, 0, 0].set(jnp.abs(inv_cov_11) + epsilon)
            inv_covs = inv_covs.at[:, 0, 1].set(inv_cov_12)
            inv_covs = inv_covs.at[:, 1, 0].set(inv_cov_12)
            inv_covs = inv_covs.at[:, 1, 1].set(jnp.abs(inv_cov_22) + epsilon)
            
            # Ensure positive definiteness
            det = inv_covs[:, 0, 0] * inv_covs[:, 1, 1] - inv_covs[:, 0, 1]**2
            min_det = 1e-6
            scale_factor = jnp.maximum(min_det / det, 1.0)
            inv_covs = inv_covs * scale_factor[:, None, None]
        
        return mus, weights, inv_covs
    
    def evaluate(self, X: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the optimized RBF model."""
        mus, weights, inv_covs = self.precompute_parameters(params)
        
        # Compute all differences at once: (N, K, 2)
        diff = X[:, None, :] - mus[None, :, :]
        
        # Compute quadratic forms efficiently
        quad = jnp.einsum('nki,kij,nkj->nk', diff, inv_covs, diff)
        
        # Compute all kernel values at once
        phi = jnp.exp(-0.5 * quad)
        
        # Weighted sum
        return jnp.dot(phi, weights)
    
    def create_loss_function(self, target: jnp.ndarray) -> Callable:
        """Create loss function for the optimized RBF model."""
        def loss_fn(params):
            prediction = self.evaluate(X, params)
            return jnp.mean((prediction - target) ** 2)
        return loss_fn

def create_adaptive_optimizer():
    """Create adaptive optimizer with different learning rates for different parameter types."""
    
    def create_optimizer(learning_rate_weights: float = 0.01, 
                        learning_rate_covariance: float = 0.001):
        """Create optimizer with different learning rates for weights vs covariance parameters."""
        
        # Create separate optimizers
        weight_optimizer = optax.adam(learning_rate_weights)
        cov_optimizer = optax.adam(learning_rate_covariance)
        
        def init_fn(params):
            return {
                'weight_opt_state': weight_optimizer.init(params),
                'cov_opt_state': cov_optimizer.init(params)
            }
        
        def update_fn(updates, opt_state):
            # Split parameters into weights and covariance parameters
            weight_params = updates[:, -1]  # weights are last column
            cov_params = updates[:, :-1]    # covariance parameters are all but last column
            
            # Update with different optimizers
            weight_updates, weight_opt_state = weight_optimizer.update(weight_params, opt_state['weight_opt_state'])
            cov_updates, cov_opt_state = cov_optimizer.update(cov_params, opt_state['cov_opt_state'])
            
            # Combine updates
            combined_updates = jnp.concatenate([cov_updates, weight_updates[:, None]], axis=1)
            
            return combined_updates, {
                'weight_opt_state': weight_opt_state,
                'cov_opt_state': cov_opt_state
            }
        
        return optax.GradientTransformation(init_fn, update_fn)
    
    return create_optimizer

def train_optimized_model(model_type: str = 'isotropic', 
                         n_kernels: int = 25, 
                         n_epochs: int = 1000,
                         learning_rate_weights: float = 0.01,
                         learning_rate_covariance: float = 0.001):
    """Train the optimized RBF model."""
    
    print(f"Training {model_type} RBF model with {n_kernels} kernels...")
    
    # Create model
    model = OptimizedRBFModel(model_type=model_type, n_kernels=n_kernels)
    
    # Create training data
    x = jnp.linspace(-1, 1, 50)
    y = jnp.linspace(-1, 1, 50)
    X, Y = jnp.meshgrid(x, y)
    target = jnp.sin(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y)
    target_flat = target.flatten()
    X_eval = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = model.initialize_parameters(key)
    
    # Create loss function
    loss_fn = model.create_loss_function(target_flat)
    
    # Create optimizer
    optimizer = create_adaptive_optimizer()(learning_rate_weights, learning_rate_covariance)
    opt_state = optimizer.init(params)
    
    # Training function
    @jax.jit
    def train_step(params, opt_state, X_eval, target):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Training loop
    loss_history = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        params, opt_state, loss = train_step(params, opt_state, X_eval, target_flat)
        loss_history.append(float(loss))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final loss: {loss_history[-1]:.6f}")
    
    return params, loss_history, training_time

def compare_model_performance():
    """Compare performance of different optimized models."""
    
    print("="*80)
    print("OPTIMIZED MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    model_types = ['isotropic', 'scaled_diagonal', 'direct_inverse']
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting {model_type} model...")
        
        # Train model
        params, loss_history, training_time = train_optimized_model(
            model_type=model_type,
            n_kernels=25,
            n_epochs=500
        )
        
        # Analyze results
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        convergence_rate = (initial_loss - final_loss) / len(loss_history)
        
        results[model_type] = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'training_time': training_time,
            'convergence_rate': convergence_rate,
            'loss_history': loss_history
        }
        
        print(f"  Initial Loss: {initial_loss:.6f}")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"  Convergence Rate: {convergence_rate:.6f}")
    
    return results

def create_performance_plots(results: Dict):
    """Create performance comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Optimized RBF Model Performance Comparison', fontsize=16, fontweight='bold')
    
    model_types = list(results.keys())
    
    # 1. Final Loss Comparison
    ax1 = axes[0, 0]
    final_losses = [results[model]['final_loss'] for model in model_types]
    bars1 = ax1.bar(model_types, final_losses, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Final Loss Comparison', fontweight='bold')
    ax1.set_ylabel('Final Loss')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, final_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.6f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Training Time Comparison
    ax2 = axes[0, 1]
    training_times = [results[model]['training_time'] for model in model_types]
    bars2 = ax2.bar(model_types, training_times, color=['gold', 'plum', 'orange'])
    ax2.set_title('Training Time Comparison', fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, training_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # 3. Convergence Rate Comparison
    ax3 = axes[1, 0]
    convergence_rates = [results[model]['convergence_rate'] for model in model_types]
    bars3 = ax3.bar(model_types, convergence_rates, color=['lightblue', 'pink', 'lightyellow'])
    ax3.set_title('Convergence Rate Comparison', fontweight='bold')
    ax3.set_ylabel('Convergence Rate (loss/epoch)')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars3, convergence_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                f'{value:.6f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Loss Curves
    ax4 = axes[1, 1]
    colors = ['blue', 'red', 'green']
    for i, model_type in enumerate(model_types):
        loss_history = results[model_type]['loss_history']
        epochs = range(len(loss_history))
        ax4.plot(epochs, loss_history, color=colors[i], label=model_type, linewidth=2)
    
    ax4.set_title('Loss Curves Comparison', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('optimized_model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_implementation_summary():
    """Print implementation summary and recommendations."""
    
    print("\n" + "="*80)
    print("IMPLEMENTATION SUMMARY")
    print("="*80)
    
    print("\nKey Improvements:")
    print("1. **Simplified Parameterization**: Reduced non-linear transformations")
    print("2. **Eliminated Matrix Operations**: No rotation matrices in isotropic model")
    print("3. **Better Numerical Stability**: Controlled parameter ranges")
    print("4. **Adaptive Learning Rates**: Different rates for weights vs covariance")
    print("5. **Faster Convergence**: Simpler gradients lead to faster optimization")
    
    print("\nModel Comparison:")
    print("- Isotropic: Fastest, least flexible")
    print("- Scaled Diagonal: Balanced speed and flexibility")
    print("- Direct Inverse: Most flexible, requires careful implementation")
    
    print("\nRecommendations:")
    print("1. Start with isotropic model for maximum speed")
    print("2. Use scaled diagonal if directional sensitivity is needed")
    print("3. Consider direct inverse only for complex problems")
    print("4. Always use adaptive learning rates")
    print("5. Monitor parameter bounds for numerical stability")

def main():
    """Main function to demonstrate optimized covariance models."""
    
    print("Optimized Covariance Model Implementation")
    print("="*80)
    print("This demonstrates simplified covariance parameterizations for faster convergence.")
    
    # Compare model performance
    results = compare_model_performance()
    
    # Create performance plots
    create_performance_plots(results)
    
    # Print implementation summary
    print_implementation_summary()
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE")
    print("="*80)
    print("""
    # Create optimized model
    model = OptimizedRBFModel(model_type='isotropic', n_kernels=25)
    
    # Initialize parameters
    params = model.initialize_parameters()
    
    # Create loss function
    loss_fn = model.create_loss_function(target)
    
    # Train with adaptive optimizer
    optimizer = create_adaptive_optimizer()(lr_weights=0.01, lr_cov=0.001)
    opt_state = optimizer.init(params)
    
    # Training loop
    for epoch in range(n_epochs):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    """)

if __name__ == "__main__":
    main()


