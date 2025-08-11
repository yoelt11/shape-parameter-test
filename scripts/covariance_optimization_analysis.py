#!/usr/bin/env python3
"""
Analysis of covariance matrix optimization challenges and solutions for faster convergence.
This script investigates why the covariance matrix is difficult to optimize and proposes solutions.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Callable, List
import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model.standard_model import generate_rbf_solutions as generate_standard_rbf

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)

def analyze_covariance_parameterization():
    """Analyze the current covariance matrix parameterization and its optimization challenges."""
    
    print("="*80)
    print("COVARIANCE MATRIX OPTIMIZATION ANALYSIS")
    print("="*80)
    
    print("\nCurrent Parameterization:")
    print("-" * 40)
    print("1. log_sigmas: (K, 2) - log of standard deviations")
    print("2. angles: (K,) - rotation angles (sigmoid transformed)")
    print("3. mus: (K, 2) - kernel centers")
    print("4. weights: (K,) - kernel weights")
    
    print("\nCovariance Matrix Construction:")
    print("-" * 40)
    print("1. sigmas = exp(log_sigmas)  # (K, 2)")
    print("2. angles = sigmoid(angles) * 2π  # (K,)")
    print("3. R = rotation_matrix(angles)  # (K, 2, 2)")
    print("4. Σ⁻¹ = R * diag(1/σ²) * Rᵀ  # (K, 2, 2)")
    
    print("\nOptimization Challenges:")
    print("-" * 40)
    print("1. **Non-linear transformations**: exp(), sigmoid(), cos(), sin()")
    print("2. **Matrix operations**: rotation, matrix multiplication")
    print("3. **Numerical instability**: 1/σ² can be very large")
    print("4. **Parameter coupling**: angles affect both dimensions")
    print("5. **Gradient complexity**: Chain rule through multiple transformations")

def create_simplified_covariance_models():
    """Create simplified covariance matrix parameterizations for easier optimization."""
    
    print("\n" + "="*80)
    print("SIMPLIFIED COVARIANCE MODELS")
    print("="*80)
    
    models = {}
    
    # Model 1: Diagonal covariance (no rotation)
    def diagonal_covariance(log_sigmas):
        """Simplest model: diagonal covariance matrices."""
        sigmas = jnp.exp(log_sigmas)  # (K, 2)
        inv_covs = jnp.zeros((log_sigmas.shape[0], 2, 2))
        inv_covs = inv_covs.at[:, 0, 0].set(1.0 / (sigmas[:, 0]**2 + 1e-6))
        inv_covs = inv_covs.at[:, 1, 1].set(1.0 / (sigmas[:, 1]**2 + 1e-6))
        return inv_covs
    
    # Model 2: Isotropic covariance (same σ in both directions)
    def isotropic_covariance(log_sigma):
        """Isotropic model: same standard deviation in both directions."""
        sigma = jnp.exp(log_sigma)  # (K,)
        inv_covs = jnp.zeros((log_sigma.shape[0], 2, 2))
        inv_covs = inv_covs.at[:, 0, 0].set(1.0 / (sigma**2 + 1e-6))
        inv_covs = inv_covs.at[:, 1, 1].set(1.0 / (sigma**2 + 1e-6))
        return inv_covs
    
    # Model 3: Scaled diagonal covariance
    def scaled_diagonal_covariance(log_sigma, scale_factors):
        """Scaled diagonal: one base σ, scaled for each direction."""
        sigma = jnp.exp(log_sigma)  # (K,)
        inv_covs = jnp.zeros((log_sigma.shape[0], 2, 2))
        inv_covs = inv_covs.at[:, 0, 0].set(scale_factors[:, 0] / (sigma**2 + 1e-6))
        inv_covs = inv_covs.at[:, 1, 1].set(scale_factors[:, 1] / (sigma**2 + 1e-6))
        return inv_covs
    
    # Model 4: Direct inverse covariance parameterization
    def direct_inv_covariance(inv_cov_params):
        """Direct parameterization of inverse covariance elements."""
        # inv_cov_params: (K, 3) - [a, b, c] for [[a, b], [b, c]]
        K = inv_cov_params.shape[0]
        inv_covs = jnp.zeros((K, 2, 2))
        inv_covs = inv_covs.at[:, 0, 0].set(inv_cov_params[:, 0])
        inv_covs = inv_covs.at[:, 0, 1].set(inv_cov_params[:, 1])
        inv_covs = inv_covs.at[:, 1, 0].set(inv_cov_params[:, 1])
        inv_covs = inv_covs.at[:, 1, 1].set(inv_cov_params[:, 2])
        return inv_covs
    
    models = {
        'diagonal': diagonal_covariance,
        'isotropic': isotropic_covariance,
        'scaled_diagonal': scaled_diagonal_covariance,
        'direct_inv': direct_inv_covariance
    }
    
    return models

def compare_parameterization_complexity():
    """Compare the complexity of different covariance parameterizations."""
    
    print("\n" + "="*80)
    print("PARAMETERIZATION COMPLEXITY COMPARISON")
    print("="*80)
    
    parameterizations = {
        'Current (Full)': {
            'parameters_per_kernel': 4,  # log_sigma_x, log_sigma_y, angle, weight
            'non_linear_transforms': 4,  # exp, sigmoid, cos, sin
            'matrix_operations': 3,  # rotation, diag_inv, matrix_mult
            'numerical_issues': 'High',  # 1/σ² can be very large
            'gradient_complexity': 'Very High'
        },
        'Diagonal': {
            'parameters_per_kernel': 3,  # log_sigma_x, log_sigma_y, weight
            'non_linear_transforms': 1,  # exp only
            'matrix_operations': 0,  # diagonal only
            'numerical_issues': 'Medium',  # 1/σ² but simpler
            'gradient_complexity': 'Low'
        },
        'Isotropic': {
            'parameters_per_kernel': 2,  # log_sigma, weight
            'non_linear_transforms': 1,  # exp only
            'matrix_operations': 0,  # diagonal only
            'numerical_issues': 'Low',  # simple 1/σ²
            'gradient_complexity': 'Very Low'
        },
        'Scaled Diagonal': {
            'parameters_per_kernel': 3,  # log_sigma, scale_x, scale_y, weight
            'non_linear_transforms': 1,  # exp only
            'matrix_operations': 0,  # diagonal only
            'numerical_issues': 'Low',  # controlled scaling
            'gradient_complexity': 'Low'
        },
        'Direct Inverse': {
            'parameters_per_kernel': 4,  # inv_cov_11, inv_cov_12, inv_cov_22, weight
            'non_linear_transforms': 0,  # no transforms
            'matrix_operations': 0,  # direct assignment
            'numerical_issues': 'Medium',  # need to ensure positive definiteness
            'gradient_complexity': 'Very Low'
        }
    }
    
    print(f"{'Model':<20} {'Params/Kernel':<12} {'Transforms':<10} {'Matrix Ops':<10} {'Num Issues':<10} {'Grad Complexity':<15}")
    print("-" * 80)
    
    for model, details in parameterizations.items():
        print(f"{model:<20} {details['parameters_per_kernel']:<12} {details['non_linear_transforms']:<10} "
              f"{details['matrix_operations']:<10} {details['numerical_issues']:<10} {details['gradient_complexity']:<15}")

def create_optimization_friendly_model():
    """Create an optimization-friendly RBF model with simplified covariance."""
    
    print("\n" + "="*80)
    print("OPTIMIZATION-FRIENDLY RBF MODEL")
    print("="*80)
    
    def create_isotropic_rbf_model():
        """Create an isotropic RBF model for easier optimization."""
        
        def initialize_isotropic_parameters(n_kernels: int = 25, key: jax.random.PRNGKey = None):
            """Initialize parameters for isotropic RBF model."""
            if key is None:
                key = jax.random.PRNGKey(42)
            
            # Create a grid of centers
            grid_size = int(jnp.sqrt(n_kernels))
            x_centers = jnp.linspace(-0.8, 0.8, grid_size)
            y_centers = jnp.linspace(-0.8, 0.8, grid_size)
            
            xx, yy = jnp.meshgrid(x_centers, y_centers)
            centers = jnp.stack([xx.flatten(), yy.flatten()], axis=1)
            
            # Initialize parameters: [mu_x, mu_y, log_sigma, weight]
            params = jnp.zeros((n_kernels, 4))
            
            # Set means (centers)
            params = params.at[:, 0:2].set(centers)
            
            # Set log_sigma (isotropic)
            params = params.at[:, 2].set(jnp.log(0.1) * jnp.ones(n_kernels))
            
            # Set weights (random initialization)
            key, subkey = jax.random.split(key)
            params = params.at[:, 3].set(jax.random.normal(subkey, (n_kernels,)) * 0.1)
            
            return params
        
        def isotropic_precompute_params(params: jnp.ndarray, epsilon: float = 1e-6):
            """Precompute parameters for isotropic RBF model."""
            mus = params[:, 0:2]  # (K, 2)
            log_sigmas = params[:, 2]  # (K,)
            weights = params[:, 3]  # (K,)
            
            # Compute isotropic sigmas
            sigmas = jnp.exp(log_sigmas)  # (K,)
            
            # Create diagonal inverse covariance matrices
            inv_covs = jnp.zeros((params.shape[0], 2, 2))
            inv_covs = inv_covs.at[:, 0, 0].set(1.0 / (sigmas**2 + epsilon))
            inv_covs = inv_covs.at[:, 1, 1].set(1.0 / (sigmas**2 + epsilon))
            
            return mus, weights, inv_covs
        
        def isotropic_evaluate(X: jnp.ndarray, params: jnp.ndarray):
            """Evaluate isotropic RBF model."""
            mus, weights, inv_covs = isotropic_precompute_params(params)
            
            # Compute all differences at once: (N, K, 2)
            diff = X[:, None, :] - mus[None, :, :]
            
            # Compute quadratic forms efficiently
            quad = jnp.einsum('nki,kij,nkj->nk', diff, inv_covs, diff)
            
            # Compute all kernel values at once
            phi = jnp.exp(-0.5 * quad)
            
            # Weighted sum
            return jnp.dot(phi, weights)
        
        return {
            'initialize': initialize_isotropic_parameters,
            'evaluate': isotropic_evaluate,
            'name': 'Isotropic RBF'
        }
    
    def create_scaled_diagonal_rbf_model():
        """Create a scaled diagonal RBF model for moderate complexity."""
        
        def initialize_scaled_diagonal_parameters(n_kernels: int = 25, key: jax.random.PRNGKey = None):
            """Initialize parameters for scaled diagonal RBF model."""
            if key is None:
                key = jax.random.PRNGKey(42)
            
            # Create a grid of centers
            grid_size = int(jnp.sqrt(n_kernels))
            x_centers = jnp.linspace(-0.8, 0.8, grid_size)
            y_centers = jnp.linspace(-0.8, 0.8, grid_size)
            
            xx, yy = jnp.meshgrid(x_centers, y_centers)
            centers = jnp.stack([xx.flatten(), yy.flatten()], axis=1)
            
            # Initialize parameters: [mu_x, mu_y, log_sigma, scale_x, scale_y, weight]
            params = jnp.zeros((n_kernels, 6))
            
            # Set means (centers)
            params = params.at[:, 0:2].set(centers)
            
            # Set log_sigma (base sigma)
            params = params.at[:, 2].set(jnp.log(0.1) * jnp.ones(n_kernels))
            
            # Set scale factors (close to 1.0 initially)
            params = params.at[:, 3:5].set(jnp.ones((n_kernels, 2)))
            
            # Set weights (random initialization)
            key, subkey = jax.random.split(key)
            params = params.at[:, 5].set(jax.random.normal(subkey, (n_kernels,)) * 0.1)
            
            return params
        
        def scaled_diagonal_precompute_params(params: jnp.ndarray, epsilon: float = 1e-6):
            """Precompute parameters for scaled diagonal RBF model."""
            mus = params[:, 0:2]  # (K, 2)
            log_sigma = params[:, 2]  # (K,)
            scale_factors = params[:, 3:5]  # (K, 2)
            weights = params[:, 5]  # (K,)
            
            # Compute base sigma
            sigma = jnp.exp(log_sigma)  # (K,)
            
            # Create scaled diagonal inverse covariance matrices
            inv_covs = jnp.zeros((params.shape[0], 2, 2))
            inv_covs = inv_covs.at[:, 0, 0].set(scale_factors[:, 0] / (sigma**2 + epsilon))
            inv_covs = inv_covs.at[:, 1, 1].set(scale_factors[:, 1] / (sigma**2 + epsilon))
            
            return mus, weights, inv_covs
        
        def scaled_diagonal_evaluate(X: jnp.ndarray, params: jnp.ndarray):
            """Evaluate scaled diagonal RBF model."""
            mus, weights, inv_covs = scaled_diagonal_precompute_params(params)
            
            # Compute all differences at once: (N, K, 2)
            diff = X[:, None, :] - mus[None, :, :]
            
            # Compute quadratic forms efficiently
            quad = jnp.einsum('nki,kij,nkj->nk', diff, inv_covs, diff)
            
            # Compute all kernel values at once
            phi = jnp.exp(-0.5 * quad)
            
            # Weighted sum
            return jnp.dot(phi, weights)
        
        return {
            'initialize': initialize_scaled_diagonal_parameters,
            'evaluate': scaled_diagonal_evaluate,
            'name': 'Scaled Diagonal RBF'
        }
    
    return {
        'isotropic': create_isotropic_rbf_model(),
        'scaled_diagonal': create_scaled_diagonal_rbf_model()
    }

def compare_optimization_performance():
    """Compare optimization performance of different covariance parameterizations."""
    
    print("\n" + "="*80)
    print("OPTIMIZATION PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create target function
    def create_target_function(x, y):
        return jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    
    # Create training data
    x = jnp.linspace(-1, 1, 30)
    y = jnp.linspace(-1, 1, 30)
    X, Y = jnp.meshgrid(x, y)
    target = create_target_function(X, Y)
    target_flat = target.flatten()
    eval_points = (X, Y)
    
    # Get simplified models
    models = create_optimization_friendly_model()
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTesting {model['name']}...")
        
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        params = model['initialize'](n_kernels=16, key=key)
        
        # Create loss function
        def loss_fn(params):
            prediction = model['evaluate'](jnp.stack([X.flatten(), Y.flatten()], axis=1), params)
            return jnp.mean((prediction - target_flat) ** 2)
        
        # Test gradient computation time
        start_time = time.time()
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(params)
        grad_time = time.time() - start_time
        
        # Test evaluation time
        start_time = time.time()
        loss = loss_fn(params)
        eval_time = time.time() - start_time
        
        # Analyze parameter gradients
        grad_norm = jnp.linalg.norm(grad)
        grad_std = jnp.std(grad)
        
        results[model_name] = {
            'grad_time': grad_time,
            'eval_time': eval_time,
            'initial_loss': loss,
            'grad_norm': grad_norm,
            'grad_std': grad_std,
            'param_count': params.size
        }
        
        print(f"  Parameters: {params.size}")
        print(f"  Initial Loss: {loss:.6f}")
        print(f"  Gradient Time: {grad_time:.4f}s")
        print(f"  Evaluation Time: {eval_time:.4f}s")
        print(f"  Gradient Norm: {grad_norm:.6f}")
        print(f"  Gradient Std: {grad_std:.6f}")
    
    return results

def create_optimization_recommendations():
    """Create recommendations for easier covariance optimization."""
    
    print("\n" + "="*80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    
    recommendations = [
        {
            'title': '1. Use Isotropic Kernels',
            'description': 'Same σ in both directions eliminates rotation complexity',
            'implementation': 'Replace log_sigmas (K,2) with log_sigma (K,)',
            'benefits': ['50% fewer parameters', 'No rotation matrices', 'Simpler gradients'],
            'trade_offs': ['Less flexible', 'Cannot capture directional features']
        },
        {
            'title': '2. Use Scaled Diagonal Kernels',
            'description': 'One base σ with scaling factors for each direction',
            'implementation': 'log_sigma (K,) + scale_factors (K,2)',
            'benefits': ['Moderate flexibility', 'No rotation complexity', 'Controlled anisotropy'],
            'trade_offs': ['Slightly more parameters', 'Still limited compared to full']
        },
        {
            'title': '3. Direct Inverse Covariance Parameterization',
            'description': 'Directly parameterize inverse covariance elements',
            'implementation': 'inv_cov_params (K,3) for [[a,b],[b,c]]',
            'benefits': ['No non-linear transforms', 'Direct optimization', 'Fast gradients'],
            'trade_offs': ['Must ensure positive definiteness', 'More parameters']
        },
        {
            'title': '4. Adaptive Learning Rates',
            'description': 'Use different learning rates for different parameter types',
            'implementation': 'Separate optimizers for weights vs covariance parameters',
            'benefits': ['Faster convergence', 'Better parameter-specific tuning'],
            'trade_offs': ['More complex setup', 'More hyperparameters']
        },
        {
            'title': '5. Parameter Bounds and Regularization',
            'description': 'Add bounds and regularization to prevent numerical issues',
            'implementation': 'Clip σ values, add L2 regularization to inverse covariances',
            'benefits': ['Numerical stability', 'Prevents overfitting'],
            'trade_offs': ['May limit expressiveness', 'More hyperparameters']
        }
    ]
    
    for rec in recommendations:
        print(f"\n{rec['title']}")
        print("-" * len(rec['title']))
        print(f"Description: {rec['description']}")
        print(f"Implementation: {rec['implementation']}")
        print(f"Benefits: {', '.join(rec['benefits'])}")
        print(f"Trade-offs: {', '.join(rec['trade_offs'])}")

def create_implementation_guide():
    """Create a practical implementation guide for easier covariance optimization."""
    
    print("\n" + "="*80)
    print("IMPLEMENTATION GUIDE")
    print("="*80)
    
    print("\nStep 1: Start with Isotropic Kernels")
    print("-" * 40)
    print("Replace the current covariance parameterization with isotropic kernels:")
    print("""
    # Current (complex)
    log_sigmas: (K, 2)  # log_sigma_x, log_sigma_y
    angles: (K,)         # rotation angles
    
    # Isotropic (simple)
    log_sigma: (K,)      # single log_sigma per kernel
    """)
    
    print("\nStep 2: Implement Scaled Diagonal if More Flexibility Needed")
    print("-" * 40)
    print("Add scaling factors for directional sensitivity:")
    print("""
    # Scaled diagonal
    log_sigma: (K,)      # base sigma
    scale_factors: (K, 2) # scaling for x and y directions
    """)
    
    print("\nStep 3: Use Adaptive Learning Rates")
    print("-" * 40)
    print("Separate optimizers for different parameter types:")
    print("""
    # Separate optimizers
    weight_optimizer = optax.adam(0.01)
    cov_optimizer = optax.adam(0.001)  # Slower for covariance
    
    # Or use different learning rates
    optimizer = optax.adam(learning_rate=0.01)
    # Apply different learning rates to different parameter groups
    """)
    
    print("\nStep 4: Add Parameter Bounds")
    print("-" * 40)
    print("Prevent numerical instability:")
    print("""
    # Clip sigma values
    log_sigma = jnp.clip(log_sigma, jnp.log(1e-3), jnp.log(10.0))
    
    # Ensure positive definiteness for direct inverse
    inv_cov_11 = jnp.abs(inv_cov_11) + 1e-6
    inv_cov_22 = jnp.abs(inv_cov_22) + 1e-6
    det = inv_cov_11 * inv_cov_22 - inv_cov_12**2
    # Ensure det > 0
    """)
    
    print("\nStep 5: Monitor Optimization")
    print("-" * 40)
    print("Track convergence and numerical stability:")
    print("""
    # Monitor these metrics
    - Loss convergence rate
    - Gradient norms for different parameter groups
    - Sigma value ranges (watch for very small/large values)
    - Condition numbers of covariance matrices
    """)

def main():
    """Main function to run the covariance optimization analysis."""
    
    print("Covariance Matrix Optimization Analysis")
    print("="*80)
    print("This analysis investigates why the covariance matrix is difficult to optimize")
    print("and proposes solutions for faster convergence.")
    
    # Analyze current parameterization
    analyze_covariance_parameterization()
    
    # Compare parameterization complexity
    compare_parameterization_complexity()
    
    # Compare optimization performance
    results = compare_optimization_performance()
    
    # Create recommendations
    create_optimization_recommendations()
    
    # Create implementation guide
    create_implementation_guide()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("The covariance matrix is difficult to optimize because:")
    print("1. Multiple non-linear transformations (exp, sigmoid, cos, sin)")
    print("2. Complex matrix operations (rotation, multiplication)")
    print("3. Numerical instability (1/σ² can be very large)")
    print("4. Parameter coupling (angles affect both dimensions)")
    print("5. Complex gradient computation through chain rule")
    
    print("\nRecommended solutions:")
    print("1. Use isotropic kernels (same σ in both directions)")
    print("2. Use scaled diagonal kernels (base σ + scaling factors)")
    print("3. Consider direct inverse covariance parameterization")
    print("4. Use adaptive learning rates for different parameter types")
    print("5. Add parameter bounds and regularization")
    
    print("\nThese changes can significantly improve convergence speed")
    print("while maintaining reasonable model flexibility.")

if __name__ == "__main__":
    main()


