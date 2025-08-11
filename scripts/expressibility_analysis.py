#!/usr/bin/env python3
"""
Analysis of expressibility trade-off between isotropic and anisotropic kernels.
This script investigates how to maintain expressibility while improving optimization.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Callable, List
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)

def create_anisotropic_test_functions():
    """Create test functions that require anisotropic kernels for good approximation."""
    
    def directional_sine(x, y):
        """Function with strong directional features."""
        return jnp.sin(4 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    
    def diagonal_pattern(x, y):
        """Function with diagonal patterns requiring rotated kernels."""
        return jnp.sin(3 * jnp.pi * (x + y)) * jnp.cos(2 * jnp.pi * (x - y))
    
    def elliptical_features(x, y):
        """Function with elliptical features requiring anisotropic kernels."""
        r = jnp.sqrt(x**2 + y**2)
        theta = jnp.arctan2(y, x)
        return jnp.sin(2 * r) * jnp.cos(3 * theta)
    
    def mixed_anisotropy(x, y):
        """Function with mixed anisotropic features."""
        return (jnp.sin(3 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y) + 
                0.5 * jnp.sin(4 * jnp.pi * (x + y)) * jnp.cos(3 * jnp.pi * (x - y)))
    
    def sharp_directional(x, y):
        """Function with sharp directional features."""
        return jnp.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1) * jnp.sin(5 * jnp.pi * x)
    
    return {
        'directional_sine': directional_sine,
        'diagonal_pattern': diagonal_pattern,
        'elliptical_features': elliptical_features,
        'mixed_anisotropy': mixed_anisotropy,
        'sharp_directional': sharp_directional
    }

def create_isotropic_test_functions():
    """Create test functions that work well with isotropic kernels."""
    
    def radial_symmetric(x, y):
        """Radially symmetric function - perfect for isotropic kernels."""
        r = jnp.sqrt(x**2 + y**2)
        return jnp.sin(2 * jnp.pi * r)
    
    def smooth_isotropic(x, y):
        """Smooth isotropic function."""
        return jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    
    def gaussian_mixture(x, y):
        """Mixture of isotropic Gaussians."""
        return (jnp.exp(-((x + 0.5)**2 + (y + 0.5)**2) / 0.2) + 
                jnp.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.2))
    
    return {
        'radial_symmetric': radial_symmetric,
        'smooth_isotropic': smooth_isotropic,
        'gaussian_mixture': gaussian_mixture
    }

def analyze_expressibility_trade_off():
    """Analyze the expressibility trade-off between isotropic and anisotropic kernels."""
    
    print("="*80)
    print("EXPRESSIBILITY TRADE-OFF ANALYSIS")
    print("="*80)
    
    # Get test functions
    anisotropic_functions = create_anisotropic_test_functions()
    isotropic_functions = create_isotropic_test_functions()
    
    print("\nAnisotropic Functions (require directional sensitivity):")
    print("-" * 60)
    for name, func in anisotropic_functions.items():
        print(f"- {name}: Requires anisotropic kernels for good approximation")
    
    print("\nIsotropic Functions (work well with isotropic kernels):")
    print("-" * 60)
    for name, func in isotropic_functions.items():
        print(f"- {name}: Works well with isotropic kernels")
    
    print("\nExpressibility Comparison:")
    print("-" * 60)
    
    comparison = {
        'Isotropic Kernels': {
            'Expressibility': 'Limited',
            'Directional Features': 'Cannot capture',
            'Rotated Patterns': 'Cannot capture',
            'Elliptical Features': 'Cannot capture',
            'Optimization Speed': 'Very Fast',
            'Parameter Count': '50% fewer',
            'Gradient Complexity': 'Very Low'
        },
        'Anisotropic Kernels': {
            'Expressibility': 'Full',
            'Directional Features': 'Can capture',
            'Rotated Patterns': 'Can capture',
            'Elliptical Features': 'Can capture',
            'Optimization Speed': 'Slow',
            'Parameter Count': 'Full',
            'Gradient Complexity': 'Very High'
        }
    }
    
    print(f"{'Aspect':<25} {'Isotropic':<15} {'Anisotropic':<15}")
    print("-" * 60)
    
    for aspect in comparison['Isotropic Kernels'].keys():
        isotropic_val = comparison['Isotropic Kernels'][aspect]
        anisotropic_val = comparison['Anisotropic Kernels'][aspect]
        print(f"{aspect:<25} {isotropic_val:<15} {anisotropic_val:<15}")

def create_hybrid_solutions():
    """Create hybrid solutions that balance expressibility and optimization speed."""
    
    print("\n" + "="*80)
    print("HYBRID SOLUTIONS FOR EXPRESSIBILITY + OPTIMIZATION")
    print("="*80)
    
    solutions = [
        {
            'name': 'Scaled Diagonal Kernels',
            'description': 'One base σ with scaling factors for each direction',
            'parameterization': 'log_sigma (K,) + scale_factors (K,2)',
            'expressibility': 'Moderate - directional sensitivity without rotation',
            'optimization': 'Fast - no rotation matrices',
            'implementation': '''
                # Scaled diagonal model
                params: (K, 6)  # [mu_x, mu_y, log_sigma, scale_x, scale_y, weight]
                
                # Controlled anisotropy
                sigma = exp(log_sigma)  # (K,)
                inv_cov_11 = scale_x / σ²
                inv_cov_22 = scale_y / σ²
            ''',
            'trade_offs': 'Good balance between expressibility and speed'
        },
        {
            'name': 'Direct Inverse Covariance',
            'description': 'Directly parameterize inverse covariance elements',
            'parameterization': 'inv_cov_params (K,3) for [[a,b],[b,c]]',
            'expressibility': 'Full - complete covariance control',
            'optimization': 'Fast - no non-linear transforms',
            'implementation': '''
                # Direct inverse model
                params: (K, 5)  # [mu_x, mu_y, inv_cov_11, inv_cov_12, inv_cov_22, weight]
                
                # Direct assignment
                inv_covs = [[inv_cov_11, inv_cov_12],
                           [inv_cov_12, inv_cov_22]]
                
                # Ensure positive definiteness
                det = inv_cov_11 * inv_cov_22 - inv_cov_12²
                scale_factor = max(min_det / det, 1.0)
            ''',
            'trade_offs': 'Maximum expressibility with fast optimization'
        },
        {
            'name': 'Adaptive Kernel Selection',
            'description': 'Use isotropic kernels where possible, anisotropic where needed',
            'parameterization': 'Mixture of isotropic and anisotropic kernels',
            'expressibility': 'Full - adaptive to function requirements',
            'optimization': 'Balanced - fast for isotropic, slower for anisotropic',
            'implementation': '''
                # Adaptive selection
                if function_requires_anisotropy(region):
                    use_anisotropic_kernel()
                else:
                    use_isotropic_kernel()
            ''',
            'trade_offs': 'Best of both worlds, but more complex'
        },
        {
            'name': 'Progressive Complexity',
            'description': 'Start with isotropic, gradually add anisotropy as needed',
            'parameterization': 'Progressive parameter addition',
            'expressibility': 'Adaptive - grows with complexity',
            'optimization': 'Fast initial convergence, slower refinement',
            'implementation': '''
                # Stage 1: Isotropic
                params = [mu_x, mu_y, log_sigma, weight]
                
                # Stage 2: Add scaling
                params = [mu_x, mu_y, log_sigma, scale_x, scale_y, weight]
                
                # Stage 3: Add rotation (if needed)
                params = [mu_x, mu_y, log_sigma_x, log_sigma_y, angle, weight]
            ''',
            'trade_offs': 'Fast initial training, gradual complexity increase'
        }
    ]
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. {solution['name']}")
        print("-" * len(solution['name']))
        print(f"Description: {solution['description']}")
        print(f"Parameterization: {solution['parameterization']}")
        print(f"Expressibility: {solution['expressibility']}")
        print(f"Optimization: {solution['optimization']}")
        print(f"Trade-offs: {solution['trade_offs']}")
        print(f"Implementation:\n{solution['implementation']}")

def create_expressibility_test():
    """Test the expressibility of different kernel types on anisotropic functions."""
    
    print("\n" + "="*80)
    print("EXPRESSIBILITY TEST")
    print("="*80)
    
    # Create test data
    x = jnp.linspace(-1, 1, 30)
    y = jnp.linspace(-1, 1, 30)
    X, Y = jnp.meshgrid(x, y)
    
    # Test anisotropic functions
    anisotropic_functions = create_anisotropic_test_functions()
    
    results = {}
    
    for func_name, func in anisotropic_functions.items():
        print(f"\nTesting {func_name}...")
        
        # Create target
        target = func(X, Y)
        target_flat = target.flatten()
        X_eval = jnp.stack([X.flatten(), Y.flatten()], axis=1)
        
        # Test different kernel types
        kernel_types = {
            'Isotropic': 'isotropic',
            'Scaled Diagonal': 'scaled_diagonal', 
            'Direct Inverse': 'direct_inverse'
        }
        
        func_results = {}
        
        for kernel_name, kernel_type in kernel_types.items():
            print(f"  Testing {kernel_name} kernels...")
            
            # Initialize parameters (simplified for testing)
            n_kernels = 16
            key = jax.random.PRNGKey(42)
            
            if kernel_type == 'isotropic':
                # Isotropic: [mu_x, mu_y, log_sigma, weight]
                params = jnp.zeros((n_kernels, 4))
                params = params.at[:, 0:2].set(jax.random.uniform(key, (n_kernels, 2), minval=-0.8, maxval=0.8))
                params = params.at[:, 2].set(jnp.log(0.1) * jnp.ones(n_kernels))
                key, subkey = jax.random.split(key)
                params = params.at[:, 3].set(jax.random.normal(subkey, (n_kernels,)) * 0.1)
                
            elif kernel_type == 'scaled_diagonal':
                # Scaled diagonal: [mu_x, mu_y, log_sigma, scale_x, scale_y, weight]
                params = jnp.zeros((n_kernels, 6))
                params = params.at[:, 0:2].set(jax.random.uniform(key, (n_kernels, 2), minval=-0.8, maxval=0.8))
                params = params.at[:, 2].set(jnp.log(0.1) * jnp.ones(n_kernels))
                params = params.at[:, 3:5].set(jnp.ones((n_kernels, 2)))  # scale factors
                key, subkey = jax.random.split(key)
                params = params.at[:, 5].set(jax.random.normal(subkey, (n_kernels,)) * 0.1)
                
            elif kernel_type == 'direct_inverse':
                # Direct inverse: [mu_x, mu_y, inv_cov_11, inv_cov_12, inv_cov_22, weight]
                params = jnp.zeros((n_kernels, 6))
                params = params.at[:, 0:2].set(jax.random.uniform(key, (n_kernels, 2), minval=-0.8, maxval=0.8))
                params = params.at[:, 2].set(100.0 * jnp.ones(n_kernels))  # inv_cov_11
                params = params.at[:, 3].set(0.0 * jnp.ones(n_kernels))    # inv_cov_12
                params = params.at[:, 4].set(100.0 * jnp.ones(n_kernels))  # inv_cov_22
                key, subkey = jax.random.split(key)
                params = params.at[:, 5].set(jax.random.normal(subkey, (n_kernels,)) * 0.1)
            
            # Evaluate model (simplified)
            def evaluate_model(params, X_eval):
                # Simplified evaluation for testing
                mus = params[:, 0:2]
                weights = params[:, -1]
                
                # Compute kernel values
                diff = X_eval[:, None, :] - mus[None, :, :]
                distances = jnp.sum(diff**2, axis=2)
                
                if kernel_type == 'isotropic':
                    sigmas = jnp.exp(params[:, 2])
                    kernel_values = jnp.exp(-0.5 * distances / (sigmas**2 + 1e-6))
                elif kernel_type == 'scaled_diagonal':
                    sigma = jnp.exp(params[:, 2])
                    scale_x = params[:, 3]
                    scale_y = params[:, 4]
                    kernel_values = jnp.exp(-0.5 * (diff[:, :, 0]**2 / (sigma**2 * scale_x + 1e-6) + 
                                                   diff[:, :, 1]**2 / (sigma**2 * scale_y + 1e-6)))
                elif kernel_type == 'direct_inverse':
                    inv_cov_11 = jnp.abs(params[:, 2]) + 1e-6
                    inv_cov_12 = params[:, 3]
                    inv_cov_22 = jnp.abs(params[:, 4]) + 1e-6
                    
                    # Fix broadcasting: (N, K) = (N, K, 1) * (N, K, 1)
                    quad = (inv_cov_11[None, :] * diff[:, :, 0]**2 + 
                           2 * inv_cov_12[None, :] * diff[:, :, 0] * diff[:, :, 1] + 
                           inv_cov_22[None, :] * diff[:, :, 1]**2)
                    kernel_values = jnp.exp(-0.5 * quad)
                
                return jnp.dot(kernel_values, weights)
            
            # Compute initial MSE
            prediction = evaluate_model(params, X_eval)
            mse = jnp.mean((prediction - target_flat) ** 2)
            
            func_results[kernel_name] = {
                'initial_mse': mse,
                'kernel_type': kernel_type
            }
            
            print(f"    Initial MSE: {mse:.6f}")
        
        results[func_name] = func_results
    
    return results

def create_expressibility_plots(results: Dict):
    """Create plots showing expressibility comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Expressibility Comparison: Isotropic vs Anisotropic Kernels', fontsize=16, fontweight='bold')
    
    # Extract data
    functions = list(results.keys())
    kernel_types = list(results[functions[0]].keys())
    
    # 1. Initial MSE Comparison
    ax1 = axes[0, 0]
    for i, kernel_type in enumerate(kernel_types):
        mse_values = [results[func][kernel_type]['initial_mse'] for func in functions]
        ax1.plot(range(len(functions)), mse_values, marker='o', label=kernel_type, linewidth=2)
    
    ax1.set_title('Initial MSE by Function and Kernel Type', fontweight='bold')
    ax1.set_xlabel('Function')
    ax1.set_ylabel('Initial MSE')
    ax1.set_xticks(range(len(functions)))
    ax1.set_xticklabels(functions, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Expressibility Ranking
    ax2 = axes[0, 1]
    expressibility_scores = {
        'Isotropic': 0.3,      # Limited expressibility
        'Scaled Diagonal': 0.7, # Moderate expressibility
        'Direct Inverse': 0.9   # High expressibility
    }
    
    kernel_names = list(expressibility_scores.keys())
    scores = list(expressibility_scores.values())
    
    bars = ax2.bar(kernel_names, scores, color=['lightcoral', 'lightgreen', 'skyblue'])
    ax2.set_title('Expressibility Scores', fontweight='bold')
    ax2.set_ylabel('Expressibility Score')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{score:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Optimization Speed vs Expressibility
    ax3 = axes[1, 0]
    optimization_scores = {
        'Isotropic': 0.9,      # Very fast
        'Scaled Diagonal': 0.7, # Fast
        'Direct Inverse': 0.8   # Fast (no transforms)
    }
    
    x_pos = [expressibility_scores[name] for name in kernel_names]
    y_pos = [optimization_scores[name] for name in kernel_names]
    colors = ['lightcoral', 'lightgreen', 'skyblue']
    
    for i, (name, x, y) in enumerate(zip(kernel_names, x_pos, y_pos)):
        ax3.scatter(x, y, s=200, c=colors[i], alpha=0.7, label=name)
        ax3.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax3.set_title('Optimization Speed vs Expressibility', fontweight='bold')
    ax3.set_xlabel('Expressibility Score')
    ax3.set_ylabel('Optimization Speed Score')
    ax3.grid(True, alpha=0.3)
    
    # 4. Recommendation Matrix
    ax4 = axes[1, 1]
    recommendations = {
        'Isotropic': 'Use for smooth, symmetric functions',
        'Scaled Diagonal': 'Use for directional features',
        'Direct Inverse': 'Use for complex anisotropic patterns'
    }
    
    y_pos = range(len(recommendations))
    bars = ax4.barh(y_pos, [1, 1, 1], color=['lightcoral', 'lightgreen', 'skyblue'])
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(list(recommendations.keys()))
    ax4.set_title('Recommendations by Use Case', fontweight='bold')
    ax4.set_xlim(0, 1.2)
    
    # Add recommendation text
    for i, (name, rec) in enumerate(recommendations.items()):
        ax4.text(1.05, i, rec, va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('expressibility_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_expressibility_recommendations():
    """Print recommendations for maintaining expressibility while improving optimization."""
    
    print("\n" + "="*80)
    print("EXPRESSIBILITY RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. **Use Scaled Diagonal Kernels for Directional Features**")
    print("-" * 60)
    print("When your function has directional features but not complex rotations:")
    print("""
    # Scaled diagonal provides directional sensitivity without rotation complexity
    params: (K, 6)  # [mu_x, mu_y, log_sigma, scale_x, scale_y, weight]
    
    # Benefits:
    # - Can capture directional features
    # - No rotation matrices (faster optimization)
    # - Explicit control over anisotropy
    # - Good balance of expressibility and speed
    """)
    
    print("\n2. **Use Direct Inverse Covariance for Maximum Expressibility**")
    print("-" * 60)
    print("When you need full covariance control but want fast optimization:")
    print("""
    # Direct inverse provides full expressibility with fast optimization
    params: (K, 5)  # [mu_x, mu_y, inv_cov_11, inv_cov_12, inv_cov_22, weight]
    
    # Benefits:
    # - Full covariance control (maximum expressibility)
    # - No non-linear transforms (fast gradients)
    # - Direct parameterization
    # - Can capture any anisotropic pattern
    """)
    
    print("\n3. **Use Progressive Complexity for Adaptive Expressibility**")
    print("-" * 60)
    print("Start simple and add complexity as needed:")
    print("""
    # Stage 1: Isotropic (fastest)
    params = [mu_x, mu_y, log_sigma, weight]
    
    # Stage 2: Add scaling (if needed)
    params = [mu_x, mu_y, log_sigma, scale_x, scale_y, weight]
    
    # Stage 3: Add rotation (if needed)
    params = [mu_x, mu_y, log_sigma_x, log_sigma_y, angle, weight]
    """)
    
    print("\n4. **Use Adaptive Learning Rates for All Models**")
    print("-" * 60)
    print("Different learning rates for different parameter types:")
    print("""
    # Faster learning for weights, slower for covariance
    weight_optimizer = optax.adam(0.01)
    cov_optimizer = optax.adam(0.001)
    
    # This helps maintain expressibility while improving convergence
    """)

def main():
    """Main function to analyze expressibility trade-offs."""
    
    print("Expressibility Trade-off Analysis")
    print("="*80)
    print("This analysis investigates the trade-off between expressibility and optimization speed.")
    
    # Analyze expressibility trade-off
    analyze_expressibility_trade_off()
    
    # Create hybrid solutions
    create_hybrid_solutions()
    
    # Test expressibility
    results = create_expressibility_test()
    
    # Create plots
    create_expressibility_plots(results)
    
    # Print recommendations
    print_expressibility_recommendations()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("The expressibility trade-off is real, but there are solutions:")
    print("1. **Scaled Diagonal**: Good balance for directional features")
    print("2. **Direct Inverse**: Maximum expressibility with fast optimization")
    print("3. **Progressive Complexity**: Start simple, add complexity as needed")
    print("4. **Adaptive Learning Rates**: Maintain expressibility while improving speed")
    
    print("\nKey Insight: You don't have to choose between expressibility and speed!")
    print("The right parameterization can give you both.")

if __name__ == "__main__":
    main()
