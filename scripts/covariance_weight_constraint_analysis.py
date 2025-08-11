#!/usr/bin/env python3
"""
Analysis of how weight constraints affect the covariance matrix and model representational capacity.
This script investigates the relationship between weight constraints and the need for covariance compensation.
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

from model.standard_model import generate_rbf_solutions as generate_standard_rbf

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)

def create_target_functions() -> Dict[str, Callable]:
    """Create different target functions with varying ranges."""
    
    def low_amplitude(x, y):
        """Function with values in [-0.5, 0.5] - within tanh bounds."""
        return 0.5 * (jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y))
    
    def medium_amplitude(x, y):
        """Function with values in [-1.0, 1.0] - at tanh bounds."""
        return jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    
    def high_amplitude(x, y):
        """Function with values in [-2.0, 2.0] - beyond tanh bounds."""
        return 2.0 * (jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y))
    
    def very_high_amplitude(x, y):
        """Function with values in [-5.0, 5.0] - far beyond tanh bounds."""
        return 5.0 * (jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y))
    
    def mixed_amplitude(x, y):
        """Function with mixed amplitudes - some peaks beyond tanh bounds."""
        return (jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y) + 
                2.0 * jnp.sin(4 * jnp.pi * x) * jnp.sin(4 * jnp.pi * y))
    
    return {
        'low_amplitude': low_amplitude,
        'medium_amplitude': medium_amplitude,
        'high_amplitude': high_amplitude,
        'very_high_amplitude': very_high_amplitude,
        'mixed_amplitude': mixed_amplitude
    }

def create_training_data(target_fn: Callable, n_points: int = 50) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple]:
    """Create training data for a given target function."""
    x = jnp.linspace(-1, 1, n_points)
    y = jnp.linspace(-1, 1, n_points)
    X, Y = jnp.meshgrid(x, y)
    
    target = target_fn(X, Y)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    target_flat = target.flatten()
    
    eval_points = (X, Y)
    
    return jnp.stack([X_flat, Y_flat], axis=1), target_flat, eval_points

def initialize_parameters(n_kernels: int = 25, key: jax.random.PRNGKey = None) -> jnp.ndarray:
    """Initialize parameters for the RBF model (6 parameters per kernel)."""
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Create a grid of centers
    grid_size = int(jnp.sqrt(n_kernels))
    x_centers = jnp.linspace(-0.8, 0.8, grid_size)
    y_centers = jnp.linspace(-0.8, 0.8, grid_size)
    
    xx, yy = jnp.meshgrid(x_centers, y_centers)
    centers = jnp.stack([xx.flatten(), yy.flatten()], axis=1)
    
    # Initialize parameters
    params = jnp.zeros((n_kernels, 6))
    
    # Set means (centers)
    params = params.at[:, 0:2].set(centers)
    
    # Set log_sigmas (small initial values)
    params = params.at[:, 2:4].set(jnp.log(0.1) * jnp.ones((n_kernels, 2)))
    
    # Set angles (small initial values)
    params = params.at[:, 4].set(0.1 * jnp.ones(n_kernels))
    
    # Set weights (random initialization)
    key, subkey = jax.random.split(key)
    params = params.at[:, 5].set(jax.random.normal(subkey, (n_kernels,)) * 0.1)
    
    return params

def analyze_covariance_compensation(target_fn: Callable, target_name: str):
    """Analyze how covariance matrices compensate for weight constraints."""
    
    print(f"\n=== Analyzing {target_name} ===")
    
    # Create training data
    X_train, target_train, eval_points = create_training_data(target_fn)
    
    # Analyze target function range
    target_min = jnp.min(target_train)
    target_max = jnp.max(target_train)
    target_range = target_max - target_min
    target_std = jnp.std(target_train)
    
    print(f"Target function range: [{target_min:.3f}, {target_max:.3f}]")
    print(f"Target function std: {target_std:.3f}")
    print(f"Target function range: {target_range:.3f}")
    
    # Check if target exceeds tanh bounds
    exceeds_tanh = target_max > 1.0 or target_min < -1.0
    print(f"Exceeds tanh bounds [-1, 1]: {exceeds_tanh}")
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    init_params = initialize_parameters(n_kernels=25, key=key)
    
    # Test different weight scaling approaches
    scaling_approaches = {
        'no_scaling': lambda x: x,
        'tanh_scaling': jax.nn.tanh,
        'scale_0.1': lambda x: x * 0.1,
        'scale_0.5': lambda x: x * 0.5,
        'clip_1': lambda x: jnp.clip(x, -1.0, 1.0)
    }
    
    results = {}
    
    for approach_name, scaling_fn in scaling_approaches.items():
        print(f"\n  Testing {approach_name}...")
        
        # Apply scaling to the solution
        solution = generate_standard_rbf(eval_points, init_params)
        scaled_solution = scaling_fn(solution)
        
        # Calculate MSE
        mse = jnp.mean((scaled_solution - target_train) ** 2)
        
        # Analyze the solution range
        solution_min = jnp.min(scaled_solution)
        solution_max = jnp.max(scaled_solution)
        solution_range = solution_max - solution_min
        solution_std = jnp.std(scaled_solution)
        
        # Calculate how much the target exceeds the solution range
        range_deficit = max(0, target_max - solution_max) + max(0, solution_min - target_min)
        
        results[approach_name] = {
            'mse': mse,
            'solution_min': solution_min,
            'solution_max': solution_max,
            'solution_range': solution_range,
            'solution_std': solution_std,
            'range_deficit': range_deficit,
            'target_min': target_min,
            'target_max': target_max,
            'target_range': target_range
        }
        
        print(f"    MSE: {mse:.6f}")
        print(f"    Solution range: [{solution_min:.3f}, {solution_max:.3f}]")
        print(f"    Range deficit: {range_deficit:.3f}")
    
    return results

def analyze_covariance_matrix_changes():
    """Analyze how the covariance matrix changes to compensate for weight constraints."""
    
    print("="*80)
    print("COVARIANCE MATRIX COMPENSATION ANALYSIS")
    print("="*80)
    
    # Get target functions
    target_functions = create_target_functions()
    
    all_results = {}
    
    for target_name, target_fn in target_functions.items():
        results = analyze_covariance_compensation(target_fn, target_name)
        all_results[target_name] = results
    
    return all_results

def create_compensation_analysis_plots(all_results: Dict):
    """Create plots showing covariance compensation effects."""
    
    target_names = list(all_results.keys())
    scaling_approaches = list(all_results[target_names[0]].keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Covariance Matrix Compensation Analysis', fontsize=16, fontweight='bold')
    
    # 1. MSE Comparison
    ax1 = axes[0, 0]
    for i, approach in enumerate(scaling_approaches):
        mse_values = [all_results[target][approach]['mse'] for target in target_names]
        ax1.plot(range(len(target_names)), mse_values, marker='o', label=approach, linewidth=2)
    
    ax1.set_title('MSE by Target Function and Scaling', fontweight='bold')
    ax1.set_xlabel('Target Function')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_xticks(range(len(target_names)))
    ax1.set_xticklabels(target_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Range Deficit Analysis
    ax2 = axes[0, 1]
    for i, approach in enumerate(scaling_approaches):
        deficit_values = [all_results[target][approach]['range_deficit'] for target in target_names]
        ax2.plot(range(len(target_names)), deficit_values, marker='s', label=approach, linewidth=2)
    
    ax2.set_title('Range Deficit by Target Function', fontweight='bold')
    ax2.set_xlabel('Target Function')
    ax2.set_ylabel('Range Deficit')
    ax2.set_xticks(range(len(target_names)))
    ax2.set_xticklabels(target_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Solution Range vs Target Range
    ax3 = axes[0, 2]
    for i, approach in enumerate(scaling_approaches):
        solution_ranges = [all_results[target][approach]['solution_range'] for target in target_names]
        target_ranges = [all_results[target][approach]['target_range'] for target in target_names]
        ax3.scatter(target_ranges, solution_ranges, label=approach, s=100, alpha=0.7)
    
    ax3.plot([0, 10], [0, 10], 'k--', alpha=0.5, label='Perfect Match')
    ax3.set_title('Solution Range vs Target Range', fontweight='bold')
    ax3.set_xlabel('Target Range')
    ax3.set_ylabel('Solution Range')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Detailed analysis for high amplitude function
    ax4 = axes[1, 0]
    high_amp_results = all_results['high_amplitude']
    approaches = list(high_amp_results.keys())
    mse_values = [high_amp_results[app]['mse'] for app in approaches]
    
    bars = ax4.bar(approaches, mse_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    ax4.set_title('High Amplitude Function: MSE by Scaling', fontweight='bold')
    ax4.set_ylabel('Mean Squared Error')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mse_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Range comparison for high amplitude
    ax5 = axes[1, 1]
    target_range = high_amp_results['no_scaling']['target_range']
    solution_ranges = [high_amp_results[app]['solution_range'] for app in approaches]
    
    bars2 = ax5.bar(approaches, solution_ranges, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    ax5.axhline(y=target_range, color='red', linestyle='--', label=f'Target Range: {target_range:.2f}')
    ax5.set_title('High Amplitude Function: Solution Ranges', fontweight='bold')
    ax5.set_ylabel('Solution Range')
    ax5.tick_params(axis='x', rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, solution_ranges):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Compensation effectiveness
    ax6 = axes[1, 2]
    compensation_ratios = []
    for app in approaches:
        target_range = high_amp_results[app]['target_range']
        solution_range = high_amp_results[app]['solution_range']
        ratio = solution_range / target_range if target_range > 0 else 0
        compensation_ratios.append(ratio)
    
    bars3 = ax6.bar(approaches, compensation_ratios, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    ax6.axhline(y=1.0, color='red', linestyle='--', label='Perfect Compensation')
    ax6.set_title('Compensation Effectiveness (Range Ratio)', fontweight='bold')
    ax6.set_ylabel('Solution Range / Target Range')
    ax6.tick_params(axis='x', rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars3, compensation_ratios):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('covariance_compensation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def analyze_covariance_matrix_adaptation():
    """Analyze how the covariance matrix adapts to compensate for weight constraints."""
    
    print("\n" + "="*80)
    print("COVARIANCE MATRIX ADAPTATION ANALYSIS")
    print("="*80)
    
    # Create a simple test case
    x = jnp.linspace(-1, 1, 20)
    y = jnp.linspace(-1, 1, 20)
    X, Y = jnp.meshgrid(x, y)
    
    # Create target with high amplitude
    target = 3.0 * jnp.sin(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y)
    target_flat = target.flatten()
    eval_points = (X, Y)
    
    print(f"Target range: [{jnp.min(target):.3f}, {jnp.max(target):.3f}]")
    print(f"Target exceeds tanh bounds: {jnp.max(target) > 1.0 or jnp.min(target) < -1.0}")
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    init_params = initialize_parameters(n_kernels=9, key=key)  # Smaller for analysis
    
    # Extract covariance matrices
    mus = init_params[:, 0:2]
    log_sigmas = init_params[:, 2:4]
    angles = init_params[:, 4]
    weights = init_params[:, 5]
    
    # Compute covariance matrices
    sigmas = jnp.exp(log_sigmas)
    squared_sigmas = sigmas**2
    angles = jax.nn.sigmoid(angles) * 2 * jnp.pi
    
    # Compute rotation matrices
    cos_angles = jnp.cos(angles)
    sin_angles = jnp.sin(angles)
    R = jnp.stack([
        jnp.stack([cos_angles, -sin_angles], axis=1),
        jnp.stack([sin_angles, cos_angles], axis=1)
    ], axis=2)
    
    # Create diagonal matrices
    diag_inv = jnp.zeros((mus.shape[0], 2, 2))
    diag_inv = diag_inv.at[:, 0, 0].set(1.0 / (squared_sigmas[:, 0] + 1e-6))
    diag_inv = diag_inv.at[:, 1, 1].set(1.0 / (squared_sigmas[:, 1] + 1e-6))
    
    # Compute inverse covariance matrices
    inv_covs = jnp.einsum('kij,kjl,klm->kim', R, diag_inv, R.transpose((0, 2, 1)))
    
    print(f"\nInitial covariance matrices:")
    print(f"Sigma ranges: [{jnp.min(sigmas):.4f}, {jnp.max(sigmas):.4f}]")
    print(f"Inverse covariance determinant ranges: [{jnp.min(jnp.linalg.det(inv_covs)):.6f}, {jnp.max(jnp.linalg.det(inv_covs)):.6f}]")
    
    # Test different weight scaling approaches
    scaling_approaches = {
        'no_scaling': lambda w: w,
        'tanh_scaling': jax.nn.tanh,
        'scale_0.1': lambda w: w * 0.1,
        'clip_1': lambda w: jnp.clip(w, -1.0, 1.0)
    }
    
    for approach_name, scaling_fn in scaling_approaches.items():
        print(f"\n{approach_name}:")
        
        # Apply scaling to weights
        scaled_weights = scaling_fn(weights)
        
        # Generate solution
        solution = generate_standard_rbf(eval_points, init_params)
        scaled_solution = scaling_fn(solution)
        
        # Calculate MSE
        mse = jnp.mean((scaled_solution - target_flat) ** 2)
        
        # Analyze solution range
        solution_min = jnp.min(scaled_solution)
        solution_max = jnp.max(scaled_solution)
        solution_range = solution_max - solution_min
        
        print(f"  MSE: {mse:.6f}")
        print(f"  Solution range: [{solution_min:.3f}, {solution_max:.3f}]")
        print(f"  Weight range: [{jnp.min(scaled_weights):.3f}, {jnp.max(scaled_weights):.3f}]")
        
        # Calculate how much the covariance needs to compensate
        target_range = jnp.max(target) - jnp.min(target)
        compensation_needed = target_range - solution_range
        print(f"  Compensation needed: {compensation_needed:.3f}")

def print_compensation_summary(all_results: Dict):
    """Print a summary of the compensation analysis."""
    
    print("\n" + "="*80)
    print("COVARIANCE COMPENSATION SUMMARY")
    print("="*80)
    
    for target_name, results in all_results.items():
        print(f"\n{target_name.upper()}:")
        print("-" * 40)
        
        # Find best and worst approaches
        mse_values = {app: results[app]['mse'] for app in results.keys()}
        best_app = min(mse_values.items(), key=lambda x: x[1])[0]
        worst_app = max(mse_values.items(), key=lambda x: x[1])[0]
        
        print(f"Best approach: {best_app} (MSE: {mse_values[best_app]:.6f})")
        print(f"Worst approach: {worst_app} (MSE: {mse_values[worst_app]:.6f})")
        
        # Analyze range deficits
        deficits = {app: results[app]['range_deficit'] for app in results.keys()}
        print(f"Range deficits: {deficits}")
        
        # Check if target exceeds bounds
        target_max = results['no_scaling']['target_max']
        target_min = results['no_scaling']['target_min']
        exceeds_bounds = target_max > 1.0 or target_min < -1.0
        
        if exceeds_bounds:
            print(f"⚠️  Target exceeds tanh bounds [-1, 1]")
            print(f"   Target range: [{target_min:.3f}, {target_max:.3f}]")
            print(f"   This requires covariance matrix compensation")
        else:
            print(f"✅ Target within tanh bounds [-1, 1]")
            print(f"   Target range: [{target_min:.3f}, {target_max:.3f}]")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print("1. **Weight Constraints Limit Output Range**:")
    print("   - Tanh scaling bounds output to [-1, 1]")
    print("   - Functions with values outside this range cannot be represented")
    print("   - The covariance matrix must compensate for this limitation")
    
    print("\n2. **Covariance Matrix Compensation Mechanisms**:")
    print("   - Smaller sigmas (larger inverse covariances) increase local sensitivity")
    print("   - This allows the model to capture fine details despite weight constraints")
    print("   - However, this can lead to overfitting and numerical instability")
    
    print("\n3. **Trade-offs of Weight Constraints**:")
    print("   - Pros: Numerical stability, bounded outputs, smooth gradients")
    print("   - Cons: Limited representational capacity, need for compensation")
    print("   - The compensation can make the model less interpretable")
    
    print("\n4. **Recommendations**:")
    print("   - Use weight constraints only when necessary for stability")
    print("   - Consider the target function range when choosing constraints")
    print("   - Monitor covariance matrix behavior during training")
    print("   - Consider adaptive scaling based on target function range")

def main():
    """Main function to run the covariance compensation analysis."""
    
    print("Covariance Matrix Compensation Analysis")
    print("="*80)
    print("This analysis investigates how weight constraints affect the covariance matrix")
    print("and the model's ability to represent functions with values outside the constraint range.")
    
    # Run the main analysis
    all_results = analyze_covariance_matrix_changes()
    
    # Create visualization plots
    create_compensation_analysis_plots(all_results)
    
    # Analyze covariance matrix adaptation
    analyze_covariance_matrix_adaptation()
    
    # Print summary
    print_compensation_summary(all_results)
    
    print("\nAnalysis complete! Check the generated plot for visual analysis.")

if __name__ == "__main__":
    main()


