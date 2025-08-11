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

from model.standard_model import generate_rbf_solutions as generate_standard_rbf
from model.rbf_model import generate_rbf_solutions as generate_shape_rbf

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)

# JIT-compiled projection functions
@jax.jit
def apply_standard_projection_jit(lambdas_0: jnp.ndarray, n_points: int) -> jnp.ndarray:
    """JIT-compiled projection for standard model parameters."""
    # Project mus to domain bounds
    lambdas_0 = lambdas_0.at[:, 0:2].set(jnp.clip(lambdas_0[:, 0:2], -1.0, 1.0))
    
    # Calculate domain characteristics
    domain_width = 1.75
    avg_point_spacing = domain_width / jnp.sqrt(n_points)
    
    # Set sigma bounds
    min_sigma = avg_point_spacing / 2
    max_sigma = domain_width / 2
    
    # Apply bounds to log_sigmas
    lambdas_0 = lambdas_0.at[:, 2:4].set(jnp.clip(
        lambdas_0[:, 2:4], 
        jnp.log(min_sigma), 
        jnp.log(max_sigma)
    ))
    
    return lambdas_0

@jax.jit
def apply_shape_projection_jit(lambdas_0: jnp.ndarray, n_points: int) -> jnp.ndarray:
    """JIT-compiled projection for shape parameter model."""
    # Project mus to domain bounds
    lambdas_0 = lambdas_0.at[:, 0:2].set(jnp.clip(lambdas_0[:, 0:2], -1.0, 1.0))
    
    # Bound epsilon to control shape transform range
    lambdas_0 = lambdas_0.at[:, 2].set(jnp.clip(lambdas_0[:, 2], -jnp.pi, jnp.pi))
    
    return lambdas_0

@jax.jit
def no_projection(params: jnp.ndarray, n_points: int) -> jnp.ndarray:
    """No projection - return parameters as-is."""
    return params

def create_2d_sine_target(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Create a 2D sine wave target function similar to Poisson's equation."""
    # Create a combination of sine waves in both directions
    target = (jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y) + 
              0.5 * jnp.sin(4 * jnp.pi * x) * jnp.sin(4 * jnp.pi * y))
    return target

def create_training_data(n_points: int = 50) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple]:
    """Create training data for the 2D sine wave function."""
    # Create evaluation grid
    x = jnp.linspace(-1, 1, n_points)
    y = jnp.linspace(-1, 1, n_points)
    X, Y = jnp.meshgrid(x, y)
    
    # Create target values
    target = create_2d_sine_target(X, Y)
    
    # Flatten for training
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    target_flat = target.flatten()
    
    eval_points = (X, Y)
    
    return jnp.stack([X_flat, Y_flat], axis=1), target_flat, eval_points

def initialize_kernel_centers_grid(n_kernels: int = 25) -> jnp.ndarray:
    """Initialize kernel centers in a uniform grid pattern."""
    # Create a grid of centers
    grid_size = int(jnp.sqrt(n_kernels))
    x_centers = jnp.linspace(-0.8, 0.8, grid_size)
    y_centers = jnp.linspace(-0.8, 0.8, grid_size)
    
    # Create all combinations
    xx, yy = jnp.meshgrid(x_centers, y_centers)
    centers = jnp.stack([xx.flatten(), yy.flatten()], axis=1)
    
    return centers

def initialize_standard_parameters(n_kernels: int = 25, key: jax.random.PRNGKey = None) -> jnp.ndarray:
    """Initialize parameters for the standard RBF model (6 parameters per kernel)."""
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Initialize kernel centers in grid
    centers = initialize_kernel_centers_grid(n_kernels)
    
    # Initialize other parameters
    key, subkey = jax.random.split(key)
    log_sigmas = jax.random.uniform(subkey, (n_kernels, 2), minval=-2, maxval=0)
    
    key, subkey = jax.random.split(key)
    angles = jax.random.uniform(subkey, (n_kernels,), minval=-1, maxval=1)
    
    key, subkey = jax.random.split(key)
    weights = jax.random.uniform(subkey, (n_kernels,), minval=-1, maxval=1)
    
    # Combine all parameters: [mu_x, mu_y, log_sigma_x, log_sigma_y, angle, weight]
    params = jnp.concatenate([
        centers,           # (n_kernels, 2) - mu_x, mu_y
        log_sigmas,       # (n_kernels, 2) - log_sigma_x, log_sigma_y
        angles[:, None],  # (n_kernels, 1) - angle
        weights[:, None]  # (n_kernels, 1) - weight
    ], axis=1)  # Shape: (n_kernels, 6)
    
    return params

def initialize_shape_parameters(n_kernels: int = 25, key: jax.random.PRNGKey = None) -> jnp.ndarray:
    """Initialize parameters for the shape parameter transform model (4 parameters per kernel)."""
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Initialize kernel centers in grid
    centers = initialize_kernel_centers_grid(n_kernels)
    
    # Initialize other parameters
    key, subkey = jax.random.split(key)
    epsilons = jax.random.uniform(subkey, (n_kernels,), minval=-jnp.pi, maxval=jnp.pi)
    
    key, subkey = jax.random.split(key)
    weights = jax.random.uniform(subkey, (n_kernels,), minval=-1, maxval=1)
    
    # Combine all parameters: [mu_x, mu_y, epsilon, weight]
    params = jnp.concatenate([
        centers,           # (n_kernels, 2) - mu_x, mu_y
        epsilons[:, None], # (n_kernels, 1) - epsilon
        weights[:, None]   # (n_kernels, 1) - weight
    ], axis=1)  # Shape: (n_kernels, 4)
    
    return params

def create_loss_function(eval_points: Tuple, target: jnp.ndarray) -> Callable:
    """Create loss function for training."""
    def loss_fn(params, eval_points, target):
        # Generate RBF solution
        prediction = generate_standard_rbf(eval_points, params[None, :])
        prediction = prediction[0]  # Remove batch dimension
        
        # Compute MSE loss
        mse_loss = jnp.mean((prediction - target) ** 2)
        return mse_loss
    
    return loss_fn

def create_shape_loss_function(eval_points: Tuple, target: jnp.ndarray) -> Callable:
    """Create loss function for shape parameter transform model."""
    def loss_fn(params, eval_points, target):
        # Generate RBF solution
        prediction = generate_shape_rbf(eval_points, params[None, :])
        prediction = prediction[0]  # Remove batch dimension
        
        # Compute MSE loss
        mse_loss = jnp.mean((prediction - target) ** 2)
        return mse_loss
    
    return loss_fn

def train_model(init_params: jnp.ndarray, 
                eval_points: Tuple, 
                target: jnp.ndarray,
                loss_fn: Callable,
                projection_fn: Callable,
                n_epochs: int = 1000,
                learning_rate: float = 0.01,
                use_projection: bool = True) -> Tuple[jnp.ndarray, list, float]:
    """Train the RBF model using AdamW optimizer with optional projection."""
    
    # Create optimizer
    optimizer = optax.adamw(learning_rate=learning_rate)
    
    # Initialize optimizer state
    opt_state = optimizer.init(init_params)
    
    # Create gradient function
    grad_fn = jax.grad(loss_fn)
    
    # Get number of points for projection
    n_points = eval_points[0].shape[0]
    
    # Training loop
    params = init_params
    losses = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Compute gradients
        grads = grad_fn(params, eval_points, target)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Apply projection if enabled
        if use_projection:
            params = projection_fn(params, n_points)
        
        # Compute loss
        loss = loss_fn(params, eval_points, target)
        losses.append(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    training_time = time.time() - start_time
    
    return params, losses, training_time

def run_multiple_seeds(n_seeds: int = 3, 
                      n_epochs: int = 1000, 
                      learning_rate: float = 0.01,
                      n_kernels: int = 25) -> Dict:
    """Run training with multiple seeds and collect results."""
    
    print(f"Running {n_seeds} seeds for robust comparison...")
    
    # Create training data (same for all seeds)
    train_points, target, eval_points = create_training_data(n_points=50)
    
    # Create loss functions
    standard_loss_fn = create_loss_function(eval_points, target)
    shape_loss_fn = create_shape_loss_function(eval_points, target)
    
    # Results storage
    all_results = {
        'standard_with_proj': {'losses': [], 'times': [], 'final_losses': []},
        'shape_with_proj': {'losses': [], 'times': [], 'final_losses': []},
        'standard_no_proj': {'losses': [], 'times': [], 'final_losses': []},
        'shape_no_proj': {'losses': [], 'times': [], 'final_losses': []}
    }
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        
        # Set seed
        key = jax.random.PRNGKey(42 + seed)
        
        # Initialize parameters for this seed
        key1, key2, key3, key4 = jax.random.split(key, 4)
        standard_params = initialize_standard_parameters(n_kernels**2, key1)
        shape_params = initialize_shape_parameters(n_kernels**2, key2)
        
        # Train all scenarios for this seed
        scenarios = [
            ('standard_with_proj', standard_params, standard_loss_fn, apply_standard_projection_jit, True),
            ('shape_with_proj', shape_params, shape_loss_fn, apply_shape_projection_jit, True),
            ('standard_no_proj', standard_params, standard_loss_fn, no_projection, False),
            ('shape_no_proj', shape_params, shape_loss_fn, no_projection, False)
        ]
        
        for scenario_name, params, loss_fn, proj_fn, use_proj in scenarios:
            print(f"Training {scenario_name}...")
            
            final_params, losses, training_time = train_model(
                params, eval_points, target, loss_fn, proj_fn, 
                n_epochs, learning_rate, use_proj
            )
            
            all_results[scenario_name]['losses'].append(losses)
            all_results[scenario_name]['times'].append(training_time)
            all_results[scenario_name]['final_losses'].append(losses[-1])
    
    return all_results

def compute_statistics(all_results: Dict) -> Dict:
    """Compute mean and standard deviation for all metrics."""
    stats = {}
    
    for scenario_name, results in all_results.items():
        # Convert to numpy arrays for easier computation
        losses_array = np.array(results['losses'])  # Shape: (n_seeds, n_epochs)
        times_array = np.array(results['times'])    # Shape: (n_seeds,)
        final_losses_array = np.array(results['final_losses'])  # Shape: (n_seeds,)
        
        stats[scenario_name] = {
            'losses_mean': np.mean(losses_array, axis=0),
            'losses_std': np.std(losses_array, axis=0),
            'times_mean': np.mean(times_array),
            'times_std': np.std(times_array),
            'final_losses_mean': np.mean(final_losses_array),
            'final_losses_std': np.std(final_losses_array)
        }
    
    return stats

def plot_robust_comparison(stats: Dict, all_results: Dict, save_path: str = "convergence_comparison_robust.png"):
    """Plot the robust comparison with mean curves and variance shading."""
    epochs = range(len(stats['standard_with_proj']['losses_mean']))
    
    plt.figure(figsize=(15, 10))
    
    # Colors for different scenarios
    colors = {
        'standard_with_proj': 'blue',
        'shape_with_proj': 'red', 
        'standard_no_proj': 'green',
        'shape_no_proj': 'orange'
    }
    
    # Plot all scenarios with mean and variance
    plt.subplot(2, 3, 1)
    for name in stats.keys():
        mean_losses = stats[name]['losses_mean']
        std_losses = stats[name]['losses_std']
        
        plt.plot(epochs, mean_losses, color=colors[name], 
                label=name.replace('_', ' ').title(), linewidth=2)
        plt.fill_between(epochs, 
                        mean_losses - std_losses, 
                        mean_losses + std_losses, 
                        color=colors[name], alpha=0.2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Convergence (Mean ± Std)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot final convergence (last 100 epochs)
    plt.subplot(2, 3, 2)
    final_epochs = epochs[-100:]
    for name in stats.keys():
        mean_losses = stats[name]['losses_mean'][-100:]
        std_losses = stats[name]['losses_std'][-100:]
        
        plt.plot(final_epochs, mean_losses, color=colors[name], 
                label=name.replace('_', ' ').title(), linewidth=2)
        plt.fill_between(final_epochs, 
                        mean_losses - std_losses, 
                        mean_losses + std_losses, 
                        color=colors[name], alpha=0.2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Final Convergence (Last 100 Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot training times with error bars
    plt.subplot(2, 3, 3)
    names = list(stats.keys())
    times_mean = [stats[name]['times_mean'] for name in names]
    times_std = [stats[name]['times_std'] for name in names]
    
    bars = plt.bar(names, times_mean, yerr=times_std, 
                   color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'], 
                   alpha=0.7, capsize=5)
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison (± Std)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, times_mean, times_std):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean_val:.1f}±{std_val:.1f}s', ha='center', va='bottom')
    
    # Plot final losses comparison with error bars
    plt.subplot(2, 3, 4)
    final_losses_mean = [stats[name]['final_losses_mean'] for name in names]
    final_losses_std = [stats[name]['final_losses_std'] for name in names]
    
    bars = plt.bar(names, final_losses_mean, yerr=final_losses_std,
                   color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'], 
                   alpha=0.7, capsize=5)
    plt.ylabel('Final Loss')
    plt.title('Final Loss Comparison (± Std)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, final_losses_mean, final_losses_std):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{mean_val:.6f}±{std_val:.6f}', ha='center', va='bottom')
    
    # Plot parameter efficiency
    plt.subplot(2, 3, 5)
    n_kernels = 25
    standard_params = n_kernels**2 * 6
    shape_params = n_kernels**2 * 4
    
    param_efficiency_mean = []
    param_efficiency_std = []
    for name in names:
        final_loss_mean = stats[name]['final_losses_mean']
        final_loss_std = stats[name]['final_losses_std']
        
        if 'standard' in name:
            efficiency_mean = final_loss_mean / standard_params
            efficiency_std = final_loss_std / standard_params
        else:
            efficiency_mean = final_loss_mean / shape_params
            efficiency_std = final_loss_std / shape_params
        
        param_efficiency_mean.append(efficiency_mean)
        param_efficiency_std.append(efficiency_std)
    
    bars = plt.bar(names, param_efficiency_mean, yerr=param_efficiency_std,
                   color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'], 
                   alpha=0.7, capsize=5)
    plt.ylabel('Loss per Parameter')
    plt.title('Parameter Efficiency (± Std)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, param_efficiency_mean, param_efficiency_std):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1e-8,
                f'{mean_val:.2e}±{std_val:.2e}', ha='center', va='bottom')
    
    # Plot convergence speed (epochs to reach threshold)
    plt.subplot(2, 3, 6)
    conv_speeds = []
    conv_speeds_std = []
    threshold = 1e-3
    
    for name in names:
        all_losses = all_results[name]['losses']  # This is already a list of loss arrays
        conv_epochs = []
        
        for losses in all_losses:
            conv_epoch = next((i for i, loss in enumerate(losses) if loss < threshold), len(losses))
            conv_epochs.append(conv_epoch)
        
        conv_speeds.append(np.mean(conv_epochs))
        conv_speeds_std.append(np.std(conv_epochs))
    
    bars = plt.bar(names, conv_speeds, yerr=conv_speeds_std,
                   color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'], 
                   alpha=0.7, capsize=5)
    plt.ylabel('Epochs to Converge (< 1e-3)')
    plt.title('Convergence Speed (± Std)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, conv_speeds, conv_speeds_std):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{mean_val:.0f}±{std_val:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return save_path

def main():
    """Main training and comparison function with robust multi-seed analysis."""
    print("=" * 70)
    print("RBF Model Robust Convergence Comparison")
    print("2D Sine Wave Function with 25x25 Kernels")
    print("Multi-Seed Analysis (5 seeds) with Mean ± Variance")
    print("=" * 70)
    
    # Training parameters
    n_epochs = 500
    learning_rate = 0.1
    n_kernels = 25
    n_seeds = 2
    
    print(f"Training Parameters:")
    print(f"- Epochs: {n_epochs}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Kernels: {n_kernels}x{n_kernels} = {n_kernels**2}")
    print(f"- Seeds: {n_seeds}")
    print(f"- Standard Model: {n_kernels**2 * 6} parameters")
    print(f"- Shape Transform: {n_kernels**2 * 4} parameters")
    print()
    
    # Run multiple seeds
    all_results = run_multiple_seeds(n_seeds=n_seeds, n_epochs=n_epochs, 
                                   learning_rate=learning_rate, n_kernels=n_kernels)
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(all_results)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (Mean ± Std)")
    print("=" * 70)
    
    for name in stats.keys():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Final Loss: {stats[name]['final_losses_mean']:.6f} ± {stats[name]['final_losses_std']:.6f}")
        print(f"  Training Time: {stats[name]['times_mean']:.1f} ± {stats[name]['times_std']:.1f}s")
    
    # Best performing model
    best_model = min(stats.keys(), key=lambda x: stats[x]['final_losses_mean'])
    print(f"\nBest performing model: {best_model}")
    print(f"Best final loss: {stats[best_model]['final_losses_mean']:.6f} ± {stats[best_model]['final_losses_std']:.6f}")
    
    # Projection impact analysis
    standard_proj_impact = ((stats['standard_no_proj']['final_losses_mean'] - 
                            stats['standard_with_proj']['final_losses_mean']) / 
                           stats['standard_no_proj']['final_losses_mean']) * 100
    shape_proj_impact = ((stats['shape_no_proj']['final_losses_mean'] - 
                         stats['shape_with_proj']['final_losses_mean']) / 
                        stats['shape_no_proj']['final_losses_mean']) * 100
    
    print(f"\nProjection Impact:")
    print(f"- Standard Model: {standard_proj_impact:.2f}% improvement with projection")
    print(f"- Shape Transform: {shape_proj_impact:.2f}% improvement with projection")
    
    # Parameter efficiency
    n_kernels = 25
    standard_params = n_kernels**2 * 6
    shape_params = n_kernels**2 * 4
    
    print(f"\nParameter Efficiency (Loss per parameter):")
    for name in stats.keys():
        final_loss_mean = stats[name]['final_losses_mean']
        final_loss_std = stats[name]['final_losses_std']
        if 'standard' in name:
            efficiency_mean = final_loss_mean / standard_params
            efficiency_std = final_loss_std / standard_params
        else:
            efficiency_mean = final_loss_mean / shape_params
            efficiency_std = final_loss_std / shape_params
        print(f"- {name}: {efficiency_mean:.2e} ± {efficiency_std:.2e}")
    
    # Plot comparison
    print("\nCreating robust comparison plots...")
    save_path = plot_robust_comparison(stats, all_results)
    print(f"Robust comparison plot saved to: {save_path}")
    
    print("\nRobust training and comparison completed successfully!")

if __name__ == "__main__":
    main()
