import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Callable
import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model.standard_model import generate_rbf_solutions as generate_standard_rbf
from model.rbf_model import generate_rbf_solutions as generate_shape_rbf

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)

# Current implementation
def current_transform(epsilon):
    """Current implementation: simple inverse relationship."""
    epsilon_tensor = jnp.sin(jnp.array(epsilon)) * 3
    log_sx = epsilon_tensor
    log_sy = -epsilon_tensor
    theta = 0 * jnp.sin(epsilon_tensor) * jnp.pi
    return log_sx, log_sy, theta

# Alternative 1: Circular sweep in log-space
def circular_sweep_transform(epsilon):
    """Circular sweep in log-space for symmetric coverage."""
    r = 1.0  # radius controls log-scale amplitude
    log_sx = r * jnp.sin(epsilon)
    log_sy = r * jnp.cos(epsilon)
    theta = (epsilon % (2 * jnp.pi))  # full smooth rotation
    return log_sx, log_sy, theta

# Alternative 2: Eccentricity + mean scale
def eccentricity_transform(epsilon):
    """Use eccentricity + mean scale for better separation."""
    mean_scale = jnp.sin(epsilon)  # cycles isotropically
    eccentricity = 0.5 * jnp.sin(2 * epsilon)  # -0.5..0.5
    log_sx = mean_scale + eccentricity
    log_sy = mean_scale - eccentricity
    theta = (epsilon % (2 * jnp.pi))
    return log_sx, log_sy, theta

# Alternative 3: Lissajous coverage
def lissajous_transform(epsilon):
    """Lissajous coverage for richer shape space exploration."""
    log_sx = jnp.sin(epsilon)          # slow oscillation
    log_sy = jnp.sin(2 * epsilon)      # faster oscillation
    theta = jnp.sin(3 * epsilon) * jnp.pi
    return log_sx, log_sy, theta

# Dictionary of transforms to test
TRANSFORMS = {
    'current': current_transform,
    'circular_sweep': circular_sweep_transform,
    'eccentricity': eccentricity_transform,
    'lissajous': lissajous_transform
}

def create_2d_sine_target(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Create a 2D sine wave target function."""
    target = (jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y) + 
              0.5 * jnp.sin(4 * jnp.pi * x) * jnp.sin(4 * jnp.pi * y))
    return target

def create_training_data(n_points: int = 50) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple]:
    """Create training data for the 2D sine wave function."""
    x = jnp.linspace(-1, 1, n_points)
    y = jnp.linspace(-1, 1, n_points)
    X, Y = jnp.meshgrid(x, y)
    
    target = create_2d_sine_target(X, Y)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    target_flat = target.flatten()
    
    eval_points = (X, Y)
    
    return jnp.stack([X_flat, Y_flat], axis=1), target_flat, eval_points

def initialize_kernel_centers_grid(n_kernels: int = 25) -> jnp.ndarray:
    """Initialize kernel centers in a uniform grid pattern."""
    grid_size = int(jnp.sqrt(n_kernels))
    x_centers = jnp.linspace(-0.8, 0.8, grid_size)
    y_centers = jnp.linspace(-0.8, 0.8, grid_size)
    
    xx, yy = jnp.meshgrid(x_centers, y_centers)
    centers = jnp.stack([xx.flatten(), yy.flatten()], axis=1)
    
    return centers

def initialize_shape_parameters(n_kernels: int = 25, key: jax.random.PRNGKey = None) -> jnp.ndarray:
    """Initialize parameters for the shape parameter transform model."""
    if key is None:
        key = jax.random.PRNGKey(42)
    
    centers = initialize_kernel_centers_grid(n_kernels)
    
    key, subkey = jax.random.split(key)
    epsilons = jax.random.uniform(subkey, (n_kernels,), minval=-jnp.pi, maxval=jnp.pi)
    
    key, subkey = jax.random.split(key)
    weights = jax.random.uniform(subkey, (n_kernels,), minval=-1, maxval=1)
    
    params = jnp.concatenate([
        centers,           # (n_kernels, 2) - mu_x, mu_y
        epsilons[:, None], # (n_kernels, 1) - epsilon
        weights[:, None]   # (n_kernels, 1) - weight
    ], axis=1)  # Shape: (n_kernels, 4)
    
    return params

def create_shape_loss_function(eval_points: Tuple, target: jnp.ndarray) -> Callable:
    """Create loss function for shape parameter transform model."""
    def loss_fn(params, eval_points, target):
        prediction = generate_shape_rbf(eval_points, params[None, :])
        prediction = prediction[0]
        mse_loss = jnp.mean((prediction - target) ** 2)
        return mse_loss
    
    return loss_fn

@jax.jit
def apply_shape_projection_jit(lambdas_0: jnp.ndarray, n_points: int) -> jnp.ndarray:
    """JIT-compiled projection for shape parameter model."""
    lambdas_0 = lambdas_0.at[:, 0:2].set(jnp.clip(lambdas_0[:, 0:2], -1.0, 1.0))
    lambdas_0 = lambdas_0.at[:, 2].set(jnp.clip(lambdas_0[:, 2], -jnp.pi, jnp.pi))
    return lambdas_0

def train_model(init_params: jnp.ndarray, 
                eval_points: Tuple, 
                target: jnp.ndarray,
                loss_fn: Callable,
                projection_fn: Callable,
                n_epochs: int = 300,
                learning_rate: float = 0.01) -> Tuple[jnp.ndarray, list, float]:
    """Train the RBF model using AdamW optimizer."""
    
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)
    grad_fn = jax.grad(loss_fn)
    n_points = eval_points[0].shape[0]
    
    params = init_params
    losses = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        grads = grad_fn(params, eval_points, target)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = projection_fn(params, n_points)
        
        loss = loss_fn(params, eval_points, target)
        losses.append(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    training_time = time.time() - start_time
    return params, losses, training_time

def analyze_transform_properties(transform_name: str, transform_fn: Callable):
    """Analyze the properties of a transform function."""
    epsilons = jnp.linspace(-jnp.pi, jnp.pi, 100)
    results = []
    
    for eps in epsilons:
        log_sx, log_sy, theta = transform_fn(eps)
        results.append({
            'epsilon': eps,
            'log_sx': log_sx,
            'log_sy': log_sy,
            'theta': theta,
            'sx': jnp.exp(log_sx),
            'sy': jnp.exp(log_sy),
            'ratio': jnp.exp(log_sx) / jnp.exp(log_sy),
            'isotropy': jnp.abs(jnp.exp(log_sx) - jnp.exp(log_sy))
        })
    
    return results

def plot_transform_comparison():
    """Plot comparison of all transform functions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    epsilons = jnp.linspace(-jnp.pi, jnp.pi, 100)
    colors = ['blue', 'red', 'green', 'orange']
    
    for idx, (name, transform_fn) in enumerate(TRANSFORMS.items()):
        ax = axes[idx]
        
        # Get transform results
        results = analyze_transform_properties(name, transform_fn)
        eps = [r['epsilon'] for r in results]
        sx = [r['sx'] for r in results]
        sy = [r['sy'] for r in results]
        ratio = [r['ratio'] for r in results]
        isotropy = [r['isotropy'] for r in results]
        
        # Plot sx, sy vs epsilon
        ax.plot(eps, sx, 'b-', label='σx', linewidth=2)
        ax.plot(eps, sy, 'r-', label='σy', linewidth=2)
        ax.plot(eps, ratio, 'g--', label='σx/σy', linewidth=1, alpha=0.7)
        ax.axhline(y=1, color='k', linestyle=':', alpha=0.5, label='Isotropic')
        
        ax.set_xlabel('ε')
        ax.set_ylabel('Scale Parameters')
        ax.set_title(f'{name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add isotropy info
        min_isotropy = min(isotropy)
        max_isotropy = max(isotropy)
        ax.text(0.02, 0.98, f'Isotropy range: [{min_isotropy:.2f}, {max_isotropy:.2f}]', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('smooth_alternatives_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_transform_comparison(n_seeds: int = 2, n_epochs: int = 300):
    """Run comparison of all transform functions."""
    print("=" * 60)
    print("Smooth Shape Parameter Transform Alternatives")
    print("=" * 60)
    
    # Create training data
    train_points, target, eval_points = create_training_data(n_points=50)
    shape_loss_fn = create_shape_loss_function(eval_points, target)
    
    # Results storage
    all_results = {}
    
    for transform_name in TRANSFORMS.keys():
        print(f"\n--- Testing {transform_name} transform ---")
        all_results[transform_name] = {
            'losses': [],
            'times': [],
            'final_losses': []
        }
        
        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}")
            
            # Set seed
            key = jax.random.PRNGKey(42 + seed)
            shape_params = initialize_shape_parameters(25**2, key)
            
            # Train model
            final_params, losses, training_time = train_model(
                shape_params, eval_points, target, shape_loss_fn, 
                apply_shape_projection_jit, n_epochs, 0.01
            )
            
            all_results[transform_name]['losses'].append(losses)
            all_results[transform_name]['times'].append(training_time)
            all_results[transform_name]['final_losses'].append(losses[-1])
    
    return all_results

def compute_transform_statistics(all_results: Dict) -> Dict:
    """Compute statistics for all transforms."""
    stats = {}
    
    for transform_name, results in all_results.items():
        losses_array = np.array(results['losses'])
        times_array = np.array(results['times'])
        final_losses_array = np.array(results['final_losses'])
        
        stats[transform_name] = {
            'losses_mean': np.mean(losses_array, axis=0),
            'losses_std': np.std(losses_array, axis=0),
            'times_mean': np.mean(times_array),
            'times_std': np.std(times_array),
            'final_losses_mean': np.mean(final_losses_array),
            'final_losses_std': np.std(final_losses_array)
        }
    
    return stats

def plot_alternatives_comparison(stats: Dict, save_path: str = "smooth_alternatives_comparison.png"):
    """Plot comparison of all alternatives."""
    epochs = range(len(stats['current']['losses_mean']))
    
    plt.figure(figsize=(15, 10))
    
    # Colors for different transforms
    colors = ['blue', 'red', 'green', 'orange']
    
    # Plot convergence curves
    plt.subplot(2, 3, 1)
    for idx, (name, stat) in enumerate(stats.items()):
        mean_losses = stat['losses_mean']
        std_losses = stat['losses_std']
        
        plt.plot(epochs, mean_losses, color=colors[idx], 
                label=name.replace('_', ' ').title(), linewidth=2)
        plt.fill_between(epochs, 
                        mean_losses - std_losses, 
                        mean_losses + std_losses, 
                        color=colors[idx], alpha=0.2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Convergence (Mean ± Std)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot final convergence
    plt.subplot(2, 3, 2)
    final_epochs = epochs[-100:]
    for idx, (name, stat) in enumerate(stats.items()):
        mean_losses = stat['losses_mean'][-100:]
        std_losses = stat['losses_std'][-100:]
        
        plt.plot(final_epochs, mean_losses, color=colors[idx], 
                label=name.replace('_', ' ').title(), linewidth=2)
        plt.fill_between(final_epochs, 
                        mean_losses - std_losses, 
                        mean_losses + std_losses, 
                        color=colors[idx], alpha=0.2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Final Convergence (Last 100 Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot training times
    plt.subplot(2, 3, 3)
    names = list(stats.keys())
    times_mean = [stats[name]['times_mean'] for name in names]
    times_std = [stats[name]['times_std'] for name in names]
    
    bars = plt.bar(names, times_mean, yerr=times_std, 
                   color=colors[:len(names)], alpha=0.7, capsize=5)
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison (± Std)')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, mean_val, std_val in zip(bars, times_mean, times_std):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean_val:.1f}±{std_val:.1f}s', ha='center', va='bottom')
    
    # Plot final losses
    plt.subplot(2, 3, 4)
    final_losses_mean = [stats[name]['final_losses_mean'] for name in names]
    final_losses_std = [stats[name]['final_losses_std'] for name in names]
    
    bars = plt.bar(names, final_losses_mean, yerr=final_losses_std,
                   color=colors[:len(names)], alpha=0.7, capsize=5)
    plt.ylabel('Final Loss')
    plt.title('Final Loss Comparison (± Std)')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, mean_val, std_val in zip(bars, final_losses_mean, final_losses_std):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{mean_val:.6f}±{std_val:.6f}', ha='center', va='bottom')
    
    # Plot convergence speed
    plt.subplot(2, 3, 5)
    conv_speeds = []
    conv_speeds_std = []
    threshold = 1e-3
    
    for name in names:
        all_losses = all_results[name]['losses']
        conv_epochs = []
        
        for losses in all_losses:
            conv_epoch = next((i for i, loss in enumerate(losses) if loss < threshold), len(losses))
            conv_epochs.append(conv_epoch)
        
        conv_speeds.append(np.mean(conv_epochs))
        conv_speeds_std.append(np.std(conv_epochs))
    
    bars = plt.bar(names, conv_speeds, yerr=conv_speeds_std,
                   color=colors[:len(names)], alpha=0.7, capsize=5)
    plt.ylabel('Epochs to Converge (< 1e-3)')
    plt.title('Convergence Speed (± Std)')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, mean_val, std_val in zip(bars, conv_speeds, conv_speeds_std):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{mean_val:.0f}±{std_val:.0f}', ha='center', va='bottom')
    
    # Plot transform properties summary
    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.5, 'Transform Properties\nAnalysis', 
             ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.title('Transform Properties')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return save_path

def main():
    """Main comparison function."""
    print("Smooth Shape Parameter Transform Alternatives Analysis")
    print("Comparing 4 different transform implementations")
    
    # Plot transform properties first
    print("\nAnalyzing transform properties...")
    plot_transform_comparison()
    
    # Run training comparison
    print("\nRunning training comparison...")
    all_results = run_transform_comparison(n_seeds=2, n_epochs=300)
    
    # Compute statistics
    stats = compute_transform_statistics(all_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (Mean ± Std)")
    print("=" * 60)
    
    for name in stats.keys():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Final Loss: {stats[name]['final_losses_mean']:.6f} ± {stats[name]['final_losses_std']:.6f}")
        print(f"  Training Time: {stats[name]['times_mean']:.1f} ± {stats[name]['times_std']:.1f}s")
    
    # Find best performing transform
    best_transform = min(stats.keys(), key=lambda x: stats[x]['final_losses_mean'])
    print(f"\nBest performing transform: {best_transform}")
    print(f"Best final loss: {stats[best_transform]['final_losses_mean']:.6f} ± {stats[best_transform]['final_losses_std']:.6f}")
    
    # Plot comparison
    print("\nCreating comparison plots...")
    save_path = plot_alternatives_comparison(stats)
    print(f"Comparison plot saved to: {save_path}")
    
    print("\nSmooth alternatives comparison completed successfully!")

if __name__ == "__main__":
    main()
