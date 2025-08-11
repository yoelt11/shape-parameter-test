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

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)

def create_2d_sine_target(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Create a 2D sine wave target function similar to Poisson's equation."""
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

def create_loss_function_with_tanh_scaling(eval_points: Tuple, target: jnp.ndarray, use_tanh: bool = False) -> Callable:
    """Create loss function with optional tanh scaling on weights."""
    def loss_fn(params, eval_points, target):
        # Generate RBF solution
        solution = generate_standard_rbf(eval_points, params)
        
        # Apply tanh scaling if requested
        if use_tanh:
            scaled_solution = jax.nn.tanh(solution)
        else:
            scaled_solution = solution
        
        # Compute MSE loss
        loss = jnp.mean((scaled_solution - target) ** 2)
        return loss
    
    return loss_fn

def train_model(init_params: jnp.ndarray, 
                eval_points: Tuple, 
                target: jnp.ndarray,
                loss_fn: Callable,
                n_epochs: int = 1000,
                learning_rate: float = 0.01) -> Tuple[jnp.ndarray, List[float], float]:
    """Train the model and return final parameters, loss history, and training time."""
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(init_params)
    
    # Training function
    @jax.jit
    def train_step(params, opt_state, eval_points, target):
        loss, grads = jax.value_and_grad(loss_fn)(params, eval_points, target)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Training loop
    params = init_params
    loss_history = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        params, opt_state, loss = train_step(params, opt_state, eval_points, target)
        loss_history.append(float(loss))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    training_time = time.time() - start_time
    
    return params, loss_history, training_time

def run_tanh_comparison(n_seeds: int = 5, n_epochs: int = 1000, learning_rate: float = 0.01, n_kernels: int = 25):
    """Run comparison between with and without tanh scaling."""
    
    # Create training data
    print("Creating training data...")
    X_train, target_train, eval_points = create_training_data(n_points=50)
    
    # Store results
    results = {
        'no_tanh': {
            'loss_histories': [],
            'final_losses': [],
            'training_times': [],
            'convergence_epochs': []
        },
        'with_tanh': {
            'loss_histories': [],
            'final_losses': [],
            'training_times': [],
            'convergence_epochs': []
        }
    }
    
    for seed in range(n_seeds):
        print(f"\n=== Running seed {seed + 1}/{n_seeds} ===")
        
        # Initialize parameters
        key = jax.random.PRNGKey(seed)
        init_params = initialize_parameters(n_kernels=n_kernels, key=key)
        
        # Test without tanh scaling
        print("  Testing without tanh scaling...")
        loss_fn_no_tanh = create_loss_function_with_tanh_scaling(eval_points, target_train, use_tanh=False)
        final_params_no_tanh, loss_history_no_tanh, training_time_no_tanh = train_model(
            init_params, eval_points, target_train, loss_fn_no_tanh, n_epochs, learning_rate
        )
        
        results['no_tanh']['loss_histories'].append(loss_history_no_tanh)
        results['no_tanh']['final_losses'].append(loss_history_no_tanh[-1])
        results['no_tanh']['training_times'].append(training_time_no_tanh)
        results['no_tanh']['convergence_epochs'].append(find_convergence_epoch(loss_history_no_tanh))
        
        # Test with tanh scaling
        print("  Testing with tanh scaling...")
        loss_fn_with_tanh = create_loss_function_with_tanh_scaling(eval_points, target_train, use_tanh=True)
        final_params_with_tanh, loss_history_with_tanh, training_time_with_tanh = train_model(
            init_params, eval_points, target_train, loss_fn_with_tanh, n_epochs, learning_rate
        )
        
        results['with_tanh']['loss_histories'].append(loss_history_with_tanh)
        results['with_tanh']['final_losses'].append(loss_history_with_tanh[-1])
        results['with_tanh']['training_times'].append(training_time_with_tanh)
        results['with_tanh']['convergence_epochs'].append(find_convergence_epoch(loss_history_with_tanh))
    
    return results

def find_convergence_epoch(loss_history: List[float], tolerance: float = 1e-6, patience: int = 50) -> int:
    """Find the epoch when the model converges."""
    if len(loss_history) < patience:
        return len(loss_history)
    
    for i in range(patience, len(loss_history)):
        recent_losses = loss_history[i-patience:i]
        if max(recent_losses) - min(recent_losses) < tolerance:
            return i - patience
    
    return len(loss_history)

def compute_statistics(results: Dict) -> Dict:
    """Compute statistics for both configurations."""
    stats = {}
    
    for config_name, config_results in results.items():
        final_losses = np.array(config_results['final_losses'])
        training_times = np.array(config_results['training_times'])
        convergence_epochs = np.array(config_results['convergence_epochs'])
        
        stats[config_name] = {
            'mean_final_loss': np.mean(final_losses),
            'std_final_loss': np.std(final_losses),
            'min_final_loss': np.min(final_losses),
            'max_final_loss': np.max(final_losses),
            'mean_training_time': np.mean(training_times),
            'std_training_time': np.std(training_times),
            'mean_convergence_epoch': np.mean(convergence_epochs),
            'std_convergence_epoch': np.std(convergence_epochs)
        }
    
    return stats

def plot_tanh_comparison(stats: Dict, results: Dict, save_path: str = "tanh_scaling_comparison.png"):
    """Plot comparison between with and without tanh scaling."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tanh Scaling Analysis: Impact on Training Performance', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    config_names = list(stats.keys())
    mean_final_losses = [stats[name]['mean_final_loss'] for name in config_names]
    std_final_losses = [stats[name]['std_final_loss'] for name in config_names]
    mean_training_times = [stats[name]['mean_training_time'] for name in config_names]
    mean_convergence_epochs = [stats[name]['mean_convergence_epoch'] for name in config_names]
    
    # 1. Final Loss Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(config_names)), mean_final_losses, yerr=std_final_losses, 
                    capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'], 
                    edgecolor=['navy', 'darkred'])
    ax1.set_title('Final Loss Comparison', fontweight='bold')
    ax1.set_ylabel('Mean Final Loss')
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels(['No Tanh', 'With Tanh'])
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars1, mean_final_losses, std_final_losses)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001, 
                f'{mean:.6f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Training Time Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(config_names)), mean_training_times, alpha=0.7, 
                    color=['lightgreen', 'orange'], edgecolor=['darkgreen', 'darkorange'])
    ax2.set_title('Training Time Comparison', fontweight='bold')
    ax2.set_ylabel('Mean Training Time (seconds)')
    ax2.set_xticks(range(len(config_names)))
    ax2.set_xticklabels(['No Tanh', 'With Tanh'])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars2, mean_training_times)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mean:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # 3. Convergence Epoch Comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(config_names)), mean_convergence_epochs, alpha=0.7, 
                    color=['plum', 'gold'], edgecolor=['purple', 'darkgoldenrod'])
    ax3.set_title('Convergence Epoch Comparison', fontweight='bold')
    ax3.set_ylabel('Mean Convergence Epoch')
    ax3.set_xticks(range(len(config_names)))
    ax3.set_xticklabels(['No Tanh', 'With Tanh'])
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars3, mean_convergence_epochs)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{mean:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Loss Curves Comparison
    ax4 = axes[1, 1]
    
    # Plot loss curves for both configurations
    for config_name in config_names:
        loss_histories = results[config_name]['loss_histories']
        mean_loss = np.mean(loss_histories, axis=0)
        std_loss = np.std(loss_histories, axis=0)
        
        epochs = range(len(mean_loss))
        color = 'blue' if config_name == 'no_tanh' else 'red'
        label = 'No Tanh' if config_name == 'no_tanh' else 'With Tanh'
        
        ax4.plot(epochs, mean_loss, color=color, label=label, linewidth=2)
        ax4.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, 
                        alpha=0.3, color=color)
    
    ax4.set_title('Loss Curves Comparison', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_tanh_summary(stats: Dict):
    """Print a summary of the tanh scaling analysis."""
    print("\n" + "="*80)
    print("TANH SCALING ANALYSIS SUMMARY")
    print("="*80)
    
    no_tanh_stats = stats['no_tanh']
    with_tanh_stats = stats['with_tanh']
    
    print(f"{'Metric':<25} {'No Tanh':<15} {'With Tanh':<15} {'Difference':<15}")
    print("-" * 80)
    
    print(f"{'Mean Final Loss':<25} {no_tanh_stats['mean_final_loss']:<15.6f} "
          f"{with_tanh_stats['mean_final_loss']:<15.6f} "
          f"{with_tanh_stats['mean_final_loss'] - no_tanh_stats['mean_final_loss']:<15.6f}")
    
    print(f"{'Std Final Loss':<25} {no_tanh_stats['std_final_loss']:<15.6f} "
          f"{with_tanh_stats['std_final_loss']:<15.6f} "
          f"{with_tanh_stats['std_final_loss'] - no_tanh_stats['std_final_loss']:<15.6f}")
    
    print(f"{'Mean Training Time (s)':<25} {no_tanh_stats['mean_training_time']:<15.2f} "
          f"{with_tanh_stats['mean_training_time']:<15.2f} "
          f"{with_tanh_stats['mean_training_time'] - no_tanh_stats['mean_training_time']:<15.2f}")
    
    print(f"{'Mean Convergence Epoch':<25} {no_tanh_stats['mean_convergence_epoch']:<15.1f} "
          f"{with_tanh_stats['mean_convergence_epoch']:<15.1f} "
          f"{with_tanh_stats['mean_convergence_epoch'] - no_tanh_stats['mean_convergence_epoch']:<15.1f}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Determine which is better
    if no_tanh_stats['mean_final_loss'] < with_tanh_stats['mean_final_loss']:
        better_config = "No Tanh"
        worse_config = "With Tanh"
        improvement = no_tanh_stats['mean_final_loss'] / with_tanh_stats['mean_final_loss']
    else:
        better_config = "With Tanh"
        worse_config = "No Tanh"
        improvement = with_tanh_stats['mean_final_loss'] / no_tanh_stats['mean_final_loss']
    
    print(f"Better performing configuration: {better_config}")
    print(f"Performance improvement: {improvement:.2f}x")
    print(f"Loss difference: {abs(with_tanh_stats['mean_final_loss'] - no_tanh_stats['mean_final_loss']):.6f}")
    
    # Analyze convergence
    if no_tanh_stats['mean_convergence_epoch'] < with_tanh_stats['mean_convergence_epoch']:
        faster_convergence = "No Tanh"
    else:
        faster_convergence = "With Tanh"
    
    print(f"Faster convergence: {faster_convergence}")
    
    # Analyze stability (lower std is more stable)
    if no_tanh_stats['std_final_loss'] < with_tanh_stats['std_final_loss']:
        more_stable = "No Tanh"
    else:
        more_stable = "With Tanh"
    
    print(f"More stable training: {more_stable}")

def main():
    """Main function to run the tanh scaling analysis."""
    print("Starting Tanh Scaling Analysis...")
    print("This analysis compares the effect of applying tanh scaling to the RBF solution output.")
    print("The tanh scaling was commented out in the original code: 'scaled_weights = weights #jax.nn.tanh(weights)'")
    
    # Run comparison
    results = run_tanh_comparison(
        n_seeds=5, 
        n_epochs=1000, 
        learning_rate=0.01, 
        n_kernels=25
    )
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Print summary
    print_tanh_summary(stats)
    
    # Create plots
    plot_tanh_comparison(stats, results)
    
    print("\nAnalysis complete! Check the generated plot for visual comparison.")
    print("\nRecommendation: Based on the results, you can decide whether to uncomment the tanh scaling in your model.")

if __name__ == "__main__":
    main()


