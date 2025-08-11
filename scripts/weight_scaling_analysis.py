import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Callable, List
import sys
import os
import time
from dataclasses import dataclass

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model.standard_model import generate_rbf_solutions as generate_standard_rbf

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)

@dataclass
class WeightScalingConfig:
    """Configuration for weight scaling analysis."""
    name: str
    scaling_fn: Callable
    description: str

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

def create_loss_function_with_scaling(eval_points: Tuple, target: jnp.ndarray, scaling_fn: Callable) -> Callable:
    """Create loss function with custom weight scaling."""
    def loss_fn(params, eval_points, target):
        # Generate RBF solution
        solution = generate_standard_rbf(eval_points, params)
        
        # Apply custom weight scaling
        scaled_solution = scaling_fn(solution)
        
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

def run_scaling_comparison(n_seeds: int = 5, n_epochs: int = 1000, learning_rate: float = 0.01, n_kernels: int = 25):
    """Run comparison of different weight scaling approaches."""
    
    # Define different scaling configurations
    scaling_configs = [
        WeightScalingConfig(
            name="No Scaling",
            scaling_fn=lambda x: x,
            description="No weight scaling applied"
        ),
        WeightScalingConfig(
            name="Tanh Scaling",
            scaling_fn=jax.nn.tanh,
            description="Apply tanh activation to weights"
        ),
        WeightScalingConfig(
            name="Sigmoid Scaling",
            scaling_fn=jax.nn.sigmoid,
            description="Apply sigmoid activation to weights"
        ),
        WeightScalingConfig(
            name="Softplus Scaling",
            scaling_fn=jax.nn.softplus,
            description="Apply softplus activation to weights"
        ),
        WeightScalingConfig(
            name="Gelu Scaling",
            scaling_fn=jax.nn.gelu,
            description="Apply GELU activation to weights"
        ),
        WeightScalingConfig(
            name="Relu Scaling",
            scaling_fn=jax.nn.relu,
            description="Apply ReLU activation to weights"
        ),
        WeightScalingConfig(
            name="Scale Factor 0.1",
            scaling_fn=lambda x: x * 0.1,
            description="Multiply weights by 0.1"
        ),
        WeightScalingConfig(
            name="Scale Factor 2.0",
            scaling_fn=lambda x: x * 2.0,
            description="Multiply weights by 2.0"
        ),
        WeightScalingConfig(
            name="Clip [-1, 1]",
            scaling_fn=lambda x: jnp.clip(x, -1.0, 1.0),
            description="Clip weights to [-1, 1] range"
        ),
        WeightScalingConfig(
            name="Normalize",
            scaling_fn=lambda x: x / (jnp.std(x) + 1e-8),
            description="Normalize weights by standard deviation"
        )
    ]
    
    # Create training data
    print("Creating training data...")
    X_train, target_train, eval_points = create_training_data(n_points=50)
    
    # Store results
    all_results = {}
    
    for config in scaling_configs:
        print(f"\n=== Testing {config.name} ===")
        print(f"Description: {config.description}")
        
        config_results = {
            'loss_histories': [],
            'final_losses': [],
            'training_times': [],
            'convergence_epochs': []
        }
        
        for seed in range(n_seeds):
            print(f"  Running seed {seed + 1}/{n_seeds}")
            
            # Initialize parameters
            key = jax.random.PRNGKey(seed)
            init_params = initialize_parameters(n_kernels=n_kernels, key=key)
            
            # Create loss function with scaling
            loss_fn = create_loss_function_with_scaling(eval_points, target_train, config.scaling_fn)
            
            # Train model
            final_params, loss_history, training_time = train_model(
                init_params, eval_points, target_train, loss_fn, n_epochs, learning_rate
            )
            
            # Store results
            config_results['loss_histories'].append(loss_history)
            config_results['final_losses'].append(loss_history[-1])
            config_results['training_times'].append(training_time)
            
            # Find convergence epoch (when loss stops improving significantly)
            convergence_epoch = find_convergence_epoch(loss_history)
            config_results['convergence_epochs'].append(convergence_epoch)
        
        all_results[config.name] = config_results
    
    return all_results, scaling_configs

def find_convergence_epoch(loss_history: List[float], tolerance: float = 1e-6, patience: int = 50) -> int:
    """Find the epoch when the model converges."""
    if len(loss_history) < patience:
        return len(loss_history)
    
    for i in range(patience, len(loss_history)):
        recent_losses = loss_history[i-patience:i]
        if max(recent_losses) - min(recent_losses) < tolerance:
            return i - patience
    
    return len(loss_history)

def compute_statistics(all_results: Dict) -> Dict:
    """Compute statistics for all configurations."""
    stats = {}
    
    for config_name, results in all_results.items():
        final_losses = np.array(results['final_losses'])
        training_times = np.array(results['training_times'])
        convergence_epochs = np.array(results['convergence_epochs'])
        
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

def plot_comparison(stats: Dict, all_results: Dict, scaling_configs: List[WeightScalingConfig], save_path: str = "weight_scaling_comparison.png"):
    """Plot comparison of different weight scaling approaches."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Weight Scaling Analysis: Training Performance Comparison', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    config_names = list(stats.keys())
    mean_final_losses = [stats[name]['mean_final_loss'] for name in config_names]
    std_final_losses = [stats[name]['std_final_loss'] for name in config_names]
    mean_training_times = [stats[name]['mean_training_time'] for name in config_names]
    mean_convergence_epochs = [stats[name]['mean_convergence_epoch'] for name in config_names]
    
    # 1. Final Loss Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(config_names)), mean_final_losses, yerr=std_final_losses, 
                    capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_title('Final Loss Comparison', fontweight='bold')
    ax1.set_ylabel('Mean Final Loss')
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars1, mean_final_losses, std_final_losses)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001, 
                f'{mean:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Training Time Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(config_names)), mean_training_times, alpha=0.7, 
                    color='lightcoral', edgecolor='darkred')
    ax2.set_title('Training Time Comparison', fontweight='bold')
    ax2.set_ylabel('Mean Training Time (seconds)')
    ax2.set_xticks(range(len(config_names)))
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars2, mean_training_times)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mean:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # 3. Convergence Epoch Comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(config_names)), mean_convergence_epochs, alpha=0.7, 
                    color='lightgreen', edgecolor='darkgreen')
    ax3.set_title('Convergence Epoch Comparison', fontweight='bold')
    ax3.set_ylabel('Mean Convergence Epoch')
    ax3.set_xticks(range(len(config_names)))
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars3, mean_convergence_epochs)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{mean:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Loss Curves for Best and Worst Performers
    ax4 = axes[1, 1]
    
    # Find best and worst performers
    best_config = min(stats.items(), key=lambda x: x[1]['mean_final_loss'])[0]
    worst_config = max(stats.items(), key=lambda x: x[1]['mean_final_loss'])[0]
    
    # Plot loss curves
    for config_name in [best_config, worst_config]:
        loss_histories = all_results[config_name]['loss_histories']
        mean_loss = np.mean(loss_histories, axis=0)
        std_loss = np.std(loss_histories, axis=0)
        
        epochs = range(len(mean_loss))
        color = 'green' if config_name == best_config else 'red'
        label = f'{config_name} (Best)' if config_name == best_config else f'{config_name} (Worst)'
        
        ax4.plot(epochs, mean_loss, color=color, label=label, linewidth=2)
        ax4.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, 
                        alpha=0.3, color=color)
    
    ax4.set_title('Loss Curves: Best vs Worst Performers', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_summary_table(stats: Dict, scaling_configs: List[WeightScalingConfig]):
    """Print a summary table of the results."""
    print("\n" + "="*80)
    print("WEIGHT SCALING ANALYSIS SUMMARY")
    print("="*80)
    
    # Sort by mean final loss
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['mean_final_loss'])
    
    print(f"{'Configuration':<20} {'Mean Loss':<12} {'Std Loss':<12} {'Mean Time':<12} {'Conv Epoch':<12}")
    print("-" * 80)
    
    for config_name, stat in sorted_stats:
        print(f"{config_name:<20} {stat['mean_final_loss']:<12.6f} {stat['std_final_loss']:<12.6f} "
              f"{stat['mean_training_time']:<12.2f} {stat['mean_convergence_epoch']:<12.1f}")
    
    print("\n" + "="*80)
    print("CONFIGURATION DESCRIPTIONS")
    print("="*80)
    
    for config in scaling_configs:
        print(f"{config.name}: {config.description}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    best_config = sorted_stats[0][0]
    worst_config = sorted_stats[-1][0]
    
    print(f"Best performing configuration: {best_config}")
    print(f"Worst performing configuration: {worst_config}")
    print(f"Best mean final loss: {sorted_stats[0][1]['mean_final_loss']:.6f}")
    print(f"Worst mean final loss: {sorted_stats[-1][1]['mean_final_loss']:.6f}")
    print(f"Performance ratio (worst/best): {sorted_stats[-1][1]['mean_final_loss'] / sorted_stats[0][1]['mean_final_loss']:.2f}")

def main():
    """Main function to run the weight scaling analysis."""
    print("Starting Weight Scaling Analysis...")
    
    # Run comparison
    all_results, scaling_configs = run_scaling_comparison(
        n_seeds=5, 
        n_epochs=1000, 
        learning_rate=0.01, 
        n_kernels=25
    )
    
    # Compute statistics
    stats = compute_statistics(all_results)
    
    # Print summary
    print_summary_table(stats, scaling_configs)
    
    # Create plots
    plot_comparison(stats, all_results, scaling_configs)
    
    print("\nAnalysis complete! Check the generated plot for visual comparison.")

if __name__ == "__main__":
    main()


