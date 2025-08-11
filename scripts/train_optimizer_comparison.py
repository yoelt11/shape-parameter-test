import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Callable, List
import sys
import os
import time
from scipy.optimize import minimize

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model.standard_model import generate_rbf_solutions as generate_standard_rbf
from model.rbf_model import generate_rbf_solutions as generate_shape_rbf

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)
key = jax.random.PRNGKey(42)

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

def train_with_optimizer(init_params: jnp.ndarray, 
                         eval_points: Tuple, 
                         target: jnp.ndarray,
                         loss_fn: Callable,
                         projection_fn: Callable,
                         optimizer_name: str,
                         n_epochs: int = 1000,
                         learning_rate: float = 0.01,
                         use_projection: bool = True) -> Tuple[jnp.ndarray, list, float]:
    """Train the RBF model using specified optimizer."""
    
    # Get number of points for projection
    n_points = eval_points[0].shape[0]
    
    if optimizer_name.lower() == 'lbfgs':
        # Use scipy's L-BFGS-B optimizer
        def scipy_loss_fn(params_flat):
            params = params_flat.reshape(init_params.shape)
            if use_projection:
                params = projection_fn(params, n_points)
            return float(loss_fn(params, eval_points, target))
        
        def scipy_grad_fn(params_flat):
            params = params_flat.reshape(init_params.shape)
            if use_projection:
                params = projection_fn(params, n_points)
            grads = jax.grad(loss_fn)(params, eval_points, target)
            return np.array(grads).flatten()
        
        start_time = time.time()
        
        # Flatten parameters for scipy
        init_params_flat = np.array(init_params).flatten()
        
        # Run L-BFGS-B optimization
        result = minimize(
            scipy_loss_fn,
            init_params_flat,
            method='L-BFGS-B',
            jac=scipy_grad_fn,
            options={'maxiter': n_epochs, 'disp': False}
        )
        
        training_time = time.time() - start_time
        
        # Reshape back to original shape
        final_params = result.x.reshape(init_params.shape)
        
        # For L-BFGS, we don't have per-epoch losses, so we'll create a simple list
        losses = [result.fun] * min(n_epochs, result.nit)
        
        return final_params, losses, training_time
    
    else:
        # Use optax optimizers
        if optimizer_name.lower() == 'adam':
            optimizer = optax.adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'adamw':
            optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Initialize optimizer state
        opt_state = optimizer.init(init_params)
        
        # Create gradient function
        grad_fn = jax.grad(loss_fn)
        
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
            
            if epoch % 200 == 0:
                print(f"  Epoch {epoch}: Loss = {loss:.6f}")
        
        training_time = time.time() - start_time
        
        return params, losses, training_time

def plot_optimizer_comparison(results: Dict, save_path: str = "optimizer_comparison.png"):
    """Plot the optimizer comparison results."""
    plt.figure(figsize=(20, 12))
    
    # Get all optimizer names
    optimizers = list(set([key.split('_')[0] for key in results.keys()]))
    models = list(set([key.split('_')[1] for key in results.keys()]))
    
    # Plot 1: Convergence comparison for each model type
    plt.subplot(2, 4, 1)
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                epochs = range(len(results[key]['losses']))
                plt.plot(epochs, results[key]['losses'], 
                        label=f'{optimizer.upper()} ({model})', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Final convergence (last 100 epochs)
    plt.subplot(2, 4, 2)
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                losses = results[key]['losses']
                if len(losses) > 100:
                    final_epochs = range(len(losses) - 100, len(losses))
                    final_losses = losses[-100:]
                else:
                    final_epochs = range(len(losses))
                    final_losses = losses
                plt.plot(final_epochs, final_losses, 
                        label=f'{optimizer.upper()} ({model})', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Final Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Training times comparison
    plt.subplot(2, 4, 3)
    times_data = {}
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                if optimizer not in times_data:
                    times_data[optimizer] = {}
                times_data[optimizer][model] = results[key]['training_time']
    
    x = np.arange(len(optimizers))
    width = 0.35
    
    for i, model in enumerate(models):
        times = [times_data.get(opt, {}).get(model, 0) for opt in optimizers]
        plt.bar(x + i*width, times, width, label=model, alpha=0.8)
    
    plt.xlabel('Optimizer')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(x + width/2, [opt.upper() for opt in optimizers])
    plt.legend()
    
    # Plot 4: Final losses comparison
    plt.subplot(2, 4, 4)
    final_losses_data = {}
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                if optimizer not in final_losses_data:
                    final_losses_data[optimizer] = {}
                final_losses_data[optimizer][model] = results[key]['losses'][-1]
    
    for i, model in enumerate(models):
        losses = [final_losses_data.get(opt, {}).get(model, float('inf')) for opt in optimizers]
        plt.bar(x + i*width, losses, width, label=model, alpha=0.8)
    
    plt.xlabel('Optimizer')
    plt.ylabel('Final Loss')
    plt.title('Final Loss Comparison')
    plt.xticks(x + width/2, [opt.upper() for opt in optimizers])
    plt.legend()
    plt.yscale('log')
    
    # Plot 5: Convergence speed (epochs to reach threshold)
    plt.subplot(2, 4, 5)
    conv_speeds = {}
    threshold = 1e-3
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                losses = results[key]['losses']
                conv_epoch = next((i for i, loss in enumerate(losses) if loss < threshold), len(losses))
                if optimizer not in conv_speeds:
                    conv_speeds[optimizer] = {}
                conv_speeds[optimizer][model] = conv_epoch
    
    for i, model in enumerate(models):
        speeds = [conv_speeds.get(opt, {}).get(model, len(results[f"{optimizers[0]}_{model}"]['losses'])) for opt in optimizers]
        plt.bar(x + i*width, speeds, width, label=model, alpha=0.8)
    
    plt.xlabel('Optimizer')
    plt.ylabel(f'Epochs to Converge (< {threshold})')
    plt.title('Convergence Speed')
    plt.xticks(x + width/2, [opt.upper() for opt in optimizers])
    plt.legend()
    
    # Plot 6: Parameter efficiency
    plt.subplot(2, 4, 6)
    n_kernels = 25
    standard_params = n_kernels**2 * 6
    shape_params = n_kernels**2 * 4
    
    efficiency_data = {}
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                final_loss = results[key]['losses'][-1]
                if model == 'standard':
                    efficiency = final_loss / standard_params
                else:
                    efficiency = final_loss / shape_params
                if optimizer not in efficiency_data:
                    efficiency_data[optimizer] = {}
                efficiency_data[optimizer][model] = efficiency
    
    for i, model in enumerate(models):
        efficiencies = [efficiency_data.get(opt, {}).get(model, float('inf')) for opt in optimizers]
        plt.bar(x + i*width, efficiencies, width, label=model, alpha=0.8)
    
    plt.xlabel('Optimizer')
    plt.ylabel('Loss per Parameter')
    plt.title('Parameter Efficiency')
    plt.xticks(x + width/2, [opt.upper() for opt in optimizers])
    plt.legend()
    plt.yscale('log')
    
    # Plot 7: Best optimizer per model
    plt.subplot(2, 4, 7)
    best_optimizers = {}
    for model in models:
        best_opt = None
        best_loss = float('inf')
        for optimizer in optimizers:
            key = f"{optimizer}_{model}"
            if key in results:
                final_loss = results[key]['losses'][-1]
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_opt = optimizer
        best_optimizers[model] = best_opt
    
    model_names = list(best_optimizers.keys())
    best_opt_names = [best_optimizers[model].upper() for model in model_names]
    colors = ['lightblue', 'lightcoral']
    
    bars = plt.bar(model_names, [1]*len(model_names), color=colors[:len(model_names)], alpha=0.7)
    plt.ylabel('Best Optimizer')
    plt.title('Best Optimizer per Model')
    
    # Add optimizer names on bars
    for bar, opt_name in zip(bars, best_opt_names):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                opt_name, ha='center', va='center', fontweight='bold')
    
    # Plot 8: Summary statistics
    plt.subplot(2, 4, 8)
    plt.axis('off')
    
    # Create summary text
    summary_text = "OPTIMIZER COMPARISON SUMMARY\n\n"
    
    for model in models:
        summary_text += f"{model.upper()} MODEL:\n"
        model_losses = []
        for optimizer in optimizers:
            key = f"{optimizer}_{model}"
            if key in results:
                final_loss = results[key]['losses'][-1]
                training_time = results[key]['training_time']
                model_losses.append((optimizer, final_loss, training_time))
        
        # Sort by final loss
        model_losses.sort(key=lambda x: x[1])
        
        for i, (opt, loss, time_val) in enumerate(model_losses):
            rank = i + 1
            summary_text += f"  {rank}. {opt.upper()}: {loss:.6f} ({time_val:.1f}s)\n"
        summary_text += "\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return save_path

def main():
    """Main training and comparison function with optimizer analysis."""
    print("=" * 70)
    print("RBF Model Optimizer Comparison")
    print("2D Sine Wave Function with 25x25 Kernels")
    print("Comparing Adam, AdamW, SGD, and L-BFGS")
    print("=" * 70)
    
    # Training parameters
    n_epochs = 1000
    learning_rate = 0.01
    n_kernels = 25
    
    # Optimizer configurations
    optimizers = ['adam', 'adamw', 'sgd', 'lbfgs']
    models = ['standard', 'shape']
    
    print(f"Training Parameters:")
    print(f"- Epochs: {n_epochs}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Kernels: {n_kernels}x{n_kernels} = {n_kernels**2}")
    print(f"- Optimizers: {', '.join([opt.upper() for opt in optimizers])}")
    print(f"- Models: {', '.join(models)}")
    print()
    
    # Create training data
    print("Creating training data...")
    train_points, target, eval_points = create_training_data(n_points=50)
    print(f"Training points: {train_points.shape}")
    print(f"Target shape: {target.shape}")
    print()
    
    # Initialize parameters
    print("Initializing parameters...")
    keys = jax.random.split(key, len(optimizers) * len(models))
    key_idx = 0
    
    # Create loss functions
    print("Creating loss functions...")
    standard_loss_fn = create_loss_function(eval_points, target)
    shape_loss_fn = create_shape_loss_function(eval_points, target)
    
    # Train all combinations
    results = {}
    
    for optimizer in optimizers:
        for model in models:
            print(f"\nTraining {optimizer.upper()} with {model} model...")
            print("-" * 50)
            
            # Initialize parameters
            if model == 'standard':
                init_params = initialize_standard_parameters(n_kernels**2, keys[key_idx])
                loss_fn = standard_loss_fn
                projection_fn = apply_standard_projection_jit
            else:
                init_params = initialize_shape_parameters(n_kernels**2, keys[key_idx])
                loss_fn = shape_loss_fn
                projection_fn = apply_shape_projection_jit
            
            key_idx += 1
            
            # Train model
            final_params, losses, training_time = train_with_optimizer(
                init_params, eval_points, target, loss_fn, projection_fn,
                optimizer, n_epochs, learning_rate, use_projection=True
            )
            
            results[f"{optimizer}_{model}"] = {
                'losses': losses,
                'training_time': training_time,
                'final_params': final_params
            }
            
            print(f"  Final Loss: {losses[-1]:.6f}")
            print(f"  Training Time: {training_time:.1f}s")
    
    print("\nTraining completed!")
    print(f"Final Losses:")
    for name, result in results.items():
        print(f"- {name}: {result['losses'][-1]:.6f} ({result['training_time']:.1f}s)")
    
    # Plot comparison
    print("\nCreating comparison plots...")
    save_path = plot_optimizer_comparison(results)
    print(f"Comparison plot saved to: {save_path}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)
    
    # Best performing optimizer per model
    for model in models:
        model_results = {k: v for k, v in results.items() if k.endswith(f"_{model}")}
        best_opt = min(model_results.keys(), key=lambda x: model_results[x]['losses'][-1])
        print(f"\nBest optimizer for {model} model: {best_opt.split('_')[0].upper()}")
        print(f"Best final loss: {model_results[best_opt]['losses'][-1]:.6f}")
    
    # Overall best performing combination
    best_combination = min(results.keys(), key=lambda x: results[x]['losses'][-1])
    print(f"\nBest overall combination: {best_combination}")
    print(f"Best final loss: {results[best_combination]['losses'][-1]:.6f}")
    
    # Optimizer ranking
    print(f"\nOptimizer Ranking (by average final loss):")
    opt_rankings = {}
    for optimizer in optimizers:
        opt_losses = [results[f"{optimizer}_{model}"]['losses'][-1] for model in models]
        avg_loss = np.mean(opt_losses)
        opt_rankings[optimizer] = avg_loss
    
    sorted_opts = sorted(opt_rankings.items(), key=lambda x: x[1])
    for i, (opt, avg_loss) in enumerate(sorted_opts):
        print(f"{i+1}. {opt.upper()}: {avg_loss:.6f}")
    
    # Training time analysis
    print(f"\nTraining Time Analysis:")
    for optimizer in optimizers:
        opt_times = [results[f"{optimizer}_{model}"]['training_time'] for model in models]
        avg_time = np.mean(opt_times)
        print(f"- {optimizer.upper()}: {avg_time:.1f}s average")
    
    print("\nTraining and comparison completed successfully!")

if __name__ == "__main__":
    main()
