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
import itertools

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

def train_with_optimizer_hyperparams(init_params: jnp.ndarray, 
                                    eval_points: Tuple, 
                                    target: jnp.ndarray,
                                    loss_fn: Callable,
                                    projection_fn: Callable,
                                    optimizer_name: str,
                                    hyperparams: Dict,
                                    n_epochs: int = 500,
                                    use_projection: bool = True) -> Tuple[jnp.ndarray, list, float]:
    """Train the RBF model using specified optimizer with hyperparameters."""
    
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
        # Use optax optimizers with hyperparameters
        learning_rate = hyperparams.get('learning_rate', 0.01)
        
        if optimizer_name.lower() == 'adam':
            beta1 = hyperparams.get('beta1', 0.9)
            beta2 = hyperparams.get('beta2', 0.999)
            optimizer = optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2)
        elif optimizer_name.lower() == 'adamw':
            weight_decay = hyperparams.get('weight_decay', 1e-4)
            beta1 = hyperparams.get('beta1', 0.9)
            beta2 = hyperparams.get('beta2', 0.999)
            optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay, b1=beta1, b2=beta2)
        elif optimizer_name.lower() == 'sgd':
            momentum = hyperparams.get('momentum', 0.9)
            nesterov = hyperparams.get('nesterov', False)
            optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
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
            
            # Early stopping if loss is very small
            if loss < 1e-6:
                break
        
        training_time = time.time() - start_time
        
        return params, losses, training_time

def hyperparameter_tuning(init_params: jnp.ndarray,
                         eval_points: Tuple,
                         target: jnp.ndarray,
                         loss_fn: Callable,
                         projection_fn: Callable,
                         optimizer_name: str,
                         n_epochs: int = 500) -> Dict:
    """Perform hyperparameter tuning for a given optimizer."""
    
    # Define hyperparameter search spaces
    if optimizer_name.lower() == 'adam':
        learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
        beta1s = [0.8, 0.9, 0.95]
        beta2s = [0.999, 0.99]
        hyperparam_combinations = list(itertools.product(learning_rates, beta1s, beta2s))
        
    elif optimizer_name.lower() == 'adamw':
        learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
        weight_decays = [1e-5, 1e-4, 1e-3]
        beta1s = [0.8, 0.9, 0.95]
        beta2s = [0.999, 0.99]
        hyperparam_combinations = list(itertools.product(learning_rates, weight_decays, beta1s, beta2s))
        
    elif optimizer_name.lower() == 'sgd':
        learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
        momentums = [0.0, 0.5, 0.9, 0.95]
        nesterovs = [True, False]
        hyperparam_combinations = list(itertools.product(learning_rates, momentums, nesterovs))
        
    elif optimizer_name.lower() == 'lbfgs':
        # L-BFGS doesn't have many hyperparameters to tune
        return {'learning_rate': 1.0, 'final_loss': float('inf')}
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    best_loss = float('inf')
    best_hyperparams = None
    best_training_time = 0
    best_losses = []
    
    print(f"  Tuning {optimizer_name.upper()} hyperparameters...")
    print(f"  Testing {len(hyperparam_combinations)} combinations...")
    
    for i, combo in enumerate(hyperparam_combinations):
        if optimizer_name.lower() == 'adam':
            hyperparams = {
                'learning_rate': combo[0],
                'beta1': combo[1],
                'beta2': combo[2]
            }
        elif optimizer_name.lower() == 'adamw':
            hyperparams = {
                'learning_rate': combo[0],
                'weight_decay': combo[1],
                'beta1': combo[2],
                'beta2': combo[3]
            }
        elif optimizer_name.lower() == 'sgd':
            hyperparams = {
                'learning_rate': combo[0],
                'momentum': combo[1],
                'nesterov': combo[2]
            }
        
        try:
            final_params, losses, training_time = train_with_optimizer_hyperparams(
                init_params, eval_points, target, loss_fn, projection_fn,
                optimizer_name, hyperparams, n_epochs, use_projection=True
            )
            
            final_loss = losses[-1]
            
            if final_loss < best_loss:
                best_loss = final_loss
                best_hyperparams = hyperparams
                best_training_time = training_time
                best_losses = losses
            
            if i % 10 == 0:
                print(f"    Progress: {i+1}/{len(hyperparam_combinations)} - Best loss: {best_loss:.6f}")
                
        except Exception as e:
            print(f"    Error with hyperparams {hyperparams}: {e}")
            continue
    
    print(f"  Best {optimizer_name.upper()} hyperparams: {best_hyperparams}")
    print(f"  Best loss: {best_loss:.6f}")
    
    return {
        'best_hyperparams': best_hyperparams,
        'best_loss': best_loss,
        'best_training_time': best_training_time,
        'best_losses': best_losses
    }

def plot_hyperparameter_tuning_results(results: Dict, save_path: str = "hyperparameter_tuning_results.png"):
    """Plot the hyperparameter tuning results."""
    plt.figure(figsize=(20, 12))
    
    optimizers = list(results.keys())
    models = list(set([key.split('_')[1] for key in results.keys()]))
    
    # Plot 1: Best loss comparison
    plt.subplot(2, 4, 1)
    best_losses = []
    opt_names = []
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                best_losses.append(results[key]['best_loss'])
                opt_names.append(f"{optimizer.upper()}\n({model})")
    
    bars = plt.bar(range(len(best_losses)), best_losses, alpha=0.7)
    plt.xlabel('Optimizer (Model)')
    plt.ylabel('Best Loss')
    plt.title('Best Loss After Hyperparameter Tuning')
    plt.xticks(range(len(opt_names)), opt_names, rotation=45)
    plt.yscale('log')
    
    # Add value labels on bars
    for bar, loss_val in zip(bars, best_losses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1e-6,
                f'{loss_val:.6f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Training time comparison
    plt.subplot(2, 4, 2)
    training_times = []
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                training_times.append(results[key]['best_training_time'])
    
    bars = plt.bar(range(len(training_times)), training_times, alpha=0.7)
    plt.xlabel('Optimizer (Model)')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time After Hyperparameter Tuning')
    plt.xticks(range(len(opt_names)), opt_names, rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Convergence curves for best hyperparameters
    plt.subplot(2, 4, 3)
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                losses = results[key]['best_losses']
                epochs = range(len(losses))
                plt.plot(epochs, losses, label=f'{optimizer.upper()} ({model})', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Convergence with Best Hyperparameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 4: Hyperparameter analysis
    plt.subplot(2, 4, 4)
    plt.axis('off')
    
    # Create hyperparameter summary text
    summary_text = "BEST HYPERPARAMETERS\n\n"
    
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                hyperparams = results[key]['best_hyperparams']
                best_loss = results[key]['best_loss']
                training_time = results[key]['best_training_time']
                
                summary_text += f"{optimizer.upper()} ({model}):\n"
                summary_text += f"  Loss: {best_loss:.6f}\n"
                summary_text += f"  Time: {training_time:.1f}s\n"
                summary_text += f"  Params: {hyperparams}\n\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # Plot 5: Optimizer ranking
    plt.subplot(2, 4, 5)
    optimizer_rankings = {}
    for optimizer in optimizers:
        opt_losses = []
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                opt_losses.append(results[key]['best_loss'])
        if opt_losses:
            optimizer_rankings[optimizer] = np.mean(opt_losses)
    
    sorted_opts = sorted(optimizer_rankings.items(), key=lambda x: x[1])
    opt_names = [opt.upper() for opt, _ in sorted_opts]
    avg_losses = [loss for _, loss in sorted_opts]
    
    bars = plt.bar(range(len(opt_names)), avg_losses, alpha=0.7)
    plt.xlabel('Optimizer')
    plt.ylabel('Average Best Loss')
    plt.title('Optimizer Ranking (Average)')
    plt.xticks(range(len(opt_names)), opt_names)
    plt.yscale('log')
    
    # Add value labels on bars
    for bar, loss_val in zip(bars, avg_losses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1e-6,
                f'{loss_val:.6f}', ha='center', va='bottom')
    
    # Plot 6: Model comparison
    plt.subplot(2, 4, 6)
    model_comparison = {}
    for model in models:
        model_losses = []
        for optimizer in optimizers:
            key = f"{optimizer}_{model}"
            if key in results:
                model_losses.append(results[key]['best_loss'])
        if model_losses:
            model_comparison[model] = np.mean(model_losses)
    
    model_names = list(model_comparison.keys())
    model_avg_losses = list(model_comparison.values())
    
    bars = plt.bar(model_names, model_avg_losses, alpha=0.7, color=['lightblue', 'lightcoral'])
    plt.xlabel('Model')
    plt.ylabel('Average Best Loss')
    plt.title('Model Comparison (Average)')
    plt.yscale('log')
    
    # Add value labels on bars
    for bar, loss_val in zip(bars, model_avg_losses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1e-6,
                f'{loss_val:.6f}', ha='center', va='bottom')
    
    # Plot 7: Training time vs Loss
    plt.subplot(2, 4, 7)
    times = []
    losses = []
    labels = []
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                times.append(results[key]['best_training_time'])
                losses.append(results[key]['best_loss'])
                labels.append(f"{optimizer.upper()}\n({model})")
    
    plt.scatter(times, losses, alpha=0.7, s=100)
    for i, label in enumerate(labels):
        plt.annotate(label, (times[i], losses[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Best Loss')
    plt.title('Training Time vs Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Summary statistics
    plt.subplot(2, 4, 8)
    plt.axis('off')
    
    # Create final summary
    final_summary = "FINAL SUMMARY\n\n"
    
    # Best overall
    all_results = []
    for optimizer in optimizers:
        for model in models:
            key = f"{optimizer}_{model}"
            if key in results:
                all_results.append((key, results[key]['best_loss'], results[key]['best_training_time']))
    
    all_results.sort(key=lambda x: x[1])  # Sort by loss
    
    final_summary += "TOP 5 COMBINATIONS:\n"
    for i, (key, loss, time_val) in enumerate(all_results[:5]):
        final_summary += f"{i+1}. {key}: {loss:.6f} ({time_val:.1f}s)\n"
    
    final_summary += f"\nBest Overall: {all_results[0][0]}\n"
    final_summary += f"Best Loss: {all_results[0][1]:.6f}\n"
    final_summary += f"Training Time: {all_results[0][2]:.1f}s\n"
    
    plt.text(0.1, 0.9, final_summary, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return save_path

def main():
    """Main hyperparameter tuning and comparison function."""
    print("=" * 70)
    print("RBF Model Hyperparameter Tuning")
    print("2D Sine Wave Function with 25x25 Kernels")
    print("Optimizing Adam, AdamW, SGD, and L-BFGS")
    print("=" * 70)
    
    # Training parameters
    n_epochs = 500  # Shorter for hyperparameter tuning
    n_kernels = 25
    
    # Optimizer configurations
    optimizers = ['adam', 'adamw', 'sgd', 'lbfgs']
    models = ['standard', 'shape']
    
    print(f"Training Parameters:")
    print(f"- Epochs: {n_epochs}")
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
    
    # Perform hyperparameter tuning for all combinations
    results = {}
    
    for optimizer in optimizers:
        for model in models:
            print(f"\nHyperparameter tuning for {optimizer.upper()} with {model} model...")
            print("-" * 60)
            
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
            
            # Perform hyperparameter tuning
            tuning_result = hyperparameter_tuning(
                init_params, eval_points, target, loss_fn, projection_fn,
                optimizer, n_epochs
            )
            
            results[f"{optimizer}_{model}"] = tuning_result
    
    print("\nHyperparameter tuning completed!")
    print(f"Best Results:")
    for name, result in results.items():
        print(f"- {name}: {result['best_loss']:.6f} ({result['best_training_time']:.1f}s)")
    
    # Plot comparison
    print("\nCreating comparison plots...")
    save_path = plot_hyperparameter_tuning_results(results)
    print(f"Comparison plot saved to: {save_path}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)
    
    # Best performing optimizer per model
    for model in models:
        model_results = {k: v for k, v in results.items() if k.endswith(f"_{model}")}
        best_opt = min(model_results.keys(), key=lambda x: model_results[x]['best_loss'])
        print(f"\nBest optimizer for {model} model: {best_opt.split('_')[0].upper()}")
        print(f"Best final loss: {model_results[best_opt]['best_loss']:.6f}")
        print(f"Best hyperparameters: {model_results[best_opt]['best_hyperparams']}")
    
    # Overall best performing combination
    best_combination = min(results.keys(), key=lambda x: results[x]['best_loss'])
    print(f"\nBest overall combination: {best_combination}")
    print(f"Best final loss: {results[best_combination]['best_loss']:.6f}")
    print(f"Best hyperparameters: {results[best_combination]['best_hyperparams']}")
    
    # Optimizer ranking
    print(f"\nOptimizer Ranking (by average best loss):")
    opt_rankings = {}
    for optimizer in optimizers:
        opt_losses = [results[f"{optimizer}_{model}"]['best_loss'] for model in models]
        avg_loss = np.mean(opt_losses)
        opt_rankings[optimizer] = avg_loss
    
    sorted_opts = sorted(opt_rankings.items(), key=lambda x: x[1])
    for i, (opt, avg_loss) in enumerate(sorted_opts):
        print(f"{i+1}. {opt.upper()}: {avg_loss:.6f}")
    
    # Training time analysis
    print(f"\nTraining Time Analysis:")
    for optimizer in optimizers:
        opt_times = [results[f"{optimizer}_{model}"]['best_training_time'] for model in models]
        avg_time = np.mean(opt_times)
        print(f"- {optimizer.upper()}: {avg_time:.1f}s average")
    
    print("\nHyperparameter tuning and comparison completed successfully!")

if __name__ == "__main__":
    main()
