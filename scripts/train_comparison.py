import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Callable
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model.standard_model import generate_rbf_solutions as generate_standard_rbf
from model.rbf_model import generate_rbf_solutions as generate_shape_rbf
from model.standard_model import apply_projection as apply_standard_projection
from model.rbf_model import apply_projection as apply_shape_projection

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)
key = jax.random.PRNGKey(42)

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
                learning_rate: float = 0.01) -> Tuple[jnp.ndarray, list]:
    """Train the RBF model using AdamW optimizer."""
    
    # Create optimizer
    optimizer = optax.adamw(learning_rate=learning_rate)
    
    # Initialize optimizer state
    opt_state = optimizer.init(init_params)
    
    # Create gradient function
    grad_fn = jax.grad(loss_fn)
    
    # Training loop
    params = init_params
    losses = []
    
    for epoch in range(n_epochs):
        # Compute gradients
        grads = grad_fn(params, eval_points, target)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Apply projection
        params = projection_fn(params, eval_points)
        
        # Compute loss
        loss = loss_fn(params, eval_points, target)
        losses.append(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    return params, losses

def plot_comparison(standard_losses: list, shape_losses: list, save_path: str = "convergence_comparison.png"):
    """Plot the convergence comparison."""
    epochs = range(len(standard_losses))
    
    plt.figure(figsize=(12, 8))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, standard_losses, 'b-', label='Standard Model (6 params)', linewidth=2)
    plt.plot(epochs, shape_losses, 'r-', label='Shape Transform (4 params)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot loss difference
    plt.subplot(2, 2, 2)
    loss_diff = jnp.array(standard_losses) - jnp.array(shape_losses)
    plt.plot(epochs, loss_diff, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference (Standard - Shape)')
    plt.title('Loss Difference Over Time')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot final convergence (last 100 epochs)
    plt.subplot(2, 2, 3)
    final_epochs = epochs[-100:]
    final_standard = standard_losses[-100:]
    final_shape = shape_losses[-100:]
    plt.plot(final_epochs, final_standard, 'b-', label='Standard Model', linewidth=2)
    plt.plot(final_epochs, final_shape, 'r-', label='Shape Transform', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Final Convergence (Last 100 Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot parameter count comparison
    plt.subplot(2, 2, 4)
    n_kernels = 25
    standard_params = n_kernels * 6
    shape_params = n_kernels * 4
    reduction = ((standard_params - shape_params) / standard_params) * 100
    
    bars = plt.bar(['Standard Model', 'Shape Transform'], 
                   [standard_params, shape_params], 
                   color=['lightblue', 'lightcoral'], alpha=0.7)
    plt.ylabel('Total Parameters')
    plt.title(f'Parameter Count Comparison\n({reduction:.1f}% reduction)')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return save_path

def main():
    """Main training and comparison function."""
    print("=" * 60)
    print("RBF Model Convergence Comparison")
    print("2D Sine Wave Function with 25x25 Kernels")
    print("=" * 60)
    
    # Training parameters
    n_epochs = 1000
    learning_rate = 0.01
    n_kernels = 25
    
    print(f"Training Parameters:")
    print(f"- Epochs: {n_epochs}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Kernels: {n_kernels}x{n_kernels} = {n_kernels**2}")
    print(f"- Standard Model: {n_kernels**2 * 6} parameters")
    print(f"- Shape Transform: {n_kernels**2 * 4} parameters")
    print()
    
    # Create training data
    print("Creating training data...")
    train_points, target, eval_points = create_training_data(n_points=50)
    print(f"Training points: {train_points.shape}")
    print(f"Target shape: {target.shape}")
    print()
    
    # Initialize parameters
    print("Initializing parameters...")
    key1, key2 = jax.random.split(key)
    standard_params = initialize_standard_parameters(n_kernels**2, key1)
    shape_params = initialize_shape_parameters(n_kernels**2, key2)
    
    print(f"Standard params shape: {standard_params.shape}")
    print(f"Shape params shape: {shape_params.shape}")
    print()
    
    # Create loss functions
    print("Creating loss functions...")
    standard_loss_fn = create_loss_function(eval_points, target)
    shape_loss_fn = create_shape_loss_function(eval_points, target)
    
    # Train standard model
    print("Training Standard Model (6 parameters per kernel)...")
    print("-" * 50)
    final_standard_params, standard_losses = train_model(
        standard_params, eval_points, target, standard_loss_fn, 
        apply_standard_projection, n_epochs, learning_rate
    )
    
    print()
    print("Training Shape Transform Model (4 parameters per kernel)...")
    print("-" * 50)
    final_shape_params, shape_losses = train_model(
        shape_params, eval_points, target, shape_loss_fn,
        apply_shape_projection, n_epochs, learning_rate
    )
    
    print()
    print("Training completed!")
    print(f"Final Standard Loss: {standard_losses[-1]:.6f}")
    print(f"Final Shape Loss: {shape_losses[-1]:.6f}")
    print()
    
    # Plot comparison
    print("Creating comparison plots...")
    save_path = plot_comparison(standard_losses, shape_losses)
    print(f"Comparison plot saved to: {save_path}")
    
    # Additional analysis
    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS")
    print("=" * 60)
    
    # Final loss comparison
    final_standard = standard_losses[-1]
    final_shape = shape_losses[-1]
    improvement = ((final_standard - final_shape) / final_standard) * 100
    
    print(f"Final Loss Comparison:")
    print(f"- Standard Model: {final_standard:.6f}")
    print(f"- Shape Transform: {final_shape:.6f}")
    print(f"- Improvement: {improvement:.2f}%")
    
    # Convergence speed analysis
    standard_conv_epoch = next((i for i, loss in enumerate(standard_losses) if loss < 1e-3), -1)
    shape_conv_epoch = next((i for i, loss in enumerate(shape_losses) if loss < 1e-3), -1)
    
    print(f"\nConvergence Speed (Loss < 1e-3):")
    print(f"- Standard Model: {standard_conv_epoch if standard_conv_epoch != -1 else 'Not reached'}")
    print(f"- Shape Transform: {shape_conv_epoch if shape_conv_epoch != -1 else 'Not reached'}")
    
    # Parameter efficiency
    param_efficiency_standard = final_standard / (n_kernels**2 * 6)
    param_efficiency_shape = final_shape / (n_kernels**2 * 4)
    
    print(f"\nParameter Efficiency (Loss per parameter):")
    print(f"- Standard Model: {param_efficiency_standard:.8f}")
    print(f"- Shape Transform: {param_efficiency_shape:.8f}")
    
    print("\nTraining and comparison completed successfully!")

if __name__ == "__main__":
    main()
