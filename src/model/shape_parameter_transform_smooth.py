import jax
import jax.numpy as jnp

# Original transform (for comparison) - Updated to log-scale
def transform_original(epsilon):
    """Original transform with sin - can be oscillatory. Now returns log-scale values."""
    epsilon_tensor = jnp.sin(jnp.array(epsilon)) * 3
    log_sx = epsilon_tensor                           # Log scale in x
    log_sy = -epsilon_tensor                          # Log scale in y (inverse relationship)
    theta = jnp.sin(epsilon_tensor) * jnp.pi           # Orientation (radians)
    return log_sx, log_sy, theta

# Alternative 1: Smooth tanh-based transform - Updated to log-scale
def transform_smooth_tanh(epsilon):
    """Smooth transform using tanh - provides smooth, bounded transformation. Now returns log-scale values."""
    epsilon_tensor = jnp.tanh(jnp.array(epsilon)) * 2  # Smooth, bounded to [-2, 2]
    log_sx = epsilon_tensor                             # Log scale in x: [-2, 2]
    log_sy = -epsilon_tensor                            # Log scale in y: [-2, 2] (inverse relationship)
    theta = jnp.tanh(epsilon_tensor) * jnp.pi          # Orientation: [-π, π]
    return log_sx, log_sy, theta

# Alternative 2: Sigmoid-based transform - Updated to log-scale
def transform_smooth_sigmoid(epsilon):
    """Smooth transform using sigmoid - provides smooth, monotonic transformation. Now returns log-scale values."""
    epsilon_tensor = jax.nn.sigmoid(jnp.array(epsilon)) * 4 - 2  # Map to [-2, 2]
    log_sx = epsilon_tensor                             # Log scale in x: [-2, 2]
    log_sy = -epsilon_tensor                            # Log scale in y: [-2, 2] (inverse relationship)
    theta = jnp.tanh(epsilon_tensor) * jnp.pi          # Orientation
    return log_sx, log_sy, theta

# Alternative 3: Linear + smooth transform - Updated to log-scale
def transform_smooth_linear(epsilon):
    """Smooth transform with linear component for better gradient flow. Now returns log-scale values."""
    # Combine linear and smooth components
    linear_component = epsilon * 0.5                    # Linear scaling
    smooth_component = jnp.tanh(epsilon) * 1.5         # Smooth component
    epsilon_tensor = linear_component + smooth_component # Combined: [-2, 2]
    
    log_sx = epsilon_tensor                             # Log scale in x: [-2, 2]
    log_sy = -epsilon_tensor                            # Log scale in y: [-2, 2] (inverse relationship)
    theta = jnp.tanh(epsilon_tensor) * jnp.pi          # Orientation
    return log_sx, log_sy, theta

# Alternative 4: Polynomial-based smooth transform - Updated to log-scale
def transform_smooth_polynomial(epsilon):
    """Smooth transform using polynomial - provides smooth, controlled transformation. Now returns log-scale values."""
    # Use cubic polynomial for smooth transformation
    epsilon_tensor = (epsilon**3 + epsilon) * 0.5       # Smooth polynomial
    epsilon_tensor = jnp.clip(epsilon_tensor, -2, 2)   # Bound to [-2, 2]
    
    log_sx = epsilon_tensor                             # Log scale in x: [-2, 2]
    log_sy = -epsilon_tensor                            # Log scale in y: [-2, 2] (inverse relationship)
    theta = jnp.tanh(epsilon_tensor) * jnp.pi          # Orientation
    return log_sx, log_sy, theta

# Alternative 5: Exponential smoothing transform - Updated to log-scale
def transform_smooth_exponential(epsilon):
    """Smooth transform using exponential smoothing - provides smooth, controlled growth. Now returns log-scale values."""
    # Use exponential smoothing for better gradient flow
    epsilon_tensor = jnp.sign(epsilon) * (1 - jnp.exp(-jnp.abs(epsilon))) * 2
    epsilon_tensor = jnp.clip(epsilon_tensor, -2, 2)   # Bound to [-2, 2]
    
    log_sx = epsilon_tensor                             # Log scale in x: [-2, 2]
    log_sy = -epsilon_tensor                            # Log scale in y: [-2, 2] (inverse relationship)
    theta = jnp.tanh(epsilon_tensor) * jnp.pi          # Orientation
    return log_sx, log_sy, theta

# Alternative 6: Adaptive smooth transform - Updated to log-scale
def transform_smooth_adaptive(epsilon):
    """Adaptive smooth transform that adjusts based on epsilon magnitude. Now returns log-scale values."""
    # Adaptive transformation based on magnitude
    magnitude = jnp.abs(epsilon)
    
    # Use jnp.where for vectorized conditional
    epsilon_tensor = jnp.where(
        magnitude < 1.0,
        epsilon * 1.5,  # For small values, use linear transformation
        jnp.sign(epsilon) * (2 - jnp.exp(-magnitude + 1))  # For large values, use smooth saturation
    )
    
    epsilon_tensor = jnp.clip(epsilon_tensor, -2, 2)   # Bound to [-2, 2]
    
    log_sx = epsilon_tensor                             # Log scale in x: [-2, 2]
    log_sy = -epsilon_tensor                            # Log scale in y: [-2, 2] (inverse relationship)
    theta = jnp.tanh(epsilon_tensor) * jnp.pi          # Orientation
    return log_sx, log_sy, theta

# Alternative 7: Multi-scale smooth transform - Updated to log-scale
def transform_smooth_multiscale(epsilon):
    """Multi-scale smooth transform for better feature capture. Now returns log-scale values."""
    # Multiple smooth scales
    coarse_scale = jnp.tanh(epsilon) * 1.5             # Coarse features
    fine_scale = jnp.tanh(epsilon * 0.5) * 0.5         # Fine features
    epsilon_tensor = coarse_scale + fine_scale          # Combined scales
    epsilon_tensor = jnp.clip(epsilon_tensor, -2, 2)   # Bound to [-2, 2]
    
    log_sx = epsilon_tensor                             # Log scale in x: [-2, 2]
    log_sy = -epsilon_tensor                            # Log scale in y: [-2, 2] (inverse relationship)
    theta = jnp.tanh(epsilon_tensor) * jnp.pi          # Orientation
    return log_sx, log_sy, theta

# Alternative 8: Gradient-optimized smooth transform - Updated to log-scale
def transform_smooth_gradient_optimized(epsilon):
    """Smooth transform designed for optimal gradient flow. Now returns log-scale values."""
    # Design for better gradient properties
    epsilon_tensor = jnp.tanh(epsilon * 0.8) * 2.5     # Smooth, well-bounded
    epsilon_tensor = epsilon_tensor + 0.1 * epsilon     # Add small linear component
    
    log_sx = epsilon_tensor                             # Log scale in x: [-2.5, 2.5]
    log_sy = -epsilon_tensor                            # Log scale in y: [-2.5, 2.5] (inverse relationship)
    theta = jnp.tanh(epsilon_tensor) * jnp.pi          # Orientation
    return log_sx, log_sy, theta
