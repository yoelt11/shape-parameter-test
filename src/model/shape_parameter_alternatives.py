import jax
import jax.numpy as jnp

# Original transform (for comparison) - Updated to log-scale
def transform_original(epsilon):
    """Original shape parameter transform - now returns log-scale values."""
    epsilon_tensor = jnp.sin(jnp.array(epsilon)) * 3
    log_sx = epsilon_tensor                           # Log scale in x
    log_sy = -epsilon_tensor                          # Log scale in y (inverse relationship)
    theta = jnp.sin(epsilon_tensor) * jnp.pi           # Orientation (radians)
    return log_sx, log_sy, theta

# Alternative 1: Linear mapping with better scaling - Updated to log-scale
def transform_linear(epsilon):
    """Linear mapping with improved scaling for faster convergence.
    
    This transform uses linear relationships that are easier for the optimizer
    to learn and may lead to faster convergence. Now returns log-scale values.
    """
    # Map epsilon to reasonable log-scale ranges
    log_sx = jnp.tanh(epsilon) * 2                     # Log scale x: [-2, 2]
    log_sy = jnp.tanh(-epsilon) * 2                    # Log scale y: [-2, 2] (inverse relationship)
    theta = jnp.tanh(epsilon) * jnp.pi                 # Orientation: [-π, π]
    return log_sx, log_sy, theta

# Alternative 2: Exponential mapping with controlled growth - Updated to log-scale
def transform_exponential(epsilon):
    """Exponential mapping with controlled growth for better exploration.
    
    This transform provides exponential scaling while keeping values
    in reasonable bounds, potentially leading to better exploration
    of the parameter space. Now returns log-scale values.
    """
    # Use controlled log-scale ranges
    log_sx = jnp.clip(epsilon, -2, 2)                  # Log scale x: [-2, 2]
    log_sy = jnp.clip(-epsilon, -2, 2)                 # Log scale y: [-2, 2] (inverse relationship)
    theta = jnp.tanh(epsilon * 0.5) * jnp.pi          # Orientation: [-π, π] with slower change
    return log_sx, log_sy, theta

# Alternative 3: Polynomial mapping with smooth transitions - Updated to log-scale
def transform_polynomial(epsilon):
    """Polynomial mapping with smooth transitions for stable optimization.
    
    This transform uses polynomial relationships that provide smooth
    transitions and may lead to more stable optimization. Now returns log-scale values.
    """
    # Use polynomial relationships for smooth log-scale mapping
    log_sx = jnp.tanh(epsilon) * 1.5                   # Log scale x: [-1.5, 1.5] with smooth transition
    log_sy = jnp.tanh(-epsilon) * 1.5                  # Log scale y: [-1.5, 1.5] (inverse relationship)
    theta = epsilon * jnp.pi / 2                       # Orientation: [-π/2, π/2] linear mapping
    return log_sx, log_sy, theta

# Alternative 4: Adaptive scaling based on domain characteristics - Updated to log-scale
def transform_adaptive(epsilon, domain_size=2.0):
    """Adaptive scaling that adjusts based on domain characteristics.
    
    This transform adapts the scaling based on the problem domain,
    potentially leading to better convergence for specific problems. Now returns log-scale values.
    """
    # Adaptive log-scale mapping based on domain size
    max_log_scale = jnp.log(domain_size / 4)  # Reasonable maximum log scale
    min_log_scale = jnp.log(0.01)            # Minimum log scale for numerical stability
    
    # Map epsilon to adaptive log-scale ranges
    log_sx = min_log_scale + (max_log_scale - min_log_scale) * jax.nn.sigmoid(epsilon)
    log_sy = min_log_scale + (max_log_scale - min_log_scale) * jax.nn.sigmoid(-epsilon)
    theta = jnp.tanh(epsilon) * jnp.pi
    return log_sx, log_sy, theta

# Alternative 5: Multi-scale transform for better feature capture - Updated to log-scale
def transform_multiscale(epsilon):
    """Multi-scale transform that can capture features at different scales.
    
    This transform provides multiple scales simultaneously,
    potentially leading to better feature capture and faster convergence. Now returns log-scale values.
    """
    # Create multiple log-scales for better feature capture
    scale_factor = jax.nn.sigmoid(epsilon)
    
    # Primary log-scale
    log_sx_primary = jnp.log(0.1) + jnp.log(2.0) * scale_factor
    log_sy_primary = jnp.log(0.1) + jnp.log(2.0) * (1 - scale_factor)
    
    # Secondary log-scale (for fine-tuning)
    log_sx_fine = jnp.log(0.05) + 0.5 * jnp.tanh(epsilon * 2)
    log_sy_fine = jnp.log(0.05) + 0.5 * jnp.tanh(-epsilon * 2)
    
    # Combine log-scales (additive in log space)
    log_sx = log_sx_primary + log_sx_fine
    log_sy = log_sy_primary + log_sy_fine
    
    theta = jnp.tanh(epsilon) * jnp.pi
    return log_sx, log_sy, theta

# Alternative 6: Frequency-based transform for oscillatory functions - Updated to log-scale
def transform_frequency(epsilon):
    """Frequency-based transform optimized for oscillatory functions.
    
    This transform is specifically designed for functions with
    oscillatory behavior (like sine waves), potentially leading
    to faster convergence for such problems. Now returns log-scale values.
    """
    # Frequency-based scaling for oscillatory functions
    freq_factor = jax.nn.sigmoid(epsilon)
    
    # Log-scale based on frequency considerations
    log_sx = jnp.log(0.2) + jnp.log(1.8) * freq_factor
    log_sy = jnp.log(0.2) + jnp.log(1.8) * (1 - freq_factor)
    
    # Orientation that adapts to frequency
    theta = jnp.sin(epsilon) * jnp.pi / 2
    
    return log_sx, log_sy, theta
