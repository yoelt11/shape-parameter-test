import jax.numpy as jnp


#def transform(epsilon):
#    """Eccentricity + mean scale transform - best performing alternative."""
#    mean_scale = jnp.sin(epsilon)  * 2.5  # cycles isotropically
#    eccentricity = 0.5 * jnp.sin(2 * epsilon)  # -0.5..0.5
#    log_sx = mean_scale + eccentricity
#    log_sy = mean_scale - eccentricity
#    theta = (epsilon % (2 * jnp.pi))
#    return log_sx, log_sy, theta

def transform(epsilon):
    #epsilon_tensor = jnp.sin(jnp.array(epsilon)) * 3  # vary
    epsilon_tensor = (jnp.array(epsilon) % (2 * jnp.pi)) 
    log_sx = jnp.cos(epsilon_tensor)                            # Scale in x
    log_sy = jnp.sin(epsilon_tensor)                          # Scale in y (note you wrote 'epsillon_tensor', assuming typo)
    #theta = jnp.sin(epsilon_tensor) * jnp.pi          # Orientation (radians)

    # Compute standard deviation
    #std_log = jnp.sqrt((log_sx**2 + log_sy**2) / 2.0 + 1e-8)  # avoid div by zero

    #log_sx_norm = log_sx / std_log
    #log_sy_norm = log_sy / std_log

    theta = (epsilon % (2 * jnp.pi))  # linear sweep from -π to π

    return log_sx, log_sy, theta

def transform_circular_sweep(epsilon):
    """Circular sweep in log-space for symmetric coverage."""
    r = 1.0  # radius controls log-scale amplitude
    log_sx = r * jnp.sin(epsilon)
    log_sy = r * jnp.cos(epsilon)
    theta = (epsilon % (2 * jnp.pi))  # full smooth rotation
    return log_sx, log_sy, theta


def transform_eccentricity(epsilon):
    """Use eccentricity + mean scale for better separation."""
    mean_scale = jnp.sin(epsilon)  # cycles isotropically
    eccentricity = 0.5 * jnp.sin(2 * epsilon)  # -0.5..0.5
    log_sx = mean_scale + eccentricity
    log_sy = mean_scale - eccentricity
    theta = (epsilon % (2 * jnp.pi))
    return log_sx, log_sy, theta


def transform_lissajous(epsilon):
    """Lissajous coverage for richer shape space exploration."""
    log_sx = jnp.sin(epsilon)          # slow oscillation
    log_sy = jnp.sin(2 * epsilon)      # faster oscillation
    theta = jnp.sin(3 * epsilon) * jnp.pi
    return log_sx, log_sy, theta


# Dictionary of all available transforms
TRANSFORMS = {
    'current': transform,
    'circular_sweep': transform_circular_sweep,
    'eccentricity': transform_eccentricity,
    'lissajous': transform_lissajous
}
