import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from dataclasses import dataclass
from functools import partial
from .shape_parameter_transform import transform
#from structures import Lambda,  computational_domain_2D


@dataclass
class Lambda:
    """RBF parameters dataclass with shape parameter transform."""
    mus: jnp.ndarray  # (K, 2)
    epsilons: jnp.ndarray  # (K,) - shape parameters
    weights: jnp.ndarray  # (K,)
    
@partial(jax.jit, static_argnums=(3,))
def precompute_params(mus: jnp.ndarray, epsilons: jnp.ndarray, weights: jnp.ndarray, epsilon: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute parameters using shape parameter transform.
    Vectorized implementation for better performance.
    
    Args:
        mus: Array of shape (K, 2) containing the means of K Gaussian kernels
        epsilons: Array of shape (K,) containing the shape parameters
        weights: Array of shape (K,) containing the weights for each kernel
        epsilon: Small constant for numerical stability
        
    Returns:
        Tuple containing:
        - mus: Array of shape (K, 2) containing the means
        - weights: Array of shape (K,) containing the weights
        - inv_covs: Array of shape (K, 2, 2) containing the inverse covariance matrices
    """
    # Apply shape parameter transform to get log_sx, log_sy, theta for all kernels
    log_sx, log_sy, theta = transform(epsilons)  # All return (K,)
    
    # Convert to sigmas (standard deviations) using exponential
    sigmas_x = jnp.exp(log_sx)  # (K,)
    sigmas_y = jnp.exp(log_sy)  # (K,)
    
    # Create sigmas array for compatibility
    sigmas = jnp.stack([sigmas_x, sigmas_y], axis=1)  # (K, 2)
    squared_sigmas = sigmas**2  # (K, 2)
    
    # Normalize angles to [0, 2π]
    angles = theta % (2 * jnp.pi)  # (K,)
    
    # Compute all rotation matrices at once
    cos_angles = jnp.cos(angles)  # (K,)
    sin_angles = jnp.sin(angles)  # (K,)
    
    # Create rotation matrices for all kernels at once: (K, 2, 2)
    R = jnp.stack([
        jnp.stack([cos_angles, -sin_angles], axis=1),  # (K, 2)
        jnp.stack([sin_angles, cos_angles], axis=1)    # (K, 2)
    ], axis=2)  # Result shape: (K, 2, 2)
    
    # Create inverse diagonal matrices for all kernels: (K, 2, 2)
    diag_inv = jnp.zeros((mus.shape[0], 2, 2))
    diag_inv = diag_inv.at[:, 0, 0].set(1.0 / (squared_sigmas[:, 0] + epsilon))
    diag_inv = diag_inv.at[:, 1, 1].set(1.0 / (squared_sigmas[:, 1] + epsilon))
    
    # Compute all inverse covariance matrices at once: (K, 2, 2)
    inv_covs = jnp.einsum('kij,kjl,klm->kim', R, diag_inv, R.transpose((0, 2, 1)))
    scaled_weights = weights #jax.nn.tanh(weights)
    return mus, scaled_weights, inv_covs

@jax.jit
def fn_evaluate(X: jnp.ndarray, mus: jnp.ndarray, weights: jnp.ndarray, inv_covs: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the Gaussian kernel function at points X using efficient batched operations.
    
    Args:
        X: Array of shape (N, 2) containing the evaluation points
        mus: Array of shape (K, 2) containing the means of K Gaussian kernels
        weights: Array of shape (K,) containing the weights for each kernel
        inv_covs: Array of shape (K, 2, 2) containing the inverse covariance matrices
    """
    # Compute all differences at once: (N, K, 2)
    diff = X[:, None, :] - mus[None, :, :]
    
    # Compute quadratic forms efficiently using einsum
    quad = jnp.einsum('nki,kij,nkj->nk', diff, inv_covs, diff)
    
    # Compute all kernel values at once
    phi = jnp.exp(-0.5 * quad)
    
    # Weighted sum
    return jnp.dot(phi, weights)  # (N,)

@jax.jit
def fn_derivatives(X: jnp.ndarray, mus: jnp.ndarray, weights: jnp.ndarray, inv_covs: jnp.ndarray, hess_epsilon: float=1e-8) -> Dict:
    """Compute derivatives of the Gaussian kernel function at points X using efficient batched operations.
    
    Args:
        X: Array of shape (N, 2) containing the evaluation points
        mus: Array of shape (K, 2) containing the means
        weights: Array of shape (K,) containing the weights
        inv_covs: Array of shape (K, 2, 2) containing the inverse covariance matrices
        hess_epsilon: Small constant for numerical stability
        
    Returns:
        Dictionary containing:
        - 'u': Function values
        - 'u_x': First derivative w.r.t. x
        - 'u_y': First derivative w.r.t. y (time)
        - 'u_xx': Second derivative w.r.t. x
        - 'u_yy': Second derivative w.r.t. y (time)
        - 'u_xy': Mixed derivative
    """
    # Compute all differences at once: (N, K, 2)
    diff = X[:, None, :] - mus[None, :, :]
    
    # Compute inv_covs @ diff for all points and kernels at once
    inv_diff = jnp.einsum('kij,nkj->nki', inv_covs, diff)  # (N, K, 2)
    
    # Compute quadratic forms for all points and kernels
    quad = jnp.sum(diff * inv_diff, axis=2)  # (N, K)
    phi = jnp.exp(-0.5 * quad)  # (N, K)
    
    # Function values
    u = jnp.dot(phi, weights)  # (N,)
    
    # First derivatives using batched operations
    grad_phi = -phi[:, :, None] * inv_diff  # (N, K, 2)
    u_x = jnp.sum(weights[None, :] * grad_phi[:, :, 0], axis=1)  # (N,)
    u_y = jnp.sum(weights[None, :] * grad_phi[:, :, 1], axis=1)  # (N,)
    
    # Optimized Hessian calculation using batched operations
    inv_diff_outer = inv_diff[:, :, :, None] * inv_diff[:, :, None, :]  # (N, K, 2, 2)
    hess_terms = weights[None, :, None, None] * phi[:, :, None, None] * (inv_diff_outer - inv_covs[None, :, :, :])
    
    # Sum over kernels and add stability term
    H = jnp.sum(hess_terms, axis=1) + hess_epsilon * jnp.eye(2)[None, :, :]  # (N, 2, 2)
    
    return {
        'u': u,
        'u_x': u_x,
        'u_y': u_y,
        'u_xx': H[:, 0, 0],
        'u_yy': H[:, 1, 1],
        'u_xy': H[:, 0, 1]
    }

@jax.jit
def rotation_matrix(angle: float) -> jnp.ndarray:
    """Compute 2D rotation matrix for given angle. 
    Note: This function is kept for reference but no longer used in the optimized implementation."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[c, -s], [s, c]])

@jax.jit
def generate_rbf_solutions(eval_points: Tuple[jnp.ndarray, jnp.ndarray], lambda_params: jnp.ndarray) -> jnp.ndarray:
    """
    Generate RBF solutions for batched lambda parameters with shape parameter transform.
    
    Args:
        eval_points: Tuple of (X, Y) meshgrid arrays for evaluation
        lambda_params: Array containing batched parameters where:
            - If shape (B, K, 4): Standard batched RBF with K kernels per sample (KERNEL GROUPS)
            - If shape (B, 4): Single kernel per sample (LEGACY SUPPORT)
            Parameter format:
            - [..., 0:2]: mus (mu_x, mu_y)
            - [..., 2]: epsilons (shape parameter)
            - [..., 3]: weights
    
    Returns:
        Array of shape (B, N) containing the RBF solutions for each batch
        - For kernel groups: Each solution combines ALL kernels in the group
        - For single kernels: Each solution uses 1 kernel
    """
    
    # Unpack eval_points and create evaluation grid
    X, Y = eval_points
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    eval_grid = jnp.stack([X_flat, Y_flat], axis=-1)  # Shape: (N, 2)
    
    # Handle different input shapes
    original_shape = lambda_params.shape
    is_kernel_groups = len(original_shape) == 3
    
    if len(lambda_params.shape) == 2:
        # Legacy case: (B, 4) - single kernel per sample
        # Reshape to (B, 1, 4) to match kernel group format
        lambda_params = lambda_params[:, None, :]
    
    # Extract components from the concatenated tensor
    mus = lambda_params[:, :, 0:2]           # (B, K, 2)
    epsilons = lambda_params[:, :, 2]        # (B, K)
    weights = lambda_params[:, :, 3]         # (B, K)
    
    def single_sample_rbf(mus, epsilons, weights):
        """Process a single sample - combine ALL kernels into one solution.
        
        For kernel groups: This combines all kernels in the group into a single RBF solution.
        This is the key fix - instead of treating kernels independently, we combine them.
        """
        # Precompute parameters for ALL kernels in this sample using shape transform
        mus_proc, weights_proc, inv_covs = precompute_params(mus, epsilons, weights)
        
        # Evaluate using ALL kernels combined - this creates ONE solution per sample
        # The fn_evaluate function already handles multiple kernels and combines them
        return fn_evaluate(eval_grid, mus_proc, weights_proc, inv_covs)
    
    # Use vmap to process all samples in the batch in parallel
    # vmap over the first axis (batch dimension) of each component
    batched_rbf = jax.vmap(single_sample_rbf, in_axes=(0, 0, 0))
    
    return batched_rbf(mus, epsilons, weights)

# # ExVperiment 6
def apply_projection(lambdas_0: jnp.ndarray, eval_points: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Apply projection to the parameters with shape parameter transform."""
    # lambdas_0 is of shape (K, 4) after vmapping (mus: 2, epsilon: 1, weight: 1)
    n_points = eval_points[0].shape[0]

    # Project mus using array updating
    lambdas_0 = lambdas_0.at[:, 0:2].set(jnp.clip(lambdas_0[:, 0:2], -1.0, 1.0))
    
    # Calculate domain characteristics
    domain_width = 1.75  # [0.5, 0.5] has width 1 Experiment 5
    avg_point_spacing = domain_width / jnp.sqrt(n_points)
    
    # For epsilon, we can apply bounds to control the range of the shape transform
    # The transform function uses sin(epsilon) * 3, so we can bound epsilon to control the output range
    # A reasonable bound would be [-π, π] to get the full range of the sine function
    lambdas_0 = lambdas_0.at[:, 2].set(jnp.clip(lambdas_0[:, 2], -jnp.pi, jnp.pi))

    return lambdas_0
