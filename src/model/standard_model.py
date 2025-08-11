import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from dataclasses import dataclass
from functools import partial
#from structures import Lambda,  computational_domain_2D


@dataclass
class Lambda:
    """RBF parameters dataclass."""
    mus: jnp.ndarray  # (K, 2)
    log_sigmas: jnp.ndarray  # (K, 2)
    angles: jnp.ndarray  # (K,)
    weights: jnp.ndarray  # (K,)
    
@partial(jax.jit, static_argnums=(4,))
def precompute_params(mus: jnp.ndarray, log_sigmas: jnp.ndarray, angles: jnp.ndarray, weights: jnp.ndarray, epsilon: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute parameters for rotation covariance matrices with independent parameters per kernel.
    Vectorized implementation for better performance.
    
    Args:
        mus: Array of shape (K, 2) containing the means of K Gaussian kernels
        log_sigmas: Array of shape (K, 2) containing the log of the standard deviations for each kernel
        angles: Array of shape (K,) containing the rotation angle for each kernel
        weights: Array of shape (K,) containing the weights for each kernel
        epsilon: Small constant for numerical stability
        
    Returns:
        Tuple containing:
        - mus: Array of shape (K, 2) containing the means
        - weights: Array of shape (K,) containing the weights
        - inv_covs: Array of shape (K, 2, 2) containing the inverse covariance matrices
    """
    # Process sigmas and angles
    sigmas = jnp.exp(log_sigmas)  # (K, 2)
    squared_sigmas = sigmas**2    # (K, 2)
    angles = jax.nn.sigmoid(angles) * 2 * jnp.pi  # (K,)
    
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
    Generate RBF solutions for batched lambda parameters with kernel group support.
    
    Args:
        eval_points: Tuple of (X, Y) meshgrid arrays for evaluation
        lambda_params: Array containing batched parameters where:
            - If shape (B, K, 6): Standard batched RBF with K kernels per sample (KERNEL GROUPS)
            - If shape (B, 6): Single kernel per sample (LEGACY SUPPORT)
            Parameter format:
            - [..., 0:2]: mus (mu_x, mu_y)
            - [..., 2:4]: log_sigmas (log_sigma_x, log_sigma_y)
            - [..., 4]: angles
            - [..., 5]: weights
    
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
        # Legacy case: (B, 6) - single kernel per sample
        # Reshape to (B, 1, 6) to match kernel group format
        lambda_params = lambda_params[:, None, :]
    
    # Extract components from the concatenated tensor
    mus = lambda_params[:, :, 0:2]           # (B, K, 2)
    log_sigmas = lambda_params[:, :, 2:4]    # (B, K, 2)
    angles = lambda_params[:, :, 4]          # (B, K)
    weights = lambda_params[:, :, 5]         # (B, K)
    
    def single_sample_rbf(mus, log_sigmas, angles, weights):
        """Process a single sample - combine ALL kernels into one solution.
        
        For kernel groups: This combines all kernels in the group into a single RBF solution.
        This is the key fix - instead of treating kernels independently, we combine them.
        """
        # Precompute parameters for ALL kernels in this sample
        mus_proc, weights_proc, inv_covs = precompute_params(mus, log_sigmas, angles, weights)
        
        # Evaluate using ALL kernels combined - this creates ONE solution per sample
        # The fn_evaluate function already handles multiple kernels and combines them
        return fn_evaluate(eval_grid, mus_proc, weights_proc, inv_covs)
    
    # Use vmap to process all samples in the batch in parallel
    # vmap over the first axis (batch dimension) of each component
    batched_rbf = jax.vmap(single_sample_rbf, in_axes=(0, 0, 0, 0))
    
    return batched_rbf(mus, log_sigmas, angles, weights)

# # ExVperiment 6
def apply_projection(lambdas_0: jnp.ndarray, eval_points: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Apply projection to the parameters."""
    # lambdas_0 is of shape (K, 6) after vmapping
    n_points = eval_points[0].shape[0]

    # Project mus using array updating
    lambdas_0 = lambdas_0.at[:, 0:2].set(jnp.clip(lambdas_0[:, 0:2], -1.0, 1.0))
    
    # Calculate domain characteristics
    domain_width = 1.75  # [0.5, 0.5] has width 1 Experiment 5
    avg_point_spacing = domain_width / jnp.sqrt(n_points)
    
    # Set minimum sigma to be related to the point spacing (to capture local features)
    min_sigma = avg_point_spacing / 2
    # Set maximum sigma to be related to the domain size (to capture global features)
    max_sigma = domain_width / 2
    
    # Apply the bounds in log space using array updating
    lambdas_0 = lambdas_0.at[:, 2:4].set(jnp.clip(
        lambdas_0[:, 2:4], 
        jnp.log(min_sigma), 
        jnp.log(max_sigma)
    ))

    return lambdas_0
