import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from dataclasses import dataclass
from functools import partial
from .shape_parameter_alternatives import (
    transform_original, transform_linear, transform_exponential, 
    transform_polynomial, transform_adaptive, transform_multiscale, 
    transform_frequency
)

@dataclass
class Lambda:
    """RBF parameters dataclass with alternative shape parameter transforms."""
    mus: jnp.ndarray  # (K, 2)
    epsilons: jnp.ndarray  # (K,) - shape parameters
    weights: jnp.ndarray  # (K,)

def create_precompute_params(transform_fn):
    """Create a precompute_params function with a specific transform."""
    @partial(jax.jit, static_argnums=(3,))
    def precompute_params(mus: jnp.ndarray, epsilons: jnp.ndarray, weights: jnp.ndarray, epsilon: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Precompute parameters using the specified shape parameter transform.
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
        log_sx, log_sy, theta = transform_fn(epsilons)  # All return (K,)
        
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
        
        return mus, jax.nn.tanh(weights), inv_covs
    
    return precompute_params

# Create precompute functions for each transform
precompute_original = create_precompute_params(transform_original)
precompute_linear = create_precompute_params(transform_linear)
precompute_exponential = create_precompute_params(transform_exponential)
precompute_polynomial = create_precompute_params(transform_polynomial)
precompute_adaptive = create_precompute_params(transform_adaptive)
precompute_multiscale = create_precompute_params(transform_multiscale)
precompute_frequency = create_precompute_params(transform_frequency)

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

def create_generate_rbf_solutions(precompute_fn):
    """Create a generate_rbf_solutions function with a specific precompute function."""
    @jax.jit
    def generate_rbf_solutions(eval_points: Tuple[jnp.ndarray, jnp.ndarray], lambda_params: jnp.ndarray) -> jnp.ndarray:
        """
        Generate RBF solutions for batched lambda parameters with alternative shape parameter transform.
        
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
            mus_proc, weights_proc, inv_covs = precompute_fn(mus, epsilons, weights)
            
            # Evaluate using ALL kernels combined - this creates ONE solution per sample
            # The fn_evaluate function already handles multiple kernels and combines them
            return fn_evaluate(eval_grid, mus_proc, weights_proc, inv_covs)
        
        # Use vmap to process all samples in the batch in parallel
        # vmap over the first axis (batch dimension) of each component
        batched_rbf = jax.vmap(single_sample_rbf, in_axes=(0, 0, 0))
        
        return batched_rbf(mus, epsilons, weights)
    
    return generate_rbf_solutions

# Create generate functions for each transform
generate_original = create_generate_rbf_solutions(precompute_original)
generate_linear = create_generate_rbf_solutions(precompute_linear)
generate_exponential = create_generate_rbf_solutions(precompute_exponential)
generate_polynomial = create_generate_rbf_solutions(precompute_polynomial)
generate_adaptive = create_generate_rbf_solutions(precompute_adaptive)
generate_multiscale = create_generate_rbf_solutions(precompute_multiscale)
generate_frequency = create_generate_rbf_solutions(precompute_frequency)

@jax.jit
def apply_projection_alternatives(lambdas_0: jnp.ndarray, n_points: int) -> jnp.ndarray:
    """Apply projection to the parameters with alternative shape parameter transforms."""
    # lambdas_0 is of shape (K, 4) after vmapping (mus: 2, epsilon: 1, weight: 1)
    
    # Project mus using array updating
    lambdas_0 = lambdas_0.at[:, 0:2].set(jnp.clip(lambdas_0[:, 0:2], -1.0, 1.0))
    
    # For epsilon, we can apply bounds to control the range of the shape transform
    # Different transforms may need different bounds
    lambdas_0 = lambdas_0.at[:, 2].set(jnp.clip(lambdas_0[:, 2], -jnp.pi, jnp.pi))

    return lambdas_0
