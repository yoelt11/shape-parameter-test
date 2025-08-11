#!/usr/bin/env python3
"""
Integration analysis: How shape parameter transforms complement expressibility optimization.
This script shows how both approaches can work together synergistically.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Callable, List
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)

def analyze_complementarity():
    """Analyze how shape parameter transforms complement expressibility optimization."""
    
    print("="*80)
    print("SHAPE TRANSFORM + EXPRESSIBILITY INTEGRATION ANALYSIS")
    print("="*80)
    
    print("\nShape Parameter Transform Approach:")
    print("-" * 50)
    print("1. **Reduces Parameter Space**: Single ε parameter controls multiple covariance aspects")
    print("2. **Structured Exploration**: Systematic coverage of shape space")
    print("3. **Smooth Transitions**: Continuous parameterization prevents local minima")
    print("4. **Controlled Complexity**: Prevents overfitting through structured parameterization")
    
    print("\nExpressibility Optimization Approach:")
    print("-" * 50)
    print("1. **Simplified Parameterization**: Eliminates complex matrix operations")
    print("2. **Faster Optimization**: Direct gradients, fewer non-linear transforms")
    print("3. **Better Numerical Stability**: Controlled parameter ranges")
    print("4. **Maintained Expressibility**: Scaled diagonal and direct inverse methods")
    
    print("\nSynergistic Benefits:")
    print("-" * 50)
    print("1. **Shape Transform + Scaled Diagonal**: Structured exploration with fast optimization")
    print("2. **Shape Transform + Direct Inverse**: Full expressibility with systematic coverage")
    print("3. **Shape Transform + Progressive Complexity**: Adaptive complexity with structured exploration")
    print("4. **Shape Transform + Adaptive Learning**: Systematic parameterization with optimized learning")

def create_integrated_solutions():
    """Create integrated solutions combining shape transforms with expressibility optimization."""
    
    print("\n" + "="*80)
    print("INTEGRATED SOLUTIONS")
    print("="*80)
    
    solutions = [
        {
            'name': 'Shape Transform + Scaled Diagonal',
            'description': 'Structured shape exploration with fast optimization',
            'parameterization': 'ε → (log_sigma, scale_x, scale_y) via shape transform',
            'benefits': [
                'Systematic shape space coverage',
                'Fast optimization (no rotation matrices)',
                'Controlled anisotropy through scaling',
                'Smooth parameter transitions'
            ],
            'implementation': '''
                # Shape transform generates base parameters
                epsilon = shape_parameter_transform(ε)
                log_sigma, scale_x, scale_y = transform_to_scaled_diagonal(epsilon)
                
                # Fast optimization with scaled diagonal
                sigma = exp(log_sigma)
                inv_cov_11 = scale_x / σ²
                inv_cov_22 = scale_y / σ²
            ''',
            'use_case': 'When you want systematic shape exploration with fast optimization'
        },
        {
            'name': 'Shape Transform + Direct Inverse',
            'description': 'Full expressibility with structured exploration',
            'parameterization': 'ε → (inv_cov_11, inv_cov_12, inv_cov_22) via shape transform',
            'benefits': [
                'Maximum expressibility',
                'Systematic covariance space coverage',
                'Fast gradients (no non-linear transforms)',
                'Controlled parameter ranges'
            ],
            'implementation': '''
                # Shape transform generates full covariance parameters
                epsilon = shape_parameter_transform(ε)
                inv_cov_11, inv_cov_12, inv_cov_22 = transform_to_direct_inverse(epsilon)
                
                # Direct assignment for fast optimization
                inv_covs = [[inv_cov_11, inv_cov_12],
                           [inv_cov_12, inv_cov_22]]
            ''',
            'use_case': 'When you need maximum expressibility with systematic exploration'
        },
        {
            'name': 'Adaptive Shape Transform',
            'description': 'Progressive complexity with structured exploration',
            'parameterization': 'ε → (stage_1_params, stage_2_params, stage_3_params)',
            'benefits': [
                'Progressive complexity addition',
                'Systematic parameter space coverage',
                'Fast initial convergence',
                'Adaptive expressibility'
            ],
            'implementation': '''
                # Stage 1: Isotropic with shape transform
                epsilon = shape_parameter_transform(ε)
                log_sigma = transform_to_isotropic(epsilon)
                
                # Stage 2: Add scaling if needed
                scale_x, scale_y = transform_to_scaling(epsilon)
                
                # Stage 3: Add rotation if needed
                theta = transform_to_rotation(epsilon)
            ''',
            'use_case': 'When you want progressive complexity with structured exploration'
        }
    ]
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. {solution['name']}")
        print("-" * len(solution['name']))
        print(f"Description: {solution['description']}")
        print(f"Parameterization: {solution['parameterization']}")
        print(f"Benefits:")
        for benefit in solution['benefits']:
            print(f"  - {benefit}")
        print(f"Use Case: {solution['use_case']}")
        print(f"Implementation:\n{solution['implementation']}")

def create_shape_transform_adaptations():
    """Create adaptations of shape transforms for different expressibility approaches."""
    
    print("\n" + "="*80)
    print("SHAPE TRANSFORM ADAPTATIONS FOR EXPRESSIBILITY")
    print("="*80)
    
    # Import shape transforms
    from model.shape_parameter_transform import TRANSFORMS
    
    adaptations = {
        'scaled_diagonal': {
            'description': 'Adapt shape transforms for scaled diagonal parameterization',
            'transforms': {
                'circular_sweep': '''
                    def transform_scaled_diagonal_circular(epsilon):
                        """Circular sweep adapted for scaled diagonal."""
                        r = 1.0
                        log_sigma = r * jnp.sin(epsilon)  # base sigma
                        scale_x = 1.0 + 0.5 * jnp.cos(epsilon)  # scaling factor
                        scale_y = 1.0 + 0.5 * jnp.sin(epsilon)  # scaling factor
                        return log_sigma, scale_x, scale_y
                ''',
                'eccentricity': '''
                    def transform_scaled_diagonal_eccentricity(epsilon):
                        """Eccentricity adapted for scaled diagonal."""
                        mean_scale = jnp.sin(epsilon)
                        eccentricity = 0.5 * jnp.sin(2 * epsilon)
                        log_sigma = mean_scale  # base sigma
                        scale_x = 1.0 + eccentricity  # scaling factor
                        scale_y = 1.0 - eccentricity  # scaling factor
                        return log_sigma, scale_x, scale_y
                ''',
                'lissajous': '''
                    def transform_scaled_diagonal_lissajous(epsilon):
                        """Lissajous adapted for scaled diagonal."""
                        log_sigma = jnp.sin(epsilon)  # base sigma
                        scale_x = 1.0 + 0.3 * jnp.sin(2 * epsilon)  # scaling factor
                        scale_y = 1.0 + 0.3 * jnp.cos(2 * epsilon)  # scaling factor
                        return log_sigma, scale_x, scale_y
                '''
            }
        },
        'direct_inverse': {
            'description': 'Adapt shape transforms for direct inverse parameterization',
            'transforms': {
                'circular_sweep': '''
                    def transform_direct_inverse_circular(epsilon):
                        """Circular sweep adapted for direct inverse."""
                        r = 100.0  # base inverse covariance value
                        inv_cov_11 = r * (1.0 + jnp.sin(epsilon))
                        inv_cov_22 = r * (1.0 + jnp.cos(epsilon))
                        inv_cov_12 = 0.0  # no correlation initially
                        return inv_cov_11, inv_cov_12, inv_cov_22
                ''',
                'eccentricity': '''
                    def transform_direct_inverse_eccentricity(epsilon):
                        """Eccentricity adapted for direct inverse."""
                        mean_scale = jnp.sin(epsilon)
                        eccentricity = 0.5 * jnp.sin(2 * epsilon)
                        inv_cov_11 = 100.0 * (1.0 + mean_scale + eccentricity)
                        inv_cov_22 = 100.0 * (1.0 + mean_scale - eccentricity)
                        inv_cov_12 = 0.0  # no correlation initially
                        return inv_cov_11, inv_cov_12, inv_cov_22
                ''',
                'lissajous': '''
                    def transform_direct_inverse_lissajous(epsilon):
                        """Lissajous adapted for direct inverse."""
                        inv_cov_11 = 100.0 * (1.0 + jnp.sin(epsilon))
                        inv_cov_22 = 100.0 * (1.0 + jnp.sin(2 * epsilon))
                        inv_cov_12 = 10.0 * jnp.sin(3 * epsilon)  # correlation
                        return inv_cov_11, inv_cov_12, inv_cov_22
                '''
            }
        }
    }
    
    for approach, details in adaptations.items():
        print(f"\n{approach.upper()} ADAPTATIONS")
        print("-" * len(approach) + "-" * 10)
        print(f"Description: {details['description']}")
        
        for transform_name, code in details['transforms'].items():
            print(f"\n{transform_name}:")
            print(code)

def create_hybrid_optimization_strategy():
    """Create a hybrid optimization strategy combining both approaches."""
    
    print("\n" + "="*80)
    print("HYBRID OPTIMIZATION STRATEGY")
    print("="*80)
    
    strategy = {
        'name': 'Shape Transform + Expressibility Optimization',
        'description': 'Combines systematic shape exploration with fast optimization',
        'stages': [
            {
                'stage': 1,
                'name': 'Shape Transform Initialization',
                'description': 'Use shape transforms to initialize parameters systematically',
                'implementation': '''
                    # Initialize with shape transform
                    epsilon = shape_parameter_transform(ε)
                    log_sigma, scale_x, scale_y = transform_to_scaled_diagonal(epsilon)
                    
                    # This provides systematic coverage of shape space
                '''
            },
            {
                'stage': 2,
                'name': 'Fast Optimization',
                'description': 'Use expressibility-optimized parameterization for fast training',
                'implementation': '''
                    # Use scaled diagonal for fast optimization
                    sigma = exp(log_sigma)
                    inv_cov_11 = scale_x / σ²
                    inv_cov_22 = scale_y / σ²
                    
                    # No rotation matrices, simple gradients
                '''
            },
            {
                'stage': 3,
                'name': 'Adaptive Learning',
                'description': 'Use different learning rates for different parameter types',
                'implementation': '''
                    # Adaptive learning rates
                    weight_optimizer = optax.adam(0.01)
                    shape_optimizer = optax.adam(0.001)  # slower for shape parameters
                    
                    # This maintains expressibility while improving convergence
                '''
            },
            {
                'stage': 4,
                'name': 'Progressive Complexity',
                'description': 'Add complexity progressively using shape transforms',
                'implementation': '''
                    # Stage 1: Isotropic with shape transform
                    log_sigma = transform_to_isotropic(epsilon)
                    
                    # Stage 2: Add scaling if needed
                    scale_x, scale_y = transform_to_scaling(epsilon)
                    
                    # Stage 3: Add rotation if needed
                    theta = transform_to_rotation(epsilon)
                '''
            }
        ],
        'benefits': [
            'Systematic shape space exploration',
            'Fast optimization with simplified parameterization',
            'Maintained expressibility through appropriate parameterization',
            'Adaptive learning for different parameter types',
            'Progressive complexity addition'
        ]
    }
    
    print(f"\n{strategy['name']}")
    print("-" * len(strategy['name']))
    print(f"Description: {strategy['description']}")
    
    print(f"\nStages:")
    for stage in strategy['stages']:
        print(f"\nStage {stage['stage']}: {stage['name']}")
        print(f"Description: {stage['description']}")
        print(f"Implementation:\n{stage['implementation']}")
    
    print(f"\nBenefits:")
    for benefit in strategy['benefits']:
        print(f"  - {benefit}")

def create_implementation_example():
    """Create a practical implementation example."""
    
    print("\n" + "="*80)
    print("PRACTICAL IMPLEMENTATION EXAMPLE")
    print("="*80)
    
    print("""
# Integrated approach: Shape Transform + Scaled Diagonal

import jax
import jax.numpy as jnp
import optax

class IntegratedRBFModel:
    def __init__(self, n_kernels=25, shape_transform='circular_sweep'):
        self.n_kernels = n_kernels
        self.shape_transform = shape_transform
    
    def initialize_with_shape_transform(self, key):
        # Create systematic epsilon values
        epsilons = jnp.linspace(0, 2*jnp.pi, self.n_kernels, endpoint=False)
        
        # Apply shape transform to get base parameters
        log_sigmas, scale_xs, scale_ys = self.apply_shape_transform(epsilons)
        
        # Initialize other parameters
        mus = jax.random.uniform(key, (self.n_kernels, 2), minval=-0.8, maxval=0.8)
        weights = jax.random.normal(key, (self.n_kernels,)) * 0.1
        
        return {
            'mus': mus,
            'log_sigmas': log_sigmas,
            'scale_xs': scale_xs,
            'scale_ys': scale_ys,
            'weights': weights
        }
    
    def apply_shape_transform(self, epsilons):
        # Apply shape transform to get scaled diagonal parameters
        if self.shape_transform == 'circular_sweep':
            r = 1.0
            log_sigmas = r * jnp.sin(epsilons)
            scale_xs = 1.0 + 0.5 * jnp.cos(epsilons)
            scale_ys = 1.0 + 0.5 * jnp.sin(epsilons)
        elif self.shape_transform == 'eccentricity':
            mean_scales = jnp.sin(epsilons)
            eccentricities = 0.5 * jnp.sin(2 * epsilons)
            log_sigmas = mean_scales
            scale_xs = 1.0 + eccentricities
            scale_ys = 1.0 - eccentricities
        
        return log_sigmas, scale_xs, scale_ys
    
    def precompute_parameters(self, params):
        # Fast scaled diagonal computation
        mus = params['mus']
        log_sigmas = params['log_sigmas']
        scale_xs = params['scale_xs']
        scale_ys = params['scale_ys']
        weights = params['weights']
        
        sigmas = jnp.exp(log_sigmas)
        inv_covs = jnp.zeros((self.n_kernels, 2, 2))
        inv_covs = inv_covs.at[:, 0, 0].set(scale_xs / (sigmas**2 + 1e-6))
        inv_covs = inv_covs.at[:, 1, 1].set(scale_ys / (sigmas**2 + 1e-6))
        
        return mus, weights, inv_covs
    
    def evaluate(self, X, params):
        # Fast evaluation
        mus, weights, inv_covs = self.precompute_parameters(params)
        diff = X[:, None, :] - mus[None, :, :]
        quad = jnp.einsum('nki,kij,nkj->nk', diff, inv_covs, diff)
        phi = jnp.exp(-0.5 * quad)
        return jnp.dot(phi, weights)

# Usage
model = IntegratedRBFModel(n_kernels=25, shape_transform='circular_sweep')
params = model.initialize_with_shape_transform(jax.random.PRNGKey(42))

# Fast optimization with adaptive learning rates
weight_optimizer = optax.adam(0.01)
shape_optimizer = optax.adam(0.001)

# Training loop with integrated approach
for epoch in range(n_epochs):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    # Apply different learning rates to different parameter types
    # ... training implementation
""")

def print_integration_summary():
    """Print summary of integration benefits."""
    
    print("\n" + "="*80)
    print("INTEGRATION SUMMARY")
    print("="*80)
    
    print("\nHow Shape Transforms Complement Expressibility Optimization:")
    print("-" * 60)
    
    complementarity = [
        {
            'aspect': 'Parameter Space Reduction',
            'shape_transform': 'Single ε controls multiple aspects',
            'expressibility': 'Simplified parameterization',
            'integration': 'Systematic exploration with fast optimization'
        },
        {
            'aspect': 'Optimization Speed',
            'shape_transform': 'Structured parameterization',
            'expressibility': 'Eliminated complex operations',
            'integration': 'Fast gradients with systematic coverage'
        },
        {
            'aspect': 'Expressibility',
            'shape_transform': 'Systematic shape space coverage',
            'expressibility': 'Maintained through appropriate parameterization',
            'integration': 'Full expressibility with structured exploration'
        },
        {
            'aspect': 'Numerical Stability',
            'shape_transform': 'Controlled parameter ranges',
            'expressibility': 'Simplified transforms',
            'integration': 'Robust optimization with controlled exploration'
        }
    ]
    
    print(f"{'Aspect':<25} {'Shape Transform':<25} {'Expressibility':<25} {'Integration':<25}")
    print("-" * 100)
    
    for comp in complementarity:
        print(f"{comp['aspect']:<25} {comp['shape_transform']:<25} {comp['expressibility']:<25} {comp['integration']:<25}")
    
    print("\nKey Integration Benefits:")
    print("-" * 30)
    print("1. **Systematic + Fast**: Shape transforms provide systematic coverage, expressibility optimization provides speed")
    print("2. **Structured + Flexible**: Shape transforms provide structure, expressibility optimization provides flexibility")
    print("3. **Controlled + Expressive**: Shape transforms provide control, expressibility optimization provides expressibility")
    print("4. **Stable + Efficient**: Shape transforms provide stability, expressibility optimization provides efficiency")

def main():
    """Main function to analyze integration of shape transforms with expressibility optimization."""
    
    print("Shape Transform + Expressibility Optimization Integration")
    print("="*80)
    print("This analysis shows how both approaches complement each other.")
    
    # Analyze complementarity
    analyze_complementarity()
    
    # Create integrated solutions
    create_integrated_solutions()
    
    # Create shape transform adaptations
    create_shape_transform_adaptations()
    
    # Create hybrid optimization strategy
    create_hybrid_optimization_strategy()
    
    # Create implementation example
    create_implementation_example()
    
    # Print integration summary
    print_integration_summary()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("The shape parameter transform approach and expressibility optimization are highly complementary:")
    print("1. **Shape transforms** provide systematic exploration and controlled parameterization")
    print("2. **Expressibility optimization** provides fast optimization and maintained flexibility")
    print("3. **Integration** gives you the best of both worlds: systematic + fast + expressive")
    print("4. **Key insight**: You can have structured exploration with fast optimization!")

if __name__ == "__main__":
    main()
