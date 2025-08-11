"""
Model implementations for shape parameter optimization.

This module contains various implementations of RBF models and shape parameter
transformation strategies for PINN applications.
"""

# Import the actual functions and classes that exist
from .rbf_model import (
    Lambda as RBFModel,
    precompute_params,
    fn_evaluate,
    fn_derivatives,
    generate_rbf_solutions,
    apply_projection
)

from .rbf_model_alternatives import (
    Lambda as RBFModelAlternatives
)

# Import transform functions (not classes)
from .shape_parameter_transform import (
    transform,
    transform_circular_sweep,
    transform_eccentricity,
    transform_lissajous,
    TRANSFORMS
)

# Import smooth transform functions
from .shape_parameter_transform_smooth import (
    transform_original,
    transform_smooth_tanh,
    transform_smooth_sigmoid,
    transform_smooth_linear,
    transform_smooth_polynomial,
    transform_smooth_exponential,
    transform_smooth_adaptive,
    transform_smooth_multiscale,
    transform_smooth_gradient_optimized
)

# Import alternative transform functions
from .shape_parameter_alternatives import (
    transform_original as transform_alt_original,
    transform_linear,
    transform_exponential,
    transform_polynomial,
    transform_adaptive,
    transform_multiscale,
    transform_frequency
)

from .standard_model import (
    Lambda as StandardModel
)

__all__ = [
    "RBFModel",
    "RBFModelAlternatives",
    "StandardModel",
    "precompute_params",
    "fn_evaluate",
    "fn_derivatives",
    "generate_rbf_solutions",
    "apply_projection",
    # Transform functions
    "transform",
    "transform_circular_sweep",
    "transform_eccentricity",
    "transform_lissajous",
    "TRANSFORMS",
    # Smooth transforms
    "transform_original",
    "transform_smooth_tanh",
    "transform_smooth_sigmoid",
    "transform_smooth_linear",
    "transform_smooth_polynomial",
    "transform_smooth_exponential",
    "transform_smooth_adaptive",
    "transform_smooth_multiscale",
    "transform_smooth_gradient_optimized",
    # Alternative transforms
    "transform_alt_original",
    "transform_linear",
    "transform_exponential",
    "transform_polynomial",
    "transform_adaptive",
    "transform_multiscale",
    "transform_frequency",
]
