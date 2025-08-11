"""
Shape Parameter Test Project

A comprehensive analysis and testing framework for shape parameter optimization
in Physics-Informed Neural Networks (PINNs) and Radial Basis Function (RBF) models.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .model import (
    RBFModel,
    RBFModelAlternatives,
    StandardModel,
    precompute_params,
    fn_evaluate,
    fn_derivatives,
    generate_rbf_solutions,
    apply_projection,
    # Transform functions
    transform,
    transform_circular_sweep,
    transform_eccentricity,
    transform_lissajous,
    TRANSFORMS,
    # Smooth transforms
    transform_original,
    transform_smooth_tanh,
    transform_smooth_sigmoid,
    transform_smooth_linear,
    transform_smooth_polynomial,
    transform_smooth_exponential,
    transform_smooth_adaptive,
    transform_smooth_multiscale,
    transform_smooth_gradient_optimized,
    # Alternative transforms
    transform_alt_original,
    transform_linear,
    transform_exponential,
    transform_polynomial,
    transform_adaptive,
    transform_multiscale,
    transform_frequency,
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
