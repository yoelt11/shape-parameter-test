"""
Shape Parameter Test Project

A comprehensive analysis and testing framework for shape parameter optimization
in Physics-Informed Neural Networks (PINNs) and Radial Basis Function (RBF) models.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .model import (
    rbf_model,
    rbf_model_alternatives,
    shape_parameter_alternatives,
    shape_parameter_transform,
    shape_parameter_transform_smooth,
    standard_model,
)

__all__ = [
    "rbf_model",
    "rbf_model_alternatives", 
    "shape_parameter_alternatives",
    "shape_parameter_transform",
    "shape_parameter_transform_smooth",
    "standard_model",
]
