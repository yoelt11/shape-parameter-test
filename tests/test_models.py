"""
Basic tests for the model implementations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.model import RBFModel, StandardModel

class TestRBFModel:
    """Test cases for RBFModel."""
    
    def test_initialization(self):
        """Test RBF model initialization."""
        model = RBFModel(
            input_dim=2,
            output_dim=1,
            num_centers=10,
            shape_parameter=1.0
        )
        
        assert model.input_dim == 2
        assert model.output_dim == 1
        assert model.num_centers == 10
        assert model.shape_parameter == 1.0
    
    def test_parameter_count(self):
        """Test parameter count calculation."""
        model = RBFModel(
            input_dim=2,
            output_dim=1,
            num_centers=5,
            shape_parameter=1.0
        )
        
        # Expected: centers (2*5) + weights (5*1) + bias (1) = 16
        expected_params = 2 * 5 + 5 * 1 + 1
        assert model.num_parameters == expected_params

class TestStandardModel:
    """Test cases for StandardModel."""
    
    def test_initialization(self):
        """Test standard model initialization."""
        model = StandardModel(
            input_dim=2,
            output_dim=1,
            hidden_layers=[10, 5]
        )
        
        assert model.input_dim == 2
        assert model.output_dim == 1
        assert len(model.hidden_layers) == 2
    
    def test_parameter_count(self):
        """Test parameter count calculation."""
        model = StandardModel(
            input_dim=2,
            output_dim=1,
            hidden_layers=[10, 5]
        )
        
        # Expected: (2*10 + 10) + (10*5 + 5) + (5*1 + 1) = 20 + 10 + 55 + 5 + 6 = 96
        expected_params = (2 * 10 + 10) + (10 * 5 + 5) + (5 * 1 + 1)
        assert model.num_parameters == expected_params

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])
