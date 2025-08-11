# Shape Parameter Test Project

This project contains comprehensive analysis and testing of shape parameter optimization techniques for Physics-Informed Neural Networks (PINNs) and Radial Basis Function (RBF) models.

## Project Structure

```
shape_parameter_test/
├── src/                    # Source code modules
│   └── model/             # Model implementations
├── scripts/                # Analysis and testing scripts
├── examples/               # Example usage and demonstrations
├── tests/                  # Unit tests and validation
├── docs/                   # Documentation and analysis reports
├── requirements/           # Dependency specifications
├── images/                 # Generated plots and visualizations
└── results/                # Experimental results and outputs
```

## Key Components

### Source Code (`src/`)
- **RBF Models**: Various RBF model implementations with different shape parameter strategies
- **Shape Parameter Transforms**: Alternative approaches to shape parameter optimization
- **Standard Models**: Baseline model implementations for comparison

### Analysis Scripts (`scripts/`)
- **PINN Comparisons**: Allen-Cahn, Navier-Stokes, Poisson, and Wave equation comparisons
- **Derivative Analysis**: Comprehensive evaluation of derivative computation methods
- **Parameter Optimization**: Shape parameter reduction and optimization strategies
- **Benchmarking**: Performance comparisons across different approaches

### Documentation (`docs/`)
- **Analysis Reports**: Detailed findings from various experiments
- **Methodology**: Explanation of approaches and techniques used
- **Results Summary**: Consolidated conclusions and recommendations

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see `requirements/requirements.txt`)

### Installation
```bash
# Clone the repository
git clone git@github.com:yoelt11/shape-parameter-test.git
cd shape_parameter_test

# Install dependencies
pip install -r requirements/requirements.txt
```

### Running Examples
```bash
# Run a specific analysis
python scripts/pinn_poisson_comparison.py

# Run derivative analysis
python scripts/comprehensive_derivative_analysis.py
```

## Key Findings

This project explores several important aspects of shape parameter optimization:

1. **Shape Parameter Transforms**: Alternative encoding strategies for better optimization
2. **Derivative Computation**: Efficient methods for computing gradients and Hessians
3. **Parameter Reduction**: Techniques to reduce the number of free parameters
4. **Convergence Analysis**: Comparison of different optimization approaches

## Results

Comprehensive results are stored in the `results/` directory, organized by:
- **Problem Type**: Allen-Cahn, Navier-Stokes, Poisson, Wave equations
- **Date**: Timestamped experimental runs
- **Parameters**: Different parameter configurations tested

## Contributing

When contributing to this project:
1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation as needed
4. Use descriptive commit messages

## License

[Add your license information here]

## Contact

[Add your contact information here]
