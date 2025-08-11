# Quick Start Guide

This guide will help you get up and running with the Shape Parameter Test Project quickly.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd shape_parameter_test
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the package
```bash
pip install -e .
```

### 4. Install development dependencies (optional)
```bash
pip install -e .[dev]
```

## Quick Test

### Run the basic example
```bash
python examples/basic_usage.py
```

### Or use the CLI
```bash
shape-parameter-test run-example
```

## Available Commands

### List all analysis scripts
```bash
shape-parameter-test list-scripts
```

### Show project information
```bash
shape-parameter-test info
```

### Run tests
```bash
make test
# or
pytest tests/
```

### Format code
```bash
make format
# or
black src/ tests/ examples/ scripts/
```

### Clean temporary files
```bash
make clean
```

## Project Structure Overview

```
shape_parameter_test/
├── src/                    # Source code
│   └── model/             # Model implementations
├── scripts/                # Analysis scripts
├── examples/               # Usage examples
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── requirements/           # Dependencies
├── images/                 # Generated plots
└── results/                # Experimental results
```

## Next Steps

1. **Explore the examples**: Check out `examples/basic_usage.py`
2. **Run analysis scripts**: Look in the `scripts/` directory
3. **Read documentation**: Check the `docs/` directory
4. **Run tests**: Ensure everything works with `make test`

## Getting Help

- Check the `README.md` for detailed information
- Look at the `docs/` directory for analysis reports
- Run `make help` to see all available commands
- Use `shape-parameter-test --help` for CLI help

## Common Issues

### Import errors
Make sure you've installed the package with `pip install -e .`

### Missing dependencies
Install all requirements with `pip install -r requirements/requirements.txt`

### Path issues
The examples assume you're running from the project root directory
