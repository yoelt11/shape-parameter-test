.PHONY: help install test clean format lint docs run-example

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e .[dev]

test:  ## Run tests
	python -m pytest tests/ -v

clean:  ## Clean up temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete

format:  ## Format code with black
	black src/ tests/ examples/ scripts/

lint:  ## Run linting with flake8
	flake8 src/ tests/ examples/ scripts/

docs:  ## Build documentation
	cd docs && make html

run-example:  ## Run the basic usage example
	python examples/basic_usage.py

setup-git:  ## Initialize git repository and make first commit
	git init
	git add .
	git commit -m "Initial commit: Organized project structure"
	@echo "Git repository initialized. You can now add a remote origin."

check-deps:  ## Check for missing dependencies
	pip check

requirements:  ## Generate requirements.txt from current environment
	pip freeze > requirements/requirements-current.txt
	@echo "Current environment requirements saved to requirements/requirements-current.txt"
