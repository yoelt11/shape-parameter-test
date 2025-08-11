"""
Command-line interface for the Shape Parameter Test Project.
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Shape Parameter Test Project CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  shape-parameter-test run-example     # Run the basic usage example
  shape-parameter-test list-scripts    # List available analysis scripts
  shape-parameter-test --help          # Show this help message
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['run-example', 'list-scripts', 'info'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='1.0.0'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'run-example':
        run_example()
    elif args.command == 'list-scripts':
        list_scripts()
    elif args.command == 'info':
        show_info()

def run_example():
    """Run the basic usage example."""
    try:
        # Add the project root to Python path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        # Import and run the example
        from examples.basic_usage import run_basic_comparison
        run_basic_comparison()
        
    except ImportError as e:
        print(f"Error importing example: {e}")
        print("Make sure you have installed the package and its dependencies.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running example: {e}")
        sys.exit(1)

def list_scripts():
    """List available analysis scripts."""
    scripts_dir = Path(__file__).parent.parent / 'scripts'
    
    if not scripts_dir.exists():
        print("Scripts directory not found.")
        return
    
    print("Available analysis scripts:")
    print("=" * 50)
    
    for script_file in sorted(scripts_dir.glob('*.py')):
        if script_file.name.startswith('__'):
            continue
            
        # Try to extract docstring
        try:
            with open(script_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                docstring = ""
                for line in lines:
                    if line.strip().startswith('"""') or line.strip().startswith("'''"):
                        if docstring:
                            docstring = line.strip().strip('"\'')
                            break
                        docstring = line.strip().strip('"\'')
                
                if docstring:
                    print(f"{script_file.name:<30} - {docstring}")
                else:
                    print(f"{script_file.name}")
        except:
            print(f"{script_file.name}")

def show_info():
    """Show project information."""
    print("Shape Parameter Test Project")
    print("=" * 40)
    print("Version: 1.0.0")
    print("Description: Comprehensive analysis and testing framework for shape parameter optimization in PINNs and RBF models")
    print("\nProject Structure:")
    print("- src/: Source code modules")
    print("- scripts/: Analysis and testing scripts")
    print("- examples/: Example usage and demonstrations")
    print("- tests/: Unit tests and validation")
    print("- docs/: Documentation and analysis reports")
    print("- requirements/: Dependency specifications")
    print("- images/: Generated plots and visualizations")
    print("- results/: Experimental results and outputs")

if __name__ == '__main__':
    main()
