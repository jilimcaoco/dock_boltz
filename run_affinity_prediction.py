#!/usr/bin/env python3
"""
Wrapper script to run affinity predictions from source directory.
This ensures the latest code from src/ is used.
"""
import sys
import os

# Add src directory to Python path FIRST so our code takes precedence
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import and run the CLI
from boltz.main_simplified import cli

if __name__ == "__main__":
    cli()
