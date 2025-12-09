#!/usr/bin/env python3
"""
Run e2e tests with environment loaded.

This script ensures that .env is loaded before pytest starts,
so that the API key checks in test fixtures work properly.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load environment FIRST
from selectools.env import load_default_env

load_default_env()

# Now run pytest
import pytest

if __name__ == "__main__":
    # Pass through any command line arguments, adding --run-e2e
    args = ["tests/test_e2e_providers.py", "--run-e2e", "-v"]
    args.extend(sys.argv[1:])
    sys.exit(pytest.main(args))
