#!/usr/bin/env python3
"""Compatibility entry point for the release's unified baseline command."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.run_baseline import main

if __name__ == '__main__':
    main()
