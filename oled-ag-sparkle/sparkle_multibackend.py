#!/usr/bin/env python3
"""Compatibility launcher; main source is scripts/sparkle_multibackend.py."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).parent / 'scripts' / 'sparkle_multibackend.py'),run_name='__main__')
