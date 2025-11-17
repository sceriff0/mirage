# Ensure the repository root and bin/ directory are on sys.path
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BIN_DIR = os.path.join(ROOT, 'bin')

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)
