"""
conftest.py
Ensures the project root is on sys.path so all test modules can import
project packages (config, ingestion, retrieval, store, eval) regardless
of the directory pytest is launched from.
"""
import sys
from pathlib import Path

# Insert the project root (the directory that contains this file) at the
# front of sys.path so that "from config.loader import ..." always resolves.
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
