"""Compatibility entry for the shared manifest-driven experiment queue."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.experiment_queue import *  # preserve the existing public queue API

if __name__ == '__main__':
    main()
