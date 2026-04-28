#!/usr/bin/env python3
"""Train the ArchGraphMAE model.

Resumable: loads from checkpoint if one exists.
Saves checkpoint after each epoch.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.train import train


if __name__ == "__main__":
    train()
