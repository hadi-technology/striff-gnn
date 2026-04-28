#!/usr/bin/env python3
"""Export trained model to ONNX and bundle artifacts."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.export.to_onnx import export_to_onnx
from src.export.metadata import export_metadata


if __name__ == "__main__":
    export_to_onnx()
    export_metadata()
    print("\nExport complete. Artifacts in artifacts/models/")
