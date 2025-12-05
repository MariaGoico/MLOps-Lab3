import pytest
from pathlib import Path

def test_model_artifacts_exist():
    """Test that required model artifacts exist before deployment"""
    model_path = Path("model.onnx")  # Adjust path as needed
    labels_path = Path("class_labels.json")  # Adjust path as needed
    
    assert model_path.exists(), f"Model file not found at {model_path}"
    assert labels_path.exists(), f"Class labels file not found at {labels_path}"