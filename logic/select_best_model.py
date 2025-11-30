"""
Script to select the best model from MLFlow registry and serialize it to ONNX format.
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import json
from pathlib import Path

# ─────────────────────────────
# CONFIGURATION
# ─────────────────────────────
MODEL_REGISTRY_NAME = "pet_classifier"
ONNX_MODEL_PATH = "./model.onnx"
CLASS_LABELS_PATH = "./class_labels.json"
RESULTS_DIR = "./results"


# ─────────────────────────────
# SETUP
# ─────────────────────────────
def setup_directories():
    """Create necessary directories."""
    Path(RESULTS_DIR).mkdir(exist_ok=True)


# ─────────────────────────────
# MODEL SELECTION
# ─────────────────────────────
def select_best_model():
    """
    Query MLFlow registry to find the best model based on validation accuracy.

    Returns:
        best_version: ModelVersion object of the best model
        best_accuracy: Validation accuracy of the best model
    """
    client = MlflowClient()

    # Search for all versions of the registered model
    model_versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")

    if not model_versions:
        raise ValueError(f"No models found with name '{MODEL_REGISTRY_NAME}'")

    print(f"Found {len(model_versions)} model versions")
    print("\nComparing models:\n")

    best_version = None
    best_accuracy = -1

    # Compare all versions
    for version in model_versions:
        run_id = version.run_id
        run = client.get_run(run_id)
        metrics = run.data.metrics

        # Get validation accuracy
        val_accuracy = metrics.get("final_val_accuracy", 0)

        print(f"Version {version.version}:")
        print(f"  Run ID: {run_id}")
        print(f"  Run Name: {run.info.run_name}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")

        # Track best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_version = version

        print()

    print(f"{'='*60}")
    print("Best Model Selected:")
    print(f"  Version: {best_version.version}")
    print(f"  Run ID: {best_version.run_id}")
    print(f"  Validation Accuracy: {best_accuracy:.4f}")
    print(f"{'='*60}\n")

    return best_version, best_accuracy


# ─────────────────────────────
# MODEL SERIALIZATION
# ─────────────────────────────
def serialize_to_onnx(best_version):
    """
    Load the best model and serialize it to ONNX format.

    Args:
        best_version: ModelVersion object of the best model
    """
    print("Loading model from MLFlow...")

    # Load model
    model_uri = f"runs:/{best_version.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)

    # Move to CPU and set to eval mode
    model = model.to("cpu")
    model.eval()

    print("Model loaded successfully")

    # Create dummy input for ONNX export
    # Input shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting model to ONNX format: {ONNX_MODEL_PATH}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,
    )

    print(f" Model exported to {ONNX_MODEL_PATH}")


# ─────────────────────────────
# CLASS LABELS EXTRACTION
# ─────────────────────────────
def save_class_labels(best_version):
    """
    Download and save the class labels for the selected model,
    automatically detecting the correct artifact file.
    """
    client = MlflowClient()
    print("Downloading class labels...")

    # List artifacts
    artifacts = client.list_artifacts(best_version.run_id)
    print("Artifacts found:")
    for art in artifacts:
        print(" -", art.path)

    # Find any JSON file containing "class_labels"
    json_candidates = [
        art.path
        for art in artifacts
        if "class_labels" in art.path and art.path.endswith(".json")
    ]

    if not json_candidates:
        raise FileNotFoundError("No class_labels JSON file found in artifacts")

    # Use the first match
    artifact_path = json_candidates[0]
    print(f"Selected artifact: {artifact_path}")

    # Download it
    local_path = client.download_artifacts(best_version.run_id, artifact_path)

    with open(local_path, "r", encoding='utf-8') as f:
        class_labels = json.load(f)

    # Save to project root
    with open(CLASS_LABELS_PATH, "w", encoding='utf-8') as f:
        json.dump(class_labels, f, indent=2)

    print(f" Class labels saved to {CLASS_LABELS_PATH}")
    print(f" Number of classes: {len(class_labels)}")

    return class_labels


# ─────────────────────────────
# SAVE RESULTS
# ─────────────────────────────
def save_results(best_version, best_accuracy, class_labels):
    """
    Save information about the selected model.

    Args:
        best_version: ModelVersion object of the best model
        best_accuracy: Validation accuracy of the best model
        class_labels: List of class labels
    """
    client = MlflowClient()
    run = client.get_run(best_version.run_id)

    results = {
        "model_version": best_version.version,
        "run_id": best_version.run_id,
        "run_name": run.info.run_name,
        "validation_accuracy": best_accuracy,
        "num_classes": len(class_labels),
        "onnx_model_path": ONNX_MODEL_PATH,
        "class_labels_path": CLASS_LABELS_PATH,
        "parameters": run.data.params,
        "metrics": run.data.metrics,
    }

    results_path = Path(RESULTS_DIR) / "best_model_info.json"
    with open(results_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f" Results saved to {results_path}")


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    """Main function to select and serialize the best model."""
    setup_directories()

    print("\n" + "=" * 60)
    print("MODEL SELECTION AND SERIALIZATION")
    print("=" * 60 + "\n")

    # Select best model
    best_version, best_accuracy = select_best_model()

    # Serialize to ONNX
    serialize_to_onnx(best_version)

    # Save class labels
    class_labels = save_class_labels(best_version)

    # Save results
    save_results(best_version, best_accuracy, class_labels)

    print("\n" + "=" * 60)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {ONNX_MODEL_PATH}")
    print(f"  - {CLASS_LABELS_PATH}")
    print(f"  - {RESULTS_DIR}/best_model_info.json")
    print("\nYou can now use these files for inference in your API.")


if __name__ == "__main__":
    main()
