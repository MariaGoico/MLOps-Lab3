"""
ONNX classifier wrapper for inference.
This module provides a class to load and use ONNX models for image classification.
"""

import json
import numpy as np
from PIL import Image
import onnxruntime as ort
from pathlib import Path


class ONNXClassifier:
    """Wrapper class for ONNX model inference."""
    
    def __init__(self, model_path: str, class_labels_path: str):
        """
        Initialize the ONNX classifier.
        
        Args:
            model_path: Path to the ONNX model file
            class_labels_path: Path to the JSON file containing class labels
        """
        # Validate files exist
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not Path(class_labels_path).exists():
            raise FileNotFoundError(f"Class labels file not found: {class_labels_path}")
        
        # Initialize ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        
        # Get input name
        self.input_name = self.session.get_inputs()[0].name
        
        # Load class labels
        with open(class_labels_path, 'r') as f:
            self.class_labels = json.load(f)
        
        print(f"  ONNX Classifier initialized")
        print(f"  Model: {model_path}")
        print(f"  Classes: {len(self.class_labels)}")
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess an image for model input.
        
        Args:
            image: PIL Image object
        
        Returns:
            Preprocessed image as numpy array
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 224x224
        image = image.resize((224, 224), Image.BILINEAR)
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Change from HWC to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image: Image.Image) -> str:
        """
        Predict the class label for an image.
        
        Args:
            image: PIL Image object
        
        Returns:
            Predicted class label as string
        """
        # Preprocess image
        preprocessed = self.preprocess(image)
        
        # Create input dictionary
        inputs = {self.input_name: preprocessed}
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Get logits (first output)
        logits = outputs[0]
        
        # Get predicted class index
        predicted_idx = int(np.argmax(logits, axis=1)[0])
        
        # Return class label
        return self.class_labels[predicted_idx]
    
    def predict_with_confidence(self, image: Image.Image) -> tuple:
        """
        Predict the class label with confidence score.
        
        Args:
            image: PIL Image object
        
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        # Preprocess image
        preprocessed = self.preprocess(image)
        
        # Create input dictionary
        inputs = {self.input_name: preprocessed}
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Get logits
        logits = outputs[0][0]
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get predicted class and confidence
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        predicted_class = self.class_labels[predicted_idx]
        
        return predicted_class, confidence


# ─────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────
# Create a global instance that can be imported
_classifier_instance = None


def get_classifier(model_path: str = "./model.onnx", 
                   class_labels_path: str = "./class_labels.json") -> ONNXClassifier:
    """
    Get or create the classifier instance (singleton pattern).
    
    Args:
        model_path: Path to ONNX model
        class_labels_path: Path to class labels JSON
    
    Returns:
        ONNXClassifier instance
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = ONNXClassifier(model_path, class_labels_path)
    
    return _classifier_instance


# For easy import
try:
    classifier = get_classifier()
except FileNotFoundError as e:
    print(f"Warning: Could not initialize classifier: {e}")
    print("Make sure to run 'select_best_model.py' first to generate the model files.")
    classifier = None


# ─────────────────────────────
# TESTING
# ─────────────────────────────
if __name__ == "__main__":
    """Test the classifier with a sample image."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python onnx_classifier.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load classifier
    clf = get_classifier()
    
    # Load image
    image = Image.open(image_path)
    
    # Get predictions
    print("\nSingle prediction:")
    predicted_class = clf.predict(image)
    print(f"  Predicted class: {predicted_class}")
    
    print("\nPrediction with confidence:")
    predicted_class, confidence = clf.predict_with_confidence(image)
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {confidence:.4f}")