import random
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path
import json

# Import ONNX classifier
try:
    from logic.onnx_classifier import classifier

    CLASSIFIER_AVAILABLE = classifier is not None
except ImportError:# pragma: no cover
    CLASSIFIER_AVAILABLE = False
    classifier = None

# Load class labels from JSON
CLASS_LABELS_PATH = Path(__file__).parent.parent / "class_labels.json"
try:
    with open(CLASS_LABELS_PATH, encoding='utf-8') as f:
        CLASS_LABELS = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):# pragma: no cover
    # Fallback if file not found
    CLASS_LABELS = [
        "Abyssinian", "American Bulldog", "American Pit Bull Terrier",
        "Basset Hound", "Beagle", "Bengal", "Birman", "Bombay", "Boxer",
        "British Shorthair", "Chihuahua", "Egyptian Mau", "English Cocker Spaniel",
        "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese",
        "Japanese Chin", "Keeshond", "Leonberger", "Maine Coon",
        "Miniature Pinscher", "Newfoundland", "Persian", "Pomeranian",
        "Pug", "Ragdoll", "Russian Blue", "Saint Bernard", "Samoyed",
        "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx",
        "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier"
    ]

# ─────────────────────────────
# PREDICTION
# ─────────────────────────────
def predict_simple(image):
    """
    Predict the class of an image using the ONNX model.
    Falls back to random prediction if model is not available.

    Args:
        image: PIL Image object or path to image

    Returns:
        str: Predicted class name
    """
    # If classifier is available, use it
    if CLASSIFIER_AVAILABLE and classifier is not None:
        try:
            # If image is a path, load it
            if isinstance(image, (str, Path)):
                image = Image.open(image)

            # Make prediction using ONNX model
            predicted_class = classifier.predict(image)
            return predicted_class
        except (FileNotFoundError, OSError, RuntimeError) as e:
            print(f"Error during prediction: {e}")
            print("Falling back to random prediction")

    # Fallback to random prediction (for backward compatibility or if model not available)
    return random.choice(CLASS_LABELS)


def predict(image):
    """
    Predict the class of an image with confidence score.

    Args:
        image: PIL Image object or path to image

    Returns:
        tuple: (predicted_class, confidence) or (predicted_class, None) if model unavailable
    """
    if CLASSIFIER_AVAILABLE and classifier is not None:
        try:
            # If image is a path, load it
            if isinstance(image, (str, Path)):
                image = Image.open(image)

            # Make prediction with confidence
            predicted_class, confidence = classifier.predict_with_confidence(image)
            return predicted_class, confidence
        except (FileNotFoundError, OSError, RuntimeError) as e:
            print(f"Error during prediction: {e}")

    # Fallback
    return predict_simple(image), None


# ─────────────────────────────
# RESIZE
# ─────────────────────────────

MIN_DIM = 28
MAX_DIM = 225


def resize(image_path: str, width: int = None, height: int = None):
    """
    Resize an image to given dimensions.
    Negative or zero dimensions must raise custom ValueErrors
    expected by the tests.
    """

    # ---- VALIDATION ----
    if width is not None:
        if width <= 0:
            raise ValueError("Width must be greater than 0")
    if height is not None:
        if height <= 0:
            raise ValueError("Height must be greater than 0")

    # Open image
    image = Image.open(image_path)

    # Random size case (if width or height missing)
    if width is None or height is None:
        width = width or random.randint(MIN_DIM, MAX_DIM)
        height = height or random.randint(MIN_DIM, MAX_DIM)

    # Perform resizing
    new_image = image.resize((width, height))
    return new_image


# ─────────────────────────────
# PREPROCESSING FUNCTIONS
# ─────────────────────────────
def to_grayscale(image: Image.Image) -> Image.Image:
    """Convert an image to grayscale."""
    return ImageOps.grayscale(image)


def normalize(image: Image.Image) -> Image.Image:
    """Normalize pixel values to [0,1]."""
    return ImageOps.autocontrast(image).point(lambda x: x / 255.0)


def random_rotate(image: Image.Image) -> Image.Image:
    """Apply random rotation between -20° and 20°."""
    angle = random.uniform(-20, 20)
    return image.rotate(angle)


def blur(image: Image.Image) -> Image.Image:
    """Apply Gaussian blur."""
    return image.filter(ImageFilter.GaussianBlur(radius=1.5))


def random_flip(image: Image.Image) -> Image.Image:
    """Flip horizontally with 50% probability."""
    if random.random() > 0.5:
        return ImageOps.mirror(image)
    return image


def preprocess(image_path):
    """
    Full preprocessing pipeline.

    Args:
        image_path (str): Path to the input image

    Returns:
        PIL.Image.Image: Preprocessed image
    """
    image = Image.open(image_path)

    # Apply preprocessing steps
    image = resize(image_path)  # random resize
    image = to_grayscale(image)  # grayscale
    image = random_rotate(image)  # slight rotation
    image = random_flip(image)  # maybe flip
    image = blur(image)  # blur a bit

    return image


# ─────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────
def ensure_output_dir():
    """Ensure the outputs directory exists."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir
