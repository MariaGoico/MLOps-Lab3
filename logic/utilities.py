import random
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path


# ─────────────────────────────
# PREDICTION
# ─────────────────────────────
def predict(image):
    """
    Randomly predict a class from the given list of classes.

    Args:
        classes (list or tuple): List of class names

    Returns:
        str: Randomly selected class name
    """
    classes = ["cat", "dog", "frog", "horse"]

    return random.choice(list(classes))


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
        import random
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
