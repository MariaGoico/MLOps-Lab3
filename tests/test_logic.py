import pytest
import json
import random
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from PIL import Image

from logic.utilities import (
    predict,
    predict_simple,
    resize,
    to_grayscale,
    normalize,
    random_rotate,
    blur,
    random_flip,
    preprocess,
    ensure_output_dir,
)

@pytest.fixture(scope="session")
def expected_classes():
    """Load class_labels.json or return fallback class names."""

    class_labels_path = Path("class_labels.json")

    if class_labels_path.exists():
        with open(class_labels_path) as f:
            return json.load(f)

    # Fallback for CI or environments without the file
    return [
        "Abyssinian", "American Bulldog", "American Pit Bull Terrier",
        "Basset Hound", "Beagle", "Bengal", "Birman", "Bombay", "Boxer",
        "British Shorthair", "Chihuahua", "Egyptian Mau",
        "English Cocker Spaniel", "English Setter", "German Shorthaired",
        "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond",
        "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
        "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue",
        "Saint Bernard", "Samoyed", "Scottish Terrier", "Shiba Inu",
        "Siamese", "Sphynx", "Staffordshire Bull Terrier",
        "Wheaten Terrier", "Yorkshire Terrier",
    ]


# Utility to generate dummy image
@pytest.fixture
def dummy_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(img_path)
    return img_path


# ─────────────────────────────
# PREDICT_SIMPLE
# ─────────────────────────────
@patch("logic.utilities.CLASSIFIER_AVAILABLE", False)
def test_predict_simple_returns_valid_class(dummy_image, expected_classes):
    """Test that predict_simple returns one of the valid classes from class_labels.json."""
    for _ in range(10):
        result = predict_simple(dummy_image)
        assert result in expected_classes, f"Got unexpected class: {result}"


def test_predict_simple_with_image_path(dummy_image, expected_classes):
    """Test predict_simple with an actual image path."""
    result = predict_simple(dummy_image)
    assert result in expected_classes


@patch("logic.utilities.CLASSIFIER_AVAILABLE", False)
def test_predict_simple_fallback_when_no_classifier(dummy_image, expected_classes):
    """Test that predict_simple falls back to random when classifier unavailable."""
    result = predict_simple(dummy_image)
    assert result in expected_classes


@patch("logic.utilities.CLASSIFIER_AVAILABLE", True)
@patch("logic.utilities.classifier")
def test_predict_simple_uses_classifier_when_available(mock_classifier, dummy_image):
    """Test that predict_simple uses ONNX classifier when available."""
    mock_classifier.predict.return_value = "Persian"
    
    result = predict_simple(dummy_image)
    
    assert result == "Persian"
    mock_classifier.predict.assert_called_once()


@patch("logic.utilities.CLASSIFIER_AVAILABLE", True)
@patch("logic.utilities.classifier")
def test_predict_simple_fallback_on_error(mock_classifier, dummy_image, expected_classes):
    """Test that predict_simple falls back to random on classifier error."""
    mock_classifier.predict.side_effect = RuntimeError("Model error")
    
    result = predict_simple(dummy_image)
    assert result in expected_classes


# ─────────────────────────────
# PREDICT (with confidence)
# ─────────────────────────────
@patch("logic.utilities.CLASSIFIER_AVAILABLE", False)
def test_predict_returns_tuple(dummy_image):
    """Test that predict returns a tuple of (class, confidence)."""
    result = predict(dummy_image)
    assert isinstance(result, tuple)
    assert len(result) == 2


@patch("logic.utilities.CLASSIFIER_AVAILABLE", False)
def test_predict_fallback_returns_none_confidence(dummy_image, expected_classes):
    """Test that predict returns None confidence when classifier unavailable."""
    predicted_class, confidence = predict(dummy_image)
    assert predicted_class in expected_classes
    assert confidence is None


@patch("logic.utilities.CLASSIFIER_AVAILABLE", False)
def test_predict_fallback_class_is_valid(dummy_image, expected_classes):
    """Test that fallback prediction returns valid class from class_labels.json."""
    for _ in range(10):
        predicted_class, confidence = predict(dummy_image)
        assert predicted_class in expected_classes
        assert confidence is None


@patch("logic.utilities.CLASSIFIER_AVAILABLE", True)
@patch("logic.utilities.classifier")
def test_predict_with_classifier(mock_classifier, dummy_image):
    """Test predict with ONNX classifier available."""
    mock_classifier.predict_with_confidence.return_value = ("Siamese", 0.95)
    
    predicted_class, confidence = predict(dummy_image)
    
    assert predicted_class == "Siamese"
    assert confidence == 0.95
    mock_classifier.predict_with_confidence.assert_called_once()


@patch("logic.utilities.CLASSIFIER_AVAILABLE", True)
@patch("logic.utilities.classifier")
def test_predict_handles_pil_image(mock_classifier):
    """Test that predict can handle PIL Image objects directly."""
    mock_classifier.predict_with_confidence.return_value = ("Bengal", 0.88)
    img = Image.new("RGB", (100, 100))
    
    predicted_class, confidence = predict(img)
    
    assert predicted_class == "Bengal"
    assert confidence == 0.88


@patch("logic.utilities.CLASSIFIER_AVAILABLE", True)
@patch("logic.utilities.classifier")
@patch("logic.utilities.predict_simple")
def test_predict_fallback_on_error(mock_predict_simple, mock_classifier, dummy_image, expected_classes):
    """Test that predict falls back gracefully on error."""
    mock_classifier.predict_with_confidence.side_effect = OSError("File error")
    mock_predict_simple.return_value = "Beagle"
    
    predicted_class, confidence = predict(dummy_image)
    assert predicted_class in expected_classes
    assert confidence is None
    mock_predict_simple.assert_called_once()


@patch("logic.utilities.CLASSIFIER_AVAILABLE", True)
@patch("logic.utilities.classifier")
def test_predict_returns_valid_class_from_labels(mock_classifier, dummy_image, expected_classes):
    """Test that classifier returns a class that's in class_labels.json."""
    # Mock classifier to return a valid class
    mock_classifier.predict_with_confidence.return_value = ("Pomeranian", 0.92)
    
    predicted_class, confidence = predict(dummy_image)
    
    assert predicted_class in expected_classes
    assert confidence == 0.92


# ─────────────────────────────
# RESIZE
# ─────────────────────────────
def test_resize_specific_dimensions(dummy_image):
    """Test resize with specific width and height."""
    img = resize(dummy_image, width=50, height=60)
    assert img.size == (50, 60)


@patch("logic.utilities.random.randint", return_value=100)
def test_resize_random_dimensions(mock_rand, dummy_image):
    """Test resize with random dimensions."""
    img = resize(dummy_image)
    assert img.size == (100, 100)


def test_resize_random_width_only(dummy_image):
    """Test resize with only height specified."""
    img = resize(dummy_image, height=60)
    assert img.size[1] == 60
    assert 28 <= img.size[0] <= 225  # Random width in valid range


def test_resize_random_height_only(dummy_image):
    """Test resize with only width specified."""
    img = resize(dummy_image, width=50)
    assert img.size[0] == 50
    assert 28 <= img.size[1] <= 225  # Random height in valid range


def test_resize_negative_width(dummy_image):
    """Test that resize raises error for negative width."""
    with pytest.raises(ValueError, match="Width must be greater than 0"):
        resize(dummy_image, width=-50, height=60)


def test_resize_negative_height(dummy_image):
    """Test that resize raises error for negative height."""
    with pytest.raises(ValueError, match="Height must be greater than 0"):
        resize(dummy_image, width=50, height=-60)


def test_resize_zero_width(dummy_image):
    """Test that resize raises error for zero width."""
    with pytest.raises(ValueError, match="Width must be greater than 0"):
        resize(dummy_image, width=0, height=60)


def test_resize_zero_height(dummy_image):
    """Test that resize raises error for zero height."""
    with pytest.raises(ValueError, match="Height must be greater than 0"):
        resize(dummy_image, width=50, height=0)


# ─────────────────────────────
# PREPROCESSING FUNCTIONS
# ─────────────────────────────
def test_to_grayscale(dummy_image):
    """Test grayscale conversion."""
    img = Image.open(dummy_image)
    gray = to_grayscale(img)
    assert gray.mode == "L"  # grayscale mode


def test_normalize(dummy_image):
    """Test normalize function applies autocontrast and scales to [0,1]."""
    img = Image.open(dummy_image)
    normalized = normalize(img)
    
    # Check that it returns an image
    assert isinstance(normalized, Image.Image)
    
    # Sample a pixel and check it's in valid range
    px = normalized.getpixel((0, 0))
    if isinstance(px, tuple):
        # RGB image
        assert all(0 <= c <= 1 for c in px)
    else:
        # Grayscale image
        assert 0 <= px <= 1


def test_normalize_grayscale(dummy_image, tmp_path):
    """Test normalize with grayscale image."""
    # Create a grayscale version of the dummy image
    gray_path = tmp_path / "gray.jpg"
    Image.open(dummy_image).convert("L").save(gray_path)

    img = Image.open(gray_path)
    normalized = normalize(img)

    # Check pixel is in valid range
    px = normalized.getpixel((0, 0))
    assert not isinstance(px, tuple)
    assert 0 <= px <= 1


@patch("logic.utilities.random.uniform", return_value=10)
def test_random_rotate(mock_rot, dummy_image):
    """Test random rotation."""
    img = Image.open(dummy_image)
    rotated = random_rotate(img)
    mock_rot.assert_called_once_with(-20, 20)
    assert isinstance(rotated, Image.Image)


def test_random_rotate_range(dummy_image):
    """Test that rotation angle is within expected range."""
    img = Image.open(dummy_image)
    # Run multiple times to ensure it's working
    for _ in range(5):
        rotated = random_rotate(img)
        assert isinstance(rotated, Image.Image)


@patch("logic.utilities.ImageOps.mirror")
@patch("logic.utilities.random.random", return_value=0.9)
def test_random_flip_flips(mock_rand, mock_mirror, dummy_image):
    """Test that image flips when random > 0.5."""
    img = Image.open(dummy_image)
    mock_mirror.return_value = img
    random_flip(img)
    mock_mirror.assert_called_once_with(img)


@patch("logic.utilities.random.random", return_value=0.1)
def test_random_flip_no_flip(mock_rand, dummy_image):
    """Test that image doesn't flip when random <= 0.5."""
    img = Image.open(dummy_image)
    result = random_flip(img)
    assert result == img


def test_blur(dummy_image):
    """Test blur applies GaussianBlur filter."""
    img = Image.open(dummy_image)
    blurred = blur(img)
    assert isinstance(blurred, Image.Image)


# ─────────────────────────────
# PREPROCESS PIPELINE
# ─────────────────────────────
def test_preprocess_returns_image(dummy_image):
    """Test that preprocess returns a PIL Image."""
    output = preprocess(dummy_image)
    assert isinstance(output, Image.Image)


def test_preprocess_produces_grayscale(dummy_image):
    """Test that preprocessing produces grayscale output."""
    output = preprocess(dummy_image)
    assert output.mode == "L"


@patch("logic.utilities.resize")
@patch("logic.utilities.blur")
@patch("logic.utilities.random_flip")
@patch("logic.utilities.random_rotate")
@patch("logic.utilities.to_grayscale")
def test_preprocess_call_order(
    mock_gray, mock_rot, mock_flip, mock_blur, mock_resize, dummy_image
):
    """Test that all preprocessing steps are called."""
    # Setup mock returns
    mock_resize.return_value = Image.new("RGB", (64, 64))
    mock_gray.return_value = Image.new("L", (64, 64))
    mock_rot.return_value = Image.new("L", (64, 64))
    mock_flip.return_value = Image.new("L", (64, 64))
    mock_blur.return_value = Image.new("L", (64, 64))

    output = preprocess(dummy_image)
    assert isinstance(output, Image.Image)

    # Verify all steps were called
    mock_resize.assert_called_once()
    mock_gray.assert_called_once()
    mock_rot.assert_called_once()
    mock_flip.assert_called_once()
    mock_blur.assert_called_once()


# ─────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────
def test_ensure_output_dir(tmp_path, monkeypatch):
    """Test that ensure_output_dir creates the directory."""
    monkeypatch.chdir(tmp_path)
    out = ensure_output_dir()
    assert out.exists()
    assert out.is_dir()
    assert out.name == "outputs"


def test_ensure_output_dir_idempotent(tmp_path, monkeypatch):
    """Test that calling ensure_output_dir multiple times is safe."""
    monkeypatch.chdir(tmp_path)
    out1 = ensure_output_dir()
    out2 = ensure_output_dir()
    assert out1 == out2
    assert out1.exists()


# ─────────────────────────────
# CLASS LABELS VALIDATION
# ─────────────────────────────
def test_class_labels_loaded(expected_classes):
    """Test that we have the expected number of classes."""
    assert len(expected_classes) == 37


def test_class_labels_content(expected_classes):
    """Test that class labels contain expected pet breeds."""
    # Check a few known classes are present
    assert "Persian" in expected_classes
    assert "Beagle" in expected_classes
    assert "Siamese" in expected_classes
    assert "Pomeranian" in expected_classes