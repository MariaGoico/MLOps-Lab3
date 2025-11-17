import click
from logic.utilities import (
    predict,
    resize,
    to_grayscale,
    normalize,
    random_rotate,
    blur,
    random_flip,
    preprocess as preprocess_pipeline,
)
from PIL import Image


# ─────────────────────────────
# MAIN GROUP
# ─────────────────────────────
@click.group(help="Main CLI tool for image classification utilities.")
def cli():
    """
    Main entry point of the CLI.

    Commands:
        - classify
        - preprocess
    """
    pass


# ─────────────────────────────
# CLASSIFICATION GROUP
# ─────────────────────────────
@cli.group(help="Commands related to classification.")
def classify():
    """Group for prediction-related commands."""
    pass


@classify.command(
    name="predict",
    help="Predict a random class from a provided list. "
         "Example: python cli.py classify predict --classes cat dog fish"
)
@click.option(
    "--classes",
    "-c",
    multiple=True,
    required=True,
    help="List of class names. Pass multiple times: -c cat -c dog -c bird",
)
def classify_predict(classes):
    """
    Predict a random class from a given list of categories.

    Example:
        python cli.py classify predict -c cat -c dog -c fish
    """
    result = predict(classes)
    click.echo(f"Predicted class: {result}")


# ─────────────────────────────
# PREPROCESSING GROUP
# ─────────────────────────────
@cli.group(help="Commands related to preprocessing images.")
def preprocess():
    """Group for preprocessing commands."""
    pass


@preprocess.command(
    name="resize",
    help="Resize an image to a random size between 28 and 225 pixels. "
         "Example: python cli.py preprocess resize input.jpg output.jpg"
)
@click.argument("image_path")
@click.argument("output_path")
def preprocess_resize(image_path, output_path):
    """
    Resize an image randomly and save the result.

    Args:
        image_path (str): Input image.
        output_path (str): Path to save resized image.
    """
    img = resize(image_path)
    img.save(output_path)
    click.echo(f"Saved resized image to: {output_path}")

# --- Grayscale Command ---
@preprocess.command(
    name="grayscale",
    help="Convert image to grayscale."
)
@click.argument("image_path")
@click.argument("output_path")
def preprocess_grayscale(image_path, output_path):
    img = Image.open(image_path)
    img = to_grayscale(img)
    img.save(output_path)
    click.echo(f"Saved grayscale image to: {output_path}")


# --- Rotate Command ---
@preprocess.command(
    name="rotate",
    help="Randomly rotate the image by up to ±20 degrees."
)
@click.argument("image_path")
@click.argument("output_path")
def preprocess_rotate(image_path, output_path):
    img = Image.open(image_path)
    img = random_rotate(img)
    img.save(output_path)
    click.echo(f"Saved rotated image to: {output_path}")


# --- Flip Command ---
@preprocess.command(
    name="flip",
    help="Randomly flip the image horizontally (50% probability)."
)
@click.argument("image_path")
@click.argument("output_path")
def preprocess_flip(image_path, output_path):
    img = Image.open(image_path)
    img = random_flip(img)
    img.save(output_path)
    click.echo(f"Saved flipped image to: {output_path}")


# --- Blur Command ---
@preprocess.command(
    name="blur",
    help="Apply Gaussian blur to the image."
)
@click.argument("image_path")
@click.argument("output_path")
def preprocess_blur(image_path, output_path):
    img = Image.open(image_path)
    img = blur(img)
    img.save(output_path)
    click.echo(f"Saved blurred image to: {output_path}")


# --- Full Pipeline Command ---
@preprocess.command(
    name="pipeline",
    help="Apply full preprocessing pipeline (resize, grayscale, rotate, flip, blur)."
)
@click.argument("image_path")
@click.argument("output_path")
def preprocess_full_pipeline(image_path, output_path):
    img = preprocess_pipeline(image_path)
    img.save(output_path)
    click.echo(f"Saved fully preprocessed image to: {output_path}")

# ─────────────────────────────
# ENTRY POINT
# ─────────────────────────────
if __name__ == "__main__":
    cli()
