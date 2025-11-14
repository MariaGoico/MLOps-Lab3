import click
from logic.utilities import predict, resize


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


# ─────────────────────────────
# ENTRY POINT
# ─────────────────────────────
if __name__ == "__main__":
    cli()
