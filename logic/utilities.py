import random
from PIL import Image, ImageOps, ImageFilter

""" method to predict the class of a given image. In this first lab, the class will 
be randomly chosen among a set of class names (of your choice)."""
def predict(class_set):
    index = random.randint(0,len(class_set))
    return list(class_set)[index]

"""Define a method to resize an image to a certain size"""
def resize(image_path):
    image = Image.open(image_path)
    random_size = random.randint(28,225)
    new_image = image.resize((random_size, random_size))
    return new_image

"""You can define more methods to preprocess an image (of your choice)"""
""" Convert an image to grayscale """
def to_grayscale(image: Image.Image) -> Image.Image:
    return ImageOps.grayscale(image)


""" Normalize pixel values to [0,1] """
def normalize(image: Image.Image):
    # Convert to float values 0–1
    return ImageOps.autocontrast(image).point(lambda x: x / 255.0)


""" Apply random rotation between -20° and 20° """
def random_rotate(image: Image.Image) -> Image.Image:
    angle = random.uniform(-20, 20)
    return image.rotate(angle)


""" Apply Gaussian blur """
def blur(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=1.5))


""" Flip horizontally with 50% probability """
def random_flip(image: Image.Image):
    if random.random() > 0.5:
        return ImageOps.mirror(image)
    return image