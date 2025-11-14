import random
from PIL import Image

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