import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    # Resize image to 224x224 for the model
    image = image.resize((224, 224))
    # Convert image to numpy array and normalize it
    image = np.array(image) / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image
