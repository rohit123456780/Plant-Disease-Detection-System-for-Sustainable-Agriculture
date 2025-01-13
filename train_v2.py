from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the saved model
model = load_model('models/plant_disease_model.h5')

# Function to predict disease from an image
def predict_disease(image_path):
    img = image.load_img(image_path, target_size=(150, 150))  # Load image and resize it
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)  # Make the prediction
    predicted_class = np.argmax(prediction, axis=1)  # Get the class index of highest probability

    # Load class labels
    class_labels = list(os.listdir('data/plantdisease/train/'))

    return class_labels[predicted_class[0]]

# Example usage
image_path = 'path_to_plant_image.jpg'  # Replace with your test image path
result = predict_disease(image_path)
print(f"The predicted disease is: {result}")
