import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
from PIL import Image

# Load the model
model = tf.keras.models.load_model('models/plant_disease_model.h5')

# Function to preprocess the image
def preprocess_image(img):
    img = load_img(img, target_size=(224, 224))  # Resize image
    img_array = img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Function to predict the disease
def predict_disease(img_array):
    prediction = model.predict(img_array)
    return np.argmax(prediction, axis=1)[0]

# Set the title of the app
st.set_page_config(page_title="Plant Disease Detection", layout="wide")
st.title("🌱 Plant Disease Detection System")
st.markdown(
    """
    <style>
        body {
            background-image: url('https://cdn.pixabay.com/photo/2015/03/26/09/54/field-690594_1280.jpg');
            background-size: cover;
            font-family: Arial, sans-serif;
            color: #2C3E50;
        }
        .header {
            text-align: center;
            padding: 20px;
            color: white;
        }
        .result {
            background-color: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            color: white;
            text-align: center;
        }
        .gallery {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }
        .gallery img {
            width: 150px;
            height: 150px;
            object-fit: cover;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "png", "jpeg"])

# Disease dictionary for mapping predictions
disease_dict = {
    0: 'Apple Scab',
    1: 'Black Rot',
    2: 'Cedar Apple Rust',
    3: 'Healthy',
    4: 'Downy Mildew',
    5: 'Late Blight',
    6: 'Leaf Blight',
    7: 'Powdery Mildew',
    8: 'Rust',
    9: 'Septoria Leaf Spot'
}

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Predict the disease
    disease_class = predict_disease(img_array)

    # Display prediction result
    st.markdown(f'<div class="result"><h3>Predicted Disease: {disease_dict[disease_class]}</h3></div>', unsafe_allow_html=True)

    # Add actionable insights
    if disease_class == 3:
        st.markdown('<div class="result"><h4>The plant is healthy! Keep up the good work!</h4></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result"><h4>Suggested Treatment: Refer to agricultural guidelines for treating {disease_dict[disease_class]}.</h4></div>', unsafe_allow_html=True)
else:
    st.write("Please upload an image to check for plant diseases.")

# Add a section to display sample images of infected leaves
st.markdown("## Examples of Infected Leaves")
st.markdown('<div class="gallery">', unsafe_allow_html=True)

# Add sample images (replace with actual paths or URLs if available)
sample_images = [
    "https://via.placeholder.com/150/FF0000/FFFFFF?text=Disease1",
    "https://via.placeholder.com/150/00FF00/FFFFFF?text=Disease2",
    "https://via.placeholder.com/150/0000FF/FFFFFF?text=Disease3",
    "https://via.placeholder.com/150/FFFF00/FFFFFF?text=Disease4",
    "https://via.placeholder.com/150/FF00FF/FFFFFF?text=Disease5"
]

for img_url in sample_images:
    st.markdown(f'<img src="{img_url}" alt="Sample Image">', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

