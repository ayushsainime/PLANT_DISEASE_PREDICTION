#%%
import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

#%%
working_dir = os.path.dirname(os.path.abspath(__file__))
working_dir 
#%%
model_path = f"{working_dir}\\model\\plant_disease_prediction_model.h5"

model_path 
# Load the pre-trained model
from tensorflow.keras.models import load_model

model = load_model(model_path, compile=False)

# loading the class names
class_indices = json.load(open(f"{working_dir}\\class_indices.json"))

class_indices
# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
import streamlit as st
from PIL import Image

# Title and Description
st.title('🌱 Plant Disease Classifier')
st.markdown("""
This application uses a deep learning model to identify **plant diseases from leaf images**.  
It covers **38 different categories of plant health and disease**, helping farmers to:

- Detect diseases at an **early stage**
- Reduce crop damage and financial losses
- Take **timely action** to control the spread
- Improve overall agricultural productivity
""")

# List of Categories
categories = [
    "Apple — Apple scab", "Apple — Black rot", "Apple — Cedar apple rust", "Apple — healthy",
    "Blueberry — healthy",
    "Cherry (including sour) — Powdery mildew", "Cherry (including sour) — healthy",
    "Corn (maize) — Cercospora leaf spot Gray leaf spot",
    "Corn (maize) — Common rust", "Corn (maize) — Northern Leaf Blight",
    "Corn (maize) — healthy",
    "Grape — Black rot", "Grape — Esca (Black Measles)", "Grape — Leaf blight (Isariopsis Leaf Spot)", "Grape — healthy",
    "Orange — Huanglongbing (Citrus greening)", "Peach — Bacterial spot", "Peach — healthy",
    "Pepper, bell — Bacterial spot", "Pepper, bell — healthy",
    "Potato — Early blight", "Potato — Late blight", "Potato — healthy",
    "Raspberry — healthy", "Soybean — healthy",
    "Squash — Powdery mildew",
    "Strawberry — Leaf scorch", "Strawberry — healthy",
    "Tomato — Bacterial spot", "Tomato — Early blight", "Tomato — Late blight",
    "Tomato — Leaf Mold", "Tomato — Septoria leaf spot",
    "Tomato — Spider mites (Two-spotted spider mite)", "Tomato — Target Spot",
    "Tomato — Yellow Leaf Curl Virus", "Tomato — Mosaic Virus", "Tomato — healthy"
]

with st.expander("View Categories"):
    for i, cat in enumerate(categories, 1):
        st.write(f"{i}. {cat}")

# File Uploader
uploaded_file = st.file_uploader("📁 Upload a leaf image (jpg, jpeg, or png)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption='Uploaded leaf')

    with col2:
        if st.button('Classify'):
            # Preprocessing and Prediction
            prediction = predict_image_class(model, uploaded_file, class_indices)
            st.success(f'✅ Disease Detected: {prediction}')
