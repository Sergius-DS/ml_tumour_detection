import tensorflow as tf
import requests
import tempfile
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Function to download the model from Google Drive
@st.cache(allow_output_mutation=True)
def load_model_from_url(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        tmp.write(response.content)
        tmp.flush()
        model = load_model(tmp.name)
    return model

# Your Google Drive download link
model_url = 'https://drive.google.com/uc?export=download&id=1qLS6t1a5R3hk5PtpvUxuQQjAHnDcUFeU'
model = load_model_from_url(model_url)

# Define class labels
class_labels = ["Healthy", "Tumor"]

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.write(f"Prediction: **{predicted_class}** with confidence {confidence:.2f}")