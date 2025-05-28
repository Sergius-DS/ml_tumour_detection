import tensorflow as tf
import requests
import tempfile
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import time
import os
import gdown  # New import

# Updated model loading function
@st.cache_resource
def load_model_from_url(url, max_retries=3):
    file_id = url.split('=')[-1]
    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                model_path = tmp.name
            st.info(f"Downloading model (attempt {attempt + 1}/{max_retries})...")
            gdown.download(id=file_id, output=model_path, quiet=False)
            st.success("Model downloaded successfully!")
            model = load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading or downloading model (attempt {attempt + 1}/{max_retries}): {e}")
            st.warning(f"Attempt {attempt + 1} failed. Retrying in 5 seconds...")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                st.error("Failed to load model after multiple attempts. Please check the URL or internet connection.")
                return None
        finally:
            if 'model_path' in locals() and os.path.exists(model_path):
                try:
                    os.unlink(model_path)
                except Exception as e:
                    print(f"Error deleting temporary file: {e}")
    return None

# Your model URL
model_url = 'https://drive.google.com/uc?export=download&id=1qLS6t1a5R3hk5PtpvUxuQQjAHnDcUFeU'

# Load model with spinner
with st.spinner("Loading model... This might take a moment."):
    model = load_model_from_url(model_url)

# Define class labels
class_labels = ["Healthy", "Tumor"]

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    # Asegúrate de usar la misma normalización que en tu entrenamiento
    # Aquí usas / 255.0, pero si usaste preprocess_input, debes usarlo
    # Si tu modelo fue entrenado con preprocess_input, usa esa
    from tensorflow.keras.applications.resnet import preprocess_input
    image_array = preprocess_input(image_array)
    return image_array

st.title("🧠 Deep Learning for Detecting Brain Tumour 🔎")

if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)

        # Interpretación de la predicción
        if predictions.shape[1] == 1:
            pred = predictions[0][0]
            if pred >= 0.5:
                predicted_class = "Tumor"
                confidence = pred
            else:
                predicted_class = "Healthy"
                confidence = 1 - pred
        elif predictions.shape[1] == 2:
            probs = predictions[0]
            predicted_index = np.argmax(probs)
            predicted_class = class_labels[predicted_index]
            confidence = probs[predicted_index]
        else:
            st.write("Unexpected model output shape:", predictions.shape)
            predicted_class = "Unknown"
            confidence = 0.0

        st.write(f"Prediction: **{predicted_class}** with confidence {confidence:.2f}")
else:
    st.write("Model loading failed. Please check the error messages above.")
