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

# Function to download the model from Google Drive
@st.cache(allow_output_mutation=True)
def load_model_from_url(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp.flush()

                model = load_model(tmp.name)
                return model

        except requests.exceptions.RequestException as e:
            print(f"Download error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                st.error(f"Failed to download model after {max_retries} attempts. Please check the URL and your internet connection.")
                return None

        except Exception as e:
            print(f"Error loading model (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                st.error(f"Failed to load model after {max_retries} attempts. Check model format and Keras version compatibility.")
                return None
        finally:
            if 'tmp' in locals():
                try:
                    os.unlink(tmp.name)
                except:
                    pass
    return None

# Your Google Drive model URL
model_url = 'https://drive.google.com/uc?export=download&id=1qLS6t1a5R3hk5PtpvUxuQQjAHnDcUFeU'

# Load the model
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

st.title("Image Classification App")

if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)

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
