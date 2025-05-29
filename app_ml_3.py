import streamlit as st
import tensorflow as tf
import requests
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import time
import os
import gdown
import base64

# Path to your background image
background_image_path = "medical_laboratory.jpg"

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Enhanced CSS with white backgrounds for text elements
def set_background(image_path):
    b64_image = get_base64_image(image_path)
    css = f"""
    <style>
    /* Set background image for the entire page */
    .stApp {{
        background-image: url("data:image/jpeg;base64,{b64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Style for the main title */
    .main-title {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Style for prediction results */
    .prediction-box {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center; /* Added for centering text inside prediction box */
    }}
    
    /* Style for file uploader area */
    .stFileUploader > div {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 8px;
    }}
    
    /* Style for info/warning messages */
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.95) !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call this function at the very top of your script
set_background(background_image_path)

# Rest of your model loading functions remain the same
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

model_url = 'https://drive.google.com/uc?export=download&id=1qLS6t1a5R3hk5PtpvUxuQQjAHnDcUFeU'

with st.spinner("Loading model... This might take a moment."):
    model = load_model_from_url(model_url)

class_labels = ["Healthy", "Tumor"]

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    from tensorflow.keras.applications.resnet import preprocess_input
    image_array = preprocess_input(image_array)
    return image_array

# Use HTML for the title with custom styling
st.markdown("""
<div class="main-title">
    <h1>🧠 Deep Learning for Detecting Brain Tumour 🔎</h1>
</div>
""", unsafe_allow_html=True)

if model is not None:
    # Create two columns: one for uploader, one for image
    col1, col2 = st.columns(2)
  
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
  
    with col2:
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

            # Store the prediction info to display outside columns
            prediction_info = f"""
            <div class="prediction-box">
                <h3>Prediction Result:</h3>
                <p><strong>{predicted_class}</strong> with confidence <strong>{confidence:.2f}</strong></p>
            </div>
            """

    # After the columns, display the prediction spanning full width
    if 'prediction_info' in locals():
        st.markdown(prediction_info, unsafe_allow_html=True)
