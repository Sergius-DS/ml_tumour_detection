import streamlit as st
import tensorflow as tf
import requests
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import gdown
import base64

# Path to your background image
background_image_path = "medical_laboratory.jpg"

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def set_background(image_path):
    b64_image = get_base64_image(image_path)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{b64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main-title {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .prediction-box {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }}
    .stFileUploader > div {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 8px;
    }}
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.95) !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background(background_image_path)

def load_model_from_url(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        gdown.download(url, tmp_file.name, quiet=False)
        model = load_model(tmp_file.name)
        os.unlink(tmp_file.name)
    return model

file_id = '1qLS6t1a5R3hk5PtpvUxuQQjAHnDcUFeU'


with st.spinner("Loading model... This might take a moment."):
    model = load_model_from_url(file_id)

class_labels = ["Healthy", "Tumor"]

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    from tensorflow.keras.applications.resnet import preprocess_input
    return preprocess_input(image_array)

# Title
st.markdown("""
<div class="main-title">
    <h1>🧠 Deep Learning for Detecting Brain Tumour 🔎</h1>
</div>
""", unsafe_allow_html=True)

# --- Layout with two columns for upload and display ---
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    predict_button = st.button("Predict")

    # Check if a new file is uploaded; if so, reset prediction
    if uploaded_file:
        if st.session_state.get('uploaded_image') != uploaded_file:
            st.session_state['uploaded_image'] = uploaded_file
            st.session_state['prediction'] = None
    elif st.session_state.get('uploaded_image'):
        uploaded_file = st.session_state['uploaded_image']
    else:
        uploaded_file = None

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', width=240)

# --- Prediction logic ---
if predict_button and uploaded_file:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert('RGB')
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    # Interpret predictions
    if predictions.shape[1] == 1:
        pred_value = predictions[0][0]
        if pred_value >= 0.5:
            predicted_class = "Tumor"
            confidence = pred_value
        else:
            predicted_class = "Healthy"
            confidence = 1 - pred_value
    elif predictions.shape[1] == 2:
        probs = predictions[0]
        predicted_index = np.argmax(probs)
        predicted_class = class_labels[predicted_index]
        confidence = probs[predicted_index]
    else:
        st.write("Unexpected model output shape:", predictions.shape)
        predicted_class = "Unknown"
        confidence = 0.0

    # Save prediction to session state
    st.session_state['prediction'] = {
        'class': predicted_class,
        'confidence': confidence
    }

# --- Display prediction result ---
if st.session_state['prediction']:
    pred = st.session_state['prediction']
    st.markdown(f"""
    <div class="prediction-box">
        <h3>Prediction Result:</h3>
        <p><strong>{pred['class']}</strong> with confidence <strong>{pred['confidence']*100:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)
