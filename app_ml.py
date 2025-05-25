import tensorflow as tf
import requests
import tempfile
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import time

# Function to download the model from Google Drive
@st.cache(allow_output_mutation=True)
def load_model_from_url(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)  # Use stream=True
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp.flush()  # Ensure data is written to disk

                model = load_model(tmp.name)
                return model  # Success!

        except requests.exceptions.RequestException as e:
            print(f"Download error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retrying
            else:
                st.error(f"Failed to download model after {max_retries} attempts. Please check the URL and your internet connection.")
                return None # Or raise the exception, depending on how you want to handle failure

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
                    import os
                    os.unlink(tmp.name) # Delete tmp file on completion
                except:
                    pass


    return None  # If all retries fail



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

if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions)

        st.write(f"Prediction: **{predicted_class}** with confidence {confidence:.2f}")
else:
    st.write("Model loading failed. Please check the error messages above.")

