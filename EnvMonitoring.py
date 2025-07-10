import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Model download link (Google Drive)
# MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_URL = "https://drive.google.com/file/d/1IVob-TNuCbFjL1fSjiTI-MlGvV-nN_5E/view?usp=sharing"
MODEL_PATH = "Modelenv.v1.h5"

# Download model if not present
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... (only once)"):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return load_model(MODEL_PATH)

model = download_and_load_model()

# Class labels
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# UI
st.title("🌍 Satellite Image Classifier")
st.markdown("Upload a satellite image to classify it as **Cloudy, Desert, Green Area, or Water**.")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((256, 256))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")
