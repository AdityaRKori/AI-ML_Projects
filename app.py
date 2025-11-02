
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown
import os

# Load the saved model
@st.cache_resource # Cache the model to avoid reloading on each interaction
def load_model():
    model_path = "heart_segmentation_model.h5"

    # ✅ Google Drive direct download link (replace YOUR_ID with your file ID)
    # You will need to replace "YOUR_FILE_ID" with the actual file ID of your model in Google Drive.
    url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"

    # Download model if not already present
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(url, model_path, quiet=False)

    # Load model
    model = tf.keras.models.load_model(model_path)
    return model

# Load model once
model = load_model()
st.success("Model loaded successfully ✅")

st.title("Heart (Atrial) Segmentation")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    original_image_np = np.array(image)

    # Resize and normalize for the model
    img_resized = cv2.resize(original_image_np, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(np.expand_dims(img_normalized, axis=-1), axis=0) # Add batch and channel dimensions

    # Perform segmentation
    prediction = model.predict(img_input)

    # Process the prediction
    predicted_mask = np.squeeze(prediction) # Remove batch and channel dimensions
    predicted_mask = (predicted_mask * 255).astype(np.uint8) # Scale back to 0-255 for visualization

    # Display results side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Image")
        st.image(original_image_np, use_column_width=True, clamp=True)

    with col2:
        st.header("Predicted Mask")
        st.image(predicted_mask, use_column_width=True, clamp=True)

