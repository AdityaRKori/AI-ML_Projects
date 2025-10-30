import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import os

# Load the saved model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("spoken_digit_model.h5")

model = load_my_model()

# Load the padding value
try:
    with open("max_pad_len.txt", "r") as f:
        max_pad_len = int(f.read())
except FileNotFoundError:
    st.error("max_pad_len.txt not found. Please run the data preprocessing steps in the notebook first.")
    st.stop()

# App title
st.title("Spoken Digit Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    # To read file as bytes:
    audio_bytes = uploaded_file.getvalue()

    # Save the uploaded file temporarily to process with librosa
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)

    try:
        # Load audio file
        y, sr = librosa.load(temp_audio_path)

        # Extract MFCCs (40 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Pad the MFCCs
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif mfccs.shape[1] > max_pad_len:
            # Trim if longer than max_pad_len (this shouldn't happen with the dataset, but good practice)
            mfccs = mfccs[:, :max_pad_len]


        # Reshape for the model
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)

        # Make prediction
        prediction = model.predict(mfccs)
        predicted_digit = np.argmax(prediction)

        # Display the result
        st.write(f"Predicted Digit: {predicted_digit}")

    except Exception as e:
        st.error(f"Error processing audio file: {e}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
