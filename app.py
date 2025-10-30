import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import os
import traceback
# Removed: from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
# Removed: import av
# Removed: import simplejson as json
import matplotlib.pyplot as plt
import seaborn as sns

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

# Removed: Option to choose input method
# Removed: input_method = st.radio("Choose input method:", ("Upload Audio File", "Record Live Audio"))

# Directly use File uploader
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
        st.write("Loading audio file...")
        y, sr = librosa.load(temp_audio_path)
        st.write(f"Audio loaded successfully with sample rate: {sr}")
        st.write(f"Audio duration: {len(y)/sr:.2f} seconds")

        # Display audio player
        st.audio(audio_bytes, format='audio/wav')

        # Display audio waveform visualization
        st.subheader("Audio Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(y=y, sr=sr, ax=ax)
        st.pyplot(fig)
        plt.close(fig) # Close the figure to prevent it from displaying again


        # Extract MFCCs (40 coefficients)
        st.write("Extracting MFCCs...")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        st.write(f"MFCCs shape: {mfccs.shape}")


        # Pad the MFCCs
        st.write(f"Padding MFCCs to length {max_pad_len}...")
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            st.write(f"Padded MFCCs shape: {mfccs.shape}")

        elif mfccs.shape[1] > max_pad_len:
            # Trim if longer than max_pad_len (this shouldn't happen with the dataset, but good practice)
            mfccs = mfccs[:, :max_pad_len]
            st.write(f"Trimmed MFCCs shape: {mfccs.shape}")


        # Reshape for the model
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        st.write(f"Reshaped MFCCs shape for model: {mfccs.shape}")

        # Make prediction
        st.write("Making prediction...")
        prediction = model.predict(mfccs)
        predicted_digit = np.argmax(prediction)

        # Display the result
        st.success(f"Predicted Digit: {predicted_digit}")

    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        st.error(traceback.format_exc()) # Display detailed traceback

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# Removed the entire elif block for "Record Live Audio"
