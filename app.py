import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences # Import pad_sequences

# Load the trained model
try:
    model = tf.keras.models.load_model('sentiment_model.keras')
    # Re-compile the model after loading
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the tokenizer
try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
except FileNotFoundError:
    st.error("Error: tokenizer.pkl not found. Please ensure it's in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

# Load the maxlen
try:
    with open('maxlen.txt', 'r') as f:
        maxlen = int(f.read())
except FileNotFoundError:
    st.error("Error: maxlen.txt not found. Please ensure it's in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"Error loading maxlen: {e}")
    st.stop()


# Store the class names
class_names = ['Negative', 'Positive']

# Re-define the exact same text-cleaning function
def clean_text(text):
    # a) uses BeautifulSoup to remove any HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # b) converts all text to lowercase
    text = text.lower()
    # c) removes all non-alphabetic characters (but keeps spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Streamlit app title
st.title("Hospital Review Sentiment Analyzer")

# Text area for user input
review_text = st.text_area("Enter a hospital review to analyze:")

# Predict button
if st.button("Predict"):
    if review_text:
        # Get the text, clean it, and put it in a list
        cleaned_text = clean_text(review_text)
        text_list = [cleaned_text]

        # Use the loaded tokenizer to convert this list to a sequence
        sequence = tokenizer.texts_to_sequences(text_list)

        # Use pad_sequences to pad the sequence to maxlen
        padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding='post', truncating='post')

        # Get the model.predict() probability
        prediction = model.predict(padded_sequence)[0][0]

        # Determine the sentiment class
        predicted_class_index = 1 if prediction > 0.5 else 0
        predicted_sentiment = class_names[predicted_class_index]

        # Display the result
        if predicted_sentiment == 'Positive':
            st.success(f"Predicted Sentiment: {predicted_sentiment} (Confidence: {prediction:.2f})")
        else:
            st.error(f"Predicted Sentiment: {predicted_sentiment} (Confidence: {1 - prediction:.2f})")
    else:
        st.warning("Please enter a review to analyze.")
