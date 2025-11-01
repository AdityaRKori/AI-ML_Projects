import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os

# Load the trained Keras model
# Use a try-except block to handle potential errors during model loading
try:
    model = tf.keras.models.load_model("heart_ecg_model.h5")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop() # Stop the app if the model fails to load

# Define the list of class names and recommendations
# Based on the output from the training, the model was trained on 6 classes: ['F', 'M', 'N', 'Q', 'S', 'V']
# The task description specifies ['F', 'N', 'Q', 'S', 'V'].
# We will use the classes the model was trained on.
# If the model was trained on a different set of classes, the class_names list should match the model's output.
# Assuming the order of classes in the model's output matches the sorted order of class names from the directory:
# ['F', 'M', 'N', 'Q', 'S', 'V']
class_names = ['F', 'M', 'N', 'Q', 'S', 'V'] # Using the specified 5 classes for the app

recommendations = {
    "N": "This pattern appears Normal. According to WHO, continue maintaining a healthy lifestyle with a balanced diet and regular exercise.",
    "S": "This pattern suggests a Supraventricular Ectopic beat. The WHO advises consulting a healthcare professional for a full evaluation to understand the cause and frequency.",
    "V": "This pattern suggests a Ventricular Ectopic beat. The WHO stresses the importance of medical consultation, as frequent ventricular beats can be serious. A doctor may check blood pressure and order further tests.",
    "F": "This pattern suggests a Fusion beat. This is complex. The WHO recommends a thorough review by a cardiologist to determine the underlying heart condition.",
    "Q": "This pattern is classified as Unknown and cannot be determined. The WHO recommends seeking an immediate in-person medical evaluation to get a clear diagnosis."
}

# Get the expected image size from the model's input shape
# The input shape includes batch size, height, width, and channels.
# We need the height and width.
image_height = model.input_shape[1]
image_width = model.input_shape[2]
image_size = (image_height, image_width)


# Set the title of the Streamlit app
st.title("Heart ECG Pattern Classifier")

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload an ECG image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

# Process the uploaded file
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded ECG Image", use_column_width=True)

    # Open and preprocess the image
    try:
        img = Image.open(uploaded_file).convert('RGB') # Ensure image is in RGB
        img = img.resize(image_size) # Resize to the expected size
        img_array = np.array(img) # Convert to NumPy array
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0 # Rescale the image data as done during training
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        st.stop() # Stop the app if image processing fails

    # Make a prediction
    predictions = model.predict(img_array)
    # Get prediction probabilities for the classes
    probabilities = predictions[0]

    # Create a pandas DataFrame to display probabilities
    # Ensure the DataFrame uses the correct class names based on the app's defined classes
    prob_df = pd.DataFrame({
        'Class': class_names,
        'Probability': probabilities[:len(class_names)] # Slice probabilities to match the 5 class names
    })
    st.write("Prediction Probabilities:")
    st.dataframe(prob_df.style.format({'Probability': '{:.4f}'}))

    # Get the predicted class index and name
    # Find the index of the highest probability within the first 5 probabilities
    predicted_class_index = np.argmax(probabilities[:len(class_names)])
    predicted_class_name = class_names[predicted_class_index]

    # Get the recommendation for the predicted class
    recommendation = recommendations.get(predicted_class_name, "No specific recommendation available for this class.")

    # Display the final diagnosis and recommendation with styling
    st.subheader("Final Diagnosis and Recommendation:")
    if predicted_class_name == 'N':
        st.success(f"Diagnosis: {predicted_class_name} - Normal Beat")
        st.success(f"Recommendation: {recommendation}")
    elif predicted_class_name in ['V', 'F']:
        st.error(f"Diagnosis: {predicted_class_name} - Abnormal Beat")
        st.error(f"Recommendation: {recommendation}")
    elif predicted_class_name in ['Q', 'S']:
         st.warning(f"Diagnosis: {predicted_class_name} - Potentially Abnormal Beat")
         st.warning(f"Recommendation: {recommendation}")
    else:
        st.info(f"Diagnosis: {predicted_class_name}")
        st.info(f"Recommendation: {recommendation}")
