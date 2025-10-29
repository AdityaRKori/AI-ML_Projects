import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load the trained model

# Load the trained model
# Assuming the model was saved as 'random_forest_model.joblib' in the previous steps
# If not, the model object 'model' from the notebook needs to be saved first.
try:
    model = joblib.load('random_forest_model.joblib')
except FileNotFoundError:
    st.error("Model file 'random_forest_model.joblib' not found. Please ensure the model is saved.")
    st.stop()

st.title("Cancer Classification App")

st.write("""
This application uses a trained machine learning model to predict cancer type (ALL or AML)
based on gene expression data.
""")

st.header("Enter Gene Expression Data")

# Define the number of features the model expects
# This should match X_train.shape[1] which is 7129
n_features = 7129 # Replace with the actual number of features if different

st.write(f"Please enter {n_features} comma-separated gene expression values.")
st.write("Example: 15,-114,2,193,-51,...")

gene_expression_input = st.text_area("Gene Expression Values")

predict_button = st.button("Predict")

if predict_button:
    if gene_expression_input:
        try:
            # Process the input data
            # Split the input string by comma and convert to float
            gene_expression_values = [float(x.strip()) for x in gene_expression_input.split(',')]

            # Check if the number of input values matches the expected number of features
            if len(gene_expression_values) != n_features:
                st.error(f"Incorrect number of gene expression values. Expected {n_features}, but got {len(gene_expression_values)}.")
            else:
                # Convert the list to a numpy array and reshape for the model
                input_data = np.array(gene_expression_values).reshape(1, -1)

                # Make prediction
                prediction = model.predict(input_data)

                # Display the prediction
                st.header("Prediction Result")
                st.success(f"The predicted cancer type is: **{prediction[0]}**")

        except ValueError:
            st.error("Invalid input. Please ensure all values are numbers and are comma-separated.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter gene expression values to get a prediction.")

st.header("About the Dataset and Model")

st.write("""
This model was trained on a gene expression dataset from Kaggle, specifically
the leukemia dataset containing samples from patients with Acute Lymphoblastic Leukemia (ALL)
and Acute Myeloid Leukemia (AML). The dataset contains expression levels for over 7,000 genes.

A Random Forest Classifier model was trained on a portion of this data to
distinguish between ALL and AML based on the gene expression profiles.
""")

st.header("Guidance on Prediction Results")

st.write("""
**Disclaimer:** This application is for informational purposes only and should
not be used as a substitute for professional medical advice, diagnosis, or treatment.

*   **If the model predicts 'ALL' or 'AML':** This indicates that the gene expression
    pattern of the input sample is similar to patterns observed in samples
    from patients with that specific type of leukemia in the training data.
*   **Important Note:** This prediction is based *only* on the provided gene
    expression data and the patterns the model learned. Many other factors
    are considered in a clinical diagnosis.
*   **What to do if the model indicates cancer:** If you receive a prediction
    of 'ALL' or 'AML', it is crucial to **consult with a qualified healthcare
    professional**. Share this information with your doctor, but do not make
    any medical decisions based solely on this application's output. Further
    medical tests and expert evaluation are necessary for a definitive diagnosis.
""")
import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load the trained model
import shap # To explain model predictions
import matplotlib.pyplot as plt # For plotting SHAP values

# Load the trained model
try:
    model = joblib.load('random_forest_model.joblib')
except FileNotFoundError:
    st.error("Model file 'random_forest_model.joblib' not found. Please ensure the model is saved.")
    st.stop()

# Create a SHAP explainer object
# Assuming X_train is available or can be re-created for the explainer
# For simplicity, we will create a dummy explainer here. In a real app,
# you would ideally save and load the explainer or use a representative background dataset.
# Since we don't have access to X_train here, we'll re-create a simple TreeExplainer
# Note: For accurate SHAP values, using a background dataset (e.g., a sample of the training data)
# with KernelExplainer or DeepExplainer is often recommended, but TreeExplainer is suitable for tree models.
try:
    explainer = shap.TreeExplainer(model)
    # For TreeExplainer, a background dataset is not strictly necessary but can be used
    # to calculate expected values. We'll proceed without a background dataset for simplicity.
except Exception as e:
    st.error(f"Error creating SHAP explainer: {e}")
    st.stop()


st.title("Cancer Classification App")

st.write("""
This application uses a trained machine learning model to predict cancer type (ALL or AML)
based on gene expression data.
""")

st.header("Enter Gene Expression Data")

# Define the number of features the model expects
# This should match X_train.shape[1] which is 7129
n_features = 7129 # Replace with the actual number of features if different

st.write(f"Please enter {n_features} comma-separated gene expression values.")
st.write("Example: 15,-114,2,193,-51,...")

gene_expression_input = st.text_area("Gene Expression Values")

predict_button = st.button("Predict")

if predict_button:
    if gene_expression_input:
        try:
            # Process the input data
            # Split the input string by comma and convert to float
            gene_expression_values = [float(x.strip()) for x in gene_expression_input.split(',')]

            # Check if the number of input values matches the expected number of features
            if len(gene_expression_values) != n_features:
                st.error(f"Incorrect number of gene expression values. Expected {n_features}, but got {len(gene_expression_values)}.")
            else:
                # Convert the list to a numpy array and reshape for the model
                input_data = np.array(gene_expression_values).reshape(1, -1)

                # Make prediction
                prediction = model.predict(input_data)

                # Display the prediction
                st.header("Prediction Result")
                st.success(f"The predicted cancer type is: **{prediction[0]}**")

                # --- SHAP Explanation ---
                st.header("Explanation of the Prediction (SHAP)")

                # Calculate SHAP values for the input data
                # The explainer expects the input data to be in the same format as the training data
                shap_values = explainer.shap_values(input_data)

                # Determine the class index for the prediction
                predicted_class_index = np.where(model.classes_ == prediction[0])[0][0]

                # Generate a SHAP force plot for the individual prediction
                # Note: force_plot works best in a Jupyter environment. For Streamlit,
                # we can use other SHAP plots like waterfall or the summary plot for individual instances.
                # Waterfall plot is suitable for explaining individual predictions.

                st.subheader(f"How Gene Expression Contributed to the '{prediction[0]}' Prediction")

                # Need feature names for the plot
                # Assuming feature names are available or can be loaded
                # For now, let's use generic names or load them if available
                # If df is available from previous steps, we can get feature names from there
                # Assuming 'Gene Accession Number' from the original df are the feature names
                try:
                     # This assumes df is available in the environment or can be loaded
                     # In a real app, you'd load this data or save feature names separately
                     # For this example, we'll use placeholder names if df is not available
                     # Let's try to load df if it's not in the environment
                     # If df is not defined, this will raise a NameError
                     feature_names = pd.read_csv("gene-expression/data_set_ALL_AML_train.csv")['Gene Accession Number'].tolist()
                except (NameError, FileNotFoundError):
                     st.warning("Could not load feature names. Using generic names.")
                     feature_names = [f"Feature {i}" for i in range(n_features)]


                # Generate and display the waterfall plot
                # Need to handle the case where shap_values is a list (multi-output)
                if isinstance(shap_values, list):
                    # Select the SHAP values for the predicted class
                    shap_values_for_plot = shap_values[predicted_class_index][0] # Select the first (and only) sample
                else:
                    # If shap_values is a numpy array (single output or binary classification)
                    shap_values_for_plot = shap_values[0] # Select the first (and only) sample

                # Ensure the expected_value is for the predicted class
                expected_value = explainer.expected_value
                if isinstance(expected_value, list):
                     expected_value_for_plot = expected_value[predicted_class_index]
                else:
                     expected_value_for_plot = expected_value


                # Create a SHAP Explanation object for the waterfall plot
                # The Explanation object needs values, base_values, data, and feature_names
                # base_values is the expected_value
                # data is the original feature values for the instance
                shap_explanation = shap.Explanation(values=shap_values_for_plot,
                                                    base_values=expected_value_for_plot,
                                                    data=input_data[0], # Select the first (and only) sample data
                                                    feature_names=feature_names)

                # Generate the waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(shap_explanation, max_display=10, show=False) # show=False to prevent immediate display
                st.pyplot(fig) # Display the matplotlib figure in Streamlit

                # --- End of SHAP Explanation ---


        except ValueError:
            st.error("Invalid input. Please ensure all values are numbers and are comma-separated.")
        except Exception as e:
            st.error(f"An error occurred during prediction or SHAP explanation: {e}")
    else:
        st.warning("Please enter gene expression values to get a prediction.")
