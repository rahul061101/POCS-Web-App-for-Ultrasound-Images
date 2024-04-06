import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import zipfile
import os

# Function to extract and load the trained model
def load_trained_model(zip_file_path):
    # Extract the model from the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("temp_model")

    # Load the model
    model = load_model("temp_model/pcos_detection_model.keras")

    # Clean up extracted files
    os.remove("temp_model/pcos_detection_model.keras")
    os.rmdir("temp_model")

    return model

# Load the trained model
model = load_trained_model("pcos_detection_model.zip")

# Function to predict image
def predict_image(path):
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255
    input_arr = np.array([img_array])
    pred = model.predict(input_arr)
    return "Affected" if pred[0][0] < 0.5 else "Not Affected"

def main():
    st.title("PCOS Detection Using Ultrasound Images")
    image = Image.open("pocs image.png")
    st.image(image)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Make prediction on the uploaded image
        prediction = predict_image(uploaded_file)
        
        # Display prediction in larger font size
        st.markdown(f"<h2>Prediction: {prediction}</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
