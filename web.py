import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("waste_classifier_model.h5")

# Function to make predictions
def predict_fun(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (224, 224))
    img_resized = np.reshape(img_resized, [-1, 224, 224, 3]) / 255.0  # Normalize
    result = np.argmax(model.predict(img_resized))
    if result == 0:
        return 'The Image Shown is Recyclable Waste'
    else:
        return 'The Image Shown is Organic Waste'

# Streamlit UI
st.set_page_config(page_title="Waste Classification App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Classification"])

if page == "Home":
    st.title("Waste Classification Using Deep Learning")
    st.write("This project uses a Convolutional Neural Network (CNN) to classify waste as either Organic or Recyclable.")
    st.write("The model is trained on a dataset of labeled images to improve environmental waste management.")
    
    st.subheader("How it Works")
    st.write("1. Upload an image of waste material.")
    st.write("2. The model analyzes the image and classifies it as Organic or Recyclable.")
    st.write("3. This helps in proper disposal and recycling efforts.")
    
    st.subheader("Connect with Me")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/aditya-kumar-3721012aa)")
    st.markdown("[Twitter](https://x.com/kaditya264?s=09)")
    st.markdown("[GitHub](https://github.com/GxAditya)")

elif page == "Classification":
    st.title("Waste Classification Page")
    st.write("Upload an image to classify it as Organic or Recyclable.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        
        if st.button("Classify Image"):
            result = predict_fun(image)
            st.subheader("Prediction:")
            st.write(result)
