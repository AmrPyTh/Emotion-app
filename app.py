#########################################
#_Libraries 
#########################################
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf 
import numpy as np
import cv2
from PIL import Image

#########################################
#_Load the model
#########################################
model1 = load_model('model-up.keras')

#########################################
#_Define class labels
#########################################
labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#########################################
#_Title
#########################################
title = "<h1 style='text-align: center; color: green; white-space: nowrap;'>Emotion Detection App Using DL</h1>" 
st.markdown(title, unsafe_allow_html=True)

#########################################
#_Upload an image
#########################################
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
#########################################
# Preprocess&Predict the image
#########################################

    def new_predict(pil_img):
        img = np.array(pil_img)                                        # Convert PIL Image to NumPy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                    # Convert to grayscale
        img = cv2.resize(img, (48, 48))                                # Resize to match model input
        img = img / 255.0                                              # Normalize pixel values to [0, 1]
        img = np.expand_dims(img, axis=0)                              # Add batch dimension
        predictions = model1.predict(img)                              # predictions
        predicted_classes = tf.argmax(predictions, axis=1).numpy()     # Get the predicted class index
        return f'Prediction: {labels[predicted_classes[0]]}'           # Return prediction label and index

    if st.button("Predict"):
        st.markdown(f"<h3 style='color: white; text-align: center;; font-size:30px;font-family: 'Times New Roman', Times, serif;'>{new_predict(image)}</h3>", unsafe_allow_html=True)
        st.balloons()
