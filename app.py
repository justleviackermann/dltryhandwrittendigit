import streamlit as st
import os

# Set the backend before importing keras
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
import numpy as np

from PIL import Image, ImageOps, ImageFilter

# ... inside your upload block ...
image = ImageOps.invert(image)

# NEW: Blur it slightly and then threshold it to "smear" the lines together
# This fills in those sprinkle holes!
image = image.filter(ImageFilter.GaussianBlur(radius=1)) 
image = image.point(lambda p: 255 if p > 100 else 0) # Hard threshold
# 1. Load the trained model
@st.cache_resource
def load_my_model():
    # Ensure this file is in your GitHub repo!
    return keras.models.load_model('mnist_model.keras')

model = load_my_model()

# UI Layout
st.title("🔢 AI Digit Classifier")
st.write("Upload a photo of a handwritten digit (0-9).")
st.info("Tip: Use a dark pen on white paper and center the digit!")

# 2. File Uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    
    # 1. Convert to grayscale
    image = raw_image.convert('L') 
    
    # 2. Invert colors
    image = ImageOps.invert(image) 
    
    # 3. NEW: Boost Contrast (Binarization)
    # This makes anything dim turn black and anything bright turn pure white
    image = image.point(lambda p: 255 if p > 140 else 0) 
    
    # 4. Resize
    image = image.resize((28, 28))
    
    # D. Display what the model sees (for debugging)
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_image, caption='Original Upload', use_container_width=True)
    with col2:
        st.image(image, caption='Model Input (28x28 Inverted)', width=150)

    # 3. Prepare for Prediction
    # Scale pixels to 0-1 range
    img_array = np.array(image) / 255.0  
    # Reshape to (1, 28, 28) - 1 image, 28 height, 28 width
    img_array = img_array.reshape(1, 28, 28) 

    # 4. Predict Button
    if st.button('Predict Number'):
        with st.spinner('The AI is thinking...'):
            prediction = model.predict(img_array)
            label = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Show result
            st.divider()
            st.header(f"Result: **{label}**")
            st.write(f"Confidence: {confidence*100:.2f}%")
            
            if confidence < 0.70:
                st.warning("The AI is a bit unsure. Try a clearer photo or a bolder pen!")