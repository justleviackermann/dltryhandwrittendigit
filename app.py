import streamlit as st
import os
import numpy as np
import tensorflow as tf
import keras
from PIL import Image, ImageOps, ImageFilter

# 1. Setup & Model Loading
os.environ["KERAS_BACKEND"] = "tensorflow"

@st.cache_resource
def load_my_model():
    return keras.models.load_model('mnist_model.keras')

model = load_my_model()

st.title("🔢 AI Digit Classifier")
st.write("Upload a photo of a handwritten digit (0-9).")

# 2. File Uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    
    # --- PREPROCESSING FACTORY ---
    
    # A. Convert to Grayscale
    image = raw_image.convert('L') 
    
    # B. Invert Colors (Black paper -> White ink)
    image = ImageOps.invert(image) 
    
    # C. THE SPRINKLE FIX: Blur then Threshold
    # This smears those white sprinkles into the orange body to make it one solid shape
    image = image.filter(ImageFilter.GaussianBlur(radius=2)) 
    image = image.point(lambda p: 255 if p > 80 else 0) 
    
    # D. Resize to MNIST standards
    image = image.resize((28, 28), resample=Image.LANCZOS)
    
    # E. Final normalization check (ensure background is pure black)
    img_array = np.array(image) / 255.0  
    img_array = img_array.reshape(1, 28, 28) 

    # --- UI DISPLAY ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_image, caption='Original Upload', use_container_width=True)
    with col2:
        st.image(image, caption='Final AI Input', width=150)

    # 4. Predict
    if st.button('Predict Number'):
        prediction = model.predict(img_array)
        label = np.argmax(prediction)
        confidence = np.max(prediction)
        
        st.divider()
        st.header(f"Result: **{label}**")
        st.write(f"Confidence: {confidence*100:.2f}%")