import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the trained model
# Using @st.cache_resource so it doesn't reload the model every time you click a button
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mnist_model.keras')

model = load_my_model()

st.title("Handwritten Digit Classifier")
st.write("Upload an image of a digit (28x28) and the model will guess what it is!")

# 2. Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 3. Preprocess for the model (The 28x28 requirement)
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0  # Scale
    img_array = img_array.reshape(1, 28, 28) # Add "batch" dimension
    
    # 4. Predict
    if st.button('Predict'):
        prediction = model.predict(img_array)
        label = np.argmax(prediction)
        confidence = np.max(prediction)
        
        st.success(f"I am {confidence*100:.2f}% sure this is a **{label}**")