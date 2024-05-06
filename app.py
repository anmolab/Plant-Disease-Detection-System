import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.title("Plant Disease Detection")

uploaded = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Input Cell Image", width=200)

    model = load_model('model.h5')

    def preprocess_image(uploaded_image):
        resized_image = uploaded_image.resize((128, 128))
        image_array = img_to_array(resized_image)
        image_array /= 255.
        return image_array

    def prediction(image_array):
        pred = model.predict(np.expand_dims(image_array, axis=0))
        return pred

    inp = preprocess_image(image)
    ans = prediction(inp)
    classes = [
    "Apple___Black_rot",
    "Tomato___Leaf_Mold",
    "Raspberry___healthy",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Apple___Apple_scab",
    "Tomato___Early_blight",
    "Strawberry___healthy",
    "Apple___Cedar_apple_rust",
    "Soybean___healthy",
    "Potato___Late_blight",
    "Peach___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Strawberry___Leaf_scorch",
    "Tomato___Tomato_mosaic_virus",
    "Grape___healthy",
    "Tomato___Target_Spot",
    "Squash___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Blueberry___healthy",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Peach___healthy",
    "Grape___Esca_(Black_Measles)",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Apple___healthy",
    "Tomato___Septoria_leaf_spot",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Cherry_(including_sour)___healthy",
    "Tomato___Bacterial_spot",
    "Potato___Early_blight",
    "Corn_(maize)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Tomato___healthy",
    "Potato___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Tomato___Late_blight",
    "Grape___Black_rot"
    ]
    st.write(ans); st.write (classes[np.argmax(ans)])