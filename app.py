import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load your model once at startup
model = keras.models.load_model(
    r"C:\Users\muralikrishna\Desktop\leafdiease\tomato_leaf.h5",
    compile=False
)

# Class names
class_names = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Yellow_Leaf_Curl_Virus",
    "Mosaic_virus",
    "Healthy"
]

st.title("Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image",  use_container_width=True)
    
    # Preprocess image
    img = load_img(uploaded_file, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    
    st.success(f"Prediction: **{predicted_class}**")
    st.write("Probabilities:", predictions.tolist())
