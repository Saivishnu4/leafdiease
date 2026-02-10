from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

# Load your model
model = keras.models.load_model(
    r"C:\Users\muralikrishna\Desktop\leafdiease\tomato_leaf.h5",
    compile=False
)

# Define your class names
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

# Path to folder with images
img_folder = r"C:\Users\muralikrishna\Desktop\Leaf\Tomato_images\images"

# Loop through all images in folder
for filename in os.listdir(img_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(img_folder, filename)
        
        # Load and preprocess image
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = class_names[predicted_index]
        
        print(f"Image: {filename}")
        print(f"Predicted class: {predicted_label}")
        print(f"Probabilities: {predictions[0]}")
        print("-" * 40)
