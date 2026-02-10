# 🍅 Tomato Leaf Disease Detection

A machine learning application that detects diseases in tomato leaves using deep learning and provides an interactive web interface built with Streamlit.

## 📋 Overview

This project uses a trained Keras model to classify tomato leaf images into 10 different categories, including 9 disease types and healthy leaves. The application can process images in batch mode or through an interactive web interface.

## 🎯 Detected Diseases

The model can identify the following conditions:

1. **Bacterial Spot**
2. **Early Blight**
3. **Late Blight**
4. **Leaf Mold**
5. **Septoria Leaf Spot**
6. **Spider Mites**
7. **Target Spot**
8. **Yellow Leaf Curl Virus**
9. **Mosaic Virus**
10. **Healthy**

## 📁 Project Structure

```
├── app.py          # Streamlit web application
├── leaf.py         # Batch processing script
├── tomato_leaf.h5  # Trained Keras model (not included)
└── README.md       # Project documentation
```

## 🛠️ Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- NumPy
- Pillow

## 📦 Installation

1. Clone this repository or download the files:
```bash
git clone <your-repository-url>
cd tomato-leaf-disease-detection
```

2. Install required packages:
```bash
pip install tensorflow streamlit numpy pillow
```

3. Place your trained model file (`tomato_leaf.h5`) in the appropriate directory and update the path in both scripts.

## 🚀 Usage

### Web Application (Recommended)

Run the Streamlit app for an interactive web interface:

```bash
streamlit run app.py
```

This will open a web browser where you can:
- Upload tomato leaf images (JPG, JPEG, PNG)
- View the uploaded image
- Get instant disease predictions
- See prediction probabilities for all classes

### Batch Processing

For processing multiple images at once:

```bash
python leaf.py
```

This script will:
- Process all images in the specified folder
- Print predictions for each image
- Display probability distributions

## ⚙️ Configuration

Before running the scripts, update the file paths:

**In `app.py` and `leaf.py`:**
```python
# Update this path to your model location
model = keras.models.load_model(
    r"C:\Users\muralikrishna\Desktop\leafdiease\tomato_leaf.h5",
    compile=False
)
```

**In `leaf.py` only:**
```python
# Update this path to your images folder
img_folder = r"C:\Users\muralikrishna\Desktop\Leaf\Tomato_images\images"
```

## 📊 Model Details

- **Input Size**: 128 x 128 pixels
- **Format**: RGB images
- **Output**: 10 classes with probability scores

## 💡 Tips

- Use clear, well-lit images of tomato leaves
- Ensure the leaf occupies most of the image frame
- Supported formats: JPG, JPEG, PNG
- The model expects images to be resized to 128x128 pixels (done automatically)

## 🔍 Example Output

```
Image: tomato_leaf_001.jpg
Predicted class: Early_blight
Probabilities: [0.02, 0.87, 0.03, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
----------------------------------------
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- Streamlit for the web application framework
- Dataset contributors for tomato leaf disease images

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This project requires a pre-trained model file (`tomato_leaf.h5`). Make sure you have trained your model or obtained one before running the application.
