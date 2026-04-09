import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = r'C:\Users\Win10\Downloads\hoa9_2\bagongDahon.keras'
IMG_SIZE = (299, 299)

CLASS_NAMES = [
    'Alstonia Scholaris', 'Arjun', 'Bael', 'Basil', 'Chinar',
    'Gauva', 'Jamun', 'Jatropha', 'Lemon', 'Mango',
    'Pomegranate', 'Pongamia Pinnata'
]


@st.cache_resource
def get_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"TrueDivide": tf.keras.layers.Lambda}
    )

model = get_model()


st.title("🌿 Plant Leaf Detection System")
st.caption("Upload a leaf image to identify its class")

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(image):
    processed = preprocess_image(image)
    preds = model.predict(processed)
    class_index = np.argmax(preds)
    confidence = np.max(preds)
    return CLASS_NAMES[class_index], confidence


uploaded_file = st.file_uploader(
    "📤 Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing..."):
        label, confidence = predict(image)

    st.success(f"🌱 Prediction: **{label}**")
    st.info(f"Confidence: {confidence:.2%}")

else:
    st.warning("Please upload an image to start prediction.")