import streamlit as st
import tensorflow as tf

model_path = 'bagongDahon.keras'

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"TrueDivide": tf.keras.layers.Lambda}
    )
    return model
model=load_model()
st.write("""
# Plant Leaf Detection System"""
)
file=st.file_uploader("Choose plant photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data, model):
    size = (299, 299)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image) / 255.0
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model) # change class names based on the new dataset
    class_names=['Alstonia Scholaris', 'Arjun', 'Bael', 'Basil', 'Chinar',
                  'Gauva', 'Jamun', 'Jatropha', 'Lemon', 'Mango', 'Pomegranate', 'Pongamia Pinnata']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)