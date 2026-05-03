import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("cnn_model.h5")

st.title("Image Classification")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(64,64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)/255.0
    prediction = model.predict(img_array)
    st.write(prediction)
