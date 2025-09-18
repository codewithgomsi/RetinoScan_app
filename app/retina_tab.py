# app/retina_tab.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

from utils import preprocess_image, load_class_names

def retina_tab():
    st.header("🧠 Retina Image Prediction")
    st.write("Upload a fundus image to classify the eye condition using the trained CNN model.")

    uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = preprocess_image(uploaded_file)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Load model and class names
        model = load_model("model/retina_model_for_app_71.08.h5")   #load_model("model/retina_model.h5")
        class_names = load_class_names("data/retina_classes.json")

        # Predict
        preds = model.predict(img)
        pred_class = class_names[str(np.argmax(preds))]


        st.success(f"🩺 Predicted Class: **{pred_class}**")
