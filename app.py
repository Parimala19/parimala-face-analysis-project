import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("Face Analysis Demo")

# Load models once
@st.cache_resource
def load_models():
    age_model = tf.keras.models.load_model("age_predictor.keras")
    skin_model = tf.keras.models.load_model("model_skin.keras")
    return age_model, skin_model

age_model, skin_model = load_models()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to model input size (assume 224x224; adjust to your model)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predictions
    age_pred = age_model.predict(img_array)
    skin_pred = skin_model.predict(img_array)

    st.subheader("Predictions")
    st.write(f"Age model output: {age_pred}")
    st.write(f"Skin model output: {skin_pred}")
