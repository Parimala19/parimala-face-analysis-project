import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_models():
    skin_model = tf.keras.models.load_model("model_skin.keras", compile=False)
    age_model  = tf.keras.models.load_model("age_predictor.keras", compile=False)
    return skin_model, age_model

skin_model, age_model = load_models()

st.title("Skin Condition + Age Prediction")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    img = image.resize((224,224))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr,0)

    skin_pred = skin_model.predict(arr)
    age_pred  = age_model.predict(arr)

    skin_classes = ["Wrinkles","Puffy Eyes","Clear Skin","Dark Spots"]
    st.write("Skin Condition:", skin_classes[np.argmax(skin_pred)])
    st.write("Predicted Age:", int(age_pred[0][0]))
