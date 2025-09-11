import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

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

    # Convert image to array and resize for model input
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predictions
    age_pred = age_model.predict(img_array)
    skin_pred = skin_model.predict(img_array)

    # Format predictions
    age_value = round(float(age_pred[0][0]), 1)  # one decimal place
    skin_classes = ["Wrinkles", "Dark Spots", "Clear Skin", "Puffy Eyes"]
    skin_index = np.argmax(skin_pred[0])
    skin_type = skin_classes[skin_index]

    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Detect faces using Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5)

    # Draw green box and prediction on faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_cv, f"{age_value}, {skin_type}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Convert back to PIL for Streamlit
    result_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    st.image(result_image, caption="Face Analysis Result", use_column_width=True)

    # Display formatted predictions
    st.subheader("Predictions")
    st.write(f"Predicted Age: {age_value}")
    st.write(f"Predicted Skin Type: {skin_type}")
