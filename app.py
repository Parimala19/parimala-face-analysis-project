import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Face Analysis Demo", layout="wide")

st.markdown("<h1 style='text-align:center;'>Face Analysis Demo</h1>", unsafe_allow_html=True)
st.write("Upload an image to predict age and skin type. The system will mark detected faces.")

@st.cache_resource
def load_models():
    age_model = tf.keras.models.load_model("age_predictor.keras")
    skin_model = tf.keras.models.load_model("model_skin.keras")
    return age_model, skin_model

age_model, skin_model = load_models()

# session state for logging
if "logs" not in st.session_state:
    st.session_state.logs = pd.DataFrame(columns=["Age", "SkinType"])

skin_classes = ["Wrinkles", "Dark Spots", "Clear Skin", "Puffy Eyes"]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Preprocess for model input
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predictions
    age_pred = age_model.predict(img_array)
    skin_pred = skin_model.predict(img_array)

    age_value = round(float(age_pred[0][0]), 1)
    skin_index = np.argmax(skin_pred[0])
    skin_type = skin_classes[skin_index]

    # Annotate image
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_cv, f"{age_value}, {skin_type}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    result_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    st.markdown("### Predictions")
    st.write(f"**Predicted Age:** {age_value}")
    st.write(f"**Predicted Skin Type:** {skin_type}")

    # Add to logs
    new_row = {"Age": age_value, "SkinType": skin_type}
    st.session_state.logs = pd.concat([st.session_state.logs, pd.DataFrame([new_row])], ignore_index=True)

    # Show annotated image
    st.image(result_image, caption="Annotated Image", use_column_width=True)

    # Download annotated image
    img_bytes = BytesIO()
    result_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    st.download_button("Download Annotated Image", data=img_bytes, file_name="annotated_image.png", mime="image/png")

    # Download CSV logs
    csv = st.session_state.logs.to_csv(index=False)
    st.download_button("Download Predictions CSV", data=csv, file_name="predictions_log.csv", mime="text/csv")
