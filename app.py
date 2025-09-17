import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import pandas as pd
from io import BytesIO

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Face Analysis", layout="wide")

# ---------- THEME TOGGLE ----------
dark_mode = st.toggle(" Dark mode")
if dark_mode:
    bg_gradient = "#0E1117"
    font_color = "#FAFAFA"
else:
    bg_gradient = "#dff1ff"
    font_color = "#4B9CD3"

# ---------- CSS ----------
st.markdown(f"""
    <style>
    .app-container {{
        background: linear-gradient(135deg, {bg_gradient}, #ffffff);
        padding: 1rem;
        font-family: "Segoe UI", sans-serif;
        color: {font_color};
    }}
    .glass-card {{
        background: rgba(255, 255, 255, 0.75);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }}
    .metric-box {{
        background-color: #4B9CD3;
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
    }}
    .stDownloadButton button {{
        background-color:#4B9CD3;
        color:white;
        font-size:16px;
        border-radius:8px;
        width:100%;
        height:50px;
    }}
    h1, h2, h3, h4, h5 {{
        color: #4B9CD3;
        text-align:center;
    }}
    </style>
    <div class='app-container'>
    """, unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown(
    "<h1>💫 Face Analysis</h1>"
    "<p style='text-align:center; font-size:18px;'>Upload a photo to predict Age & Skin Condition</p>",
    unsafe_allow_html=True
)

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    age_model = tf.keras.models.load_model("age_predictor.keras")
    skin_model = tf.keras.models.load_model("model_skin.keras")
    return age_model, skin_model

age_model, skin_model = load_models()

if "logs" not in st.session_state:
    st.session_state.logs = pd.DataFrame(columns=["Age", "SkinType"])

# Ensure correct class order
skin_classes = ["clear face", "dark spots", "puffy eyes", "wrinkles"]

# ---------- Upload Section ----------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(" Choose an image file", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Preprocess
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
    faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        cv2.putText(img_cv, f"{age_value}, {skin_type}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img_cv, f"{age_value}, {skin_type}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    result_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # ---------- Display Side by Side ----------
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader("Original")
        st.image(image, use_column_width=True)
    with col_img2:
        st.subheader("Annotated")
        st.image(result_image, use_column_width=True)

    # ---------- Metrics ----------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='metric-box'>Predicted Age<br>{age_value}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'>Skin Type<br>{skin_type.title()}</div>", unsafe_allow_html=True)

    # ---------- Logs ----------
    new_row = {"Age": age_value, "SkinType": skin_type}
    st.session_state.logs = pd.concat([st.session_state.logs, pd.DataFrame([new_row])], ignore_index=True)

    # ---------- Downloads ----------
    st.markdown("###  Download Results")
    c1, c2 = st.columns(2)
    with c1:
        img_bytes = BytesIO()
        result_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        st.download_button("⬇️ Annotated Image", data=img_bytes,
                           file_name="annotated_image.png", mime="image/png")
    with c2:
        csv = st.session_state.logs.to_csv(index=False)
        st.download_button("⬇️ Predictions CSV", data=csv,
                           file_name="predictions_log.csv", mime="text/csv")

# close the top-level div
st.markdown("</div>", unsafe_allow_html=True)
