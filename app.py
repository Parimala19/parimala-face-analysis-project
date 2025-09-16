# app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Face Age & Skin Analyzer", layout="wide")

# ---------- DARK / LIGHT THEME ----------
# Streamlit’s built-in theme switcher is in Settings > Theme.
# To mimic toggle inside the app:
theme = st.toggle("🌗 Dark mode")
if theme:
    st.markdown(
        """
        <style>
        body {background-color: #0E1117; color: #FAFAFA;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------- HEADER ----------
st.markdown(
    "<h1 style='text-align:center;'>🧑‍⚕️ Face Age & Skin Analyzer</h1>",
    unsafe_allow_html=True,
)
st.write("Upload a face photo to estimate **age** and **skin condition**, and download the annotated result.")

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Dummy model outputs for illustration
def dummy_inference(image: Image.Image):
    # Replace this with your model’s prediction logic
    # Age (float)
    age = float(np.random.uniform(12, 60))
    # Skin condition probabilities
    probs = np.random.dirichlet(np.ones(4), size=1)[0]
    return age, probs

# Labels for skin types
skin_labels = ["Wrinkles", "Dark Spots", "Clear Skin", "Puffy Eyes"]

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # -------- INFERENCE --------
    age_pred, skin_probs = dummy_inference(image)
    age_pred = round(age_pred, 1)
    skin_idx = int(np.argmax(skin_probs))
    skin_label = skin_labels[skin_idx]
    skin_conf = skin_probs[skin_idx] * 100

    # -------- ANNOTATE IMAGE --------
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    # Draw simple rectangle (demo) – replace with your bbox coordinates
    w, h = annotated.size
    draw.rectangle([w*0.3, h*0.3, w*0.7, h*0.7], outline="lime", width=4)
    font = ImageFont.load_default()
    text = f"Age: {age_pred} | {skin_label}"
    draw.text((10, 10), text, fill="lime", font=font)

    # --------- UI LAYOUT ----------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Annotated Image")
        st.image(annotated, use_column_width=True)

    st.markdown("### Prediction Results")
    st.write(f"**Predicted Age:** {age_pred} years")
    st.write(f"**Skin Condition:** {skin_label} ({skin_conf:.1f}% confidence)")

    # --------- DOWNLOAD OPTIONS ----------
    # Annotated image
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="📥 Download Annotated Image",
        data=byte_im,
        file_name="annotated.png",
        mime="image/png",
    )

    # CSV predictions
    df = pd.DataFrame(
        {
            "Age": [age_pred],
            "SkinCondition": [skin_label],
            "Confidence(%)": [round(skin_conf, 1)],
        }
    )
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Prediction CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )

# ---------- FOOTER ----------
st.markdown(
    "<hr><p style='text-align:center;'>© 2025 Face Age & Skin Analyzer</p>",
    unsafe_allow_html=True,
)
