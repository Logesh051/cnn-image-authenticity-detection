import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI Image Authenticity Detector",
    page_icon="🛡️",
    layout="wide"
)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "ai_image_detector.keras",
        compile=False
    )

model = load_model()

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#07111f,#0a1630);
}

.block-container{
    padding-top:2rem;
}

.title{
    text-align:center;
    color:white;
    font-size:58px;
    font-weight:700;
}

.subtitle{
    text-align:center;
    color:#d1d5db;
    font-size:22px;
    margin-bottom:25px;
}

.card{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:20px;
    padding:20px;
}

.result-green{
    background:rgba(0,255,100,0.15);
    border-radius:15px;
    padding:20px;
    color:#4ade80;
    font-size:34px;
    font-weight:bold;
    text-align:center;
}

.result-red{
    background:rgba(255,0,0,0.15);
    border-radius:15px;
    padding:20px;
    color:#ff6b6b;
    font-size:34px;
    font-weight:bold;
    text-align:center;
}

.center{
    text-align:center;
}

.small{
    color:#d1d5db;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown(
    """
    <div class="title">
    🛡️ AI Image Authenticity Detector
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload an image and our AI model will analyze it to determine whether it is
    <span style="color:#4ade80;"><b>Real (Genuine)</b></span>
    or
    <span style="color:#ff6b6b;"><b>AI Generated</b></span>.
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# UPLOAD SECTION
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# PREDICTION
# ----------------------------
if uploaded_file:

    original_image = Image.open(uploaded_file).convert("RGB")

    original_size = original_image.size

    image_format = Image.open(uploaded_file).format

    file_size = round(
        uploaded_file.size / (1024 * 1024),
        2
    )

    # Model preprocessing
    image = original_image.resize((224, 224))

    img = np.array(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(
        img,
        verbose=0
    )

    score = float(prediction[0][0])

    left, right = st.columns([1, 2])

    # ----------------------------
    # IMAGE PANEL
    # ----------------------------
    with left:

        st.markdown(
            "### 📷 Uploaded Image"
        )

        st.image(
            original_image,
            width=300
        )

    # ----------------------------
    # RESULT PANEL
    # ----------------------------
    with right:

        st.markdown(
            "### 📊 Analysis Result"
        )

        # Swap labels if required
        if score > 0.5:

            result = "AI GENERATED IMAGE"
            confidence = score * 100

            st.markdown(
                """
                <div class="result-red">
                ⚠️ AI GENERATED IMAGE
                </div>
                """,
                unsafe_allow_html=True
            )

            summary = (
                "The uploaded image contains patterns commonly found in AI-generated content."
            )

        else:

            result = "REAL IMAGE (GENUINE)"
            confidence = (1 - score) * 100

            st.markdown(
                """
                <div class="result-green">
                ✅ REAL IMAGE (GENUINE)
                </div>
                """,
                unsafe_allow_html=True
            )

            summary = (
                "The uploaded image appears to contain natural visual patterns and is likely a genuine photograph."
            )

        st.write("")

        st.subheader("Confidence Score")

        st.progress(confidence / 100)

        st.markdown(
            f"""
            <h1 style='text-align:center;color:#4ade80'>
            {confidence:.1f}%
            </h1>
            """,
            unsafe_allow_html=True
        )

        st.write("---")

        st.markdown("### 📋 Details")

        st.write("🤖 **Model Used:** CNN (Custom Trained)")
        st.write(
            f"📅 **Analyzed On:** {datetime.now().strftime('%d %B %Y, %I:%M %p')}"
        )
        st.write(
            f"🖼️ **Image Resolution:** {original_size[0]} x {original_size[1]}"
        )
        st.write(
            f"📦 **File Size:** {file_size} MB"
        )
        st.write(
            f"📄 **Format:** {image_format}"
        )
        st.write(
            "🎯 **Prediction Threshold:** 50%"
        )

        st.write("---")

        st.markdown("### 📝 Analysis Summary")

        st.write(summary)

        st.info(
            "This tool uses a deep learning model trained on real and AI-generated images. Results may not be 100% accurate."
        )