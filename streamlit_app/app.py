# streamlit_app/app.py
import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
from tf_keras.models import load_model 

# Page Config
st.set_page_config(
    page_title="SkinLesion AI - Skin Cancer Detection",
    page_icon="medkit",
    layout="centered",
    initial_sidebar_state="collapsed"  # hide sidebar
)


# Custom CSS - Professional & Readable
st.markdown("""
<style>
    .main {background-color: #f8f9fa; padding: 20px;}
    .header {font-size: 48px; text-align: center; color: #1e3a8a; font-weight: bold;}
    .subtitle {text-align: center; color: #495057; font-size: 18px;}
    .stButton>button {
        background-color: #007bff; color: white; border-radius: 12px;
        height: 60px; font-size: 18px; font-weight: bold; width: 100%;
    }
    .stButton>button:hover {background-color: #0056b3;}
    .lesion-box {
        padding: 20px; border-left: 6px solid #007bff;
        background-color: #e8f4fc;  /* ← رنگ تیره‌تر و خواناتر */
        color: #1e3a8a;              /* ← متن تیره */
        border-radius: 12px;
        margin: 15px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        font-size: 16px;
    }
    .risk-high {color: #dc3545; font-weight: bold;}
    .risk-moderate {color: #fd7e14; font-weight: bold;}
    .risk-low {color: #28a745; font-weight: bold;}
    .footer {text-align: center; color: #6c757d; font-size: 14px; margin-top: 50px;}
</style>
""", unsafe_allow_html=True)


# Focal Loss
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        pt = y_true * y_pred + (1-y_true)*(1-y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1-pt, gamma) * tf.math.log(pt))
    return loss


# Model Loader
@st.cache_resource
def load_model():
    model_url = "https://drive.google.com/uc?export=download&id=1H7ioE6wU2uWSGL3V37egrJTo3RE6DRqu"
    model_path = "/tmp/best_final_model.keras"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model (387 MB)... This may take 2-3 minutes on first load."):
            try:
                response = requests.get(model_url, timeout=300)
                response.raise_for_status()
                with open(model_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
                st.stop()

    try:
       # when use tf: model = tf.keras.models.load_model --> for use on local
       # when use tf-keras --> for using streamlit cloud because tf is heavy
        model = load_model(
            model_path,
            custom_objects={'focal_loss': focal_loss},
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()


# Class Info
class_info = {
    "AK": {"name": "Actinic Keratosis (AK)", "risk": "Pre-cancerous", "desc": "Rough, scaly patches on sun-exposed skin. Can progress to SCC.", "source": "AAD"},
    "BCC": {"name": "Basal Cell Carcinoma (BCC)", "risk": "Low metastasis", "desc": "Most common skin cancer. Pearly bump. Rarely spreads.", "source": "Skin Cancer Foundation"},
    "BKL": {"name": "Benign Keratosis (BKL)", "risk": "Benign", "desc": "Seborrheic keratosis, solar lentigo. Warty, stuck-on appearance.", "source": "DermNet NZ"},
    "DF": {"name": "Dermatofibroma (DF)", "risk": "Benign", "desc": "Firm, reddish-brown bump on legs. Harmless.", "source": "BAD"},
    "MEL": {"name": "Melanoma (MEL)", "risk": "High risk", "desc": "Most dangerous. Asymmetrical, irregular, multi-colored.", "source": "NCI"},
    "NV": {"name": "Melanocytic Nevus (NV)", "risk": "Benign (mostly)", "desc": "Common mole. Round, even color, stable.", "source": "ISIC"},
    "SCC": {"name": "Squamous Cell Carcinoma (SCC)", "risk": "Moderate risk", "desc": "Scaly patches, open sores. Can metastasize.", "source": "Mayo Clinic"},
    "VASC": {"name": "Vascular Lesion (VASC)", "risk": "Benign", "desc": "Red, raised, may bleed. Usually harmless.", "source": "DermNet NZ"}
}

class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']


# Header
st.markdown("<h1 class='header'>SkinLesion AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced skin cancer detection using ResNet50 – 75% accuracy on ISIC 2019</p>", unsafe_allow_html=True)
st.markdown("---")


# Load Model 
model = load_model()  

# ============================
# Step 1: Choose Input Method
# ============================
st.markdown("### Step 1: Select Input Method")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Upload Image"):
        st.session_state.input_mode = "upload"
        st.session_state.image = None
        st.session_state.prediction = None
with col2:
    if st.button("Take Photo"):
        st.session_state.input_mode = "camera"
        st.session_state.image = None
        st.session_state.prediction = None
with col3:
    if st.button("Clear All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]

st.markdown("---")

# ============================
# Step 2: Upload / Capture Image
# ============================
if 'input_mode' in st.session_state:
    st.markdown("### Step 2: Provide Image")
    img = None
    
    if st.session_state.input_mode == "upload":
        uploaded = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key="uploader")
        if uploaded:
            img = Image.open(uploaded)
            
    elif st.session_state.input_mode == "camera":
        img = st.camera_input("Take a photo of the lesion", key="camera")
        if img:
            img = Image.open(img)
    
    if img is not None:
        st.session_state.image = img
        st.image(img, caption="Your image", width=300)

# ============================
# Step 3: Predict Button
# ============================
if 'image' in st.session_state:
    st.markdown("### Step 3: Analyze Image")
    if st.button("Predict Now", type="primary", key="predict_btn"):
        img = st.session_state.image
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = preprocess_input(img_array) 
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner("Analyzing..."):
            preds = model.predict(img_array)[0]
            pred_idx = np.argmax(preds)
            pred_class = class_names[pred_idx]
            confidence = preds[pred_idx]
            
        st.session_state.prediction = {
            "class": pred_class,
            "confidence": confidence,
            "probs": preds
        }
        st.success("Analysis complete!")
        st.rerun() 

# ============================
# Display Results
# ============================
if 'prediction' in st.session_state and st.session_state.prediction is not None:
    pred = st.session_state.prediction
    st.markdown(f"## **Prediction: {pred['class']}**")
    st.markdown(f"### Confidence: **{pred['confidence']:.1%}**")
    
    prob_dict = {name: float(p) for name, p in zip(class_names, pred['probs'])}
    st.bar_chart(prob_dict)
    
    info = class_info[pred['class']]
    risk_class = "risk-high" if "High" in info["risk"] else "risk-moderate" if "Moderate" in info["risk"] else "risk-low"
    st.markdown(f"""
    <div class='lesion-box'>
        <h3>{info['name']}</h3>
        <p><strong>Risk Level:</strong> <span class='{risk_class}'>{info['risk']}</span></p>
        <p>{info['desc']}</p>
        <p><small><em>Source: {info['source']}</em></small></p>
    </div>
    """, unsafe_allow_html=True)
else:
    if 'image' in st.session_state:
        st.info("Image is ready. Click **Predict Now** to analyze.")

# ============================
# Footer
# ============================
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p><strong>Important:</strong> This is a research tool. Always consult a dermatologist.</p>
    <p>© 2025 Elnaz Parsaei | ResNet50 + TensorFlow + Streamlit</p>
</div>
""", unsafe_allow_html=True)