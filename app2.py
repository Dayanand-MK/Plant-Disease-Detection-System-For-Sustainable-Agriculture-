import streamlit as st  
import tensorflow as tf  
import numpy as np  
import os  
import cv2  
from PIL import Image  

# Load the appropriate model based on version
@st.cache_resource
def load_model(version):
    if version == "V1.0":
        return tf.keras.models.load_model('CNN_Model_V1.keras')
    else:
        return tf.keras.models.load_model('CNN_Model_V2.keras')

# Preprocess and predict
def model_predict(image_path, model):
    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3  

    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = np.array(img).astype('float32') / 255.0  
    img = img.reshape(1, H, W, C)  

    prediction = model.predict(img)
    confidence = np.max(prediction) * 100  
    result_index = np.argmax(prediction, axis=-1)[0]

    return result_index, confidence

# UI Customization
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;  
        color: white;
    }
    .css-18e3th9 {
        padding: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Setup
st.sidebar.title("🌱 Plant Disease Detection")
app_mode = st.sidebar.selectbox("📌 Select Page:", ["Home", "Disease Detection V 1.0", "Disease Detection V 2.0"])

# Top Banner
img = Image.open('Prienai_forest.png')  
st.image(img, use_container_width=True)  

if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center; color: lightgreen;'>🍃 Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.markdown("## 🌟 Features of Our App")
    st.markdown("- ✅ Uses Deep Learning for Plant Disease Detection")
    st.markdown("- 🖼️ Upload an image and get instant results")
    st.markdown("- 🌱 Designed for Sustainable Agriculture")

    with st.expander("📌 How to Use This App?"):
        st.write("1. Upload an image of a plant leaf.")
        st.write("2. Click 'Predict' to analyze the disease.")
        st.write("3. Get the disease name and confidence score.")

elif "Disease Detection" in app_mode:
    version = "V1.0" if "1.0" in app_mode else "V2.0"
    model = load_model(version)
    st.header(f"🌿 Plant Disease Detection {version}")

    test_image = st.file_uploader("📤 Choose an image:")

    if test_image is not None:
        save_path = os.path.join(os.getcwd(), 'uploaded_image.jpg')
        with open(save_path, 'wb') as f:
            f.write(test_image.getbuffer())

        col1, col2 = st.columns(2)
        with col1:
            st.image(test_image, use_container_width=True)

        with col2:
            if st.button("🔍 Predict"):
                with st.spinner("Processing Image... ⏳"):
                    result_index, confidence = model_predict(save_path, model)
                    class_name = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy'
                    ]

                    st.success(f"✅ Model predicts: **{class_name[result_index]}**")
                    st.info(f"📊 Confidence: **{confidence:.2f}%**")
