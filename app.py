# Import necessary libraries
import streamlit as st  
import tensorflow as tf  
import numpy as np  
import os  
import cv2  

# Load the model and preprocess
def model_predict(image_path):
    model = tf.keras.models.load_model('CNN_Model_V1.keras')

    # Read the input image
    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3  # dimensions

    # Resize image to match input size
    img = cv2.resize(img, (H, W))

    # Convert image from BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to array
    img = np.array(img).astype('float32')  

    # Normalize pixel values
    img = img / 255.0  

    # Reshape image
    img = img.reshape(1, H, W, C)

    # Perform prediction 
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Streamlit sidebar
st.sidebar.title("🌱 Plant Disease Detection System for Sustainable Agriculture")  

# Dropdown menu
app_mode = st.sidebar.selectbox('Select Page', ('Home', 'Disease Detection'))  

# Display an image at top
from PIL import Image
img = Image.open('Prienai_forest.png') 
st.image(img) 

# Home Page
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == 'Disease Detection':
    st.header("🌿 Plant Disease Detection System for Sustainable Agriculture")  # Page title

    # Upload image
    test_image = st.file_uploader("📤 Choose an image:")

    # save uploaded image
    if test_image is not None:
        save_path = os.path.join(os.getcwd(), 'test_image.name')  #save location
        print(save_path) 

        # Save the uploaded image locally
        with open(save_path, 'wb') as f:
            f.write(test_image.getbuffer())

    # To display the image
    if st.button("🖼️ Show Image"):
        st.image(test_image, width=4, use_container_width=True)

    # To make predictions
    if st.button("🔍 Predict"):
        st.write("🔎 Our Prediction:")
        
        # prediction result
        result_index = model_predict(save_path)
        print(result_index) 

        # List of plant diseases
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

        # Display prediction result
        st.spinner()
        st.success(f"✅ Model predicts: **{class_name[result_index]}**")
