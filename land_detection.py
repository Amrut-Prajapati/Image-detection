import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Set page config
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0f5298;
    }
    .prediction-box {
        background-color: #f0f8ff;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e8f4f8;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model function with caching
@st.cache_resource
def load_classification_model():
    """Load the trained model"""
    try:
        model = load_model('Modelenv.v1.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure 'Modelenv.v1.h5' is in the same directory as this script.")
        return None

# Image preprocessing function
def preprocess_image(img):
    """Preprocess image for prediction"""
    img = img.resize((255, 255))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Prediction function
def predict_image(model, img_array):
    """Make prediction on preprocessed image"""
    prediction = model.predict(img_array)
    return prediction

# Class names
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Color mapping for classes
COLOR_MAP = {
    'Cloudy': '#87CEEB',
    'Desert': '#F4A460',
    'Green_Area': '#32CD32',
    'Water': '#1E90FF'
}

def main():
    # Main title
    st.markdown('<h1 class="main-header">🛰️ Satellite Image Land Cover Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Image Classification", "Model Information", "About"])
    
    # Load model
    model = load_classification_model()
    
    if page == "Image Classification":
        classification_page(model)
    elif page == "Model Information":
        model_info_page(model)
    else:
        about_page()

def classification_page(model):
    """Main classification page"""
    st.markdown('<h2 class="sub-header">Upload and Classify Satellite Images</h2>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please check if 'Modelenv.v1.h5' exists in the directory.")
        return
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a satellite image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a satellite image for land cover classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("🔍 Classify Image"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    img_array = preprocess_image(image_display)
                    
                    # Make prediction
                    prediction = predict_image(model, img_array)
                    predicted_class_idx = np.argmax(prediction[0])
                    predicted_class = CLASS_NAMES[predicted_class_idx]
                    confidence = prediction[0][predicted_class_idx] * 100
                    
                    # Store results in session state
                    st.session_state.prediction = prediction[0]
                    st.session_state.predicted_class = predicted_class
                    st.session_state.confidence = confidence
    
    with col2:
        st.markdown("### Results")
        
        if hasattr(st.session_state, 'prediction'):
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Prediction: {st.session_state.predicted_class}</h3>
                <p><strong>Confidence:</strong> {st.session_state.confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create confidence chart
            fig = go.Figure(data=[
                go.Bar(
                    x=CLASS_NAMES,
                    y=st.session_state.prediction * 100,
                    marker_color=[COLOR_MAP[class_name] for class_name in CLASS_NAMES],
                    text=[f"{conf:.1f}%" for conf in st.session_state.prediction * 100],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Confidence Scores for All Classes",
                xaxis_title="Land Cover Type",
                yaxis_title="Confidence (%)",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed results
            st.markdown("### Detailed Results")
            results_df = pd.DataFrame({
                'Land Cover Type': CLASS_NAMES,
                'Confidence (%)': st.session_state.prediction * 100
            })
            results_df = results_df.sort_values('Confidence (%)', ascending=False)
            st.dataframe(results_df, use_container_width=True)
        
        else:
            st.info("Upload an image and click 'Classify Image' to see results.")

def model_info_page(model):
    """Model information page"""
    st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded.")
        return
    
    # Model architecture
    st.markdown("### Model Architecture")
    
    # Create a string buffer to capture model summary
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    
    st.code(model_summary, language='text')
    
    # Model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Details")
        st.write(f"**Total Parameters:** {model.count_params():,}")
        st.write(f"**Input Shape:** {model.input_shape}")
        st.write(f"**Output Shape:** {model.output_shape}")
        st.write(f"**Number of Classes:** {len(CLASS_NAMES)}")
    
    with col2:
        st.markdown("### Classes")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"**{i}:** {class_name}")
    
    # Training information
    st.markdown("### Training Information")
    st.write("""
    - **Dataset:** Satellite images with 4 different land cover types
    - **Training Split:** 80% training, 20% testing
    - **Image Size:** 255x255 pixels
    - **Batch Size:** 32
    - **Epochs:** 25
    - **Optimizer:** Adam
    - **Loss Function:** Categorical Crossentropy
    """)

def about_page():
    """About page"""
    st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🛰️ Satellite Image Land Cover Classification
    
    This application uses a Convolutional Neural Network (CNN) to classify satellite images into four different land cover types:
    
    - **🌧️ Cloudy:** Areas covered by clouds
    - **🏜️ Desert:** Arid and desert regions
    - **🌿 Green Area:** Vegetation and forested areas
    - **💧 Water:** Water bodies like lakes, rivers, and oceans
    
    ### 🔧 Technical Details
    
    The model is built using TensorFlow/Keras with the following architecture:
    - Multiple Convolutional layers with ReLU activation
    - MaxPooling layers for downsampling
    - Dropout for regularization
    - Dense layers for final classification
    
    ### 📊 How to Use
    
    1. **Upload Image:** Go to the "Image Classification" page and upload a satellite image
    2. **Classify:** Click the "Classify Image" button to analyze the image
    3. **View Results:** See the predicted land cover type with confidence scores
    
    ### 🎯 Model Performance
    
    The model was trained on a dataset of satellite images and achieved good accuracy in distinguishing between different land cover types. The confidence scores help you understand how certain the model is about its predictions.
    
    ### 🚀 Technologies Used
    
    - **TensorFlow/Keras:** Deep learning framework
    - **Streamlit:** Web application framework
    - **Plotly:** Interactive visualizations
    - **PIL/Pillow:** Image processing
    - **NumPy/Pandas:** Data manipulation
    """)

if __name__ == "__main__":
    main()