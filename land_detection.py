import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


import gdown  # Make sure this is at the top of your file

@st.cache_resource
def load_trained_model():
    """Download model from Google Drive and load it"""
    file_id = "10fj2KgjXvvJbnh15rINSy3N4kPprlLIy"  # ⬅️ REPLACE with your real file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "Modelenv.v1.h5"
    
    if not os.path.exists(output):
        st.info("Downloading model from Google Drive...")
        gdown.download(url, output, quiet=False)
    
    try:
        model = load_model(output)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Set page configuration
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
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading to improve performance
@st.cache_resource
def load_trained_model():
    """Load the pre-trained model"""
    try:
        model = load_model('Modelenv.v1.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure 'Modelenv.v1.h5' is in the same directory as this script.")
        return None

def preprocess_image(uploaded_image):
    """Preprocess the uploaded image for prediction"""
    try:
        # Convert to RGB if needed
        if uploaded_image.mode != 'RGB':
            uploaded_image = uploaded_image.convert('RGB')
        
        # Resize image to model input size
        img_resized = uploaded_image.resize((255, 255))
        
        # Convert to array and normalize
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array, img_resized
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

def predict_image(model, img_array):
    """Make prediction on the preprocessed image"""
    try:
        prediction = model.predict(img_array)
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def create_prediction_chart(prediction, class_names):
    """Create a bar chart showing prediction probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=prediction[0] * 100,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Land Cover Type",
        yaxis_title="Confidence (%)",
        template="plotly_white",
        height=400
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🛰️ Satellite Image Land Cover Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## About This App")
    st.sidebar.markdown("""
    This application uses a Convolutional Neural Network (CNN) to classify satellite images into four land cover types:
    - **Cloudy**: Cloud-covered areas
    - **Desert**: Arid/desert regions
    - **Green Area**: Vegetation/forest areas
    - **Water**: Water bodies
    """)
    
    st.sidebar.markdown("## Model Information")
    st.sidebar.markdown("""
    - **Architecture**: CNN with 3 Conv2D layers
    - **Input Size**: 255x255 pixels
    - **Classes**: 4 land cover types
    - **Training**: 25 epochs with data augmentation
    """)
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if 'Modelenv.v1.h5' exists in the current directory.")
        return
    
    # Class names
    class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Upload Satellite Image</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a satellite image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a satellite image to classify its land cover type"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write(f"**Image Size:** {uploaded_image.size}")
            st.write(f"**Image Mode:** {uploaded_image.mode}")
            st.write(f"**File Size:** {uploaded_file.size} bytes")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown('<h2 class="sub-header">Classification Results</h2>', unsafe_allow_html=True)
            
            # Preprocess image
            with st.spinner('Processing image...'):
                img_array, img_resized = preprocess_image(uploaded_image)
            
            if img_array is not None:
                # Make prediction
                with st.spinner('Classifying image...'):
                    prediction = predict_image(model, img_array)
                
                if prediction is not None:
                    # Get predicted class
                    predicted_class_idx = np.argmax(prediction[0])
                    predicted_class = class_names[predicted_class_idx]
                    confidence = prediction[0][predicted_class_idx] * 100
                    
                    # Display results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### Predicted Class: **{predicted_class}**")
                    st.markdown(f"### Confidence: **{confidence:.2f}%**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Create and display prediction chart
                    fig = create_prediction_chart(prediction, class_names)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show all probabilities
                    st.markdown("### Detailed Probabilities")
                    prob_df = pd.DataFrame({
                        'Land Cover Type': class_names,
                        'Probability (%)': prediction[0] * 100
                    })
                    prob_df = prob_df.sort_values('Probability (%)', ascending=False)
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
    
    # Additional information section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">How to Use</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Step 1: Upload Image**
        - Click on "Browse files" or drag and drop
        - Supported formats: JPG, JPEG, PNG, BMP, TIFF
        - The image will be automatically resized to 255x255 pixels
        """)
    
    with col2:
        st.markdown("""
        **Step 2: View Results**
        - The model will classify the image automatically
        - See the predicted land cover type and confidence score
        - View detailed probabilities for all classes
        """)
    
    with col3:
        st.markdown("""
        **Step 3: Interpret Results**
        - Higher confidence indicates more certain predictions
        - The bar chart shows probability distribution
        - Consider the context of your specific use case
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and TensorFlow")

if __name__ == "__main__":
    main()