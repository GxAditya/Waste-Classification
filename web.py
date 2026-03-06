import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("waste_classifier_model.keras")

# Function to make predictions
def predict_fun(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (96, 96))  # match MobileNetV2 input size
    img = img / 255.0  # normalize as done in training
    img = np.reshape(img, [-1, 96, 96, 3])

    result = model.predict(img)[0][0]

    if result < 0.5:
        return 'The Image Shown is Organic Waste'      # class 0
    else:
        return 'The Image Shown is Recyclable Waste'   # class 1


# Streamlit UI with custom CSS for distinctive design
st.set_page_config(page_title="Waste Classification App", page_icon=":recycle:", layout="wide")

# Add custom CSS for distinctive aesthetics
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        
        :root {
            --primary-color: #2D6A4F;
            --secondary-color: #40916C;
            --accent-color: #95D5B2;
            --bg-gradient-start: #F8FDF9;
            --bg-gradient-end: #E8F5EE;
            --text-primary: #1B4332;
            --text-secondary: #52796F;
            --card-bg: #FFFFFF;
            --shadow: 0 8px 32px rgba(45, 106, 79, 0.12);
        }
        
        * {
            font-family: 'Outfit', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
        }
        
        .main > div {
            max-width: 900px;
            padding-left: 50px;
            padding-right: 50px;
            margin: 0 auto;
        }
        
        .stMarkdown {
            text-align: center;
        }
        
        .stTitle {
            text-align: center;
            font-family: 'Space Grotesk', sans-serif;
        }
        
        div[data-testid="stFileUploader"] {
            text-align: center;
        }
        
        .app-icon-container {
            display: flex;
            justify-content: center;
            margin-bottom: 24px;
        }
        
        .app-icon {
            width: 180px;
            height: 180px;
            padding: 15px;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            box-shadow: var(--shadow);
        }
        
        .info-text {
            font-size: 1.15rem;
            line-height: 1.7;
            margin-bottom: 20px;
            color: var(--text-secondary);
        }
        
        .button-spacing {
            margin-top: 40px;
        }
        
        .social-container {
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-bottom: 24px;
        }
        
        .social-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 44px;
            height: 44px;
            border-radius: 12px;
            background: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            text-decoration: none;
            border: none;
            cursor: pointer;
        }
        
        .social-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(45, 106, 79, 0.2);
        }
        
        .social-btn svg {
            width: 20px;
            height: 20px;
            fill: var(--text-primary);
        }
        
        .feature-card {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 28px;
            margin: 15px 0;
            box-shadow: var(--shadow);
            border: 1px solid rgba(45, 106, 79, 0.08);
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(45, 106, 79, 0.15);
        }
        
        .step-row {
            display: flex;
            align-items: center;
            margin: 16px 0;
        }
        
        .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 36px;
            height: 36px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            font-weight: 600;
            font-size: 14px;
            margin-right: 16px;
        }
        
        .step-text {
            font-size: 1.05rem;
            color: var(--text-secondary);
        }
        
        .cta-button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            padding: 16px 48px;
            border-radius: 14px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(45, 106, 79, 0.3);
            font-family: 'Space Grotesk', sans-serif;
        }
        
        .cta-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(45, 106, 79, 0.4);
        }
        
        .header-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.8rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 8px;
            background: linear-gradient(135deg, var(--text-primary), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.6rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 20px;
        }
        
        .prediction-result {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 20px 32px;
            border-radius: 16px;
            font-size: 1.2rem;
            font-weight: 500;
            margin-top: 20px;
            box-shadow: 0 8px 24px rgba(45, 106, 79, 0.25);
            text-align: center;
        }
        
        .back-button {
            background: transparent;
            color: var(--text-secondary);
            border: 2px solid var(--accent-color);
            padding: 10px 24px;
            border-radius: 10px;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .back-button:hover {
            background: var(--accent-color);
            color: var(--text-primary);
        }
        
        .upload-area {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 40px;
            box-shadow: var(--shadow);
            border: 2px dashed var(--accent-color);
            text-align: center;
        }
        
        .how-it-works {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 32px;
            margin-top: 32px;
            box-shadow: var(--shadow);
        }
    </style>
""", unsafe_allow_html=True)

# SVG Icons as strings
linkedin_icon = """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>"""

twitter_icon = """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>"""

github_icon = """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>"""

arrow_left_icon = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>"""

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

if st.session_state.page == "Home":
    # Social links container - now using buttons with icons
    social_container = st.container()
    with social_container:
        st.markdown(f"""
            <div class="social-container">
                <a href="https://www.linkedin.com/in/aditya-kumar-3721012aa" target="_blank" class="social-btn" title="LinkedIn">
                    {linkedin_icon}
                </a>
                <a href="https://x.com/kaditya264?s=09" target="_blank" class="social-btn" title="Twitter">
                    {twitter_icon}
                </a>
                <a href="https://github.com/GxAditya" target="_blank" class="social-btn" title="GitHub">
                    {github_icon}
                </a>
            </div>
        """, unsafe_allow_html=True)
    
    # Add app icon with animation
    st.markdown("""
        <div class="app-icon-container">
            <img src='https://raw.githubusercontent.com/GxAditya/Waste-Classification/main/waste.png' class='app-icon'>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content with distinctive typography
    st.markdown("<h1 class='header-title'>Waste Classification Using Deep Learning</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text' style='text-align: center;'>This project uses a Convolutional Neural Network (CNN) to classify waste as either Organic or Recyclable.</p>", unsafe_allow_html=True)
    st.markdown("<p class='info-text' style='text-align: center;'>The model is trained on a dataset of labeled images to improve environmental waste management.</p>", unsafe_allow_html=True)
    
    # How it Works section with styled cards
    st.markdown("<h3 class='section-title' style='text-align: center;'>How it Works</h3>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="how-it-works">
            <div class="step-row">
                <span class="step-number">1</span>
                <span class="step-text">Upload an image of waste material.</span>
            </div>
            <div class="step-row">
                <span class="step-number">2</span>
                <span class="step-text">The model analyzes the image and classifies it as Organic or Recyclable.</span>
            </div>
            <div class="step-row">
                <span class="step-number">3</span>
                <span class="step-text">This helps in proper disposal and recycling efforts.</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Center the "Try It Now" button - styled as CTA
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; margin-top: 40px;'>", unsafe_allow_html=True)
        if st.button("Try It Now", use_container_width=True):
            st.session_state.page = "Classification"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "Classification":
    st.markdown("<h1 class='header-title' style='text-align: center;'>Waste Classification</h1>", unsafe_allow_html=True)
    
    # Back button with icon
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()
    
    st.markdown("<p class='info-text' style='text-align: center;'>Upload an image to classify it as Organic or Recyclable.</p>", unsafe_allow_html=True)
    
    # Upload area with custom styling
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Center the image and prediction
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Classify Image", use_container_width=True):
                result = predict_fun(image)
                st.markdown(f"<div class='prediction-result'>Prediction: {result}</div>", unsafe_allow_html=True)
