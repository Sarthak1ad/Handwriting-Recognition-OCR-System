import os
import sys
import time
import cv2
import numpy as np
import streamlit as st

# Add the project root to python search path to resolve src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_image
from src.ocr_pipeline import run_ocr_pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="Handwritten OCR Dashboard",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Injection for Premium Glassmorphism Design
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Gradient */
    .title-gradient {
        background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    
    /* Custom Sidebar Card */
    .sidebar-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Metrics Card */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #818cf8;
        margin: 0.2rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar UI Configuration
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### ✏️ Pipeline Settings")
    st.write("Tune parameters for raw image preprocessing.")
    
    # Preprocessing toggles
    enable_deskew = st.checkbox("Enable Auto-Deskewing", value=True, help="Automatically rotates and aligns tilted handwriting.")
    save_outputs = st.checkbox("Save Visualization", value=True, help="Draws colored bounding boxes for line, word, and character boundaries.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### 🧬 EMNIST CNN Model")
    st.write("**Architecture:** Custom PyTorch CNN")
    st.write("**Classes:** 47 (Digits + Letters merged)")
    st.write("**Checkpoint:** `ocr_model.pth` (~1.7 MB)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar footer
    st.info("Built using OpenCV and PyTorch. Fully modular deep learning OCR pipeline.")

# 4. Main Body Header
st.markdown('<h1 class="title-gradient">Handwritten OCR System</h1>', unsafe_allow_html=True)
st.write("Upload an image of handwritten text (lines or paragraphs) to segment it and recognize the text using our custom deep learning classifier.")

# 5. Image Uploader File Input
uploaded_file = st.file_uploader("Upload Image (PNG, JPEG, JPG)", type=["png", "jpg", "jpeg"])

# Manage state for clearing results
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = None

if uploaded_file is not None:
    # Read bytes and convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Layout splits: Image Preview on the Left, Recognized Output on the Right
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("🖼️ Input Processing Stages")
        
        # Tabs for visual progression
        tab_orig, tab_bin, tab_vis = st.tabs(["Original Upload", "Binarized Mask", "Segmented Bounding Boxes"])
        
        with tab_orig:
            st.image(image, channels="BGR", use_container_width=True)
            
        with tab_bin:
            # Generate preprocessed mask in real-time
            binary = preprocess_image(image, deskew=enable_deskew)
            st.image(binary, caption="Inverted Binary Mask (White text on black)", use_container_width=True)
            
        with tab_vis:
            # We will show the visualization after pipeline runs
            vis_container = st.empty()
            vis_container.info("Click the 'Recognize Text' button in the right column to view bounding boxes.")

    with col2:
        st.subheader("📝 OCR Recognized Output")
        
        # Button trigger
        rec_button = st.button("🚀 Recognize Text", use_container_width=True)
        
        # Path configuration
        model_path = "models/ocr_model.pth"
        vis_path = "data/output/streamlit_vis.png"
        
        if rec_button or st.session_state.ocr_results is not None:
            # Run OCR if button is clicked
            if rec_button:
                if not os.path.exists(model_path):
                    st.error("❌ Trained model file not found in `models/ocr_model.pth`. Please run training first!")
                else:
                    with st.spinner("Analyzing image structure and running CNN classifier..."):
                        start_time = time.time()
                        text, char_metadata = run_ocr_pipeline(
                            image_path_or_np=image,
                            model_path=model_path,
                            save_vis=save_outputs,
                            vis_path=vis_path
                        )
                        elapsed = time.time() - start_time
                        
                        # Calculate statistics
                        avg_confidence = np.mean([char["confidence"] for char in char_metadata]) if char_metadata else 0.0
                        char_count = len(char_metadata)
                        
                        # Store in state
                        st.session_state.ocr_results = {
                            "text": text,
                            "elapsed": elapsed,
                            "char_count": char_count,
                            "avg_confidence": avg_confidence
                        }
            
            # Show results if stored in state
            results = st.session_state.ocr_results
            if results:
                # Update visual tab
                if save_outputs and os.path.exists(vis_path):
                    with tab_vis:
                        st.image(vis_path, caption="Blue: Lines | Green: Words | Red: Characters", use_container_width=True)
                
                # Display metrics cards
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Processing Time</div>
                        <div class="metric-value">{results['elapsed']:.3f}s</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Detected Characters</div>
                        <div class="metric-value">{results['char_count']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Avg. Confidence</div>
                        <div class="metric-value">{results['avg_confidence']:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("")
                
                # Output Text Box
                st.text_area("Recognized Text:", value=results['text'], height=200)
                
                # Action Buttons (Download & Clear)
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    st.download_button(
                        label="📥 Download Recognized Text",
                        data=results['text'],
                        file_name="ocr_output.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with action_col2:
                    clear_btn = st.button("🧹 Clear Results", use_container_width=True)
                    if clear_btn:
                        st.session_state.ocr_results = None
                        st.rerun()
else:
    # Initial landing helper message
    st.session_state.ocr_results = None
    st.info("ℹ️ Please upload an image of handwritten text using the file uploader above to begin.")
