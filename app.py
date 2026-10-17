import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from config import CANVAS_SIZE, STROKE_WIDTH
from model_loader import load_pytorch_model, load_tensorflow_model
from preprocessing import preprocess_canvas
from predictor import predict
from logger import log_prediction

# PAGE CONFIG
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✏️",
    layout="wide"
)

# TITLE
st.title("MNIST Digit Classifier")
st.write("Draw a digit, customize canvas settings, and compare models side-by-side.")

# 1. FRAMEWORK SELECTION (CHECKBOXES)
st.subheader("Select Models to Run:")
col_pt, col_tf, _ = st.columns([1.2, 1.2, 5]) 
with col_pt:
    use_pt = st.checkbox("PyTorch Model", value=True)
with col_tf:
    use_tf = st.checkbox("TensorFlow Model", value=False)


# LOAD MODEL
# Cache the model so it doesn't reload on every canvas update
@st.cache_resource
def get_pt_model(): return load_pytorch_model()
@st.cache_resource
def get_tf_model(): return load_tensorflow_model()

pt_model = get_pt_model() if use_pt else None
tf_model = get_tf_model() if use_tf else None

# SIDEBAR
st.sidebar.header("Drawing Settings")
stroke_width = st.sidebar.slider(
    "Brush size",
    min_value=5,
    max_value=40,
    value=STROKE_WIDTH
)

stroke_color = st.sidebar.color_picker("Brush Color", "#000000")
bg_color = st.sidebar.color_picker("Background Color", "#FFFFFF")
# brush_blur = st.sidebar.slider("Brush Blur (Softness)", 0, 5, 0)


# CLEAR BUTTON
# Initialize session state for canvas key
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# Increment key to reset the canvas
if st.button("Clear / Reset"):
    st.session_state.canvas_key += 1
    st.rerun()
    
# In st_canvas(), use the dynamic key
# key=f"digit_canvas_{st.session_state.canvas_key}"

# CANVAS
canvas_result = st_canvas(
    fill_color="rgba(255,255,255,0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    drawing_mode="freedraw",
    #update_streamlit=True,
    #realtime_update=True,
    return_image_data=True,
    key=f"digit_canvas_{st.session_state.canvas_key}"
)

# PREDICTION
if canvas_result.image_data is not None:
    if canvas_result.image_data[..., :3].max() > 0:
        
        if not use_pt and not use_tf:
            st.warning("Please select at least one model framework from the top checkboxes!")
        else:
            num_cols = sum([use_pt, use_tf])
            result_cols = st.columns(num_cols)
            col_index = 0
            
            # --- PYTORCH RESULT ---
            if use_pt:
                with result_cols[col_index]:
                    st.markdown("### 🔥 PyTorch Results")
                    if pt_model is None:
                        st.error("PyTorch model could not be loaded.")
                    else:
                        tensor_pt, img_pt = preprocess_canvas(canvas_result.image_data, "pytorch")
                        res_pt = predict(pt_model, tensor_pt, "pytorch")

                        log_prediction(res_pt[0]['digit'], res_pt)                        
                        st.metric("Top Prediction", res_pt[0]['digit'], f"{res_pt[0]['probability'] * 100:.2f}%")
                        
                        chart_pt = pd.DataFrame({
                            "Digit": [str(r["digit"]) for r in res_pt],
                            "Probability": [r["probability"] for r in res_pt]
                        }).set_index("Digit")
                        st.bar_chart(chart_pt)
                        
                        st.image(img_pt, caption="Preprocessed Input (PyTorch)", width=150)
                col_index += 1
                
            # --- TENSORFLOW RESULT ---
            if use_tf:
                with result_cols[col_index]:
                    st.markdown("### ⚡ TensorFlow Results")
                    if tf_model is None:
                        st.error("TensorFlow model could not be loaded.")
                    else:
                        tensor_tf, img_tf = preprocess_canvas(canvas_result.image_data, "tensorflow")
                        res_tf = predict(tf_model, tensor_tf, "tensorflow")
                        
                        log_prediction(res_tf[0]['digit'], res_tf)
                        st.metric("Top Prediction", res_tf[0]['digit'], f"{res_tf[0]['probability'] * 100:.2f}%")
                        
                        chart_tf = pd.DataFrame({
                            "Digit": [str(r["digit"]) for r in res_tf],
                            "Probability": [r["probability"] for r in res_tf]
                        }).set_index("Digit")
                        st.bar_chart(chart_tf)
                        
                        st.image(img_tf, caption="Preprocessed Input (TensorFlow)", width=150)




    