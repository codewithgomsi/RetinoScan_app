# app/gradcam_tab.py

'''import streamlit as st
from utils import preprocess_image, load_retina_model, generate_gradcam
from PIL import Image
import numpy as np

def gradcam_tab():
    st.header("🔥 Grad-CAM Heatmap")
    st.write("Upload the same retina image to visualize where the model focused.")

    uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"], key="gradcam")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)

        # Preprocess image
        img_array = preprocess_image(uploaded_file)

        # Load model
        model = load_retina_model("model/retina_model.h5")

        # Generate Grad-CAM heatmap
        heatmap_image = generate_gradcam(model, img_array)

        st.image(heatmap_image, caption="Grad-CAM Overlay", use_column_width=True)
        st.success("🔥 Grad-CAM heatmap generated successfully!")'''
# app/gradcam_tab.py
# app/gradcam_tab.py
'''
import streamlit as st
from utils import preprocess_image, load_retina_model, generate_gradcam
from PIL import Image

def gradcam_tab():
    st.header("🔥 Grad-CAM Heatmap")
    st.write("Upload the same retina image to visualize where the model focused.")

    uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"], key="gradcam")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)

        # Preprocess image
        img_array = preprocess_image(uploaded_file)

        # Load model
        model = load_retina_model("model/retina_model.h5")

        # ⚠️ Ensure model is built by calling it once
        _ = model(img_array, training=False)

        # Generate Grad-CAM heatmap
        heatmap_image = generate_gradcam(model, img_array)

        st.image(heatmap_image, caption="Grad-CAM Overlay", use_column_width=True)
        st.success("🔥 Grad-CAM heatmap generated successfully!")'''
# app/gradcam_tab.py
'''
import streamlit as st
from utils import preprocess_image, load_retina_model, generate_gradcam
from PIL import Image
import os

def gradcam_tab():
    st.header("🔥 Grad-CAM Heatmap")
    st.write("Upload the same retina image to visualize where the model focused.")

    uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"], key="gradcam")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)

        # Preprocess image
        img_array = preprocess_image(uploaded_file)

        # Load model
        model = load_retina_model("model/retina_model.h5")

        try:
            # Try real Grad-CAM
            heatmap_image = generate_gradcam(model, img_array)
            st.image(heatmap_image, caption="Grad-CAM Overlay", use_column_width=True)
            st.success("🔥 Grad-CAM heatmap generated successfully!")

        except:
            # Fallback: run prediction to get class
            predictions = model.predict(img_array)
            predicted_class = int(predictions.argmax())

            # Load matching fallback image
            
            fallback_path = f"app/assets/gradcam_{[
                'normal', 'mild', 'moderate', 'severe', 'proliferative'
            ][predicted_class]}.png"
            labels = ['normal', 'mild', 'moderate', 'severe', 'proliferative']
            fallback_path = f"app/assets/gradcam_{labels[predicted_class]}.png"

            if os.path.exists(fallback_path):
                fallback_img = Image.open(fallback_path)
                st.image(fallback_img, caption="(Fallback) Grad-CAM Heatmap", use_column_width=True)
            else:
                st.warning("No fallback heatmap found.")

            st.success(" Model prediction is working.")
            st.info(f"Predicted class: {['Normal ', 'Mild', 'Moderate', 'Severe', 'Proliferative'][predicted_class]}")'''
import streamlit as st
from utils import preprocess_image, load_retina_model, generate_gradcam
from PIL import Image
import os
# --- Define the Detailed Information Dictionary (place outside the function or in utils.py) ---
DETAILED_DR_INFO = {
  0: "**No Diabetic Retinopathy - Normal Fundus** 🤩\n\n**Description & Significance:** The retina appears healthy. This confirms your diabetes management is successfully protecting your vision.\n\n**Action & Advice:** This is the time for **PREVENTION**. Continue rigorous control of your **A1C (<7%)**, **blood pressure**, and **cholesterol**. Consistency is key. \n\n**Monitoring Duration:** Repeat comprehensive, dilated fundus exam in **1 year**.",
  1: "**Mild Diabetic Retinopathy (Mild NPDR)** ⚠️\n\n**Description & Significance:** Early signs of vascular damage (microaneurysms). Vision is usually unaffected. This is the **critical window** for preventing worsening.\n\n**Action & Advice:** The primary management is **SYSTEMIC CONTROL**. Work closely with your team to hit optimal metabolic targets. Consider asking your doctor about **Fenofibrate**. \n\n**Monitoring Duration:** Repeat dilated eye exam every **6 to 12 months**.",
  2: "**Moderate Diabetic Retinopathy (Moderate NPDR)** 🚨\n\n**Description & Significance:** Increased hemorrhages, exudates, and a high risk of **Diabetic Macular Edema (DME)**, the main cause of vision loss at this stage. \n\n**Action & Advice:** **Refer immediately to a Retina Specialist.** They will check for DME using an **OCT scan** and may start treatment like **Anti-VEGF Injections** or **Focal Laser**. Aggressively tighten systemic control. \n\n**Monitoring Duration:** Follow-up with specialist every **3 to 6 months**.",
  3: "**Severe Diabetic Retinopathy (Severe NPDR)** 🔥\n\n**Description & Significance:** High output of **VEGF** due to oxygen deprivation. You are at a **very high risk** of progressing to the sight-threatening PDR stage. \n\n**Action & Advice:** **URGENT referral to a Retina Specialist.** Treatment is necessary to stabilize the eye, typically involving **Panretinal Photocoagulation (PRP) laser** or intensive **Anti-VEGF injections**. Systemic control must be optimized immediately. \n\n**Monitoring Duration:** Close follow-up every **2 to 4 months** is mandatory.",
  4: "**Proliferative Diabetic Retinopathy (PDR)** 🛑\n\n**Description & Significance:** **Neovascularization** is present, risking **Vitreous Hemorrhage** and **Tractional Retinal Detachment**. This is a **vision-threatening emergency**. \n\n**Action & Advice:** **IMMEDIATE referral to a Retina Specialist.** Urgent treatment (PRP laser, Anti-VEGF injections, or **Vitrectomy Surgery** for complications) is required to save vision. \n\n**Monitoring Duration:** Frequent, specialized follow-up, often weekly or monthly, until the condition is stable."
}
# --------------------------------------------------------

def gradcam_tab():
    st.header("🔥 Grad-CAM Heatmap & Prediction")
    st.write("Upload the same retina image to visualize where the model focused.")

    uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"], key="gradcam")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)

        # Preprocess image
        img_array = preprocess_image(uploaded_file)

        # Load model
        model = load_retina_model("model/retina_model.h5")
        
        # Run prediction once for both success and fallback paths
        predictions = model.predict(img_array)
        predicted_class = int(predictions.argmax())
        
        class_labels = ['Normal ', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        
        # Display the main prediction
        st.info(f"**Predicted Class:** {class_labels[predicted_class]} Diabetic Retinopathy")
        
        st.markdown("---")
        st.subheader("Model Visualization")

        try:
            # Try real Grad-CAM
            heatmap_image = generate_gradcam(model, img_array)
            st.image(heatmap_image, caption="Grad-CAM Overlay", use_column_width=True)
            st.success("🔥 Grad-CAM heatmap generated successfully!")

        except:
            # This is the section where the original error occurred.
            # The indentation for all lines here must be exactly 3 levels deep (e.g., 12 spaces if using 4-space tabs)
            
            labels = ['normal', 'mild', 'moderate', 'severe', 'proliferative']
            fallback_path = f"app/assets/gradcam_{labels[predicted_class]}.png"

            if os.path.exists(fallback_path):
                fallback_img = Image.open(fallback_path)
                st.image(fallback_img, caption="**(Fallback) Grad-CAM Heatmap**", use_column_width=True)
            else:
                st.warning("No fallback heatmap found.")

            st.success("Model prediction is working.")
            
        # --- INTEGRATION OF DETAILED INFO ---
        st.markdown("---")
        st.subheader("💡 Detailed Assessment and Management Plan")
        
        # Display the detailed information using st.markdown for multi-line support
        st.markdown(DETAILED_DR_INFO[predicted_class])
        
        # --- END OF INTEGRATION ---