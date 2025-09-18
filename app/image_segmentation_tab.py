# Import the necessary libraries. We need:
# - streamlit for creating the web app interface
# - numpy for numerical operations (especially for handling images)
# - cv2 (OpenCV) for image processing tasks like segmentation
# - KMeans from scikit-learn for the clustering algorithm
# --- NEW LIBRARIES FOR AI ANALYSIS ---
# - pipeline from transformers: This is our tool to easily use powerful AI models from Hugging Face.
# - Image from PIL: A library to help handle images, which we need to prepare the image for the AI model.
import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from transformers import pipeline
from PIL import Image

def image_segmentation_tab():
    """
    This is the main function that Streamlit will run to display the 
    'Image Segmentation' tab in your web app.
    """
    
    # --- 1. SET UP THE PAGE ---
    st.title("Image Segmentation")
    st.write("Here, you can upload an image and apply different techniques to segment it.")
    st.info("Segmentation helps identify objects and boundaries within an image.")

    # --- 2. IMAGE UPLOAD ---
    uploaded_file = st.file_uploader("Upload your image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.image(img_rgb, caption='Your Original Image', use_column_width=True)
        st.markdown("---") 

        # --- 3. CHOOSE SEGMENTATION METHOD ---
        method = st.selectbox(
            "Select a Segmentation Method",
            ["Otsu's Thresholding","Canny Edge Detection", "K-Means Clustering"]
        )

        st.subheader("Segmentation Result")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption='Original')

        # --- 4. APPLY THE CHOSEN METHOD ---
        # (This part is exactly the same as before)
        if method == "Otsu's Thresholding":
            gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            _, segmented_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            with col2:
                st.image(segmented_img, caption="Otsu's Segmentation")
            st.markdown("""
            **What it does:** Otsu's method simplifies the image into black and white to separate an object from its background.
            """)
        elif method == "Canny Edge Detection":
            gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)
            with col2:
                st.image(edges, caption="Canny Edges")
            st.markdown("""
            *What it does:* Canny Edge Detection traces the outlines of objects where brightness changes sharply.
            """)
        elif method == "K-Means Clustering":
            pixels = img_rgb.reshape((-1, 3)).astype(np.float32)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
            segmented_img = segmented_pixels.reshape(img_rgb.shape).astype(np.uint8)
            with col2:
                st.image(segmented_img, caption="K-Means Segmentation (3 Clusters)")
            st.markdown("""
            **What it does:** K-Means groups all the colors in the image into a few main ones, creating clear colored segments.
            """)
            
        st.markdown("---") # Adds another visual separator line

        # --- 5. BONUS: AI-POWERED IMAGE ANALYSIS ---
        # This whole section is new. It adds the "Analyze with AI" button and functionality.
        st.subheader("What does an AI think is in the image?")
        
        # We create a button. The code inside the 'if' statement only runs when the button is clicked.
        if st.button("Analyze with AI"):
            
            # This shows a "Please wait..." message while the AI is working, so you know it's not stuck.
            with st.spinner('Please wait, the AI model is analyzing the image...'):
                try:
                    # This is the most important line from Hugging Face.
                    # It downloads and prepares a powerful, pre-trained AI model from Google.
                    # The "image-classification" part tells the pipeline what we want to do.
                    image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
                    
                    # The AI model needs the image in a specific format called a "PIL Image".
                    # This line converts our image (which is a NumPy array) into that format.
                    pil_image = Image.fromarray(img_rgb)
                    
                    # Here, we give our image to the AI model. It analyzes it and gives us back its predictions.
                    results = image_classifier(pil_image)
                    
                    # Once done, we show a success message.
                    st.success("Analysis Complete!")
                    st.write("The AI identified the following:")
                    
                    # The 'results' are a list. We loop through each prediction to display it neatly.
                    for result in results:
                        # We format the label and the score to be more readable.
                        label = result['label'].replace("_", " ").title()
                        score = f"{result['score'] * 100:.2f}%"
                        st.write(f"- **{label}** with {score} confidence.")
                
                # If something goes wrong (like you have no internet), this will show a helpful error message.
                except Exception as e:
                    st.error("Sorry, an error occurred. Please check your internet connection and try again.")



