# --- Import necessary libraries ---
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# --- Helper function to apply augmentations ---
# This function will take an image and the slider values and return the transformed image.
def augment_image(image, rotation, zoom, h_shift, v_shift, flip):
    """Applies a series of transformations to a PIL Image."""
    # Convert image to RGBA to handle rotations without black corners
    image = image.convert("RGBA")

    # 1. Rotation
    rotated_image = image.rotate(rotation, resample=Image.BICUBIC, expand=True)
    
    # 2. Zoom
    w, h = rotated_image.size
    zoom_factor = 1 / zoom
    crop_w = int(w * zoom_factor)
    crop_h = int(h * zoom_factor)
    # Crop from the center
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    zoomed_image = rotated_image.crop((left, top, right, bottom))
    # Resize back to original dimensions
    zoomed_image = zoomed_image.resize((w, h), Image.BICUBIC)

    # 3. Shift
    # Create a transparent canvas 3x the size of the image to avoid cropping during shifts
    expanded_w, expanded_h = w * 3, h * 3
    shifted_image_canvas = Image.new("RGBA", (expanded_w, expanded_h))
    shifted_image_canvas.paste(zoomed_image, (w, h))
    
    # Calculate the new crop box based on the shift
    crop_left = w - h_shift
    crop_top = h - v_shift
    crop_right = (w * 2) - h_shift
    crop_bottom = (h * 2) - v_shift
    
    shifted_image = shifted_image_canvas.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    # 4. Horizontal Flip
    if flip:
        final_image = ImageOps.mirror(shifted_image)
    else:
        final_image = shifted_image
        
    # Create a white background and paste the RGBA image onto it for final display
    final_with_bg = Image.new("RGB", final_image.size, "WHITE")
    final_with_bg.paste(final_image, (0, 0), final_image)

    return final_with_bg

# --- Main function for the Streamlit tab ---
def augmentation_playground_tab():
    st.title("🛠️ Data Augmentation Playground")
    st.write("This tab demonstrates our 'secret weapon' for training. Use the sliders to see how we can create thousands of new training examples from a single image.")
    st.info("This technique forces the model to learn the real patterns of a disease, not just memorize pictures.")
    st.markdown("---")

    # --- Sidebar for controls ---
    st.sidebar.header("Augmentation Controls")
    rotation = st.sidebar.slider("Rotation (degrees)", -45, 45, 0, key="rot")
    zoom = st.sidebar.slider("Zoom", 0.7, 1.3, 1.0, 0.05, key="zoom")
    h_shift = st.sidebar.slider("Horizontal Shift", -50, 50, 0, key="hshift")
    v_shift = st.sidebar.slider("Vertical Shift", -50, 50, 0, key="vshift")
    flip = st.sidebar.checkbox("Horizontal Flip", key="flip")

    # --- Main content area ---
    
    # The file uploader is the primary way to add an image.
    uploaded_file = st.file_uploader("Upload an image to begin:", type=["png", "jpg", "jpeg"])
    
    # The rest of the page will only be displayed if a file has been uploaded.
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        
        # --- Display the images in a two-column layout ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_image, caption="Original", use_column_width=True)

        with col2:
            st.subheader("Augmented Image")
            # We use a try-except block to catch any errors during the image processing
            try:
                augmented_image = augment_image(original_image, rotation, zoom, h_shift, v_shift, flip)
                st.image(augmented_image, caption="Transformed", use_column_width=True)
            except Exception as e:
                # If an error occurs, we display it clearly to the user.
                st.error(f"An error occurred during augmentation: {e}")
    else:
        # If no image is uploaded, display a helpful prompt.
        st.info("Please upload an image using the widget above to see the augmentation in action.")

