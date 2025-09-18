# This script's only purpose is to download the AI model from Hugging Face
# and save it to your computer's cache.

# We import the 'pipeline' tool, just like in our main app.
from transformers import pipeline

def download():
    print("Starting AI model download...")
    print("This might take a few minutes depending on your internet speed. Please be patient.")
    
    # This is the line that triggers the download.
    # It will fetch the model and all its necessary parts.
    pipeline("image-classification", model="google/vit-base-patch16-224")
    
    print("\n-------------------------------------------")
    print("Model downloaded and saved successfully!")
    print("You can now run your main Streamlit app.")
    print("-------------------------------------------")

if __name__ == "__main__":
    download()