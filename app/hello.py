# test_libraries.py

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from cv2 import *
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import load_model
    import streamlit as st
    from sklearn.model_selection import train_test_split
    from PIL import Image
    print("✅ All libraries imported successfully!")
except ImportError as e:
    print("❌ Library import failed:", e)
