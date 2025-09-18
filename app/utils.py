# app/utils.py
'''from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def load_class_names(json_path):
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names
import tensorflow as tf
import cv2
import matplotlib.cm as cm
import io

def load_retina_model(model_path):
    return tf.keras.models.load_model(model_path)

def generate_gradcam(model, processed_img):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(index=-1).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_img)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Load original image again (not normalized)
    original = processed_img[0] * 255.0
    original = np.uint8(original)

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    return superimposed_img'''
# app/utils.py
'''import numpy as np
import json
from PIL import Image
import tensorflow as tf
import cv2
import matplotlib.cm as cm
from tensorflow.keras.preprocessing import image

# Preprocess image for prediction
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Load class names from JSON
def load_class_names(json_path):
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names

# Load trained model
def load_retina_model(model_path):
    return tf.keras.models.load_model(model_path)

# Generate Grad-CAM heatmap overlay
def generate_gradcam(model, processed_img):
    # Get last convolutional layer by name or index
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_img)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Prepare original image again (from preprocessed)
    original = processed_img[0] * 255.0
    original = np.uint8(original)

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    return superimposed_img
# End of utils.py'''
# app/utils.py

from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import tensorflow as tf
import cv2

# 1. Preprocess uploaded retina image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# 2. Load class names from JSON
def load_class_names(json_path):
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names

# ✅ 3. Load model and call once (to initialize outputs)
def load_retina_model(model_path):
    model = tf.keras.models.load_model(model_path)
    dummy_input = tf.zeros((1, 224, 224, 3))
    _ = model(dummy_input)  # Force call to build layers
    return model

# ✅ 4. Grad-CAM generator
def generate_gradcam(model, processed_img):
    # Automatically get last conv layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer = layer
            break

    grad_model = tf.keras.models.Model(
        [model.inputs], [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_img)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap + 1e-10)
    heatmap = heatmap.numpy()

    # Overlay on original
    original = processed_img[0] * 255.0
    original = np.uint8(original)

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    return superimposed_img

