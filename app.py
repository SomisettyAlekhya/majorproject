import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
from glob import glob
import os
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Brain Tumor Classification
def load_model_class():
    model = keras.models.load_model('effnet.h5')
    return model

def classify_brain_tumor(image, model):
    image = ImageOps.fit(image, (150, 150), Image.ANTIALIAS)  # Resize to (150, 150)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    pred = model.predict(img_reshape)
    pred = np.argmax(pred)
    return pred


# Brain Tumor Segmentation
im_height = 150
im_width = 150
smooth = 100

def dice_coef(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return ((2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def iou_loss(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    return -iou(y_true, y_pred)

def load_segmentation_model():
    model_seg = load_model('unet_128_mri_seg.hdf5', custom_objects={'iou_loss': iou_loss, 'iou': iou, 'dice_coef': dice_coef})
    return model_seg

# Streamlit App
st.write("""
         # BRAIN TUMOR CLASSIFICATION AND SEGMENTATION
         """
         )

file = st.file_uploader("Upload the image to be classified and segmented", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    # Open and display the original uploaded image
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # Classification
    classification_model = load_model_class()
    classification_result = classify_brain_tumor(image, classification_model)

    # Segmentation
    segmentation_model = load_segmentation_model()
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img[np.newaxis, :, :, :]
    seg_result = segmentation_model.predict(img)
    color = np.array([255, 255, 0], dtype='uint8')
    seg_result = np.squeeze(seg_result) > 0.5
    masked_img = np.where(seg_result[..., None], color, img)
    masked_img = np.squeeze(masked_img)
    img_class_bgr = (masked_img * 255).astype(np.uint8)
    img_class_bgr = cv2.cvtColor(img_class_bgr, cv2.COLOR_RGB2BGR)

    # Display the resized and masked image
    st.image([img_class_bgr], clamp=True, channels="BGR")

    # Classification Result
    labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
    classification_result = labels[classification_result]
    st.write("Classification Result: ", classification_result)
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np

# Define the average_image_size as you did in your original code
average_image_size = (400, 400, 3)
# Replace this with your actual values

# Add your model loading code here
# Example:
# from tensorflow.keras.models import load_model
# model = load_model('your_model_path')

# Function to make predictions
import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Load the pre-trained model
model = load_model('C:/Users/cherr/chest_xray.h5')

# Streamlit app title
st.title('PNEUMONIA DETECTION')

# Upload an image for classification
uploaded_image = st.file_uploader('Upload a Chest X-ray Image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    # Perform classification on the uploaded image
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_data = preprocess_input(img_array)
    
    # Make predictions using the model
    classes = model.predict(img_data)
    
    # Interpret the results
    result = int(classes[0][0])
    
    # Display the result
    if result == 0:
        st.write("Prediction: Person is Affected By PNEUMONIA")
    else:
        st.write("Prediction: Result is Normal")


