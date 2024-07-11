from flask import Flask, request, render_template, send_file, jsonify
from sklearn.preprocessing import Binarizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, SpatialDropout2D, Concatenate, ReLU
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import os
from datetime import datetime

# matplotlib.use('Agg')


app = Flask(__name__)

IMAGE_SIZE = (384, 512)

#declare unet model

def unet_model():
    input_layer = Input(shape=(*IMAGE_SIZE, 3), name='Input_Layer')
    
    conv_1 = Conv2D(8, 5, padding='same', activation=ReLU(), name='Conv_1')(input_layer)
    pool_1 = MaxPool2D(name='Max_Pool_1')(conv_1)
    conv_2 = Conv2D(16, 5, padding='same', activation=ReLU(), name='Conv_2')(pool_1)
    pool_2 = MaxPool2D(name='Max_Pool_2')(conv_2)
    spd_1 = SpatialDropout2D(0.1, name='SPD_1')(pool_2)
    
    conv_3 = Conv2D(32, 4, padding='same', activation=ReLU(), name='Conv_3')(spd_1)
    pool_3 = MaxPool2D(name='Max_Pool_3')(conv_3)
    conv_4 = Conv2D(64, 4, padding='same', activation=ReLU(), name='Conv_4')(pool_3)
    pool_4 = MaxPool2D(name='Max_Pool_4')(conv_4)
    spd_2 = SpatialDropout2D(0.1, name='SPD_2')(pool_4)
    
    conv_5 = Conv2D(128, 3, padding='same', activation=ReLU(), name='Conv_5')(spd_2)
    pool_5 = MaxPool2D(name='Max_Pool_5')(conv_5)
    conv_6 = Conv2D(256, 3, padding='same', activation=ReLU(), name='Conv_6')(pool_5)
    pool_7 = MaxPool2D(name='Max_Pool_6')(conv_6)
    spd_3 = SpatialDropout2D(0.1, name='SPD_3')(pool_7)
    
    conv_7 = Conv2D(512, 2, padding='same', activation=ReLU(), name='Conv_7')(spd_3)
    pool_7 = MaxPool2D(name='Max_Pool_7')(conv_7)
    
    conv_t_1 = Conv2DTranspose(256, 2, padding='same', strides=2, activation=ReLU(), name='Conv_T_1')(pool_7)
    concat_1 = Concatenate(name='Concat_1')([conv_t_1, spd_3])
    spd_4 = SpatialDropout2D(0.1, name='SPD_4')(concat_1)
    
    conv_t_2 = Conv2DTranspose(128, 3, padding='same', strides=2, activation=ReLU(), name='Conv_T_2')(spd_4)
    conv_t_3 = Conv2DTranspose(64, 3, padding='same', strides=2, activation=ReLU(), name='Conv_T_3')(conv_t_2)
    concat_2 = Concatenate(name='Concat_2')([conv_t_3, spd_2])
    spd_5 = SpatialDropout2D(0.1, name='SPD_5')(concat_2)
    
    conv_t_4 = Conv2DTranspose(32, 4, padding='same', strides=2, activation=ReLU(), name='Conv_T_4')(spd_5)
    conv_t_5 = Conv2DTranspose(16, 4, padding='same', strides=2, activation=ReLU(), name='Conv_T_5')(conv_t_4)
    concat_3 = Concatenate(name='Concat_3')([conv_t_5, spd_1])
    spd_6 = SpatialDropout2D(0.1, name='SPD_6')(concat_3)
    
    conv_t_6 = Conv2DTranspose(8, 5, padding='same', strides=2, activation=ReLU(), name='Conv_T_6')(spd_6)
    conv_t_7 = Conv2DTranspose(1, 5, padding='same', strides=2, activation='sigmoid', name='Conv_T_7')(conv_t_6)    
    
    return Model(inputs=input_layer, outputs=conv_t_7, name='U-net_Segmentation')
                 
model = unet_model()

# Memuat model dengan custom objects
model.load_weights('model/ReLU-B3-P5-Weight.h5')

def preprocess_image(image):
    img = tf.io.decode_image(image.read(), channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    img = preprocess_image(file)
    
    prediction = model.predict(img)

    pred_mask = Binarizer(threshold=0.5).transform(prediction.reshape(-1, 1)).reshape(prediction.shape)
    
    # Convert prediction to binary mask (0 or 1)
    binary_prediction = (pred_mask > 0.5).astype(np.uint8)
    
    # Convert binary mask to image (assuming single channel)
    pil_image_masked = Image.fromarray(binary_prediction[0, :, :, 0] * 255)
    
    # Convert PIL image to base64 string
    buffered = BytesIO()
    pil_image_masked.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Convert original image (img) to base64 string
    original_img = Image.fromarray((img[0] * 255).astype(np.uint8))
    buffered_original = BytesIO()
    original_img.save(buffered_original, format="JPEG")
    original_img_str = base64.b64encode(buffered_original.getvalue()).decode('utf-8')
    
    # Send back JSON response with both images
    return jsonify({
        'original_image': original_img_str,
        'masked_image': img_str
    })