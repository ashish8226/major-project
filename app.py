#from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model,model_from_json
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/my_model.h5'
model = load_model(MODEL_PATH)
model._make_predict_function()   
print('model loaded')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        
        
        # Get the file from post request
        f = request.files['file']
        from keras.preprocessing import image
        test_image=image.load_img(f,target_size=(64,64))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=model.predict(test_image)
        print(result[0][0]*100)
        if result[0][0] >= 0.5:
            prediction='Pneumonia '
        else:
            prediction='Normal '
        return prediction
        
    return None


if __name__ == '__main__':
    app.run()

