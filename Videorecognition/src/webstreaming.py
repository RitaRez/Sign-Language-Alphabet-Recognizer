import os, sys
import numpy as np
import matplotlib.pyplot as plt

from constants import BOX_SIZE, NUMBER_OF_PIXELS
from importlib import import_module 
from flask import Flask, render_template, Response, render_template_string
from camera import Camera
from skimage import io, color
from recognizer import NeuralNetwork
from random import random
from PIL import Image

app = Flask(__name__)
cam = Camera()

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html", answer=' ')

def gen(camera):
    while True:
        frame, jpeg = camera.get_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')

@app.route("/video_feed")
def video_feed():
	return Response(gen(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict_answer')
def predict_answer():
    current_frame, jpeg = cam.get_frame()
    gray_frame = color.rgb2gray(current_frame)
    pic_number = str(int(np.floor(random()*10000)))

    pic_name = '../../dataset/pic'+pic_number+'.jpg'
    gray_pic_name = '../../dataset/gray'+pic_number+'.jpg'
    gray_croped_name = '../../dataset/gray_croped'+pic_number+'.jpg'

    #io.imsave(pic_name, current_frame)
    io.imsave(gray_pic_name, gray_frame)

    gray_frame = Image.open(gray_pic_name)
    gray_frame_crop = gray_frame.crop((80,225,80+BOX_SIZE,225+BOX_SIZE))
    gray_frame_crop.thumbnail((NUMBER_OF_PIXELS, NUMBER_OF_PIXELS), Image.ANTIALIAS)
    gray_frame_crop.save(gray_croped_name)
    pix = np.array(gray_frame_crop.getdata()).reshape(1, NUMBER_OF_PIXELS, NUMBER_OF_PIXELS, 1 )/255.0
    #pix = np.array([np.reshape(i, (NUMBER_OF_PIXELS, NUMBER_OF_PIXELS)) for i in (np.array(gray_frame_crop.getdata()))])
    prediction = ' '
    
    model = NeuralNetwork()
    #pix = model.data_resizing(pix)
    prediction = model.predict_pic(pix)
    print('\n\n\n\n',prediction)
    prediction = 'Letter Shown: ' + prediction
    return prediction


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)                    