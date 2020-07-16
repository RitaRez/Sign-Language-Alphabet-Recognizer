import os
import numpy as np

from importlib import import_module 
from flask import Flask, render_template, Response, render_template_string
from camera import Camera
from skimage import io, color
from recognizer import NeuralNetwork
from random import random

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

    io.imsave(pic_name, current_frame)
    io.imsave(gray_pic_name, gray_frame)

    #model = NeuralNetwork()
    print('\n\n\n', gray_frame.shape)
    # model.predict_pic(current_frame)
    prediction = 'Letter Shown: A'
    return prediction


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)                    