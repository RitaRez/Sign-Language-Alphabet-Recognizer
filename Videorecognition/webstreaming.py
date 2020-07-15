import os

from importlib import import_module 
from flask import Flask, render_template, Response
from src.camera import Camera
from src.recognizer import NeuralNetwork
from random import random

app = Flask(__name__)
cam = Camera()

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

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
    pic_name = random().round().astype(int)
    io.imsave('../../dataset/jpeg'+pic_name, jpeg)
    io.imsave('../../dataset/'+pic_name, current_frame)

    model = NeuralNetwork()
    model.predict_pic(current_frame)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)                    