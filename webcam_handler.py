import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from skimage import io

os.environ['DISPLAY'] = ':0'

def get_image_from_webcam1():
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        order = input()
        if (order == '\n') or (order == ' ') or cv.waitKey(1) == ord('q'):
            print('Taking picture.')
            cap.release()
            cv.destroyAllWindows()
            io.imsave('dataset/webcam_pictures.jpeg', frame)
            return frame
    
    
def get_image_from_webcam():
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    order = input()
    if (order == '\n') or (order == ' ') or cv.waitKey(1) == ord('q'):
        print('Taking picture.')
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break    
        cap.release()
        cv.destroyAllWindows()
        io.imsave('dataset/webcam_pictures.jpeg', frame)



frame = get_image_from_webcam1()
