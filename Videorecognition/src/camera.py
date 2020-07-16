import cv2 as cv

from constants import BOX_SIZE

class Camera(object):
    
    cam = 0

    def __init__(self):
        self.video = cv.VideoCapture(self.cam) 
        if not self.video.isOpened():
            print("Cannot open camera")
            exit()
                          
    def __del__(self):
        self.video.release()
        cv.destroyAllWindows() 

    def get_frame(self):
        ret, frame = self.video.read() 
        cv.rectangle(frame,(80,225),(80+BOX_SIZE,225+BOX_SIZE),(0,255,0),2)
        ret, jpeg = cv.imencode('.jpg', frame)
        
        return frame, jpeg.tobytes()

