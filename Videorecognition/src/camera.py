import cv2 as cv

face_cascade=cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

BOX_SIZE = 180

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
       #extracting frames
        ret, frame = self.video.read()
        # frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,
        # interpolation=cv2.INTER_AREA)                    
        # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        # for (x,y,w,h) in face_rects:
        #  cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #  break        # encode OpenCV raw frame to jpg and displaying it
        # ret, jpeg = cv2.imencode('.jpg', frame)
        # return jpeg.tobytes()   
        cv.rectangle(frame,(80,225),(80+BOX_SIZE,225+BOX_SIZE),(0,255,0),2)
        ret, jpeg = cv.imencode('.jpg', frame)
        return frame, jpeg.tobytes()

