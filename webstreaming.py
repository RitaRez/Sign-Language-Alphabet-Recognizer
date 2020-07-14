from flask import Flask
from flask import render_template
from flask import Response
app = Flask(__name__)

def get_image_from_webcam():
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
    

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(get_image_from_webcam(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")