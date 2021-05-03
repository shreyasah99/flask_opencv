#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras 



face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
ds_factor=0.6

class_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
my_model = tf.keras.models.load_model('shreyas_scratch_model.h5')


#Initialize the Flask app
app = Flask(__name__)


camera = cv2.VideoCapture(0)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)



def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
           
            #frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,
            #interpolation=cv2.INTER_AREA)                    
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face_roi=face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in face_roi:
             draw_border(frame, (x,y),(x+w,y+h),(0,0,204), 2,15, 10)
             
             img_color_crop = frame[y:y+h,x:x+w]
             final_image = cv2.resize(img_color_crop, (48,48))
             final_image = np.expand_dims(final_image, axis = 0)
             final_image = final_image/255.0
        
             prediction = my_model.predict(final_image)
             label=class_labels[prediction.argmax()]
             cv2.putText(frame,label, (x+20, y-40), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2, (92,79,19),3)
                 
             
             break
            # encode OpenCV raw frame to jpg and displaying it
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            
@app.route('/')
def index():
    return render_template('index.html')
    
    

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    

if __name__ == "__main__":
    app.run(debug=True)