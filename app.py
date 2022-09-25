import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from flask import Flask,render_template,Response

app = Flask(__name__, 
            static_url_path = '', 
            static_folder = 'static')

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
age_model = load_model('age_model_200epochs.h5')
gender_model = load_model('gender_model_200epochs.h5')
gender_labels = ['Male', 'Female']

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            x1, y1, x2, y2 = x, y, x+w, y+h
            cv2.rectangle(frame,(x,y),(x2, y2),(50, 168, 151),  2)
            cv2.line (frame, (x1,y1), (x1+30, y1), (50, 168, 151), 6)
            cv2.line (frame, (x1,y1), (x1, y1+30), (50, 168, 151), 6)
            cv2.line (frame, (x2,y1), (x2-30, y1), (50, 168, 151), 6)
            cv2.line (frame, (x2,y1), (x2, y1+30), (50, 168, 151), 6)
            cv2.line (frame, (x1,y2), (x1+30, y2), (50, 168, 151), 6)
            cv2.line (frame, (x1,y2), (x1, y2-30), (50, 168, 151), 6)
            cv2.line (frame, (x2,y2), (x2-30, y2), (50, 168, 151), 6)
            cv2.line (frame, (x2,y2), (x2, y2-30), (50, 168, 151), 6)
            
            roi_gray = gray[y1:y2, x1:x2]
            roi_gray = cv2.resize(roi_gray,(48,48), interpolation = cv2.INTER_AREA)
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis = 0)  # (1, 48, 48, 1)

            #Gender
            roi_color = frame[y1:y2, x1:x2]
            roi_color = cv2.resize(roi_color,(200,200), interpolation = cv2.INTER_AREA)
            gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
            gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
            gender_label = gender_labels[gender_predict[0]] 
            
            #Age
            age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
            age = round(age_predict[0,0])
            
            label = "{},{}".format(gender_label, age)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            ret,buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_snap():
    while True:        
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            x1, y1, x2, y2 = x, y, x+w, y+h
            cv2.rectangle(frame,(x,y),(x2, y2),(50, 168, 151),  2)
            cv2.line (frame, (x1,y1), (x1+30, y1), (50, 168, 151), 6)
            cv2.line (frame, (x1,y1), (x1, y1+30), (50, 168, 151), 6)
            cv2.line (frame, (x2,y1), (x2-30, y1), (50, 168, 151), 6)
            cv2.line (frame, (x2,y1), (x2, y1+30), (50, 168, 151), 6)
            cv2.line (frame, (x1,y2), (x1+30, y2), (50, 168, 151), 6)
            cv2.line (frame, (x1,y2), (x1, y2-30), (50, 168, 151), 6)
            cv2.line (frame, (x2,y2), (x2-30, y2), (50, 168, 151), 6)
            cv2.line (frame, (x2,y2), (x2, y2-30), (50, 168, 151), 6)
            
            roi_gray = gray[y1:y2, x1:x2]
            roi_gray = cv2.resize(roi_gray,(48,48), interpolation = cv2.INTER_AREA)
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis = 0)  # (1, 48, 48, 1)

            #Gender
            roi_color = frame[y1:y2, x1:x2]
            roi_color = cv2.resize(roi_color,(200,200), interpolation = cv2.INTER_AREA)
            gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
            gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
            gender_label = gender_labels[gender_predict[0]] 
            
            #Age
            age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
            age = round(age_predict[0,0])
            
            label = "{},{}".format(gender_label, age)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            ret,buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
        return(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snap')
def snap():
    return Response(generate_frames_snap(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug = True)    
