from flask import Flask, render_template, Response
import cv2
import numpy as np
from fer import FER

app = Flask(__name__)

# Initialize emotion detector
emotion_detector = FER()

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame to RGB for FER
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect emotions
            results = emotion_detector.detect_emotions(rgb_frame)
            
            # Draw results on frame
            for face in results:
                x, y, w, h = face["box"]
                emotions = face["emotions"]
                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion]
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{dominant_emotion} ({confidence:.0%})"
                cv2.putText(frame, label, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)