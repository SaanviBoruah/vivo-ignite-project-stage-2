from flask import Flask, render_template, request
import cv2
import numpy as np
from fer import FER
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Initialize detector with MTCNN (no OpenCV face detection)
detector = FER(mtcnn=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded file
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    # Process image
    img = cv2.imread(filename)
    results = detector.detect_emotions(img)

    # Draw results
    output_img = img.copy()
    for face in results:
        x, y, w, h = face["box"]
        emotions = face["emotions"]
        dominant = max(emotions, key=emotions.get)
        
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output_img, dominant, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Save processed image
    output_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
    cv2.imwrite(output_filename, output_img)

    return render_template('result.html',
                          original=filename,
                          processed=output_filename,
                          results=results)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
