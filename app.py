from flask import Flask, render_template, request, redirect, url_for, session
import os
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
detector = YOLO("yolov8n.pt")

app.config.from_pyfile('config.py')
app.secret_key = 'your_secret_key_here'  # Needed for session management

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        output_path = process_and_save_video(filepath)
        session['video_path'] = output_path  # Store the output path in the session
        return redirect(url_for('results'))
    return redirect(request.url)

@app.route('/results')
def results():
    video_path = session.get('video_path', None)  # Retrieve the output path from the session
    return render_template('results.html', video_path=video_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

def process_and_save_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.track(frame, persist=True, conf=0.7)
        frame_ = results[0].plot()  # Adjust as necessary to ensure bounding boxes are drawn
        out.write(frame_)

    cap.release()
    out.release()
    return output_path

if __name__ == "__main__":
    app.run(debug=True)
