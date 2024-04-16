from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
detector = YOLO("yolov8n.pt")

app.config.from_pyfile('config.py')

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
        # Ensure the upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Perform object detection
        results = detector.predict(filepath)
        results.save()  # Saves detected frames/images
        return redirect(url_for('results'))
    return redirect(request.url)
@app.route('/results')
def results():
    return render_template('results.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

if __name__ == "__main__":
    app.run(debug=True)
