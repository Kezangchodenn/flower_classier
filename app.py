import os
import cv2
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from img2vec_pytorch import Img2Vec # type: ignore
import joblib
import numpy as np

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained model
model = joblib.load('svm_flower_classifier.pkl')

# Initialize img2vec
img2vec = Img2Vec(model='resnet-18')

def extract_features(image):
    # Convert to PIL Image and extract features
    image_pil = Image.fromarray(image)
    features = img2vec.get_vec(image_pil)
    return features

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the saved file
        image = cv2.imread(filepath)
        
        if image is not None:
            # Extract features and predict
            features = extract_features(image).reshape(1, -1)
            prediction = model.predict(features)
            return redirect(url_for('output', value=int(prediction[0]), image_path=filepath))
    
    return redirect(url_for('index'))

@app.route('/output')
def output():
    value = request.args.get('value', default=-1, type=int)
    image_path = request.args.get('image_path')
    return render_template('output.html', value=value, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
