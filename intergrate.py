from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from mediapipe_processing import process_image  # Assuming this is where the MediaPipe code is

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/men')
def men():
    return render_template('men.html')

@app.route('/women')
def women():
    return render_template('women.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files or 'height' not in request.form:
        return "Missing image or height input", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    user_height_cm = float(request.form['height'])
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    body_type = request.form.get("bodyType")
    clothing_type = request.form.get("clothingType")
    brand = request.form.get("brand")
    fit = request.form.get("fit")

    measurements, size_prediction = process_image(image_path, user_height_cm, body_type, clothing_type, brand, fit)


    if measurements is None:
        return "Error: No pose detected.", 400  

    return render_template('result.html', measurements=measurements, size=size_prediction)

if __name__ == '__main__':
    app.run(debug=True)
