from flask import Flask, request, render_template, jsonify
import os
import base64
from werkzeug.utils import secure_filename

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
    if 'image_data' not in request.form or 'height' not in request.form:
        return "Missing image or height input", 400

    # Extract form data
    user_height_cm = float(request.form['height'])
    body_type = request.form.get("bodyType")
    clothing_type = request.form.get("clothingType")
    brand = request.form.get("brand")
    fit = request.form.get("fit")
    source = request.form.get("source")  # Determine if it's from men.html or women.html

    # Process the image (Base64 decoding)
    image_data = request.form['image_data']
    image_data = image_data.replace("data:image/png;base64,", "")  # Remove header
    image_bytes = base64.b64decode(image_data)

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], "captured_image.png")
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # Import the correct backend processing file
    if source == "men":
        from backend_men import process_image
    else:
        from backend_women import process_image

    # Call the processing function
    measurements, size_prediction = process_image(image_path, user_height_cm, body_type, clothing_type, brand, fit)

    if measurements is None:
        return "Error: No pose detected.", 400  

    return render_template('result.html', measurements=measurements, size=size_prediction)

if __name__ == '__main__':
    app.run(debug=True)
