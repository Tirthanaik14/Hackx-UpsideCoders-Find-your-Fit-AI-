from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
import backend_men, backend_women  # Separate backend scripts for men and women

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/men', methods=['GET', 'POST'])
def men():
    if request.method == 'POST':
        image = request.files['image']
        size = backend_men.predict_size(image)
        return render_template('result.html', size=size, gender='Men')
    return render_template('men.html')

@app.route('/women', methods=['GET', 'POST'])
def women():
    if request.method == 'POST':
        image = request.files['image']
        size = backend_women.predict_size(image)
        return render_template('result.html', size=size, gender='Women')
    return render_template('women.html')

if __name__ == '__main__':
    app.run(debug=True)
