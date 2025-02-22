from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained Logistic Regression model
with open('size_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load brand-specific size charts
brand_size_chart = pd.read_csv('brand_size_chart.csv')

# Function to predict body measurements & clothing size
def predict_size(height, body_type, brand, fit_type, clothing_type):
    input_data = pd.DataFrame([[height, body_type, brand, fit_type, clothing_type]],
                              columns=['height', 'body_type', 'brand', 'fit_type', 'clothing_type'])

    # Predict measurements (shoulder width, chest, waist)
    predicted_measurements = model.predict(input_data)

    # Extract measurements
    shoulder_width, chest, waist = predicted_measurements[0]

    # Find the recommended size using the brand size chart
    size_chart = brand_size_chart[
        (brand_size_chart['brand'] == brand) & 
        (brand_size_chart['clothing_type'] == clothing_type)
    ]
    
    # Find the closest size match
    size = size_chart.loc[
        (size_chart['shoulder_width_min'] <= shoulder_width) & (size_chart['shoulder_width_max'] >= shoulder_width) &
        (size_chart['chest_min'] <= chest) & (size_chart['chest_max'] >= chest) &
        (size_chart['waist_min'] <= waist) & (size_chart['waist_max'] >= waist),
        'size'
    ].values

    # Default size if no exact match
    recommended_size = size[0] if len(size) > 0 else "Unknown"

    return shoulder_width, chest, waist, recommended_size

# API Endpoint: Accepts height, body type, brand, fit type, clothing type
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        height = float(data['height'])
        body_type = data['body_type']
        brand = data['brand']
        fit_type = data['fit_type']
        clothing_type = data['clothing_type']

        shoulder_width, chest, waist, size = predict_size(height, body_type, brand, fit_type, clothing_type)

        response = {
            "shoulder_width": shoulder_width,
            "chest": chest,
            "waist": waist,
            "recommended_size": size
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
