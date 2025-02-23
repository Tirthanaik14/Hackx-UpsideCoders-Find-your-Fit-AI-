from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Load MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

@app.route('/process', methods=['POST'])
def process_image():
    # Load the captured image
    image = cv2.imread("photo.jpg")

    if image is None:
        return jsonify({"error": "Image not found"})

    # Convert image to RGB (required by MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with Pose Estimation
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return jsonify({"error": "No pose detected"})

    # Extract key points for measurement
    landmarks = results.pose_landmarks.landmark
    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)

    # Calculate Bust Width (convert from relative to absolute)
    bust_width_cm = calculate_distance(left_shoulder, right_shoulder) * 100  # Example scale

    # Dummy values for now
    estimated_size = "M"

    return render_template('res.html', bust_width=bust_width_cm, estimated_size=estimated_size)

if __name__ == '__main__':
    app.run(debug=True)
