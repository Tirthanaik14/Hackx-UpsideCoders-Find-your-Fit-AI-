from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/measure', methods=['POST'])
def measure():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return jsonify({'error': 'Could not open webcam'})

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Failed to capture image'})

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return jsonify({'error': 'No pose detected'})

    landmarks = results.pose_landmarks.landmark

    shoulder_width = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x -
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) * frame.shape[1]

    bust_size = abs(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x -
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) * frame.shape[1]

    waist_size = bust_size * 0.9  # Example approximation
    hips_size = bust_size * 1.1  # Example approximation

    size_chart = {
        'XS': (30, 32),
        'S': (33, 35),
        'M': (36, 38),
        'L': (39, 41),
        'XL': (42, 44)
    }

    recommended_size = 'Unknown'
    for size, (min_bust, max_bust) in size_chart.items():
        if min_bust <= bust_size <= max_bust:
            recommended_size = size
            break

    measurements = {
        'shoulder': round(shoulder_width, 2),
        'bust': round(bust_size, 2),
        'waist': round(waist_size, 2),
        'hips': round(hips_size, 2)
    }

    return render_template('result.html', measurements=measurements, size=recommended_size)

if __name__ == '__main__':
    app.run(debug=True)

