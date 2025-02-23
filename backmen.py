import cv2
import mediapipe as mp
import numpy as np
from google.colab.patches import cv2_imshow

def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        bust_width = calculate_distance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
        bust_width_cm = bust_width * 100
        
        cv2.putText(frame, f'Bust Width: {bust_width_cm:.2f} cm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2_imshow(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Size Recommendation System
size_chart = {
    ('T-shirt', 'Nike'): {'Slim': {38: 'S', 40: 'M', 42: 'L'}, 'Regular': {38: 'M', 40: 'L', 42: 'XL'}},
    ('Shirt', 'Adidas'): {'Slim': {38: 'S', 40: 'M', 42: 'L'}, 'Regular': {38: 'M', 40: 'L', 42: 'XL'}}
}

user_choice = 'T-shirt'  # Example input
user_brand = 'Nike'
user_fit = 'Slim'

bust_size_rounded = round(bust_width_cm)
recommended_size = size_chart.get((user_choice, user_brand), {}).get(user_fit, {}).get(bust_size_rounded, 'Unknown')

print(f'Recommended Size: {recommended_size}')
