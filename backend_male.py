import cv2
import numpy as np
import mediapipe as mp
from google.colab.patches import cv2_imshow

# Load MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load the captured image
image = cv2.imread("photo.jpg")

# Convert image to RGB (required by MediaPipe)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process with Pose Estimation
results = pose.process(image_rgb)

def calculate_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def estimate_size(user_choice, user_brand, user_fit, bust_width_cm):
    """Estimate clothing size based on user preferences and bust width."""
    size_chart = {
        ('tops', 'shirts'): {
            'puma': {
                'slim': [(70, 81, 'XS'), (81, 90, 'S'), (91, 95, 'M'), (96, 100, 'L'), (101, 110, 'XL')],
                'oversized': [(90, 95, 'XS'), (96, 101, 'S'), (102, 107, 'M'), (112, 119, 'L'), (119, 126, 'XL')],
                'regular': [(81, 88, 'XS'), (89, 96, 'S'), (97, 104, 'M'), (105, 113, 'L'), (114, 123, 'XL')]
            },
            'nike': {
                'slim': [(70, 81, 'XS'), (81, 90, 'S'), (91, 95, 'M'), (96, 100, 'L'), (101, 110, 'XL')],
                'oversized': [(90, 95, 'XS'), (96, 101, 'S'), (102, 107, 'M'), (112, 119, 'L'), (119, 126, 'XL')],
                'regular': [(81, 88, 'XS'), (89, 96, 'S'), (97, 104, 'M'), (105, 113, 'L'), (114, 123, 'XL')]
            },
            'adidas': {
                'slim': [(70, 81, 'XS'), (81, 90, 'S'), (91, 95, 'M'), (96, 100, 'L'), (101, 110, 'XL')],
                'oversized': [(90, 95, 'XS'), (96, 101, 'S'), (102, 107, 'M'), (112, 119, 'L'), (119, 126, 'XL')],
                'regular': [(81, 88, 'XS'), (89, 96, 'S'), (97, 104, 'M'), (105, 113, 'L'), (114, 123, 'XL')]
            }
        },
        'pants': {
            'puma': {
                'slim': [(70, 81, 'XS'), (81, 90, 'S'), (91, 95, 'M'), (96, 100, 'L'), (101, 110, 'XL')],
                'oversized': [(90, 95, 'XS'), (96, 101, 'S'), (102, 107, 'M'), (112, 119, 'L'), (119, 126, 'XL')],
                'regular': [(81, 88, 'XS'), (89, 96, 'S'), (97, 104, 'M'), (105, 113, 'L'), (114, 123, 'XL')]
            },
            'nike': {
                'slim': [(70, 81, 'XS'), (81, 90, 'S'), (91, 95, 'M'), (96, 100, 'L'), (101, 110, 'XL')],
                'oversized': [(90, 95, 'XS'), (96, 101, 'S'), (102, 107, 'M'), (112, 119, 'L'), (119, 126, 'XL')],
                'regular': [(81, 88, 'XS'), (89, 96, 'S'), (97, 104, 'M'), (105, 113, 'L'), (114, 123, 'XL')]
            },
            'adidas': {
                'slim': [(70, 81, 'XS'), (81, 90, 'S'), (91, 95, 'M'), (96, 100, 'L'), (101, 110, 'XL')],
                'oversized': [(90, 95, 'XS'), (96, 101, 'S'), (102, 107, 'M'), (112, 119, 'L'), (119, 126, 'XL')],
                'regular': [(81, 88, 'XS'), (89, 96, 'S'), (97, 104, 'M'), (105, 113, 'L'), (114, 123, 'XL')]
            }
        }
    }

    if user_choice in size_chart and user_brand in size_chart[user_choice] and user_fit in size_chart[user_choice][user_brand]:
        for lower, upper, size in size_chart[user_choice][user_brand][user_fit]:
            if lower <= bust_width_cm <= upper:
                return size, None  # Return size and None for message

    return None, "Size not found"  # Return None for size and message

# Process the image and extract measurements
if results.pose_landmarks:
    # Draw landmarks on image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Extract key points for measurement
    landmarks = results.pose_landmarks.landmark
    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
    right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
    top_head = (landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y)
    feet = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)

    # Approximate bust using midpoint between shoulders and hips
    left_bust = (left_shoulder[0] * 0.75 + left_hip[0] * 0.25, left_shoulder[1] * 0.75 + left_hip[1] * 0.25)
    right_bust = (right_shoulder[0] * 0.75 + right_hip[0] * 0.25, right_shoulder[1] * 0.75 + right_hip[1] * 0.25)

    # Approximate waist using a point between the bust and hips
    left_waist = (left_shoulder[0] * 0.4 + left_hip[0] * 0.6, left_shoulder[1] * 0.4 + left_hip[1] * 0.6)
    right_waist = (right_shoulder[0] * 0.4 + right_hip[0] * 0.6, right_shoulder[1] * 0.4 + right_hip[1] * 0.6)

    # Extend hip measurement to the PELVIC WIDTH
    left_outer_thigh = (left_hip[0], left_hip[1])
    right_outer_thigh = (right_hip[0], right_hip[1])

    # Get pixel measurements
    shoulder_width_px = int(calculate_distance(left_shoulder, right_shoulder))
    hip_width_px = int(calculate_distance(left_outer_thigh, right_outer_thigh) * 3)  # Improved hip width
    bust_width_px = int(calculate_distance(left_bust, right_bust))
    waist_width_px = int(calculate_distance(left_waist, right_waist))
    body_height_px = int(calculate_distance(top_head, feet))  # Full body pixel height

    # Ask user for their real height in cm
    user_height_cm = float(input("Enter your height in cm: "))
    user_brand = input("Enter your preferred brand: ")
    user_choice = input("Enter your preferred cloth type: ")
    user_fit = input("Enter your preferred fit: ")

    # Convert pixel distances to real-world cm
    shoulder_width_cm = int((shoulder_width_px / body_height_px) * user_height_cm)
    hip_width_cm = int((hip_width_px / body_height_px) * user_height_cm)
    bust_width_cm = int((bust_width_px / body_height_px) * user_height_cm * 4)
    waist_width_cm = int((waist_width_px / body_height_px) * user_height_cm)

    # Show the processed image with landmarks
    cv2_imshow(image)

    # Call the estimate_size function
    estimated_size, message = estimate_size(user_choice, user_brand, user_fit, bust_width_cm)
    print(f"Estimated Size: {estimated_size}, Message: {message}")
