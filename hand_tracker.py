import cv2
import mediapipe as mp
import numpy as np

# Helper function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a) # First point
    b = np.array(b) # Mid point
    c = np.array(c) # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Default status
    form_is_good = True
    error_message = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Check for Low Wrist
            wrist_y = hand_landmarks.landmark[0].y
            knuckle_y = hand_landmarks.landmark[9].y
            if wrist_y > knuckle_y + 0.05:
                form_is_good = False
                error_message = "LOW WRIST DETECTED"

            # 2. Check for Flat Fingers (Loop through Index, Middle, Ring, Pinky)
            # PIP joints are landmarks 6, 10, 14, 18
            for pip_idx in [6, 10, 14, 18]:
                mcp = [hand_landmarks.landmark[pip_idx-1].x, hand_landmarks.landmark[pip_idx-1].y]
                pip = [hand_landmarks.landmark[pip_idx].x, hand_landmarks.landmark[pip_idx].y]
                dip = [hand_landmarks.landmark[pip_idx+1].x, hand_landmarks.landmark[pip_idx+1].y]
                
                angle = calculate_angle(mcp, pip, dip)
                if angle > 165: # Threshold for 'flat' finger
                    form_is_good = False
                    if error_message == "": error_message = "FLAT FINGER DETECTED"

            # Set colors based on form
            # Green (0, 255, 0) if good, Red (0, 0, 255) if bad
            status_color = (0, 255, 0) if form_is_good else (0, 0, 255)
            line_color = (0, 255, 0) if form_is_good else (255, 255, 255) # White or Green lines

            # Draw the skeleton with the dynamic color
            mp_draw.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=status_color, thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=line_color, thickness=2, circle_radius=2)
            )

    # Update on-screen text
    text_to_show = "CORRECT POSITION DETECTED" if form_is_good else error_message
    text_color = (0, 255, 0) if form_is_good else (0, 0, 255)
    
    cv2.putText(image, text_to_show, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    cv2.imshow('DreamPlay Vision Feedback', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
