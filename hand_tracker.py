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

    # Remove horizontal flip to fix the "reversed" camera view
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Dictionary to store status for the physical hands
    statuses = {"RH": {"msg": "Correct", "color": (0, 255, 0)}, 
                "LH": {"msg": "Correct", "color": (0, 255, 0)}}

    if results.multi_hand_landmarks:
        # Sort hands by X-coordinate to ensure RH is always the one closer to the camera
        # (Assuming camera is on the right side of the piano looking left, higher X is RH)
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x, reverse=True)
        
        for i, hand_landmarks in enumerate(sorted_hands):
            label = "RH" if i == 0 else "LH" # Hand with higher X is RH
            
            # Perspect-adjusted wrist threshold
            # Use a smaller buffer (0.02) for the LH (index 1) because it's further away
            buffer = 0.05 if label == "RH" else 0.02 
            
            wrist_y = hand_landmarks.landmark[0].y
            knuckle_y = hand_landmarks.landmark[9].y
            
            # Detection Logic
            error = ""
            # Check Low Wrist
            if wrist_y > knuckle_y + buffer:
                error = "Low Wrist"
            
            # Check Flat Finger
            # PIP joints are landmarks 6, 10, 14, 18
            for pip_idx in [6, 10, 14, 18]:
                mcp = [hand_landmarks.landmark[pip_idx-1].x, hand_landmarks.landmark[pip_idx-1].y]
                pip = [hand_landmarks.landmark[pip_idx].x, hand_landmarks.landmark[pip_idx].y]
                dip = [hand_landmarks.landmark[pip_idx+1].x, hand_landmarks.landmark[pip_idx+1].y]
                
                angle = calculate_angle(mcp, pip, dip)
                if angle > 165:
                    if error == "": error = "Flat Finger"

            # Update colors and messages
            if error != "":
                statuses[label]["msg"] = f"Incorrect ({error})"
                statuses[label]["color"] = (0, 0, 255) # Red
            else:
                statuses[label]["msg"] = "Correct"
                statuses[label]["color"] = (0, 255, 0) # Green

            # Draw skeleton with dynamic Line and Landmark colors
            # connection_drawing_spec controls the white lines you want to turn Red/Green
            mp_draw.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=statuses[label]["color"], thickness=2, circle_radius=2), # Dots
                mp_draw.DrawingSpec(color=statuses[label]["color"], thickness=2) # Lines
            )

    # Display Independent Results
    cv2.putText(image, f"RH: {statuses['RH']['msg']}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, statuses['RH']['color'], 2)
    cv2.putText(image, f"LH: {statuses['LH']['msg']}", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, statuses['LH']['color'], 2)

    cv2.imshow('DreamPlay Perspective-Corrected Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
