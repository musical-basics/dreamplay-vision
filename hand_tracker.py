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

    # 1. FIX REVERSED CAMERA: Removed cv2.flip
    # image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Initialize independent statuses
    hand_reports = {} 
    hand_colors = {}

    if results.multi_hand_landmarks and results.multi_handedness:
        # Zip ensures we match the landmark to the classification
        for idx, (hand_landmarks, handedness_info) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            # Get hand label (Left or Right)
            handedness = handedness_info.classification[0].label
            
            # Reset local flags for this specific hand
            current_error = ""
            is_flat = False

            # Check Low Wrist
            wrist_y = hand_landmarks.landmark[0].y
            knuckle_y = hand_landmarks.landmark[9].y
            if wrist_y > knuckle_y + 0.05:
                current_error = "Low Wrist"

            # Check Flat Finger
            for pip_idx in [6, 10, 14, 18]:
                mcp = [hand_landmarks.landmark[pip_idx-1].x, hand_landmarks.landmark[pip_idx-1].y]
                pip = [hand_landmarks.landmark[pip_idx].x, hand_landmarks.landmark[pip_idx].y]
                dip = [hand_landmarks.landmark[pip_idx+1].x, hand_landmarks.landmark[pip_idx+1].y]
                
                angle = calculate_angle(mcp, pip, dip)
                if angle > 165:
                    is_flat = True
                    if current_error == "": current_error = "Flat Finger"

            # 2 & 3. UPDATE INDEPENDENT MESSAGES
            if current_error != "":
                hand_reports[handedness] = f"Incorrect ({current_error})"
                hand_colors[handedness] = (0, 0, 255) # Red
            else:
                hand_reports[handedness] = "Correct"
                hand_colors[handedness] = (0, 255, 0) # Green

            # Draw skeleton with independent hand color
            mp_draw.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=hand_colors[handedness], thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            )

    # Display separate text for RH and LH if they are detected
    y_pos = 50
    for hand_label in ["Right", "Left"]:
        if hand_label in hand_reports:
            text = f"{'RH' if hand_label == 'Right' else 'LH'}: {hand_reports[hand_label]}"
            color = hand_colors[hand_label]
            cv2.putText(image, text, (50, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_pos += 40

    cv2.imshow('DreamPlay Dual-Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
