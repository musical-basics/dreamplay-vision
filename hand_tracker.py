import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)

    # Draw the hand annotations on the image.
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the 'skeleton' on the screen
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- PIANO LOGIC: TRACKING THE WRIST ---
            # Landmark 0 is the Wrist. Landmark 9 is the middle knuckle.
            wrist_y = hand_landmarks.landmark[0].y
            knuckle_y = hand_landmarks.landmark[9].y

            # In Computer Vision, Y increases downward. 
            # If wrist_y > knuckle_y, the wrist is LOWER than the knuckles (Potential Strain).
            if wrist_y > knuckle_y + 0.05: # Adding a small buffer
                cv2.putText(image, "LOW WRIST DETECTED", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('DreamPlay Vision Prototype', image)
    if cv2.waitKey(5) & 0xFF == 27: # Press 'Esc' to quit
        break

cap.release()
cv2.destroyAllWindows()
