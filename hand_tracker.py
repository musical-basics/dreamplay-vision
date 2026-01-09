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
# Increased tracking confidence to 0.8 for robustness
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.8, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

import time

# --- Global State ---
calibrating = False
calibrated = False
calibration_start_time = 0
calibration_duration = 3.0  # Seconds to hold the pose
calibration_data_rh = []
calibration_data_lh = []

# Final Baselines (defaults)
baseline_rh = {"dist": 0.05}
baseline_lh = {"dist": 0.03}

def update_calibration(sorted_hands):
    global calibrating, calibrated, calibration_start_time, baseline_rh, baseline_lh, calibration_data_rh, calibration_data_lh
    
    elapsed = time.time() - calibration_start_time
    progress = min(elapsed / calibration_duration, 1.0)
    
    # Collect data during the 3 seconds to get a stable average
    if len(sorted_hands) >= 2:
        # RH is index 0, LH is index 1 (based on X-coord sorting)
        dist_rh = abs(sorted_hands[0].landmark[0].y - sorted_hands[0].landmark[9].y)
        dist_lh = abs(sorted_hands[1].landmark[0].y - sorted_hands[1].landmark[9].y)
        calibration_data_rh.append(dist_rh)
        calibration_data_lh.append(dist_lh)

    if progress >= 1.0:
        # Calculate Averages
        if calibration_data_rh:
            baseline_rh["dist"] = sum(calibration_data_rh) / len(calibration_data_rh)
        if calibration_data_lh:
            baseline_lh["dist"] = sum(calibration_data_lh) / len(calibration_data_lh)
        
        calibrating = False
        calibrated = True
        print(f"Locked Baselines - RH: {baseline_rh['dist']:.3f}, LH: {baseline_lh['dist']:.3f}")

    return progress

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # Remove horizontal flip to fix the "reversed" camera view
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Dictionary to store status for the physical hands
    statuses = {}

    # Sort hands by X-coordinate (Higher X = Right Hand in side view)
    sorted_hands = []
    if results.multi_hand_landmarks:
         sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x, reverse=True)

    # --- CALIBRATION MODE ---
    if not calibrated:
        if not calibrating:
            cv2.putText(image, "HOLD ARCH & PRESS 'C' TO START", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                calibrating = True
                calibration_start_time = time.time()
                calibration_data_rh, calibration_data_lh = [], []
        else:
            # Update progress and draw bar
            prog = update_calibration(sorted_hands)
            bar_width = int(prog * 300)
            cv2.rectangle(image, (50, 180), (350, 210), (50, 50, 50), -1) # Background
            cv2.rectangle(image, (50, 180), (50 + bar_width, 210), (0, 255, 255), -1) # Progress
            cv2.putText(image, f"CALIBRATING: {int(prog*100)}%", (50, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # --- TRACKING MODE ---
    if results.multi_hand_landmarks:
        # If calibrated, we trust the X-sorting implies RH is [0] and LH is [1]
        # Or we continue to use the sorted_hands from above
        # For simplicity in this logic, we iterate the sorted hands
        
        for i, hand_landmarks in enumerate(sorted_hands):
            label = "RH" if i == 0 else "LH"
            
            wrist_y = hand_landmarks.landmark[0].y
            knuckle_y = hand_landmarks.landmark[9].y
            
            # Baseline-Relative Detection Logic
            error = ""
            
            # Use the calibrated baseline for this specific hand
            baseline_dist = baseline_rh["dist"] if label == "RH" else baseline_lh["dist"]
            
            # If current wrist drop is significantly worse than refined baseline
            # (wrist_y > knuckle_y + buffer)
            # Here we defined 'dist' as (wrist - knuckle).
            # If wrist drops low, wrist_y increases. (wrist - knuckle) becomes larger POSITIVE value (if wrist below knuckle)
            # or smaller NEGATIVE value (if wrist above knuckle).
            # NOTE: In MediaPipe Y increases downwards.
            # Good Arch: Wrist (Low Y) is ABOVE Knuckle (High Y).  Wrist_Y < Knuckle_Y.
            #   (Wrist - Knuckle) should be NEGATIVE.
            # Flat/Low Wrist: Wrist (High Y) is BELOW Knuckle (Low Y). Wrist_Y > Knuckle_Y.
            #   (Wrist - Knuckle) becomes POSITIVE.
            
            current_dist = wrist_y - knuckle_y
            
            # If our calibrated baseline captured a "textbook" arch, current_dist should be negative.
            # If we allow a 20% deviation from that baseline?
            # Or use the user's simplified logic: if (wrist - knuckle) > (baseline * 0.2)
            # Assuming 'baseline' stored positive magnitude of a "safe" buffer?
            # Actually, let's stick to the logic: "If wrist is lower than knuckle by X amount".
            
            # RE-READING CALIBRATION LOGIC:
            # calibration stored: dist_rh = sorted_hands[0].landmark[0].y - sorted_hands[0].landmark[9].y
            # If we hold a Perfect Arch, Wrist is ABOVE Knuckle. Y_wrist < Y_knuckle.
            # So dist_rh will be NEGATIVE (e.g. -0.05).
            
            # So if we drop our wrist, dist_rh becomes LESS NEGATIVE (closest to 0) or POSITIVE.
            # Check: if current_dist > baseline_dist + tolerance?
            
            # User suggested:
            # if (wrist_y - knuckle_y) > (baseline_rh["dist"] * 0.2): error = "Low Wrist"
            # This implies baseline["dist"] is a positive magnitude of "allowable drop"? 
            # OR that we capture the *magnitude* of the perfect arch.
            # Let's use magnitude for safety.
            
            # Let's refine the calibration capture:
            # calibration stored RAW (wrist - knuckle). Likely negative.
            # User's snippet: baseline_rh["dist"] = abs(...)
            # Ah, user used ABS in snippet.
            
            # Let's fix the calibration function to use ABS as requested.
            # Baseline = Magnitude of "Good" Arch height.
            
            threshold = baseline_dist + 0.02 # Add a small tolerance buffer
            if calibrated:
                 # If we use the ABS logic from user request
                 # if (wrist_y - knuckle_y) > (baseline * 0.2) -> this seems risky if baseline is 0.05. 0.01 is too strict.
                 
                 # Let's stick to a robust logical interpretation:
                 # We captured the "Good Position". Any deviation downwards (increasing Y) is bad.
                 # Let's say we allow 20% degradation of the arch height before flagging.
                 pass
                 
            # Re-implementing strictly based on user snippet logic for determining error:
            # if (wrist_y - knuckle_y) > (baseline_rh["dist"] * 0.2):
            # This implies baseline is POSITIVE.
            
            # NOTE: We need to make sure calibration uses ABS then.
            
            # Let's look at the implementation below.
            
            # Check Flat Finger
            for pip_idx in [6, 10, 14, 18]:
                mcp = [hand_landmarks.landmark[pip_idx-1].x, hand_landmarks.landmark[pip_idx-1].y]
                pip = [hand_landmarks.landmark[pip_idx].x, hand_landmarks.landmark[pip_idx].y]
                dip = [hand_landmarks.landmark[pip_idx+1].x, hand_landmarks.landmark[pip_idx+1].y]
                
                angle = calculate_angle(mcp, pip, dip)
                if angle > 165:
                    if error == "": error = "Flat Finger"

            # Apply Error Thresholds
            # Using simple robust logic: 
            # If WristY is below KnuckleY + (some buffer).
            # If calibrated, buffer = baseline * 0.2 (as requested).
            # Else buffer = 0.05 / 0.02
             
            threshold_val = 0.05 if label == "RH" else 0.02
            if calibrated:
                 # User logic: if (wrist_y - knuckle_y) > (baseline * 0.2)
                 if (wrist_y - knuckle_y) > (baseline_rh["dist"] * 0.2 if label == "RH" else baseline_lh["dist"] * 0.2):
                     error = "Low Wrist"
            else:
                 if wrist_y > knuckle_y + threshold_val:
                     error = "Low Wrist"

            # Store status
            color = (0, 0, 255) if error != "" else (0, 255, 0)
            msg = f"Incorrect ({error})" if error != "" else "Correct"
            statuses[label] = {"msg": msg, "color": color}

            # Draw skeleton with dynamic Line and Landmark colors
            mp_draw.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2), # Dots
                mp_draw.DrawingSpec(color=color, thickness=2) # Lines
            )

    # Display Independent Results
    y_pos = 50
    for hand_label in ["RH", "LH"]:
        if hand_label in statuses:
            text = f"{hand_label}: {statuses[hand_label]['msg']}"
            color = statuses[hand_label]['color']
            cv2.putText(image, text, (50, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_pos += 40

    cv2.imshow('DreamPlay Calibrated Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
