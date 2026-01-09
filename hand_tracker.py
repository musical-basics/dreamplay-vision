import cv2
import mediapipe as mp
import numpy as np
import time

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

# --- State Constants ---
STATE_WAITING = 0      # Waiting for 2 hands to appear
STATE_COUNTDOWN = 1    # Hands found, counting down
STATE_CALIBRATING = 2  # Recording data
STATE_ACTIVE = 3       # Normal monitoring

# --- Global State ---
current_state = STATE_WAITING
timer_start = 0
calibration_duration = 3.0
countdown_duration = 3.0

calibration_data_rh = []
calibration_data_lh = []
baseline_rh = {"dist": 0.05}
baseline_lh = {"dist": 0.03}
prev_hand_pos = None

def check_stability(sorted_hands):
    """Checks if hands are held still enough to start calibration."""
    global prev_hand_pos
    STABILITY_THRESHOLD = 0.02 # How much hand movement is allowed to trigger auto-start
    
    if len(sorted_hands) < 2: return False
    
    # Track the wrist position (Landmark 0) of the front hand (RH [0])
    current_pos = np.array([sorted_hands[0].landmark[0].x, sorted_hands[0].landmark[0].y])
    
    if prev_hand_pos is None:
        prev_hand_pos = current_pos
        return False
    
    dist = np.linalg.norm(current_pos - prev_hand_pos)
    prev_hand_pos = current_pos
    return dist < STABILITY_THRESHOLD

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # Remove horizontal flip to fix the "reversed" camera view
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Prepare sorting
    sorted_hands = []
    if results.multi_hand_landmarks:
         # Sort by X-coordinate (Higher X = Right Hand in side view)
         sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x, reverse=True)

    # --- Manual Reset ---
    if cv2.waitKey(1) & 0xFF == ord('c'):
        current_state = STATE_COUNTDOWN
        timer_start = time.time()
        # Reset data
        calibration_data_rh, calibration_data_lh = [], []
        print("Manual Reset Triggered")

    # --- STATE MACHINE ---
    
    if current_state == STATE_WAITING:
        cv2.putText(image, "PLACE BOTH HANDS ON KEYS", (100, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Check if we should start countdown
        if len(sorted_hands) == 2 and check_stability(sorted_hands):
            current_state = STATE_COUNTDOWN
            timer_start = time.time()
            # Clear data just in case
            calibration_data_rh, calibration_data_lh = [], []
            print("Auto-Detect: Starting Countdown...")

    elif current_state == STATE_COUNTDOWN:
        elapsed = time.time() - timer_start
        remaining = max(0, int(countdown_duration - elapsed + 1))
        
        # Big countdown centered
        cv2.putText(image, f"CALIBRATING IN: {remaining}", (150, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
        
        if elapsed >= countdown_duration:
            current_state = STATE_CALIBRATING
            timer_start = time.time()
            calibration_data_rh, calibration_data_lh = [], []

    elif current_state == STATE_CALIBRATING:
        elapsed = time.time() - timer_start
        prog = min(elapsed / calibration_duration, 1.0)
        
        # Collect data
        if len(sorted_hands) >= 2:
            # RH is index 0, LH is index 1
            dist_rh = abs(sorted_hands[0].landmark[0].y - sorted_hands[0].landmark[9].y)
            dist_lh = abs(sorted_hands[1].landmark[0].y - sorted_hands[1].landmark[9].y)
            calibration_data_rh.append(dist_rh)
            calibration_data_lh.append(dist_lh)
        
        # Draw Progress Bar
        bar_width = int(prog * 400)
        cv2.rectangle(image, (100, 300), (500, 330), (50, 50, 50), -1) 
        cv2.rectangle(image, (100, 300), (100 + bar_width, 330), (0, 255, 0), -1)
        cv2.putText(image, "HOLD STILL...", (100, 290), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if elapsed >= calibration_duration:
            # Compute averages
            if calibration_data_rh:
                baseline_rh["dist"] = sum(calibration_data_rh) / len(calibration_data_rh)
            if calibration_data_lh:
                baseline_lh["dist"] = sum(calibration_data_lh) / len(calibration_data_lh)
            
            current_state = STATE_ACTIVE
            print(f"Locked Baselines - RH: {baseline_rh['dist']:.3f}, LH: {baseline_lh['dist']:.3f}")

    elif current_state == STATE_ACTIVE:
        # --- TRACKING LOGIC ---
        statuses = {}
        
        # Only process if we have hands (or hold last state? let's just process what we see)
        if results.multi_hand_landmarks:
            # We continue to use X-sorting. If one hand disappears, 
            # the remaining hand becomes index 0 (RH).
            # Limitation: losing one hand might mislabel the other if strictly X-sorted.
            # But for this prototype, we assume user keeps hands in frame for monitoring.
            
            for i, hand_landmarks in enumerate(sorted_hands):
                # Only handle up to 2 hands
                if i > 1: break
                label = "RH" if i == 0 else "LH"
                
                wrist_y = hand_landmarks.landmark[0].y
                knuckle_y = hand_landmarks.landmark[9].y
                
                # Use calibrated baseline
                baseline_dist = baseline_rh["dist"] if label == "RH" else baseline_lh["dist"]
                
                error = ""
                # Logic: If (wrist - knuckle) exceeds 20% of the baseline magnitude
                # (Assuming baseline captured a 'good' distance magnitude)
                threshold = baseline_dist * 0.2
                current_dist = wrist_y - knuckle_y
                
                if current_dist > threshold:
                    error = "Low Wrist"
                    
                # Check Flat Finger
                for pip_idx in [6, 10, 14, 18]:
                    mcp = [hand_landmarks.landmark[pip_idx-1].x, hand_landmarks.landmark[pip_idx-1].y]
                    pip = [hand_landmarks.landmark[pip_idx].x, hand_landmarks.landmark[pip_idx].y]
                    dip = [hand_landmarks.landmark[pip_idx+1].x, hand_landmarks.landmark[pip_idx+1].y]
                    
                    angle = calculate_angle(mcp, pip, dip)
                    if angle > 165:
                        if error == "": error = "Flat Finger"

                # Store status
                color = (0, 0, 255) if error != "" else (0, 255, 0)
                msg = f"Incorrect ({error})" if error != "" else "Correct"
                statuses[label] = {"msg": msg, "color": color}

                # Draw skeleton
                mp_draw.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=color, thickness=2)
                )

        # Display Text
        y_pos = 50
        for hand_label in ["RH", "LH"]:
            if hand_label in statuses:
                text = f"{hand_label}: {statuses[hand_label]['msg']}"
                color = statuses[hand_label]['color']
                cv2.putText(image, text, (50, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_pos += 40

    cv2.imshow('DreamPlay Auto-Calib Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
