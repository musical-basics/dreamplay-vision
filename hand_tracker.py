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

# --- CONFIGURATION ---
# --- CONFIGURATION ---
font_scale = 1.0
thickness = 2
score_font_scale = 1.5
score_thickness = 4

# --- SCORING HELPERS ---
def calculate_score(current_val, baseline_val, max_deviation):
    # Calculate how far we are from 'Perfect' (baseline)
    # Score drops as we deviate from baseline. 
    # For wrist: deviation is how much LOWER (y increases) we are compared to baseline 'dist'.
    # For fingers: deviation is how much straighter (angle closer to 180) we are compared to baseline angle.
    
    deviation = abs(current_val - baseline_val)
    # Clamp deviation to max_deviation
    deviation = min(deviation, max_deviation)
    
    # Linear drop: 0 deviation = 100%, max_deviation = 0%
    score = 100 - (deviation / max_deviation * 100)
    return int(max(0, score))

def get_status_color(score):
    if score >= 90: return (0, 255, 0)      # Green (Excellent)
    if score >= 80: return (0, 255, 255)    # Yellow (Acceptable)
    if score >= 70: return (0, 165, 255)    # Orange (Danger)
    return (0, 0, 255)                      # Red (Critical)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # Remove horizontal flip to fix the "reversed" camera view
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Prepare sorting
    sorted_hands = []
    if results.multi_hand_landmarks:
         # STRICT POSITIONAL SORTING: RR Hand = Higher X
         sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x, reverse=True)

    # --- Manual Reset ---
    if cv2.waitKey(1) & 0xFF == ord('c'):
        current_state = STATE_COUNTDOWN
        timer_start = time.time()
        calibration_data_rh, calibration_data_lh = [], []
        print("Manual Reset Triggered")

    # --- STATE MACHINE ---
    if current_state == STATE_WAITING:
        cv2.putText(image, "PLACE BOTH HANDS ON KEYS", (100, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if len(sorted_hands) == 2 and check_stability(sorted_hands):
            current_state = STATE_COUNTDOWN
            timer_start = time.time()
            calibration_data_rh, calibration_data_lh = [], []

    elif current_state == STATE_COUNTDOWN:
        elapsed = time.time() - timer_start
        remaining = max(0, int(countdown_duration - elapsed + 1))
        cv2.putText(image, f"CALIBRATING IN: {remaining}", (150, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        if elapsed >= countdown_duration:
            current_state = STATE_CALIBRATING
            timer_start = time.time()
            calibration_data_rh, calibration_data_lh = [], []

    elif current_state == STATE_CALIBRATING:
        elapsed = time.time() - timer_start
        prog = min(elapsed / calibration_duration, 1.0)
        
        if len(sorted_hands) >= 2:
            # RH [0], LH [1]
            dist_rh = sorted_hands[0].landmark[0].y - sorted_hands[0].landmark[9].y
            dist_lh = sorted_hands[1].landmark[0].y - sorted_hands[1].landmark[9].y
            calibration_data_rh.append(dist_rh)
            calibration_data_lh.append(dist_lh)
        
        bar_width = int(prog * 400)
        cv2.rectangle(image, (100, 300), (500, 330), (50, 50, 50), -1) 
        cv2.rectangle(image, (100, 300), (100 + bar_width, 330), (0, 255, 0), -1)
        cv2.putText(image, "HOLD PERECT FORM...", (100, 290), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if elapsed >= calibration_duration:
            if calibration_data_rh:
                baseline_rh["dist"] = sum(calibration_data_rh) / len(calibration_data_rh)
            if calibration_data_lh:
                baseline_lh["dist"] = sum(calibration_data_lh) / len(calibration_data_lh)
            
            # Start with ideal angles (approximating naturally curved fingers ~140-160 deg)
            # We didn't explicitly calibrate angle in previous steps, let's assume 150 is ideal baseline
            # Or we could have measured it. For now, we assume user held perfect form.
            baseline_rh["angle"] = 140
            baseline_lh["angle"] = 140
            
            current_state = STATE_ACTIVE
            print(f"Baselines - RH Dist: {baseline_rh['dist']:.3f}")

    elif current_state == STATE_ACTIVE:
        if sorted_hands:
            for i, hand_landmarks in enumerate(sorted_hands):
                if i > 1: break
                label = "RH" if i == 0 else "LH"
                
                # 1. WRIST SCORE
                # Measure (WristY - KnuckleY)
                # Ideally, Wrist is ABOVE knuckle (lower Y). So (WristY - KnuckleY) should be negative.
                # If Wrist drops, Y increases, (WristY - KnuckleY) becomes larger (more positive).
                current_dist = hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y
                base_dist = baseline_rh["dist"] if label == "RH" else baseline_lh["dist"]
                
                # A drop in form means current_dist > base_dist.
                # Deviation only matters if it's POSITIVE shift (flattening arch).
                # If current_dist < base_dist (wrist even higher), that's fine/excellent.
                wrist_deviation = max(0, current_dist - base_dist)
                
                # Max deviation allowed: ~0.15 (arbitrary units of 'drop')
                wrist_score = calculate_score(current_dist, base_dist, 0.15)
                # If we are strictly better (higher arch), clamp to 100
                if current_dist < base_dist: wrist_score = 100

                # 2. FINGER SCORE
                # Check PIP angles (average of index, middle, ring, pinky)
                total_angle = 0
                count = 0
                for pip_idx in [6, 10, 14, 18]:
                    mcp = [hand_landmarks.landmark[pip_idx-1].x, hand_landmarks.landmark[pip_idx-1].y]
                    pip = [hand_landmarks.landmark[pip_idx].x, hand_landmarks.landmark[pip_idx].y]
                    dip = [hand_landmarks.landmark[pip_idx+1].x, hand_landmarks.landmark[pip_idx+1].y]
                    total_angle += calculate_angle(mcp, pip, dip)
                    count += 1
                avg_angle = total_angle / count
                
                # Ideal is ~140. Flat is 180.
                # Deviation towards 180 is bad.
                angle_deviation = max(0, avg_angle - 140) 
                # Max deviation: 40 degrees (140 to 180)
                finger_score = calculate_score(avg_angle, 140, 40)
                
                # Aggregate Score
                total_score = (wrist_score + finger_score) // 2
                
                # Determine Colors
                main_color = get_status_color(total_score)
                wrist_color = get_status_color(wrist_score)
                finger_color = get_status_color(finger_score)

                # Draw Skeleton
                mp_draw.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=main_color, thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=main_color, thickness=2)
                )

                # UI Layout
                if label == "LH":
                    x_base = 20
                else:
                    # Right align
                    x_base = image.shape[1] - 350
                
                # Main Score
                cv2.putText(image, f"{label}: {int(total_score)}%", (x_base, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, main_color, score_thickness)
                
                # Sub-scores
                cv2.putText(image, f"Wrist: {int(wrist_score)}%", (x_base, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, wrist_color, 2)
                cv2.putText(image, f"Fingers: {int(finger_score)}%", (x_base, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, finger_color, 2)

    cv2.imshow('DreamPlay Continuous Scoring', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
