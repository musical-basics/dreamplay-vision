import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# --- CONFIGURATION ---
font_scale = 1.0
thickness = 2
score_font_scale = 1.5
score_thickness = 4

# --- SCORING HELPERS ---
def get_pro_score(val, baseline, is_wrist=True):
    """
    Advanced Scoring using Exponential/Sigmoid decay.
    """
    if is_wrist:
        # WRIST: High sensitivity.
        # Deviation > 0 means drop.
        # Val is y-dist. Baseline is y-dist.
        # If val > baseline, it's a drop.
        deviation = val - baseline
        if deviation <= 0: return 100
        
        # Exponential penalty
        # deviation of 0.05 should be bad.
        # exp(-15 * 0.05) = 0.47 * 100 = 47.
        # exp(-30 * 0.05) = 0.22 * 100 = 22.
        # User wants "High sensitivity".
        return int(100 * np.exp(-20 * deviation))
    else:
        # FINGERS: Forgiving Sigmoid.
        # Val is Angle (degrees). Baseline is Ideal Angle (~140).
        # We only care if angle approaches 180.
        # Normalize deviation to 0..1 range (140->180 is 40 deg range)
        
        # If angle < 150, it's perfect (100).
        if val < 155: return 100
        
        # Map 155..185 to some range
        # We want 160 to be ~85-90
        # We want 175 to be ~10-20
        
        # Using the user's sigmoid idea: 100 / (1 + exp(k*(dev - threshold)))
        # Let's normalize val: 
        norm_val = (val - 140) / 40.0 # 140->0, 180->1.0
        
        # Threshold: point where score is 50%. Let's say at 170 deg (0.75)
        # Slope k: steepness.
        
        # 100 / (1 + exp(15 * (norm_val - 0.75)))
        # At 160 (0.5): exp(15 * -0.25) = exp(-3.75) ~ 0.02. 100/1.02 ~ 98.
        # At 170 (0.75): exp(0) = 1. 100/2 = 50.
        # At 175 (0.875): exp(15 * 0.125) = exp(1.875) ~ 6.5. 100/7.5 ~ 13.
        
        score = 100 / (1 + np.exp(15 * (norm_val - 0.75)))
        return int(score)

def get_status_color(score):
    if score >= 90: return (0, 255, 0)      # Green (Excellent)
    if score >= 80: return (0, 255, 255)    # Yellow (Acceptable)
    if score >= 70: return (0, 165, 255)    # Orange (Danger)
    return (0, 0, 255)                      # Red (Critical)

# --- GRAPH HELPER ---
# History for 120 frames
graph_history = {
    "RH": {"total": deque([100]*120, maxlen=120), "wrist": deque([100]*120, maxlen=120), "finger": deque([100]*120, maxlen=120)},
    "LH": {"total": deque([100]*120, maxlen=120), "wrist": deque([100]*120, maxlen=120), "finger": deque([100]*120, maxlen=120)}
}

GRAPH_W, GRAPH_H = 400, 200 # Bigger
SECTION_H = GRAPH_H // 3    # 3 equal sections

def draw_area_graph(img, history_data, side="RH"):
    margin = 30
    x_start = img.shape[1] - GRAPH_W - margin if side == "RH" else margin
    y_start = img.shape[0] - GRAPH_H - margin
    
    # Draw Legend Labels
    # Order from Top to Bottom: Total, Wrist, Finger
    # But y grows down. So Top is y_start.
    
    labels = [("TOTAL", (255, 255, 255)), ("WRIST", (255, 200, 0)), ("FINGER", (0, 255, 100))]
    keys = ["total", "wrist", "finger"]
    
    for i, (label_text, label_color) in enumerate(labels):
        s_y = y_start + (i * SECTION_H)
        
        # Draw Section Background
        cv2.rectangle(img, (x_start, s_y), (x_start + GRAPH_W, s_y + SECTION_H), (30, 30, 30), -1)
        # Separator line
        cv2.line(img, (x_start, s_y + SECTION_H), (x_start + GRAPH_W, s_y + SECTION_H), (100, 100, 100), 1)

        # Draw Legend Text OUTSIDE the graph
        text_x = x_start - 80 if side=="RH" else x_start + GRAPH_W + 10
        cv2.putText(img, label_text, (text_x, s_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
        
        # Plot Area
        key = keys[i]
        data = history_data[key]
        
        for t, score in enumerate(data):
            moment_color = get_status_color(score)
            
            # Map t (0..119) to x
            line_x = x_start + int(t * (GRAPH_W / 120))
            
            # Height based on score (0..100) -> (0..SECTION_H)
            bar_h = int((score / 100) * SECTION_H)
            
            # Draw vertical line from bottom of section UP
            # Bottom is s_y + SECTION_H
            cv2.line(img, (line_x, s_y + SECTION_H), (line_x, s_y + SECTION_H - bar_h), moment_color, 2)

# --- TRACKING SETUP ---
# Helper function
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.8, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# --- State Constants ---
STATE_WAITING = 0
STATE_COUNTDOWN = 1
STATE_CALIBRATING = 2
STATE_ACTIVE = 3

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

# --- PERSISTENT TRACKING STATE ---
tracked_hands = {"RH": None, "LH": None} 

def check_stability(hands_list):
    """Checks if hands are held still enough to start calibration."""
    global prev_hand_pos
    STABILITY_THRESHOLD = 0.02 
    
    if len(hands_list) < 2: return False
    
    current_pos = np.array([hands_list[0].landmark[0].x, hands_list[0].landmark[0].y])
    
    if prev_hand_pos is None:
        prev_hand_pos = current_pos
        return False
    
    dist = np.linalg.norm(current_pos - prev_hand_pos)
    prev_hand_pos = current_pos
    return dist < STABILITY_THRESHOLD

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # Remove horizontal flip
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    detected_hands = []
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            detected_hands.append(hl)

    # --- Manual Reset ---
    if cv2.waitKey(1) & 0xFF == ord('c'):
        current_state = STATE_COUNTDOWN
        timer_start = time.time()
        calibration_data_rh, calibration_data_lh = [], []
        print("Manual Reset Triggered")

    # --- STATE HANDLER ---
    if current_state == STATE_WAITING:
        detected_hands.sort(key=lambda h: h.landmark[0].x, reverse=True)
        cv2.putText(image, "PLACE BOTH HANDS ON KEYS", (100, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if len(detected_hands) == 2 and check_stability(detected_hands):
            current_state = STATE_COUNTDOWN
            timer_start = time.time()
            calibration_data_rh, calibration_data_lh = [], []
        
        for hl in detected_hands: mp_draw.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)

    elif current_state == STATE_COUNTDOWN:
        detected_hands.sort(key=lambda h: h.landmark[0].x, reverse=True)
        elapsed = time.time() - timer_start
        remaining = max(0, int(countdown_duration - elapsed + 1))
        cv2.putText(image, f"CALIBRATING IN: {remaining}", (150, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        for hl in detected_hands: mp_draw.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)
        if elapsed >= countdown_duration:
            current_state = STATE_CALIBRATING
            timer_start = time.time()
            calibration_data_rh, calibration_data_lh = [], []

    elif current_state == STATE_CALIBRATING:
        detected_hands.sort(key=lambda h: h.landmark[0].x, reverse=True)
        elapsed = time.time() - timer_start
        prog = min(elapsed / calibration_duration, 1.0)
        
        if len(detected_hands) >= 2:
            d_rh = detected_hands[0].landmark[0].y - detected_hands[0].landmark[9].y
            d_lh = detected_hands[1].landmark[0].y - detected_hands[1].landmark[9].y
            calibration_data_rh.append(d_rh)
            calibration_data_lh.append(d_lh)
            
            tracked_hands["RH"] = np.array([detected_hands[0].landmark[0].x, detected_hands[0].landmark[0].y])
            tracked_hands["LH"] = np.array([detected_hands[1].landmark[0].x, detected_hands[1].landmark[0].y])
        
        bar_width = int(prog * 400)
        cv2.rectangle(image, (100, 300), (500, 330), (50, 50, 50), -1) 
        cv2.rectangle(image, (100, 300), (100 + bar_width, 330), (0, 255, 0), -1)
        cv2.putText(image, "HOLD PERECT FORM...", (100, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for hl in detected_hands: mp_draw.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)

        if elapsed >= calibration_duration:
            if calibration_data_rh: baseline_rh["dist"] = sum(calibration_data_rh) / len(calibration_data_rh)
            if calibration_data_lh: baseline_lh["dist"] = sum(calibration_data_lh) / len(calibration_data_lh)
            current_state = STATE_ACTIVE
            print("Calibration Complete. Tracking Locked.")

    elif current_state == STATE_ACTIVE:
        # Match Hands
        def get_wrist(hl): return np.array([hl.landmark[0].x, hl.landmark[0].y])
        final_assignments = {}
        
        if detected_hands:
            if tracked_hands["RH"] is not None and tracked_hands["LH"] is not None:
                matches = []
                for idx, hl in enumerate(detected_hands):
                    pos = get_wrist(hl)
                    d_rh = np.linalg.norm(pos - tracked_hands["RH"])
                    d_lh = np.linalg.norm(pos - tracked_hands["LH"])
                    matches.append( (d_rh, "RH", idx) )
                    matches.append( (d_lh, "LH", idx) )
                
                matches.sort(key=lambda x: x[0])
                assigned_ids = set()
                assigned_idxs = set()
                
                for dist, target_id, det_idx in matches:
                    if target_id not in assigned_ids and det_idx not in assigned_idxs:
                        final_assignments[target_id] = detected_hands[det_idx]
                        tracked_hands[target_id] = get_wrist(detected_hands[det_idx])
                        assigned_ids.add(target_id)
                        assigned_idxs.add(det_idx)

        # Process Each Hand
        for label in ["RH", "LH"]:
            hand_landmarks = final_assignments.get(label)
            
            if hand_landmarks:
                # 1. Wrist Score (Exp penalty)
                current_dist = hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y
                base_dist = baseline_rh["dist"] if label == "RH" else baseline_lh["dist"]
                wrist_score = get_pro_score(current_dist, base_dist, is_wrist=True)

                # 2. Finger Score (Sigmoid)
                total_angle = 0
                count = 0
                for pip_idx in [6, 10, 14, 18]:
                    mcp = [hand_landmarks.landmark[pip_idx-1].x, hand_landmarks.landmark[pip_idx-1].y]
                    pip = [hand_landmarks.landmark[pip_idx].x, hand_landmarks.landmark[pip_idx].y]
                    dip = [hand_landmarks.landmark[pip_idx+1].x, hand_landmarks.landmark[pip_idx+1].y]
                    total_angle += calculate_angle(mcp, pip, dip)
                    count += 1
                avg_angle = total_angle / count
                finger_score = get_pro_score(avg_angle, 140, is_wrist=False)

                # Total
                total_score = (wrist_score + finger_score) // 2
                
                # Update History
                graph_history[label]["total"].append(total_score)
                graph_history[label]["wrist"].append(wrist_score)
                graph_history[label]["finger"].append(finger_score)
                
                # Colors
                main_color = get_status_color(total_score)
                wrist_color = get_status_color(wrist_score)
                finger_color = get_status_color(finger_score)
                
                # Draw
                mp_draw.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=main_color, thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=main_color, thickness=2)
                )

                # UI Text
                x_base = 20 if label == "LH" else image.shape[1] - 350
                
                cv2.putText(image, f"{label}: {int(total_score)}%", (x_base, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, main_color, score_thickness)
                cv2.putText(image, f"Wrist: {int(wrist_score)}%", (x_base, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, wrist_color, 2)
                cv2.putText(image, f"Fingers: {int(finger_score)}%", (x_base, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, finger_color, 2)
            else:
                pass

        # Always draw graphs if history exists
        if len(graph_history["RH"]["total"]) > 1:
            draw_area_graph(image, graph_history["RH"], side="RH")
        if len(graph_history["LH"]["total"]) > 1:
            draw_area_graph(image, graph_history["LH"], side="LH")

    cv2.imshow('DreamPlay Continuous Scoring', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
