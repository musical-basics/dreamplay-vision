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
def calculate_wrist_score(current_val, baseline_val):
    """
    Stricter scoring for Wrist.
    If current_val > baseline_val (wrist dropping), penalize heavily.
    If current_val <= baseline_val (wrist higher), 100% score (NO penalty for high wrist in this context, 
    though extremely high wrist might be bad, usually "Low Wrist" is the main error).
    """
    # deviation positive = drop
    deviation = current_val - baseline_val
    
    if deviation <= 0:
        return 100
    
    # We want a noticeable drop even for small deviation.
    # User said 0:14 was "clearly low" but got 92%.
    # Let's make the penalty multiplier stronger.
    # Deviation is in normalized coords (0.0 to 1.0).
    # A drop of 0.05 is significant. 
    # 0.05 * x = penalty. If we want 0.05 to be ~70% score (30 penalty), x = 600.
    
    penalty = deviation * 800 # Strong penalty
    score = 100 - penalty
    return int(max(0, score))

def calculate_finger_score(avg_angle):
    """
    Forgiving scoring for Fingers.
    Only penalize if angle is approaching 180 (flat).
    Ignore over-curving (angle < 140).
    """
    # Threshold where we start penalizing
    # User said 0:18 fingers were curved but got 6%.
    # Old logic penalized deviation from 150 in both directions.
    # New logic: penalize only if > 160.
    
    start_penalty_angle = 160
    if avg_angle <= start_penalty_angle:
        return 100
    
    # Range 160 to 180 is bad.
    # 180 should be 0 score.
    # 20 degrees range.
    
    deviation = avg_angle - start_penalty_angle
    # Map 0..20 to 0..100 penalty
    penalty = (deviation / 20.0) * 100
    score = 100 - penalty
    return int(max(0, score))

def get_status_color(score):
    if score >= 90: return (0, 255, 0)      # Green (Excellent)
    if score >= 80: return (0, 255, 255)    # Yellow (Acceptable)
    if score >= 70: return (0, 165, 255)    # Orange (Danger)
    return (0, 0, 255)                      # Red (Critical)

# --- GRAPH HELPER ---
# History for 120 frames (approx 4-5 seconds at 30fps)
graph_history = {
    "RH": {"total": deque([100]*120, maxlen=120), "wrist": deque([100]*120, maxlen=120), "fingers": deque([100]*120, maxlen=120)},
    "LH": {"total": deque([100]*120, maxlen=120), "wrist": deque([100]*120, maxlen=120), "fingers": deque([100]*120, maxlen=120)}
}

def draw_triple_graph(img, data, side="RH"):
    """Draws a scrolling triple-line graph."""
    width, height = 300, 120 # Enlarged as requested
    margin = 30
    
    # Position
    x_start = img.shape[1] - width - margin if side == "RH" else margin
    y_start = img.shape[0] - height - margin
    
    # Draw Background
    cv2.rectangle(img, (x_start, y_start), (x_start + width, y_start + height), (30, 30, 30), -1)
    
    # Labels
    cv2.putText(img, f"{side} TREND", (x_start + 5, y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Helper to plot one deque
    def plot_line(history, color, thickness=1):
        if not history: return
        pts = []
        for i, val in enumerate(history):
            px = x_start + int(i * (width / 120))
            py = y_start + height - int(val * (height / 100))
            pts.append((px, py))
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], False, color, thickness)
            
    # Draw 3 lines
    # WRIST: Blue (Cyan-ish)
    plot_line(data["wrist"], (255, 255, 0), 1) 
    # FINGERS: Green (or a distinct color) -> User said "Green/Yellow Line". Let's use Magenta for contrast or Green.
    # Graphs often use Red/Green/Blue. Let's use:
    # Wrist: Cyan (255, 255, 0)
    # Fingers: Magenta (255, 0, 255)
    # Total: White (255, 255, 255)
    
    plot_line(data["fingers"], (255, 0, 255), 1)
    plot_line(data["total"], (255, 255, 255), 2)
    
    # Legend
    # We can add small dots/text if needed, but keeping it clean for now.

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
                # 1. Wrist Score (Stricter)
                current_dist = hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y
                base_dist = baseline_rh["dist"] if label == "RH" else baseline_lh["dist"]
                wrist_score = calculate_wrist_score(current_dist, base_dist)

                # 2. Finger Score (Forgiving of curvature)
                total_angle = 0
                count = 0
                for pip_idx in [6, 10, 14, 18]:
                    mcp = [hand_landmarks.landmark[pip_idx-1].x, hand_landmarks.landmark[pip_idx-1].y]
                    pip = [hand_landmarks.landmark[pip_idx].x, hand_landmarks.landmark[pip_idx].y]
                    dip = [hand_landmarks.landmark[pip_idx+1].x, hand_landmarks.landmark[pip_idx+1].y]
                    total_angle += calculate_angle(mcp, pip, dip)
                    count += 1
                avg_angle = total_angle / count
                finger_score = calculate_finger_score(avg_angle)

                # Total
                total_score = (wrist_score + finger_score) // 2
                
                # Update History
                graph_history[label]["total"].append(total_score)
                graph_history[label]["wrist"].append(wrist_score)
                graph_history[label]["fingers"].append(finger_score)
                
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
                # If hand lost, maybe append last score or 0? 
                # Let's append 0 to show drop in graph or nothing?
                # User says "stop registering". We won't update graph to pause it.
                pass

        # Always draw graphs if history exists
        if len(graph_history["RH"]["total"]) > 1:
            draw_triple_graph(image, graph_history["RH"], side="RH")
        if len(graph_history["LH"]["total"]) > 1:
            draw_triple_graph(image, graph_history["LH"], side="LH")

    cv2.imshow('DreamPlay Continuous Scoring', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
