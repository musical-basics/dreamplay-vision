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
def calculate_forgiving_score(current_val, baseline_val, tolerance):
    """
    Calculates a score (0-100) that is forgiving for small deviations 
    but drops sharply for large errors (Sigmoid-like behavior).
    """
    deviation = abs(current_val - baseline_val)
    
    # "Forgiving Zone": If deviation is within 50% of tolerance, barely penalize.
    if deviation < (tolerance * 0.5):
        # Linear drop from 100 to ~90 in the safe zone
        # (deviation / tolerance) * 20 -> max 10% penalty
        penalty = (deviation / tolerance) * 20
        return int(max(0, 100 - penalty))
    
    # "Danger Zone": Deviating beyond safe zone
    # Scale the remaining deviation
    # We want 100% tolerance to equate to ~0 score? Or low score?
    # User requested: "anything from 80-90 as acceptable"
    
    # Let's map normalized deviation (d/t) to score:
    # 0.0 -> 100
    # 0.5 -> 90
    # 1.0 -> 0 (or low)
    
    # Standard linear mapping after 0.5
    # ratio goes from 0.5 to 1.0
    # we want score to go from 90 to 0
    
    ratio = deviation / tolerance
    if ratio > 1.0: return 0
    
    # Map [0.5, 1.0] to [90, 0]
    # (ratio - 0.5) / 0.5 -> 0 to 1
    normalized = (ratio - 0.5) * 2
    score = 90 - (normalized * 90)
    
    return int(max(0, score))

def get_status_color(score):
    if score >= 90: return (0, 255, 0)      # Green (Excellent)
    if score >= 80: return (0, 255, 255)    # Yellow (Acceptable)
    if score >= 70: return (0, 165, 255)    # Orange (Danger)
    return (0, 0, 255)                      # Red (Critical)

# --- GRAPH HELPER ---
# History for 100 frames
score_history = {"RH": deque([100]*100, maxlen=100), "LH": deque([100]*100, maxlen=100)}

def draw_pip_graph(img, scores, side="RH"):
    """Draws a scrolling line graph in the corner of the screen."""
    width, height = 200, 80
    margin = 20
    # Position: Bottom Left for LH, Bottom Right for RH
    x_start = img.shape[1] - width - margin if side == "RH" else margin
    y_start = img.shape[0] - height - margin
    
    # Draw Background (Semi-transparent black ideally, but solid for now)
    cv2.rectangle(img, (x_start, y_start), (x_start + width, y_start + height), (30, 30, 30), -1)
    
    # Plot Score Line
    if not scores: return

    points = []
    for i, s in enumerate(scores):
        # Map i (0-99) to x
        px = x_start + int((i / 99) * width)
        # Map s (0-100) to y (inverted)
        py = y_start + height - int((s / 100) * height)
        points.append((px, py))
    
    # Draw polyline
    # Color based on latest score
    line_color = get_status_color(scores[-1])
    
    if len(points) > 1:
        cv2.polylines(img, [np.array(points)], False, line_color, 2)
        
    # Label
    cv2.putText(img, f"{side} History", (x_start + 5, y_start + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

# --- TRACKING SETUP ---
# Helper function to calculate the angle between three points
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
baseline_rh = {"dist": 0.05, "angle": 140}
baseline_lh = {"dist": 0.03, "angle": 140}
prev_hand_pos = None # For stability check

# --- PERSISTENT TRACKING STATE ---
# We store the last normalized (x, y) centroid for RH and LH
tracked_hands = {"RH": None, "LH": None} 

def check_stability(hands_list):
    """Checks if hands are held still enough to start calibration."""
    global prev_hand_pos
    STABILITY_THRESHOLD = 0.02 
    
    # hands_list is expected to be X-sorted initially
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
    
    # 1. Get List of Detected Hands
    detected_hands = []
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            detected_hands.append(hl)

    # 2. Logic Handler based on State
    
    # --- Manual Reset ---
    if cv2.waitKey(1) & 0xFF == ord('c'):
        current_state = STATE_COUNTDOWN
        timer_start = time.time()
        calibration_data_rh, calibration_data_lh = [], []
        print("Manual Reset Triggered")

    # --- STATE: WAITING ---
    if current_state == STATE_WAITING:
        display_hands = []
        # In waiting mode, we sort by X just to find the "Right" and "Left" candidates
        detected_hands.sort(key=lambda h: h.landmark[0].x, reverse=True)
        display_hands = detected_hands
        
        cv2.putText(image, "PLACE BOTH HANDS ON KEYS", (100, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if len(detected_hands) == 2 and check_stability(detected_hands):
            current_state = STATE_COUNTDOWN
            timer_start = time.time()
            calibration_data_rh, calibration_data_lh = [], []

        # Simple Draw
        for hl in detected_hands:
            mp_draw.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)

    # --- STATE: COUNTDOWN ---
    elif current_state == STATE_COUNTDOWN:
        # Same X-sort logic for consistency during prep
        detected_hands.sort(key=lambda h: h.landmark[0].x, reverse=True)
        
        elapsed = time.time() - timer_start
        remaining = max(0, int(countdown_duration - elapsed + 1))
        
        cv2.putText(image, f"CALIBRATING IN: {remaining}", (150, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        
        for hl in detected_hands:
            mp_draw.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)
            
        if elapsed >= countdown_duration:
            current_state = STATE_CALIBRATING
            timer_start = time.time()
            calibration_data_rh, calibration_data_lh = [], []

    # --- STATE: CALIBRATING ---
    elif current_state == STATE_CALIBRATING:
        # X-Sort is strictly enforced here to Identify RH/LH
        detected_hands.sort(key=lambda h: h.landmark[0].x, reverse=True)
        
        elapsed = time.time() - timer_start
        prog = min(elapsed / calibration_duration, 1.0)
        
        if len(detected_hands) >= 2:
            # Capture Distances
            d_rh = detected_hands[0].landmark[0].y - detected_hands[0].landmark[9].y
            d_lh = detected_hands[1].landmark[0].y - detected_hands[1].landmark[9].y
            calibration_data_rh.append(d_rh)
            calibration_data_lh.append(d_lh)
            
            # INITIALIZE TRACKING CENTROIDS
            # Lock the positions of RH and LH at the end of calibration
            tracked_hands["RH"] = np.array([detected_hands[0].landmark[0].x, detected_hands[0].landmark[0].y])
            tracked_hands["LH"] = np.array([detected_hands[1].landmark[0].x, detected_hands[1].landmark[0].y])
        
        # Draw UI
        bar_width = int(prog * 400)
        cv2.rectangle(image, (100, 300), (500, 330), (50, 50, 50), -1) 
        cv2.rectangle(image, (100, 300), (100 + bar_width, 330), (0, 255, 0), -1)
        cv2.putText(image, "HOLD PERECT FORM...", (100, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for hl in detected_hands:
            mp_draw.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)

        if elapsed >= calibration_duration:
             # Finalize Baselines
            if calibration_data_rh:
                baseline_rh["dist"] = sum(calibration_data_rh) / len(calibration_data_rh)
            if calibration_data_lh:
                baseline_lh["dist"] = sum(calibration_data_lh) / len(calibration_data_lh)
            
            # Assume ideal angle ~150 (curved) or measure it if we wanted
            baseline_rh["angle"] = 150
            baseline_lh["angle"] = 150
            
            current_state = STATE_ACTIVE
            print("Calibration Complete. Tracking Locked.")

    # --- STATE: ACTIVE ---
    elif current_state == STATE_ACTIVE:
        # PERSISTENT TRACKING LOGIC
        # Match detected hands to stored tracked_hands["RH"] and tracked_hands["LH"]
        
        # Helper to get wrist pos
        def get_wrist(hl): return np.array([hl.landmark[0].x, hl.landmark[0].y])
        
        final_assignments = {} # "RH": hl, "LH": hl
        
        if detected_hands:
            if tracked_hands["RH"] is not None and tracked_hands["LH"] is not None:
                # We have 2 known IDs. 
                # If we detect 2 hands, simple bijective matching.
                # If 1 hand, find closest.
                
                # Calculate distances matrix
                # Rows: [RH_prev, LH_prev], Cols: [detected hands]
                
                matches = [] # tuples (dist, target_id, detection_idx)
                
                for idx, hl in enumerate(detected_hands):
                    pos = get_wrist(hl)
                    d_rh = np.linalg.norm(pos - tracked_hands["RH"])
                    d_lh = np.linalg.norm(pos - tracked_hands["LH"])
                    matches.append( (d_rh, "RH", idx) )
                    matches.append( (d_lh, "LH", idx) )
                
                # Sort by smallest distance
                matches.sort(key=lambda x: x[0])
                
                assigned_idxs = set()
                assigned_ids = set()
                
                for dist, target_id, det_idx in matches:
                    if target_id not in assigned_ids and det_idx not in assigned_idxs:
                        # Found a match
                        final_assignments[target_id] = detected_hands[det_idx]
                        
                        # Update the tracked position
                        tracked_hands[target_id] = get_wrist(detected_hands[det_idx])
                        
                        assigned_ids.add(target_id)
                        assigned_idxs.add(det_idx)
        else:
            # No hands? Don't clear tracked_hands immediately? 
            # Or assume layout holds. For now, let's keep tracked_hands as last known.
            # But we won't draw anything.
            pass

        # Update History (if hands missing, append last score or 0? better to append 0 or last known?)
        # Let's append current score if visible, else 0? Or 100?
        # User said "when hands taken off, stop registering".
        # So if hand missing, maybe don't update graph or append None?
        # Let's append last value for continuity or 0 for "absent". 
        # Actually user wants to see "score over time". 
        # If absence means "break", let's hold the graph steady or hide it?
        # Let's just update if present.

        for label in ["RH", "LH"]:
            hand_landmarks = final_assignments.get(label)
            
            if hand_landmarks:
                # --- CALCULATE METRICS ---
                
                # 1. Wrist
                current_dist = hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y
                base_dist = baseline_rh["dist"] if label == "RH" else baseline_lh["dist"]
                
                # If wrist is HIGHER (current < base), that's good -> deviation 0.
                # If wrist is LOWER (current > base), deviation > 0.
                wrist_deviation = max(0, current_dist - base_dist)
                # Tolerance ~0.15 seems valid
                wrist_score = calculate_forgiving_score(current_dist, base_dist, 0.15)
                # Correction: if we are better than baseline, score 100
                if current_dist < base_dist: wrist_score = 100

                # 2. Fingers
                total_angle = 0
                count = 0
                for pip_idx in [6, 10, 14, 18]:
                    mcp = [hand_landmarks.landmark[pip_idx-1].x, hand_landmarks.landmark[pip_idx-1].y]
                    pip = [hand_landmarks.landmark[pip_idx].x, hand_landmarks.landmark[pip_idx].y]
                    dip = [hand_landmarks.landmark[pip_idx+1].x, hand_landmarks.landmark[pip_idx+1].y]
                    total_angle += calculate_angle(mcp, pip, dip)
                    count += 1
                avg_angle = total_angle / count
                
                # Ideal ~150. Flat is 180.
                finger_deviation = max(0, avg_angle - 150)
                # Tolerance 35 degrees (150 -> 185)
                finger_score = calculate_forgiving_score(avg_angle, 150, 35)

                # Total
                total_score = (wrist_score + finger_score) // 2
                
                # Update History
                score_history[label].append(total_score)
                
                # Colors
                main_color = get_status_color(total_score)
                wrist_color = get_status_color(wrist_score)
                finger_color = get_status_color(finger_score)
                
                # --- DRAW ---
                mp_draw.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=main_color, thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=main_color, thickness=2)
                )

                # --- UI TEXT ---
                if label == "LH":
                    x_base = 20
                else:
                    x_base = image.shape[1] - 350
                
                cv2.putText(image, f"{label}: {int(total_score)}%", (x_base, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, main_color, score_thickness)
                cv2.putText(image, f"Wrist: {int(wrist_score)}%", (x_base, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, wrist_color, 2)
                cv2.putText(image, f"Fingers: {int(finger_score)}%", (x_base, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, finger_color, 2)
                
                # --- PIP GRAPH ---
                draw_pip_graph(image, score_history[label], side=label)

    cv2.imshow('DreamPlay Continuous Scoring', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
