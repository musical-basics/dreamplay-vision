import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
from pathlib import Path
from collections import deque
from datetime import datetime
import subprocess
import re

# --- CAMERA HELPERS ---
def get_camera_names():
    devices = []
    try:
        result = subprocess.run(['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''], 
                                stderr=subprocess.PIPE, text=True)
        output = result.stderr
        lines = output.split('\n')
        parsing_video = False
        for line in lines:
            if "video devices:" in line:
                parsing_video = True
                continue
            if "audio devices:" in line:
                break
            if parsing_video:
                match = re.search(r'\[(\d+)\]\s+(.+)', line)
                if match:
                    idx = int(match.group(1))
                    name = match.group(2).strip()
                    devices.append({'index': idx, 'name': name})
    except Exception as e:
        print(f"Error fetching camera names: {e}")
    
    if not devices:
        devices = [{'index': 0, 'name': "Camera 0"}, {'index': 1, 'name': "Camera 1"}]
    return devices

camera_list = get_camera_names()
selected_cam_list_idx = 0 # Index in our camera_list, not the hardware ID
dropdown_open = False

# --- CONSTANTS & GLOBALS ---
# ... (Continuing with globals update to ensure state consistency)
font_scale = 1.0
thickness = 2
score_font_scale = 1.5
score_thickness = 4

# --- PATHS ---
DOCS_PATH = Path.home() / "Documents" / "DreamPlay"
DOCS_PATH.mkdir(parents=True, exist_ok=True)

# ... (Keeping Scoring Helpers and Graph Helper same) ... 
# We need to jump to the state variables to update `active_camera_index` usage.
# Instead of `active_camera_index` being the hardware id, let's track `selected_cam_list_idx` pointing to `camera_list`.
# But for continuity with existing main/globals, let's keep `active_camera_index` as the HARDWARE ID,
# and add `current_list_selection` for UI.

active_camera_index = 0 
if camera_list:
    active_camera_index = camera_list[0]['index']


# --- SCORING HELPERS ---
def get_pro_score(val, baseline, is_wrist=True):
    """
    Advanced Scoring using Exponential/Sigmoid decay.
    """
    if is_wrist:
        # WRIST: High sensitivity. Exponential penalty.
        deviation = val - baseline
        if deviation <= 0: return 100
        # exp(-20 * 0.05) ~ 0.36 -> 36 score.
        score = int(100 * np.exp(-20 * deviation))
        return max(0, score)
    else:
        # FINGERS: Forgiving Sigmoid.
        # Ideally < 155 is perfect.
        if val < 155: return 100
        
        # Normalize: 140->0, 180->1 (range 40)
        norm_val = (val - 140) / 40.0
        # Sigmoid centered roughly at 170deg (0.75)
        # Score = 100 / (1 + exp(15 * (x - 0.75)))
        score = int(100 / (1 + np.exp(15 * (norm_val - 0.75))))
        return max(0, score)

def get_status_color(score):
    if score >= 90: return (0, 255, 0)      # Green
    if score >= 80: return (0, 255, 255)    # Yellow
    if score >= 70: return (0, 165, 255)    # Orange
    return (0, 0, 255)                      # Red

# --- GRAPH HELPER ---
# History for 120 frames
graph_history = {
    "RH": {"total": deque([100]*120, maxlen=120), "wrist": deque([100]*120, maxlen=120), "finger": deque([100]*120, maxlen=120)},
    "LH": {"total": deque([100]*120, maxlen=120), "wrist": deque([100]*120, maxlen=120), "finger": deque([100]*120, maxlen=120)}
}

GRAPH_W, GRAPH_H = 400, 200 
SECTION_H = GRAPH_H // 3    

def draw_area_graph(img, history_data, side="RH"):
    margin = 30
    x_start = img.shape[1] - GRAPH_W - margin if side == "RH" else margin
    y_start = img.shape[0] - GRAPH_H - margin
    
    labels = [("TOTAL", (255, 255, 255)), ("WRIST", (255, 200, 0)), ("FINGER", (0, 255, 100))]
    keys = ["total", "wrist", "finger"]
    
    for i, (label_text, label_color) in enumerate(labels):
        s_y = y_start + (i * SECTION_H)
        
        # Background
        cv2.rectangle(img, (x_start, s_y), (x_start + GRAPH_W, s_y + SECTION_H), (30, 30, 30), -1)
        # Separator
        cv2.line(img, (x_start, s_y + SECTION_H), (x_start + GRAPH_W, s_y + SECTION_H), (100, 100, 100), 1)

        # Legend
        text_x = x_start - 80 if side=="RH" else x_start + GRAPH_W + 10
        cv2.putText(img, label_text, (text_x, s_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
        
        # Area Fill
        key = keys[i]
        data = history_data[key]
        
        for t, score in enumerate(data):
            moment_color = get_status_color(score)
            line_x = x_start + int(t * (GRAPH_W / 120))
            bar_h = int((score / 100) * SECTION_H)
            
            # Draw line up from bottom of section
            cv2.line(img, (line_x, s_y + SECTION_H), (line_x, s_y + SECTION_H - bar_h), moment_color, 2)

# --- TRACKING HELPERS ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

# --- STATE CONSTANTS ---
STATE_MENU = "MENU"
STATE_LOAD_PROFILE = "LOAD_PROFILE"
STATE_WAITING = "WAITING"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_CALIBRATING = "CALIBRATING"
STATE_SAVE_PROMPT = "SAVE_PROMPT"
STATE_SETTINGS = "SETTINGS"
STATE_ACTIVE = "ACTIVE"

current_state = STATE_MENU

# --- GLOBAL DATA ---
baseline_rh = {"dist": 0.05}
baseline_lh = {"dist": 0.03}
tracked_hands = {"RH": None, "LH": None}
prev_hand_pos = None

calibration_data_rh = []
calibration_data_lh = []
timer_start = 0
calibration_duration = 3.0
countdown_duration = 3.0
active_camera_index = 0

# --- USER INPUT / MOUSE ---
mouse_click = None # (x, y) coordinates of last click, reset after processing

def mouse_callback(event, x, y, flags, param):
    global mouse_click
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click = (x, y)

# --- UI WIDGETS ---
def is_clicked(rect, click_point):
    if not click_point: return False
    x, y, w, h = rect
    cx, cy = click_point
    return (x <= cx <= x + w) and (y <= cy <= y + h)

def draw_button(img, rect, text, color=(60, 60, 60)):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)
    
    # Center text
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# --- PROFILE MANAGER ---
def save_profile_file():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = DOCS_PATH / f"Profile_{timestamp}.json"
    data = {
        "rh_baseline": baseline_rh,
        "lh_baseline": baseline_lh,
        "timestamp": timestamp
    }
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Saved: {filename}")
    return filename.name

def load_profiles():
    files = sorted(DOCS_PATH.glob("*.json"), key=os.path.getmtime, reverse=True)
    return files

def load_profile_data(filepath):
    global baseline_rh, baseline_lh
    with open(filepath, 'r') as f:
        data = json.load(f)
        baseline_rh = data.get("rh_baseline", {"dist": 0.05})
        baseline_lh = data.get("lh_baseline", {"dist": 0.03})
    print(f"Loaded: {filepath}")

# --- MAIN LOOP ---
def main():
    global current_state, timer_start, mouse_click, prev_hand_pos
    global calibration_data_rh, calibration_data_lh, tracked_hands
    global baseline_rh, baseline_lh
    global baseline_rh, baseline_lh
    global active_camera_index, dropdown_open, camera_list

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.8, max_num_hands=2)
    mp_draw = mp.solutions.drawing_utils

    # Refresh camera list on start
    camera_list = get_camera_names()
    selected_cam_list_idx = 0
    # Align selection with active hardware index
    for i, dev in enumerate(camera_list):
        if dev['index'] == active_camera_index:
            selected_cam_list_idx = i
            break

    cap = cv2.VideoCapture(active_camera_index)
    
    cv2.namedWindow('DreamPlay Vision')
    cv2.setMouseCallback('DreamPlay Vision', mouse_callback)

    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        
        # Flip logic removed as per v0.1 (Assuming correct camera mount)
        # But for menu interaction, mirroring can be confusing if using mouse vs hand.
        # User is using mouse? We assume mouse for menu.

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process Hands only if needed (Optimization)
        results = None
        if current_state not in [STATE_MENU, STATE_LOAD_PROFILE, STATE_SETTINGS]:
            results = hands.process(image_rgb)

        detected_hands = []
        if results and results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                detected_hands.append(hl)

        # --- STATE MACHINE ---

        if current_state == STATE_MENU:
            # Draw Menu Overlay
            # Ideally blur detection background or just dark screen
            image.fill(30) # Dark Grey
            
            cv2.putText(image, "DREAMPLAY AI COACH", (image.shape[1]//2 - 250, 150), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 2)
            
            # Buttons
            cx = image.shape[1] // 2 - 200
            start_y = 250
            btn_w, btn_h = 400, 60
            gap = 80
            
            btn_play = (cx, start_y, btn_w, btn_h)
            btn_load = (cx, start_y + gap, btn_w, btn_h)
            btn_settings = (cx, start_y + gap*2, btn_w, btn_h)
            btn_exit = (cx, start_y + gap*3, btn_w, btn_h)
            
            draw_button(image, btn_play, "BEGIN PLAYING")
            draw_button(image, btn_load, "LOAD PROFILE")
            draw_button(image, btn_settings, "SETTINGS")
            draw_button(image, btn_exit, "EXIT")
            
            if mouse_click:
                if is_clicked(btn_play, mouse_click):
                    current_state = STATE_WAITING
                elif is_clicked(btn_load, mouse_click):
                    current_state = STATE_LOAD_PROFILE
                elif is_clicked(btn_settings, mouse_click):
                    current_state = STATE_SETTINGS
                elif is_clicked(btn_exit, mouse_click):
                    break
                mouse_click = None

        elif current_state == STATE_SETTINGS:
            # Dropdown UI
            cv2.putText(image, "SETTINGS", (50, 80), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(image, "SELECT INPUT DEVICE:", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # Dropdown Box
            box_x, box_y = 100, 210
            box_w, box_h = 400, 50
            main_box = (box_x, box_y, box_w, box_h)
            
            # Display current selection safely
            current_name = "Unknown"
            if 0 <= selected_cam_list_idx < len(camera_list):
                current_name = camera_list[selected_cam_list_idx]['name']
            
            cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 60, 60), -1)
            cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (200, 200, 200), 2)
            cv2.putText(image, current_name, (box_x + 15, box_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "v", (box_x + box_w - 30, box_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Back Button
            btn_back = (50, image.shape[0]-80, 100, 50)
            draw_button(image, btn_back, "BACK")

            # Handle Clicks
            if mouse_click:
                if is_clicked(btn_back, mouse_click):
                    current_state = STATE_MENU
                    dropdown_open = False
                elif is_clicked(main_box, mouse_click):
                    dropdown_open = not dropdown_open
                
                # Check dropdown items if open
                if dropdown_open:
                    for i, dev in enumerate(camera_list):
                        opt_y = box_y + box_h + (i * 45)
                        opt_rect = (box_x, opt_y, box_w, 45)
                        if is_clicked(opt_rect, mouse_click):
                            # Update indices
                            selected_cam_list_idx = i
                            active_camera_index = dev['index']
                            
                            cap.release()
                            cap = cv2.VideoCapture(active_camera_index)
                            dropdown_open = False
                            print(f"Switched to camera: {dev['name']} (Index {active_camera_index})")
                
                mouse_click = None

            # Draw Options if Open (On top of everything)
            if dropdown_open:
                for i, dev in enumerate(camera_list):
                    opt_y = box_y + box_h + (i * 45)
                    color = (80, 80, 80)
                    if i == selected_cam_list_idx: color = (0, 100, 0)
                    
                    cv2.rectangle(image, (box_x, opt_y), (box_x + box_w, opt_y + 45), color, -1)
                    cv2.rectangle(image, (box_x, opt_y), (box_x + box_w, opt_y + 45), (150, 150, 150), 1)
                    cv2.putText(image, dev['name'], (box_x + 15, opt_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        elif current_state == STATE_LOAD_PROFILE:
            image.fill(30)
            cv2.putText(image, "SELECT PROFILE", (image.shape[1]//2 - 150, 80), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
            
            profiles = load_profiles()
            # Show top 5
            cx = image.shape[1] // 2 - 200
            start_y = 150
            btn_w, btn_h = 400, 50
            
            msg = "No profiles found."
            if not profiles:
                cv2.putText(image, msg, (cx, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            
            for i, p in enumerate(profiles[:5]):
                rect = (cx, start_y + i*60, btn_w, btn_h)
                draw_button(image, rect, p.name)
                
                if mouse_click and is_clicked(rect, mouse_click):
                    load_profile_data(p)
                    # Skip calibration, go straight to active?
                    # Needs stability check or assume active?
                    # Let's go to waiting (active check) but with data loaded.
                    # Or straight to Active if we trust it.
                    # Let's go to WAITING but flag that we have data (maybe skip calib trigger?)
                    # User expects "Load" -> "Ready".
                    # Let's go to ACTIVE, but user needs to put hands in place.
                    # Actually, better to go to WAITING, but display "PROFILE LOADED: <name>"
                    # And maybe a "Start" button?
                    # Or just auto-start tracking.
                    # Let's just update baselines and go to ACTIVE.
                    current_state = STATE_ACTIVE
                    print("Profile loaded, starting...")
                    mouse_click = None
                    break
            
            # Back Button
            btn_back = (50, 50, 100, 50)
            draw_button(image, btn_back, "BACK")
            if mouse_click and is_clicked(btn_back, mouse_click):
                current_state = STATE_MENU
                mouse_click = None

        elif current_state == STATE_WAITING:
            # Sort X
            detected_hands.sort(key=lambda h: h.landmark[0].x, reverse=True)
            
            # Prompt Overlay
            msg = "PLACE HANDS TO CALIBRATE"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_TRIPLEX, 1.2, 2)
            tx = (image.shape[1] - tw) // 2
            ty = 150
            
            # Semi-transparent BG
            overlay = image.copy()
            cv2.rectangle(overlay, (tx - 20, ty - th - 20), (tx + tw + 20, ty + 20), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            
            cv2.putText(image, msg, (tx, ty), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 255), 2)

            # Draw Hands
            for hl in detected_hands: mp_draw.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)

            # Check Stability
            def check_stability_local(hands_list):
                global prev_hand_pos
                if len(hands_list) < 2: return False
                curr = np.array([hands_list[0].landmark[0].x, hands_list[0].landmark[0].y])
                if prev_hand_pos is None:
                    prev_hand_pos = curr
                    return False
                dist = np.linalg.norm(curr - prev_hand_pos)
                prev_hand_pos = curr
                return dist < 0.02
            
            if len(detected_hands) == 2 and check_stability_local(detected_hands):
                current_state = STATE_COUNTDOWN
                timer_start = time.time()
                calibration_data_rh, calibration_data_lh = [], []
            
            # Back to menu via 'm'
            cv2.putText(image, "[M] Menu", (50, image.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        elif current_state == STATE_COUNTDOWN:
            detected_hands.sort(key=lambda h: h.landmark[0].x, reverse=True)
            elapsed = time.time() - timer_start
            rem = max(0, int(countdown_duration - elapsed + 1))
            
            cv2.putText(image, f"CALIBRATING IN: {rem}", (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
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
                # Capture
                d_rh = detected_hands[0].landmark[0].y - detected_hands[0].landmark[9].y
                d_lh = detected_hands[1].landmark[0].y - detected_hands[1].landmark[9].y
                calibration_data_rh.append(d_rh)
                calibration_data_lh.append(d_lh)
                
                # Snapshot tracked positions
                tracked_hands["RH"] = np.array([detected_hands[0].landmark[0].x, detected_hands[0].landmark[0].y])
                tracked_hands["LH"] = np.array([detected_hands[1].landmark[0].x, detected_hands[1].landmark[0].y])

            # Progress Bar
            bar_w = int(prog * 400)
            cv2.rectangle(image, (100, 300), (500, 330), (50, 50, 50), -1) 
            cv2.rectangle(image, (100, 300), (100 + bar_w, 330), (0, 255, 0), -1)
            cv2.putText(image, "HOLD PERFECT FORM...", (100, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for hl in detected_hands: mp_draw.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)
            
            if elapsed >= calibration_duration:
                # Finalize
                if calibration_data_rh: baseline_rh["dist"] = sum(calibration_data_rh) / len(calibration_data_rh)
                if calibration_data_lh: baseline_lh["dist"] = sum(calibration_data_lh) / len(calibration_data_lh)
                
                # Auto-Save and Continue
                fname = save_profile_file()
                print(f"Auto-saved profile: {fname}")
                current_state = STATE_ACTIVE

        # elif current_state == STATE_SAVE_PROMPT:
            # Removed for seamless UX as per user request
            # pass

        elif current_state == STATE_ACTIVE:
            # Persistent Tracking Logic
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
            
            # --- SCORING & DRAWING ---
            for label in ["RH", "LH"]:
                hand_landmarks = final_assignments.get(label)
                if hand_landmarks:
                    # 1. Wrist
                    c_dist = hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y
                    b_dist = baseline_rh["dist"] if label == "RH" else baseline_lh["dist"]
                    w_score = get_pro_score(c_dist, b_dist, is_wrist=True)
                    
                    # 2. Finger
                    total_angle = 0
                    count = 0
                    for pip_idx in [6, 10, 14, 18]:
                        mcp = [hand_landmarks.landmark[pip_idx-1].x, hand_landmarks.landmark[pip_idx-1].y]
                        pip = [hand_landmarks.landmark[pip_idx].x, hand_landmarks.landmark[pip_idx].y]
                        dip = [hand_landmarks.landmark[pip_idx+1].x, hand_landmarks.landmark[pip_idx+1].y]
                        total_angle += calculate_angle(mcp, pip, dip)
                        count += 1
                    avg_angle = total_angle / count
                    f_score = get_pro_score(avg_angle, 140, is_wrist=False)

                    # Total
                    total_score = (w_score + f_score) // 2
                    
                    # History
                    graph_history[label]["total"].append(total_score)
                    graph_history[label]["wrist"].append(w_score)
                    graph_history[label]["finger"].append(f_score)
                    
                    # Draw
                    color = get_status_color(total_score)
                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                           mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                                           mp_draw.DrawingSpec(color=color, thickness=2))
                    
                    # UI Text
                    x_base = 20 if label == "LH" else image.shape[1] - 350
                    # Main
                    cv2.putText(image, f"{label}: {int(total_score)}%", (x_base, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, color, score_thickness)
                    # Sub-scores
                    cv2.putText(image, f"Wrist: {int(w_score)}%", (x_base, 130), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, get_status_color(w_score), 2)
                    cv2.putText(image, f"Fingers: {int(f_score)}%", (x_base, 170), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, get_status_color(f_score), 2)
                    
                # Always draw graphs
                if len(graph_history[label]["total"]) > 1:
                    draw_area_graph(image, graph_history[label], side=label)

            # Manual Reset ('C') or Menu ('M')
            cv2.putText(image, "[M] Menu", (50, image.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        # Global Key Handling
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            if current_state == STATE_SETTINGS:
                current_state = STATE_MENU
            elif current_state == STATE_MENU:
                break
            else:
                formatted_state = STATE_MENU
                current_state = STATE_MENU
        elif key == ord('m'):
            current_state = STATE_MENU
        elif key == ord('c') and current_state == STATE_ACTIVE:
            current_state = STATE_COUNTDOWN
            timer_start = time.time()
            calibration_data_rh, calibration_data_lh = [], []
            print("Manual Re-Calibration")
        
        # CAMERA SWITCHING (Only in Settings)
        elif current_state == STATE_SETTINGS:
            if key == ord('=') or key == ord('+'):
                active_camera_index += 1
                cap.release()
                cap = cv2.VideoCapture(active_camera_index)
            elif key == ord('-') or key == ord('_'):
                if active_camera_index > 0:
                    active_camera_index -= 1
                    cap.release()
                    cap = cv2.VideoCapture(active_camera_index)

        cv2.imshow('DreamPlay Vision', image)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
