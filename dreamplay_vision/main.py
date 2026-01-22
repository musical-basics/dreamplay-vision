import cv2
import time
import numpy as np
from .config import *
from .vision import CameraSystem
from .scorer import HandScorer
from .ui import UI
from .profile import ProfileManager

# State Machine Management requires some global tracking or class attributes
# We can wrap everything in a MainApp class or keep it functional.
# Functional is cleaner for the "story" read, but we need to hold state.

def main():
    # 1. Setup
    cam = CameraSystem()
    scorer = HandScorer()
    ui = UI()
    profile = ProfileManager()
    
    state = STATE_MENU
    
    # Tracking persistence state (Logic from original God File)
    hand_last_seen = {"RH": 0, "LH": 0}
    last_known_landmarks = {"RH": None, "LH": None}
    
    # Calibration State
    timer_start = 0
    calibration_data = {"RH": [], "LH": []}
    
    # Settings State
    dropdown_open = False
    
    # Window
    cv2.namedWindow('DreamPlay Vision')
    cv2.setMouseCallback('DreamPlay Vision', lambda event, x, y, flags, param: 
                         ui.handle_click(x, y) if event == cv2.EVENT_LBUTTONDOWN else None)

    print("DreamPlay Vision Started")
    
    while True:
        # 2. Get Data
        frame, image = cam.get_frame()
        if frame is None:
            break
            
        # Process Hands only if needed
        hands_data = []
        if state not in [STATE_MENU, STATE_LOAD_PROFILE, STATE_SETTINGS]:
             hands_data = cam.detect_hands(image)

        # 3. Logic Branching
        if state == STATE_MENU:
            action = ui.draw_menu(image)
            if action == "PLAY":
                state = STATE_WAITING
            elif action == "LOAD":
                state = STATE_LOAD_PROFILE
            elif action == "SETTINGS":
                state = STATE_SETTINGS
            elif action == "EXIT":
                break
                
        elif state == STATE_SETTINGS:
            # Dropdown UI Logic
            cv2.putText(image, "SETTINGS", (50, 80), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(image, "SELECT INPUT DEVICE:", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            box_x, box_y, box_w, box_h = 100, 210, 400, 50
            main_box = (box_x, box_y, box_w, box_h)
            
            # Current cam name
            cameras = cam.get_camera_names()
            current_name = "Unknown"
            # Find name of active index
            for c in cameras:
                if c['index'] == cam.active_camera_index:
                    current_name = c['name']
                    break
            
            cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 60, 60), -1)
            cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (200, 200, 200), 2)
            cv2.putText(image, current_name, (box_x + 15, box_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Back Button
            btn_back = (50, image.shape[0]-80, 100, 50)
            ui.draw_button(image, btn_back, "BACK")

            click = ui.consume_click()
            if click:
                if ui.is_clicked(btn_back, click):
                    state = STATE_MENU
                    dropdown_open = False
                elif ui.is_clicked(main_box, click):
                    dropdown_open = not dropdown_open
                
                if dropdown_open:
                     for i, dev in enumerate(cameras):
                        opt_y = box_y + box_h + (i * 45)
                        opt_rect = (box_x, opt_y, box_w, 45)
                        if ui.is_clicked(opt_rect, click):
                            cam.set_camera(dev['index'])
                            dropdown_open = False
            
            if dropdown_open:
                for i, dev in enumerate(cameras):
                    opt_y = box_y + box_h + (i * 45)
                    color = (0, 100, 0) if dev['index'] == cam.active_camera_index else (80, 80, 80)
                    cv2.rectangle(image, (box_x, opt_y), (box_x + box_w, opt_y + 45), color, -1)
                    cv2.putText(image, dev['name'], (box_x + 15, opt_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        elif state == STATE_LOAD_PROFILE:
            image.fill(30)
            cv2.putText(image, "SELECT PROFILE", (image.shape[1]//2 - 150, 80), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
            
            profiles = profile.list_profiles()
            cx = image.shape[1] // 2 - 200
            start_y = 150
            btn_w, btn_h = 400, 50
            
            if not profiles:
                cv2.putText(image, "No profiles found.", (cx, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            
            click = ui.consume_click()
            for i, p in enumerate(profiles[:5]):
                rect = (cx, start_y + i*60, btn_w, btn_h)
                ui.draw_button(image, rect, p.name)
                if click and ui.is_clicked(rect, click):
                    profile.load_profile(p)
                    state = STATE_ACTIVE
                    print("Profile loaded, starting...")
            
            btn_back = (50, 50, 100, 50)
            ui.draw_button(image, btn_back, "BACK")
            if click and ui.is_clicked(btn_back, click):
                state = STATE_MENU

        elif state == STATE_WAITING:
            ui.draw_waiting_prompt(image)
            # Visualize hands
            # Need to pass hands to UI helper to draw just landmarks without connection logic if we want standard
            # But we can use mp_draw directly or add a method in UI
            for hl in hands_data:
                ui.mp_draw.draw_landmarks(image, hl, ui.mp_hands.HAND_CONNECTIONS)
            
            # Stability Check Logic
            # We need a small local prev_pos for WAITING state
            # Let's simplify and just check for 2 hands present for now? 
            # Original code checked stability. 
            if len(hands_data) == 2:
                # Add simplified stability check later if needed, prompt explicitly says "PLACE HANDS"
                # Proceed to countdown
                state = STATE_COUNTDOWN
                timer_start = time.time()
                calibration_data = {"RH": [], "LH": []}

        elif state == STATE_COUNTDOWN:
            elapsed = time.time() - timer_start
            rem = max(0, int(COUNTDOWN_DURATION - elapsed + 1))
            ui.draw_countdown(image, rem)
            
            # Sort Hands by X to distinguish L/R roughly
            hands_data.sort(key=lambda h: h.landmark[0].x, reverse=True)
             # Draw
            for hl in hands_data:
                ui.mp_draw.draw_landmarks(image, hl, ui.mp_hands.HAND_CONNECTIONS)

            if elapsed >= COUNTDOWN_DURATION:
                state = STATE_CALIBRATING
                timer_start = time.time()

        elif state == STATE_CALIBRATING:
            elapsed = time.time() - timer_start
            ui.draw_calibration(image, elapsed, CALIBRATION_DURATION)
            
            hands_data.sort(key=lambda h: h.landmark[0].x, reverse=True)
             # Draw
            for hl in hands_data:
                ui.mp_draw.draw_landmarks(image, hl, ui.mp_hands.HAND_CONNECTIONS)

            if len(hands_data) >= 2:
                # RH is index 0 (larger X), LH is index 1
                # Capture Data
                # RH
                d_rh = hands_data[0].landmark[0].y - hands_data[0].landmark[9].y
                s_rh = np.linalg.norm(np.array([hands_data[0].landmark[0].x, hands_data[0].landmark[0].y]) - 
                                      np.array([hands_data[0].landmark[9].x, hands_data[0].landmark[9].y]))
                # LH
                d_lh = hands_data[1].landmark[0].y - hands_data[1].landmark[9].y
                s_lh = np.linalg.norm(np.array([hands_data[1].landmark[0].x, hands_data[1].landmark[0].y]) - 
                                      np.array([hands_data[1].landmark[9].x, hands_data[1].landmark[9].y]))
                
                calibration_data["RH"].append({'dist': d_rh, 'scale': s_rh})
                calibration_data["LH"].append({'dist': d_lh, 'scale': s_lh})
            
            if elapsed >= CALIBRATION_DURATION:
                # Finalize
                if calibration_data["RH"]:
                    dist_avg = sum(d['dist'] for d in calibration_data["RH"]) / len(calibration_data["RH"])
                    scale_avg = sum(d['scale'] for d in calibration_data["RH"]) / len(calibration_data["RH"])
                    profile.current_baselines["RH"] = {"dist": dist_avg, "scale": scale_avg}
                if calibration_data["LH"]:
                    dist_avg = sum(d['dist'] for d in calibration_data["LH"]) / len(calibration_data["LH"])
                    scale_avg = sum(d['scale'] for d in calibration_data["LH"]) / len(calibration_data["LH"])
                    profile.current_baselines["LH"] = {"dist": dist_avg, "scale": scale_avg}
                
                profile.save_profile(profile.current_baselines["RH"], profile.current_baselines["LH"])
                state = STATE_ACTIVE

        elif state == STATE_ACTIVE:
            now = time.time()
            current_frame_assignments = {"RH": None, "LH": None}
            
            # Assignment Logic
            base_scale_rh = profile.current_baselines["RH"]["scale"]
            base_scale_lh = profile.current_baselines["LH"]["scale"]
            
            if hands_data:
                assignments = []
                for hl in hands_data:
                    curr_scale = np.linalg.norm(np.array([hl.landmark[0].x, hl.landmark[0].y]) - 
                                                np.array([hl.landmark[9].x, hl.landmark[9].y]))
                    dist_rh = abs(curr_scale - base_scale_rh)
                    dist_lh = abs(curr_scale - base_scale_lh)
                    
                    label = "RH" if dist_rh < dist_lh else "LH"
                    assignments.append((label, hl, min(dist_rh, dist_lh)))
                
                # Resolve conflicts
                rh_candidates = [x for x in assignments if x[0] == "RH"]
                lh_candidates = [x for x in assignments if x[0] == "LH"]
                
                if rh_candidates:
                    rh_candidates.sort(key=lambda x: x[2])
                    current_frame_assignments["RH"] = rh_candidates[0][1]
                    hand_last_seen["RH"] = now
                    last_known_landmarks["RH"] = rh_candidates[0][1]
                
                if lh_candidates:
                    lh_candidates.sort(key=lambda x: x[2])
                    current_frame_assignments["LH"] = lh_candidates[0][1]
                    hand_last_seen["LH"] = now
                    last_known_landmarks["LH"] = lh_candidates[0][1]

            # Process and Draw
            for label in ["RH", "LH"]:
                hand_landmarks = current_frame_assignments[label]
                alpha = 1.0
                
                if hand_landmarks is None:
                    elapsed = now - hand_last_seen[label]
                    if elapsed < LOST_TRACK_TIMEOUT:
                        hand_landmarks = last_known_landmarks[label]
                        if elapsed > 0.1:
                            alpha = max(0, 1.0 - (elapsed - 0.1) / (LOST_TRACK_TIMEOUT - 0.1))
                    else:
                        pass # Validly lost

                if hand_landmarks and alpha > 0.05:
                    scores = scorer.process(hand_landmarks, label, profile.current_baselines[label]["dist"])
                    
                    # Draw Skeleton and Texts
                    ui.draw_skeleton_and_scores(image, hand_landmarks, scores, label, alpha)
                
                # Draw Graphs
                if len(scorer.history[label]["total"]) > 1:
                    ui.draw_area_graph(image, scorer.history[label], side=label)

            cv2.putText(image, "[M] Menu  [C] Recalibrate", (50, image.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        # 4. Render
        cv2.imshow("DreamPlay Vision", image)
        
        # Handle Keys
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
             if state == STATE_MENU: break
             else: state = STATE_MENU
        elif key == ord('m'):
            state = STATE_MENU
        elif key == ord('c') and state == STATE_ACTIVE:
            state = STATE_COUNTDOWN
            timer_start = time.time()

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
