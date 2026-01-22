import cv2
import numpy as np
import mediapipe as mp
from .config import *

class UI:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mouse_click = None

    def handle_click(self, x, y):
        self.mouse_click = (x, y)

    def is_clicked(self, rect, click_point):
        if not click_point: return False
        x, y, w, h = rect
        cx, cy = click_point
        return (x <= cx <= x + w) and (y <= cy <= y + h)

    def consume_click(self):
        # Helper to get and clear click
        click = self.mouse_click
        self.mouse_click = None
        return click

    def draw_button(self, img, rect, text, color=(60, 60, 60)):
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)
        
        # Center text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def draw_menu(self, image):
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
        
        self.draw_button(image, btn_play, "BEGIN PLAYING")
        self.draw_button(image, btn_load, "LOAD PROFILE")
        self.draw_button(image, btn_settings, "SETTINGS")
        self.draw_button(image, btn_exit, "EXIT")

        click = self.consume_click()
        if click:
            if self.is_clicked(btn_play, click): return "PLAY"
            elif self.is_clicked(btn_load, click): return "LOAD"
            elif self.is_clicked(btn_settings, click): return "SETTINGS"
            elif self.is_clicked(btn_exit, click): return "EXIT"
        return None

    def draw_area_graph(self, img, history_data, side="RH"):
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
            
            if len(data) < 2: continue # Need points to draw

            # We need to map enumerate index 't' to graph range (0 to 120 -> 0 to GRAPH_W)
            # data length is up to 120
            
            for t, score in enumerate(data):
                # Using hardcoded 120 from history size
                line_x = x_start + int(t * (GRAPH_W / 120)) 
                bar_h = int((score / 100) * SECTION_H)
                
                # Draw line up from bottom of section
                # Optimization needed? This draws 120 lines per section.
                # Actually, this is what the original code did. 
                # Could be optimized to polydraw but let's stick to working logic.
                
                # We need a function to get color from score, but color logic is simple enough or imported?
                # The original used get_status_color. We can duplicate or pass it in.
                
                # Let's simple duplicate get_status_color logic here for speed or import
                moment_color = (0, 0, 255)
                if score >= 90: moment_color = (0, 255, 0)
                elif score >= 80: moment_color = (0, 255, 255)
                elif score >= 70: moment_color = (0, 165, 255)

                cv2.line(img, (line_x, s_y + SECTION_H), (line_x, s_y + SECTION_H - bar_h), moment_color, 2)

    def draw_skeleton_and_scores(self, image, hand_landmarks, scores, label, alpha=1.0):
        if not hand_landmarks: return

        # Helper for color
        def get_color(s):
            if s >= 90: return (0, 255, 0)
            if s >= 80: return (0, 255, 255)
            if s >= 70: return (0, 165, 255)
            return (0, 0, 255)

        total_score = scores['total']
        color = get_color(total_score)
        
        # Draw Skeleton
        if alpha < 0.99:
            stencil = image.copy()
            self.mp_draw.draw_landmarks(stencil, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                   self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                                   self.mp_draw.DrawingSpec(color=color, thickness=2))
            cv2.addWeighted(stencil, alpha, image, 1 - alpha, 0, image)
        else:
            self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                   self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                                   self.mp_draw.DrawingSpec(color=color, thickness=2))

        # UI Text with Alpha
        x_base = 20 if label == "LH" else image.shape[1] - 350
        
        def apply_alpha(clr, a):
            return (int(clr[0]*a), int(clr[1]*a), int(clr[2]*a))
        
        txt_color = apply_alpha(color, alpha)
        
        cv2.putText(image, f"{label}: {int(total_score)}%", (x_base, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, SCORE_FONT_SCALE, txt_color, SCORE_THICKNESS)
        
        w_score = scores['wrist']
        f_score = scores['finger']
        
        cv2.putText(image, f"Wrist: {int(w_score)}%", (x_base, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, apply_alpha(get_color(w_score), alpha), 2)
        cv2.putText(image, f"Fingers: {int(f_score)}%", (x_base, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, apply_alpha(get_color(f_score), alpha), 2)

    def draw_calibration(self, image, elapsed, duration):
        prog = min(elapsed / duration, 1.0)
        bar_w = int(prog * 400)
        cv2.rectangle(image, (100, 300), (500, 330), (50, 50, 50), -1) 
        cv2.rectangle(image, (100, 300), (100 + bar_w, 330), (0, 255, 0), -1)
        cv2.putText(image, "HOLD PERFECT FORM...", (100, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def draw_countdown(self, image, rem):
        cv2.putText(image, f"CALIBRATING IN: {rem}", (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

    def draw_waiting_prompt(self, image):
        msg = "PLACE HANDS TO CALIBRATE"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_TRIPLEX, 1.2, 2)
        tx = (image.shape[1] - tw) // 2
        ty = 150
        
        overlay = image.copy()
        cv2.rectangle(overlay, (tx - 20, ty - th - 20), (tx + tw + 20, ty + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        cv2.putText(image, msg, (tx, ty), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 255), 2)
