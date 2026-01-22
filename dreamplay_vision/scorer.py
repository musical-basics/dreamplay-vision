import numpy as np

class HandScorer:
    def __init__(self):
        # We don't strictly need history here if it's stored in UI/Main for graphing, 
        # but the plan suggestion implied managing it or pure math. 
        # The user's prompt said "Manage history internally" in the class.
        from collections import deque
        self.history = {
            "RH": {"total": deque([100]*120, maxlen=120), "wrist": deque([100]*120, maxlen=120), "finger": deque([100]*120, maxlen=120)},
            "LH": {"total": deque([100]*120, maxlen=120), "wrist": deque([100]*120, maxlen=120), "finger": deque([100]*120, maxlen=120)}
        }

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360-angle
        return angle

    def get_pro_score(self, val, baseline, is_wrist=True):
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

    def get_status_color(self, score):
        if score >= 90: return (0, 255, 0)      # Green
        if score >= 80: return (0, 255, 255)    # Yellow
        if score >= 70: return (0, 165, 255)    # Orange
        return (0, 0, 255)                      # Red

    def process(self, hand_landmarks, label, baseline_dist):
        """
        Process a single hand's landmarks and return scores.
        Updates internal history.
        """
        if not hand_landmarks:
            return None

        # 1. Wrist
        c_dist = hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y
        w_score = self.get_pro_score(c_dist, baseline_dist, is_wrist=True)
        
        # 2. Finger
        total_angle = 0
        count = 0
        # Indices for MCP, PIP, DIP
        for pip_idx in [6, 10, 14, 18]:
            mcp = [hand_landmarks.landmark[pip_idx-1].x, hand_landmarks.landmark[pip_idx-1].y]
            pip = [hand_landmarks.landmark[pip_idx].x, hand_landmarks.landmark[pip_idx].y]
            dip = [hand_landmarks.landmark[pip_idx+1].x, hand_landmarks.landmark[pip_idx+1].y]
            total_angle += self.calculate_angle(mcp, pip, dip)
            count += 1
        avg_angle = total_angle / count
        f_score = self.get_pro_score(avg_angle, 140, is_wrist=False) # 140 is approx baseline for straight fingers

        # Total
        total_score = (w_score + f_score) // 2
        
        # Update History
        self.history[label]["total"].append(total_score)
        self.history[label]["wrist"].append(w_score)
        self.history[label]["finger"].append(f_score)
        
        return {
            "total": total_score,
            "wrist": w_score,
            "finger": f_score
        }
