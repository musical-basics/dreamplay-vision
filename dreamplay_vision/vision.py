import cv2
import mediapipe as mp
import subprocess
import re

class CameraSystem:
    def __init__(self, camera_index=0):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.8, 
            max_num_hands=2
        )
        self.active_camera_index = camera_index
        self.cap = cv2.VideoCapture(self.active_camera_index)
        
    def get_camera_names(self):
        devices = []
        try:
            # Mac specific
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
            devices = [{'index': 0, 'name': "Default Camera"}]
        return devices

    def set_camera(self, index):
        if index != self.active_camera_index:
            self.cap.release()
            self.active_camera_index = index
            self.cap = cv2.VideoCapture(self.active_camera_index)
            print(f"Switched to camera index {index}")

    def get_frame(self):
        if not self.cap.isOpened():
            return None, None
            
        success, image = self.cap.read()
        if not success:
            return None, None
            
        # Flip logic removed as per original v0.1 (assuming correct mount)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_rgb

    def detect_hands(self, image_rgb):
        results = self.hands.process(image_rgb)
        detected_hands = []
        if results and results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                detected_hands.append(hl)
        return detected_hands

    def release(self):
        self.cap.release()
