import cv2
import time
import argparse
import sys
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Import your existing modules
from dreamplay_vision.scorer import HandScorer
from dreamplay_vision.ui import UI
from dreamplay_vision.profile import ProfileManager
from dreamplay_vision.vision import CameraSystem # We'll reuse the detection logic mostly
import mediapipe as mp
import numpy as np

def process_video(input_path, output_path, profile_name=None):
    # 1. Setup Resources
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        min_detection_confidence=0.5, # Slightly lower for fast moving video
        min_tracking_confidence=0.5, 
        max_num_hands=2
    )
    
    scorer = HandScorer()
    ui = UI()
    profile_mgr = ProfileManager()
    
    # 2. Load Calibration Profile
    # Essential: We need to know the user's hand size/baseline without running a calibration phase
    if profile_name:
        profile_path = next((p for p in profile_mgr.list_profiles() if p.name == profile_name), None)
        if profile_path:
            profile_mgr.load_profile(profile_path)
            baselines = profile_mgr.current_baselines
        else:
            print(f"Warning: Profile '{profile_name}' not found. Using defaults.")
            baselines = profile_mgr.current_baselines
    else:
        # Load most recent
        profiles = profile_mgr.list_profiles()
        if profiles:
            print(f"Loading most recent profile: {profiles[0].name}")
            profile_mgr.load_profile(profiles[0])
            baselines = profile_mgr.current_baselines
        else:
            print("No profiles found. Using default generic baselines.")
            baselines = profile_mgr.current_baselines

    # 3. Open Video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Video Info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing: {width}x{height} @ {fps}fps ({total_frames} frames)")

    # 4. Setup Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1' for mac sometimes
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Tracking State (Persisted across frames)
    # We reuse the logic from main.py's STATE_ACTIVE
    hand_last_seen = {"RH": 0, "LH": 0}
    last_known_landmarks = {"RH": None, "LH": None}
    
    # Progress bar
    pbar = tqdm(total=total_frames, unit="frame") if tqdm else None

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_idx += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # A. Detect
        results = hands_detector.process(image_rgb)
        detected_hands = []
        if results.multi_hand_landmarks:
            detected_hands = list(results.multi_hand_landmarks)

        # B. Assign Hands (RH vs LH) using Profile Baselines
        current_frame_assignments = {"RH": None, "LH": None}
        base_scale_rh = baselines["RH"]["scale"]
        base_scale_lh = baselines["LH"]["scale"]

        if detected_hands:
            assignments = []
            for hl in detected_hands:
                # Calculate scale of this hand
                curr_scale = np.linalg.norm(
                    np.array([hl.landmark[0].x, hl.landmark[0].y]) - 
                    np.array([hl.landmark[9].x, hl.landmark[9].y])
                )
                dist_rh = abs(curr_scale - base_scale_rh)
                dist_lh = abs(curr_scale - base_scale_lh)
                
                label = "RH" if dist_rh < dist_lh else "LH"
                assignments.append((label, hl, min(dist_rh, dist_lh)))
            
            # Simple conflict resolution
            rh_candidates = [x for x in assignments if x[0] == "RH"]
            lh_candidates = [x for x in assignments if x[0] == "LH"]
            
            if rh_candidates:
                rh_candidates.sort(key=lambda x: x[2])
                current_frame_assignments["RH"] = rh_candidates[0][1]
                # Update persistence
                last_known_landmarks["RH"] = rh_candidates[0][1]
                hand_last_seen["RH"] = frame_idx # Use frame count instead of time for video
            
            if lh_candidates:
                lh_candidates.sort(key=lambda x: x[2])
                current_frame_assignments["LH"] = lh_candidates[0][1]
                last_known_landmarks["LH"] = lh_candidates[0][1]
                hand_last_seen["LH"] = frame_idx

        # C. Score & Draw
        # (We bypass the ghosting alpha logic for cleaner video, or implement it using frame deltas)
        for label in ["RH", "LH"]:
            hand_landmarks = current_frame_assignments[label]
            
            # Simple "Hold" logic: if lost for < 15 frames, show last known
            alpha = 1.0
            if hand_landmarks is None and last_known_landmarks[label] is not None:
                frames_since = frame_idx - hand_last_seen[label]
                if frames_since < 15: # ~0.5 seconds at 30fps
                    hand_landmarks = last_known_landmarks[label]
                    alpha = 0.5 # Dim ghost
            
            if hand_landmarks:
                # Assuming scorer.process returns something compatible with ui.draw...
                # Note: Scorer uses calibration 'dist' which we have in baselines
                scores = scorer.process(hand_landmarks, label, baselines[label]["dist"])
                ui.draw_skeleton_and_scores(frame, hand_landmarks, scores, label, alpha)
            
            # Draw Graphs
            if len(scorer.history[label]["total"]) > 1:
                ui.draw_area_graph(frame, scorer.history[label], side=label)

        # D. Write Frame
        out.write(frame)
        
        # E. Preview (Optional - slows down processing but looks cool)
        cv2.imshow('Processing Video...', frame)
        if cv2.waitKey(1) & 0xFF == 27: # Stop on ESC
            break
        
        if pbar: pbar.update(1)

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    if pbar: pbar.close()
    print(f"\nDone! Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process piano performance video.")
    parser.add_argument("input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video file")
    parser.add_argument("--profile", type=str, default=None, help="Name of specific profile JSON to use (optional)")
    
    args = parser.parse_args()
    
    # Ensure paths exist
    inp = Path(args.input)
    if not inp.exists():
        print(f"Input file does not exist: {inp}")
        sys.exit(1)
        
    process_video(inp, args.output, args.profile)
