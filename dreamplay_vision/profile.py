import json
import os
from datetime import datetime
from .config import DOCS_PATH

class ProfileManager:
    def __init__(self):
        self.current_baselines = {
            "RH": {"dist": 0.05, "scale": 0.05}, 
            "LH": {"dist": 0.03, "scale": 0.03}
        }

    def save_profile(self, baseline_rh, baseline_lh):
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
        self.current_baselines["RH"] = baseline_rh
        self.current_baselines["LH"] = baseline_lh
        return filename.name

    def list_profiles(self):
        files = sorted(DOCS_PATH.glob("*.json"), key=os.path.getmtime, reverse=True)
        return files

    def load_profile(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Default fallback if keys missing
            self.current_baselines["RH"] = data.get("rh_baseline", {"dist": 0.05, "scale": 0.05})
            self.current_baselines["LH"] = data.get("lh_baseline", {"dist": 0.03, "scale": 0.03})
        print(f"Loaded: {filepath}")
        return self.current_baselines
