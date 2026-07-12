import numpy as np
import pandas as pd
import os
from PIL import Image
import glob
from config import VIDEO_PATHS

class DataLoader:
    def __init__(self, video_name, normalize_method = "minmax"):
        self.video_name = video_name
        self.video_config = VIDEO_PATHS[video_name]
        self.frames_dir = self.video_config["frames_dir"]
        print(f"Frames directory for {video_name}: {self.frames_dir}")
        self.ref_dir = self.video_config["ref_dir"]
        self.csv_path = self.video_config["csv_path"]
        self.normalize_method = normalize_method

    def load_frames(self):
        frames_files = sorted(glob.glob(os.path.join(self.frames_dir, '*.TIFF*')))

        frames = []
        for f in frames_files:
            img = Image.open(f)
            arr = np.array(img)
            frames.append(arr)
        frames = np.stack(frames)
        if self.normalize_method == "minmax":
            lo, hi = frames.min(), frames.max()
        elif self.normalize_method == "percentile":
            lo, hi = np.percentile(frames, 1), np.percentile(frames, 99)
        return frames, lo, hi

    def load_csv(self):
        df = pd.read_csv(self.csv_path)
        return df
    
    def load_reference_images_list(self):
        ref_files = sorted(glob.glob(os.path.join(self.ref_dir, '*.TIFF*')))  # adjust extension if different (.png, .bmp, etc.)
        return ref_files  
    
if __name__ == "__main__":
    video_name = "video1"  # Change this to "video2" or "video3" as needed
    loader = DataLoader(video_name)
    
    frames, lo, hi = loader.load_frames()
    print(f"Loaded {frames.shape[0]} frames for {video_name}.")
    print(f"Normalization range: [{lo}, {hi}]")
    
    df = loader.load_csv()
    print(f"Loaded CSV with shape: {df.shape} for {video_name}.")
    
    ref_images_list = loader.load_reference_images_list()
    print(f"Loaded {len(ref_images_list)} reference images for {video_name}.")