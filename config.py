import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

VIDEO_PATHS = {
    "video1": {
        "frames_dir": os.path.join(BASE_DATA_DIR, "Video_1", "Video_1", "images"),
        "ref_dir": os.path.join(BASE_DATA_DIR, "Video_1", "Video_1", "ref_images"),
        "csv_path": os.path.join(BASE_DATA_DIR, "Video_1", "Video_1", "Video1.csv"), 
    },
    "video2": {
        "frames_dir": os.path.join(BASE_DATA_DIR, "Video_2", "Video_2", "images"),
        "ref_dir": os.path.join(BASE_DATA_DIR, "Video_2", "Video_2", "ref_images"),
        "csv_path": os.path.join(BASE_DATA_DIR, "Video_2", "Video_2", "Video2.csv"),
    },
    "video3": {
        "frames_dir": os.path.join(BASE_DATA_DIR, "Video_3", "Video_3", "images"),
        "ref_dir": os.path.join(BASE_DATA_DIR, "Video_3", "Video_3", "ref_images"),
        "csv_path": os.path.join(BASE_DATA_DIR, "Video_3", "Video_3", "Video3.csv"),
    },
}