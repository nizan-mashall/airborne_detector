# %%
import numpy as np
from matplotlib import pyplot as plt
from dataloader.loader import DataLoader
from detector_b.temporal_residual_detector import TemporalResidualDetector
from evaluation.evaluate import evaluate_detector, plot_l1_curve, plot_gt_vs_pred, plot_IoU_curve
from evaluation.evaluate import save_detection_video

video_name = "video3"
loader = DataLoader(video_name)
frames, lo, hi = loader.load_frames()
df = loader.load_csv()
ref_files = loader.load_reference_images_list()
output_path=r"C:\Users\nizan\Desktop\Work\SAIL_Lab\Airborne_Detector\output_videos\video3_temporal_residual_detector.mp4"


detector = TemporalResidualDetector(
    lo=lo, hi=hi,
    gap_candidates=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    max_reject_distance=80,
    min_history_points_to_enforce=2,
)

l1_list, idx_list, mean_l1, gt_list, pred_list, IoU_list, mask_list, conf_list = evaluate_detector(
    detector, frames, df, ref_files, start_idx=15, end_idx=len(frames)-1, step=1
)

idx_frames=[710, 720, 730, 740, 745, 749]

plot_l1_curve(l1_list, idx_list, title="L1 error vs. frame index for Temporal Residual Detector ")
plot_IoU_curve(IoU_list, idx_list)
plot_gt_vs_pred(gt_list, pred_list, idx_list, idx_frames=idx_frames,
                frames=frames, mask_list=mask_list, lo=lo, hi=hi)

save_detection_video(frames, idx_list, pred_list, conf_list, mask_list, lo, hi,
                      output_path=output_path, fps=20)