from dataloader.loader import DataLoader
from detector_a.contrast_based_detector import ContrastBasedDetector
from evaluation.evaluate import evaluate_detector, plot_l1_curve, plot_gt_vs_pred, plot_IoU_curve

video_name = "video1"
loader = DataLoader(video_name)
frames, lo, hi = loader.load_frames()
df = loader.load_csv()
ref_files = loader.load_reference_images_list()


mpcm_detector = ContrastBasedDetector(lo=lo, hi=hi, threshold_std_factor=6, margin_px=50)

l1_list, idx_list, mean_l1, gt_list, pred_list, IoU_list, mask_list = evaluate_detector(
    mpcm_detector, frames, df, ref_files, start_idx=0, end_idx=len(frames)-1, step=1
)

idx_frames = [210, 220, 230, 2240, 310, 810, 990, 1000, 1010]
plot_l1_curve(l1_list, idx_list, title="Video 2, contrast-based detector")
plot_IoU_curve(IoU_list, idx_list)
plot_gt_vs_pred(gt_list, pred_list, idx_list, idx_frames, frames, mask_list, lo, hi)
