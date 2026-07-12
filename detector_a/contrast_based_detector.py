import numpy as np
from matplotlib import pyplot as plt
from preprocess.normalize_frame import normalize_frame
from detector_a.skyline_detector import estimate_skyline_line
from detector_a.mpcm import detect_from_mpcm_bordered, compute_mpcm
from detector_a.refine_segmentation import refine_segmentation


class ContrastBasedDetector():
    """
    Solution A: MPCM contrast-based detector, with horizon-based terrain
    suppression and contour-based segmentation refinement.
    """

    def __init__(self, lo, hi, threshold_std_factor=6, margin_px=15):
        self.lo = lo
        self.hi = hi
        self.threshold_std_factor = threshold_std_factor
        self.margin_px = margin_px

    def predict(self, frames, frame_idx):
        raw_frame = frames[frame_idx]
        frame_norm = normalize_frame(raw_frame, lo=self.lo, hi=self.hi)

        _, _, _, _, flattened_frame = estimate_skyline_line(
            frame_norm, raw_frame=raw_frame, margin_px=self.margin_px
        )

        mpcm_map = compute_mpcm(flattened_frame)
        detections, binary_mask, _ = detect_from_mpcm_bordered(
            mpcm_map, threshold_std_factor=self.threshold_std_factor
        )

        if not detections:
            return (-1, -1), None, None

        top = detections[0]

        refined_mask = refine_segmentation(frame_norm, top['mask'], threshold_percentile=90, iou_threshold=0.1)

        ys, xs = np.where(refined_mask)
        if len(xs) == 0:
            return (-1, -1), None, None

        refined_x, refined_y = xs.mean(), ys.mean()

        return (refined_x, refined_y), refined_mask, top['confidence']


if __name__ == "__main__":
    from dataloader.loader import DataLoader

    video_name = "video1"
    loader = DataLoader(video_name)
    frames, lo, hi = loader.load_frames()

    detector = ContrastBasedDetector(lo=lo, hi=hi)
    (x, y), mask, conf = detector.predict(frames, 1010)
    print(f"Detection: ({x:.1f}, {y:.1f}), confidence={conf:.2f}")
    plt.figure(figsize=(10, 5))
    plt.imshow(mask, cmap='gray')
    plt.show()