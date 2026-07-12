from collections import deque
import numpy as np

from preprocess.normalize_frame import normalize_frame
from detector_b.homography import estimate_homography_motion_clustered, transform_point

class SimpleHistoryTracker:
    def __init__(self, history_length=3, outlier_reject_radius=80):
        self.history = deque(maxlen=history_length)
        self.outlier_reject_radius = outlier_reject_radius

    def estimate_position(self, frames, frame_idx, lo, hi, min_inliers_to_trust=30):
        if len(self.history) == 0:
            return None, None, None, 0

        f_now = normalize_frame(frames[frame_idx], lo, hi)
        warped_estimates = []

        for (hist_frame_idx, hx, hy) in self.history:
            if hist_frame_idx == frame_idx:
                warped_estimates.append((hx, hy))
                continue
            f_hist = normalize_frame(frames[hist_frame_idx], lo, hi)
            H, n_inliers = estimate_homography_motion_clustered(f_now, f_hist)
            if H is not None and n_inliers is not None and n_inliers >= min_inliers_to_trust:
                wx, wy = transform_point((hx, hy), H)
                warped_estimates.append((wx, wy))

        if len(warped_estimates) == 0:
            return None, None, None, 0
        if len(warped_estimates) == 1:
            return warped_estimates[0][0], warped_estimates[0][1], self.outlier_reject_radius, 1

        xs = np.array([p[0] for p in warped_estimates])
        ys = np.array([p[1] for p in warped_estimates])
        median_x, median_y = np.median(xs), np.median(ys)

        dists = np.hypot(xs - median_x, ys - median_y)
        inlier_mask = dists <= self.outlier_reject_radius
        n_inliers_final = int(inlier_mask.sum())

        if n_inliers_final == 0:
            return median_x, median_y, self.outlier_reject_radius, 0

        final_x = xs[inlier_mask].mean()
        final_y = ys[inlier_mask].mean()
        spread = (xs[inlier_mask].std() + ys[inlier_mask].std()) if n_inliers_final > 1 else self.outlier_reject_radius / 2

        return final_x, final_y, spread, n_inliers_final

    def update(self, x, y, frame_idx):
        self.history.append((frame_idx, x, y))

    def reset(self):
        self.history.clear()