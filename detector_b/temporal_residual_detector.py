import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN

from preprocess.normalize_frame import normalize_frame
from detector_b.homography import estimate_homography_motion_clustered, transform_point
from detector_b.history_tracker import SimpleHistoryTracker


def detect_via_gap_consensus_simple_history(frames, frame_idx, lo, hi, gap_candidates=[1, 2, 3, 4, 5],
                                              z_threshold=6, border_margin=40, agreement_radius=15,
                                              history_tracker=None, contour_threshold_percentile=90,
                                              contour_max_area=100000, max_reject_distance=80,
                                              min_history_points_to_enforce=2):

    f_ref = normalize_frame(frames[frame_idx], lo, hi)
    candidate_detections = []

    threshold_val = np.percentile(f_ref, contour_threshold_percentile)
    binary = (f_ref > threshold_val).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) <= contour_max_area]

    for gap in gap_candidates:
        neighbor_idx = frame_idx - gap
        if neighbor_idx < 0:
            continue
        f_moving = normalize_frame(frames[neighbor_idx], lo, hi)
        H, n_selected = estimate_homography_motion_clustered(f_ref, f_moving)
        if H is None:
            continue

        h, w = f_ref.shape
        warped = cv2.warpPerspective(f_moving, H, (w, h), flags=cv2.INTER_LINEAR)
        residual = np.abs(f_ref.astype(np.float32) - warped.astype(np.float32))
        residual[:border_margin, :] = 0
        residual[-border_margin:, :] = 0
        residual[:, :border_margin] = 0
        residual[:, -border_margin:] = 0
        if residual.max() == 0:
            continue

        max_loc = np.unravel_index(np.argmax(residual), residual.shape)
        max_y, max_x = max_loc

        best_contour, best_dist = None, np.inf
        for c in contours:
            if cv2.contourArea(c) == 0:
                continue
            signed_dist = cv2.pointPolygonTest(c, (float(max_x), float(max_y)), True)
            dist = 0.0 if signed_dist >= 0 else abs(signed_dist)
            if dist < best_dist:
                best_dist = dist
                best_contour = c

        if best_contour is None or best_dist > agreement_radius:
            fallback_threshold = np.percentile(residual, 90)
            binary_mask = residual > fallback_threshold
            labeled, n_components = ndimage.label(binary_mask)
            comp_label = labeled[max_y, max_x]
            if comp_label == 0:
                continue
            comp_mask = labeled == comp_label
        else:
            comp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(comp_mask, [best_contour], -1, 1, thickness=-1)
            comp_mask = comp_mask.astype(bool)

        if comp_mask.sum() == 0:
            continue

        conf = residual[comp_mask].max()
        ys, xs = np.where(comp_mask)
        candidate_detections.append({'gap': gap, 'x': xs.mean(), 'y': ys.mean(),
                                      'conf': conf, 'mask': comp_mask})

    if len(candidate_detections) == 0:
        return (-1, -1), None, None, 0

    expected_x, expected_y, spread, n_inlier_history = (None, None, None, 0)
    if history_tracker is not None:
        expected_x, expected_y, spread, n_inlier_history = history_tracker.estimate_position(
            frames, frame_idx, lo, hi
        )

    enforce_rejection = (expected_x is not None and n_inlier_history >= min_history_points_to_enforce)

    final_candidates = candidate_detections
    if enforce_rejection:
        filtered = [d for d in candidate_detections
                    if np.hypot(d['x'] - expected_x, d['y'] - expected_y) <= max_reject_distance]
        if len(filtered) > 0:
            final_candidates = filtered

    positions = np.array([[d['x'], d['y']] for d in final_candidates])
    clustering = DBSCAN(eps=agreement_radius, min_samples=1).fit(positions)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)

    best_label = unique_labels[np.argmax(counts)]
    agreement_count = counts.max()
    group = [d for d, l in zip(final_candidates, labels) if l == best_label]
    best = max(group, key=lambda d: d['conf'])

    if history_tracker is not None:
        history_tracker.update(best['x'], best['y'], frame_idx)

    return (best['x'], best['y']), best['mask'], best['conf'], agreement_count


class TemporalResidualDetector():
    def __init__(self, lo, hi, gap_candidates=[1, 2, 3, 4, 5], z_threshold=6,
                 border_margin=40, agreement_radius=15, contour_threshold_percentile=90,
                 contour_max_area=100000, max_reject_distance=80,
                 min_history_points_to_enforce=2, history_length=3, outlier_reject_radius=80):
        self.lo, self.hi = lo, hi
        self.gap_candidates = gap_candidates
        self.z_threshold = z_threshold
        self.border_margin = border_margin
        self.agreement_radius = agreement_radius
        self.contour_threshold_percentile = contour_threshold_percentile
        self.contour_max_area = contour_max_area
        self.max_reject_distance = max_reject_distance
        self.min_history_points_to_enforce = min_history_points_to_enforce
        self.tracker = SimpleHistoryTracker(history_length=history_length,
                                              outlier_reject_radius=outlier_reject_radius)

    def predict(self, frames, frame_idx):
        (pred_x, pred_y), pred_mask, conf, agreement = detect_via_gap_consensus_simple_history(
            frames, frame_idx, self.lo, self.hi, gap_candidates=self.gap_candidates,
            z_threshold=self.z_threshold, border_margin=self.border_margin,
            agreement_radius=self.agreement_radius, history_tracker=self.tracker,
            contour_threshold_percentile=self.contour_threshold_percentile,
            contour_max_area=self.contour_max_area, max_reject_distance=self.max_reject_distance,
            min_history_points_to_enforce=self.min_history_points_to_enforce
        )
        return (pred_x, pred_y), pred_mask, conf