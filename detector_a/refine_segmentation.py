import cv2
import numpy as np

import matplotlib.pyplot as plt
from dataloader.loader import DataLoader
from preprocess.normalize_frame import normalize_frame
from detector_a.mpcm import compute_mpcm
from detector_a.mpcm import detect_from_mpcm_bordered

import cv2
import numpy as np


def refine_segmentation(norm_frame, mask, threshold_percentile=90, iou_threshold=0.1,
                         border_width=20):
    '''
    Refine the segmentation mask by finding the best matching contour in the normalized frame.
    Required because the MPCM averaging and big object masking might preduce sparce results.
    '''

    if mask is None or mask.sum() == 0:
        return mask

    h, w = norm_frame.shape

    bordered_frame = norm_frame.copy()
    bordered_frame[:border_width, :] = 0
    bordered_frame[-border_width:, :] = 0
    bordered_frame[:, :border_width] = 0
    bordered_frame[:, -border_width:] = 0

    threshold_val = np.percentile(norm_frame, threshold_percentile)
    binary = (bordered_frame > threshold_val).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask

    best_contour, best_score = None, -1
    seed_area = mask.sum()

    for c in contours:
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [c], -1, 1, thickness=-1)
        contour_mask = contour_mask.astype(bool)

        intersection = (mask & contour_mask).sum()
        if intersection == 0:
            continue

        score = intersection / seed_area

        if score > best_score:
            best_score = score
            best_contour = c

    if best_contour is None or best_score < iou_threshold:
        return mask

    refined_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(refined_mask, [best_contour], -1, 1, thickness=-1)
    return refined_mask.astype(bool)


if __name__ == "__main__":

    video_name = "video1"
    loader = DataLoader(video_name)
    frames, lo, hi = loader.load_frames()

    frame_idx = 1010
    raw_frame = frames[frame_idx]
    frame_norm = normalize_frame(raw_frame, lo=lo, hi=hi)

    mpcm_map = compute_mpcm(raw_frame)
    detections, _, _ = detect_from_mpcm_bordered(mpcm_map, threshold_std_factor=6)

    if detections:
        top = detections[0]
        refined_mask = refine_segmentation(frame_norm, top['mask'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(frame_norm, cmap='gray')
        axes[0].contour(top['mask'], colors='yellow', linewidths=2)
        axes[0].set_title(f"Original mask (area={top['mask'].sum()})")
        axes[0].axis('off')

        axes[1].imshow(frame_norm, cmap='gray')
        axes[1].contour(refined_mask, colors='red', linewidths=2)
        axes[1].set_title(f"Refined mask (area={refined_mask.sum()})")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("No detections found on this frame.")