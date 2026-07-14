import math
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocess.normalize_frame import normalize_frame


def load_gt_mask(frame_idx, ref_images_list):
    ref_raw = np.array(Image.open(ref_images_list[frame_idx]))
    ref_8bit = ((ref_raw.astype(np.float32) - ref_raw.min()) /
                (ref_raw.max() - ref_raw.min() + 1e-8) * 255).astype(np.uint8)
    _, ref_binary = cv2.threshold(ref_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ref_binary > 0


def compute_iou(pred_mask, gt_mask):
    if pred_mask is None:
        return 0.0
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return intersection / union if union > 0 else 0.0


def evaluate_detector(detector, frames, df, ref_images_list, start_idx=10, end_idx=None, step=1):

    l1_list, idx_list, gt_list, pred_list, IoU_list, mask_list, conf_list = [], [], [], [], [], [], []

    for frame_idx in tqdm(range(start_idx, end_idx, step), desc=detector.__class__.__name__):
        gt_x = df.loc[frame_idx, 'X_t(pixels)']
        gt_y = df.loc[frame_idx, 'Y_t(pixels)']

        (det_x, det_y), final_mask, conf = detector.predict(frames, frame_idx)

        l1 = (abs(gt_x - det_x) + abs(gt_y - det_y)) / 2

        gt_mask = load_gt_mask(frame_idx, ref_images_list)
        iou_score = compute_iou(final_mask, gt_mask)

        IoU_list.append(iou_score)
        l1_list.append(l1)
        idx_list.append(frame_idx)
        gt_list.append((gt_x, gt_y))
        pred_list.append((det_x, det_y))
        mask_list.append((gt_mask, final_mask))
        conf_list.append(conf)
    mean_l1 = np.mean(l1_list)

    return l1_list, idx_list, mean_l1, gt_list, pred_list, IoU_list, mask_list,conf_list


def plot_l1_curve(l1_list, idx_list, title="L1 error vs. frame index"):
    plt.figure(figsize=(12, 5))
    plt.plot(idx_list, l1_list)
    plt.axhline(np.mean(l1_list), color='red', linestyle='--', label=f'Mean = {np.mean(l1_list):.2f}')
    plt.axhline(0, color='green', linestyle='-.')
    plt.xlabel('Frame index')
    plt.ylabel('L1 error (pixels)')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_gt_vs_pred(gt_list, pred_list, idx_list, idx_frames, frames, mask_list, lo, hi, max_cols=3):
    n = len(idx_frames)
    n_cols = min(max_cols, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))

    # Normalize axes into a flat 1D array regardless of n_rows/n_cols combination
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(-1)

    for ax_i, frame_idx in enumerate(idx_frames):
        ax = axes[ax_i]

        if frame_idx not in idx_list:
            print(f"Warning: frame {frame_idx} not in evaluated range, skipping.")
            ax.axis('off')
            continue
        pos = idx_list.index(frame_idx)

        gt_x, gt_y = gt_list[pos]
        pred_x, pred_y = pred_list[pos]
        gt_mask, pred_mask = mask_list[pos]

        ax.imshow(normalize_frame(frames[frame_idx], lo, hi), cmap='gray')
        ax.contour(gt_mask, colors='lime', linewidths=2)
        ax.contour(pred_mask, colors='red', linewidths=2)
        ax.scatter([gt_x], [gt_y], c='lime', marker='+', s=200, label='GT')
        ax.scatter([pred_x], [pred_y], c='red', marker='x', s=100, label='Pred')
        ax.set_title(f'Frame {frame_idx}')
        ax.legend()
        ax.axis('off')

    # Hide any unused subplot axes (when n doesn't evenly fill the grid)
    for ax_i in range(n, len(axes)):
        axes[ax_i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_IoU_curve(IoU_list, idx_list):
    plt.figure(figsize=(12, 5))
    plt.plot(idx_list, IoU_list)
    plt.axhline(np.mean(IoU_list), color='red', linestyle='--', label=f'Mean IoU = {np.mean(IoU_list):.3f}')
    plt.xlabel('Frame index')
    plt.ylabel('IoU')
    plt.title('IoU vs. frame index')
    plt.legend()
    plt.show()

def save_detection_video(frames, idx_list, pred_list, conf_list, mask_list, lo, hi, output_path,
                          fps=15, contour_color=(0, 0, 255), point_color=(0, 255, 0),
                          show_frame_idx=True):
  
    if len(idx_list) == 0:
        raise ValueError("idx_list is empty - nothing to write.")
 
    h, w = frames[idx_list[0]].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
 
    for pos, frame_idx in enumerate(idx_list):
        norm_frame = normalize_frame(frames[frame_idx], lo, hi)
        frame_bgr = cv2.cvtColor(norm_frame, cv2.COLOR_GRAY2BGR)
 
        _, pred_mask = mask_list[pos]
        pred_x, pred_y = pred_list[pos]
 
        if pred_mask is not None:
            mask_u8 = pred_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_bgr, contours, -1, contour_color, 1)
 
        if pred_x != -1 and pred_y != -1:
            cv2.drawMarker(frame_bgr, (int(round(pred_x)), int(round(pred_y))),
                            point_color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
 
        if show_frame_idx:
            cv2.putText(frame_bgr, f"Frame {frame_idx}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
 
        writer.write(frame_bgr)
 
    writer.release()
    print(f"Saved video with {len(idx_list)} frames to {output_path}")
