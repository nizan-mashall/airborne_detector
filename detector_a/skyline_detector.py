import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from dataloader.loader import DataLoader
from config import VIDEO_PATHS
from preprocess.normalize_frame import normalize_frame

def estimate_skyline_line(frame_norm, raw_frame=None, margin_px=15,
                            angle_step=2, n_offsets=40, canny_low=20, canny_high=60):

    blurred = cv2.bilateralFilter(frame_norm, d=9, sigmaColor=50, sigmaSpace=50)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    edges_bool = edges > 0
    h, w = edges.shape
    yy, xx = np.mgrid[0:h, 0:w]
    total = edges_bool.size

    best_score, best_line = -np.inf, None
    best_dense_mask = None

    for angle_deg in np.arange(0, 180, angle_step):
        angle_rad = np.deg2rad(angle_deg)
        vx, vy = np.cos(angle_rad), np.sin(angle_rad)
        nx, ny = -vy, vx
        proj = xx * nx + yy * ny
        offsets = np.linspace(proj.min(), proj.max(), n_offsets)

        for offset in offsets:
            mask_a = proj < offset
            n_a = mask_a.sum()
            n_b = total - n_a
            if n_a < 0.02 * total or n_b < 0.02 * total:
                continue

            weight_a, weight_b = n_a / total, n_b / total
            density_a = edges_bool[mask_a].mean()
            density_b = edges_bool[~mask_a].mean()

            score = weight_a * weight_b * (density_a - density_b) ** 2

            if score > best_score:
                best_score, best_line = score, (angle_deg, offset, nx, ny)
                best_dense_mask = mask_a if density_a > density_b else ~mask_a

    flattened_frame = None
    if raw_frame is not None and best_dense_mask is not None:
        expanded_mask = ndimage.binary_dilation(best_dense_mask, iterations=margin_px)
        flattened_frame = raw_frame.copy().astype(np.float32)
        flattened_frame[expanded_mask] = raw_frame.max()

    return best_line, edges, best_score, best_dense_mask, flattened_frame

def plot_horizon_line_side_by_side(frame_norm, edges, best_line, highlighted_frame=None):
    angle_deg, offset, nx, ny = best_line
    h, w = frame_norm.shape

    vx, vy = -ny, nx
    if abs(nx) > 1e-6:
        x0, y0 = offset / nx, 0
    else:
        x0, y0 = 0, offset / ny

    t = max(h, w) * 2
    p1 = (x0 - t * vx, y0 - t * vy)
    p2 = (x0 + t * vx, y0 + t * vy)

    imgs = [frame_norm, edges]
    titles = ['Frame', 'Canny edges']
    if highlighted_frame is not None:
        imgs.append(highlighted_frame)
        titles.append('Dense side highlighted')

    fig, axes = plt.subplots(1, len(imgs), figsize=(7 * len(imgs), 6))
    if len(imgs) == 1:
        axes = [axes]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap='gray')
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=2)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(f'{title} (angle={angle_deg}°)')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    video_name = "video1" 
    loader = DataLoader(video_name)
    frames, lo, hi = loader.load_frames()

    raw_frame = frames[1000]
    frame_norm = normalize_frame(raw_frame, lo, hi)
    best_line, edges, score, dense_mask, flattened_frame = estimate_skyline_line(
    frame_norm, raw_frame=raw_frame, margin_px=15
    )
    plot_horizon_line_side_by_side(frame_norm, edges, best_line, highlighted_frame=flattened_frame)