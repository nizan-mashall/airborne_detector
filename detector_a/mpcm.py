import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def compute_mpcm(frame, patch_sizes=[3, 5, 7]):
    '''
    Computes the Multi-Patch Contrast Map (MPCM), which highlights local bright targets in the image.
    The boxFilter handles the local mean computation efficiently, while the next part computes
    the contrast using matrix operations, avoiding explicit loops over pixels.
    The loop over patch_sizes selects the maximum contrast across multiple scales,
    which is useful for detecting targets that change in size.
    '''
    frame_f = frame.astype(np.float32)
    h, w = frame_f.shape
    mpcm_final = np.zeros((h, w), dtype=np.float32)

    for L in patch_sizes:
        mean_img = cv2.boxFilter(frame_f, ddepth=-1, ksize=(L, L), normalize=True)

        pad = L
        padded = np.pad(mean_img, pad, mode='edge')

        def shifted(dy, dx):
            return padded[pad+dy:pad+dy+h, pad+dx:pad+dx+w]

        m_c  = mean_img                     # center
        m_n  = shifted(-L, 0)                # north
        m_s  = shifted(L, 0)                 # south
        m_e  = shifted(0, L)                 # east
        m_w  = shifted(0, -L)                # west
        m_ne = shifted(-L, L)                # north-east
        m_nw = shifted(-L, -L)               # north-west
        m_se = shifted(L, L)                 # south-east
        m_sw = shifted(L, -L)                # south-west

        c1 = np.minimum(m_c - m_n, m_c - m_s)
        c2 = np.minimum(m_c - m_e, m_c - m_w)
        c3 = np.minimum(m_c - m_ne, m_c - m_sw)
        c4 = np.minimum(m_c - m_nw, m_c - m_se)

        scale_contrast = np.maximum(np.maximum(c1, c2), np.maximum(c3, c4))
        scale_contrast = np.clip(scale_contrast, 0, None) 

        mpcm_final = np.maximum(mpcm_final, scale_contrast)

    return mpcm_final

def detect_from_mpcm(mpcm_map, threshold=None, threshold_std_factor=6):
    '''
    Extracting detections from the MPCM contrast map. 
    Seperate to multiple components using connected component labeling, 
    and then compute the centroid, confidence, and area for each component.
    '''
    if threshold is None:
        threshold = mpcm_map.mean() + threshold_std_factor * mpcm_map.std()

    binary_mask = mpcm_map > threshold
    labeled, n_components = ndimage.label(binary_mask)

    detections = []
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        ys, xs = np.where(comp_mask)
        centroid_x, centroid_y = xs.mean(), ys.mean()
        confidence = mpcm_map[comp_mask].max()  / mpcm_map.max()
        #confidence = mpcm_map[comp_mask].max()
        area = comp_mask.sum()
        detections.append({'x': centroid_x, 'y': centroid_y, 'confidence': confidence,
                    'area': area, 'mask': comp_mask})
    detections.sort(key=lambda d: -d['confidence'])
    return detections, binary_mask, threshold

def detect_from_mpcm_bordered(mpcm_map, threshold_std_factor=6, border_margin=10):
    mpcm_map = mpcm_map.copy()
    mpcm_map[:border_margin, :] = 0
    mpcm_map[-border_margin:, :] = 0
    mpcm_map[:, :border_margin] = 0
    mpcm_map[:, -border_margin:] = 0
    return detect_from_mpcm(mpcm_map, threshold_std_factor=threshold_std_factor)


def test_mpcm_on_frame(frame_idx, frames, df, lo, hi):
    frame = frames[frame_idx]
    norm_frame = normalize_frame(frame, lo, hi)

    mpcm_map = compute_mpcm(frame)  # run on raw (not normalized) values for real contrast
    detections, binary_mask, threshold = detect_from_mpcm_bordered(mpcm_map)

    gt_x = df.loc[frame_idx, 'X_t(pixels)']
    gt_y = df.loc[frame_idx, 'Y_t(pixels)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(norm_frame, cmap='gray')
    # axes[0].scatter([gt_x], [gt_y], c='lime', marker='+', s=200, label='GT')
    if detections:
        axes[0].scatter([detections[0]['x']], [detections[0]['y']], c='red', marker='x', s=150, label='Top detection')
    axes[0].set_title(f'Frame {frame_idx}: normalized + GT/detection')
    axes[0].legend()
    axes[0].axis('off')

    axes[1].imshow(mpcm_map, cmap='inferno')
    axes[1].set_title(f'MPCM contrast map\n(threshold={threshold:.2f})')
    axes[1].axis('off')

    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title(f'Thresholded detections\n({len(detections)} candidate(s))')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


    return detections

if __name__ == "__main__":
    from dataloader.loader import DataLoader
    from preprocess.normalize_frame import normalize_frame

    video_name = "video1"  
    loader = DataLoader(video_name)
    frames, lo, hi = loader.load_frames()
    df = loader.load_csv()

    frame_idx = 800
    detections = test_mpcm_on_frame(frame_idx, frames, df, lo, hi)