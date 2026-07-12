import numpy as np
import matplotlib.pyplot as plt

def normalize_frame(frame, lo, hi):

    norm = (frame.astype(np.float32) - lo) / (hi - lo)
    norm = np.clip(norm, 0, 1)
    return (norm * 255).astype(np.uint8)

if __name__ == "__main__":
    from dataloader.loader import DataLoader

    video_name = "video1"  
    loader = DataLoader(video_name)
    frames, lo, hi = loader.load_frames()

    raw_frame = frames[800]
    frame_norm = normalize_frame(raw_frame, lo, hi)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)    
    plt.imshow(raw_frame, cmap='gray')
    plt.title('Original Frame')
    plt.subplot(1, 2, 2)
    plt.imshow(frame_norm, cmap='gray')
    plt.title('Normalized Frame')
    plt.show()