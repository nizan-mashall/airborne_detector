# Airborne Detector

Detects an airborne target in fixed-mount camera video, with two independent detection algorithms sharing the same evaluation pipeline.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
## Data

Create a `data/` folder at the project root and place each video's folder inside it, matching the structure below (this must match `VIDEO_PATHS` in `config.py`):

```
data/
├── Video_1/Video_1/{images, ref_images, Video1.csv}
├── Video_2/Video_2/{images, ref_images, Video2.csv}
└── Video_3/Video_3/{images, ref_images, Video3.csv}
```

## Detector A - Contrast-based

Finds the target as a small local bright spot by comparing each pixel's neighborhood contrast across multiple patch sizes (MPCM), after masking out the terrain and refining the detected region against its matching contour in the image.

```powershell
python -m scripts.evaluate_detector_a
```

## Detector B - Temporal-residual

Estimates camera ego-motion (feature matching + homography) between the current frame and several earlier frames, flags the pixels that don't move consistently with the background as target candidates, and cross-checks candidates against the target's recent trajectory to reject outliers.

```powershell
python -m scripts.evaluate_detector_b
```

## Output

Each script plots:
    - L1 error between detection and GT.
    - IoT of the segmentation predicted and the GT mask.
    - Frames with predicted detection, segmentation and GT.

