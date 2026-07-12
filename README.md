# Airborne Detector

Detects an airborne target in fixed-mount camera video, with two independent detection algorithms sharing the same evaluation pipeline.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Detector A — Contrast-based

Finds the target as a small local bright spot by comparing each pixel's neighborhood contrast across multiple patch sizes (MPCM), after masking out the terrain and refining the detected region against its matching contour in the image.

```powershell
python -m scripts.evaluate_detector_a
```

## Detector B — Gap consensus + history rejection

Estimates camera ego-motion (feature matching + homography) between the current frame and several earlier frames, flags the pixels that don't move consistently with the background as target candidates, and cross-checks candidates against the target's recent trajectory to reject outliers.

```powershell
python -m scripts.evaluate_detector_b
```

## Output

Each script plots:
    - L1 error between detection and GT.
    - IoT of the segmentation predicted and the GT mask.
    - Frames with predicted detection, segmentation and GT.

