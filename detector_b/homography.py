import cv2
import numpy as np
from sklearn.cluster import DBSCAN
''' Calculate the homography matrix between two frames using ORB feature, 
    matching and calculate motion vectors, then use DBSCAN clustering to avoid using target 
    features for homography estimation. 
'''

def estimate_homography_motion_clustered(frame_ref, frame_moving, max_features=2000,
                                           good_match_percent=0.3, eps=2.0, min_samples=8):

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(frame_ref, None)
    kp2, des2 = orb.detectAndCompute(frame_moving, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)
    n_good = max(8, int(len(matches) * good_match_percent))
    good_matches = matches[:n_good]
    if len(good_matches) < 8:
        return None, None

    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    motion_vectors = pts1 - pts2  

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(motion_vectors)
    labels = clustering.labels_

    if (labels != -1).sum() == 0:
        return None, None  # no cluster found at all

    # Find the largest cluster (most common motion pattern)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    cluster_mask = labels == largest_cluster_label

    cluster_pts1 = pts1[cluster_mask]
    cluster_pts2 = pts2[cluster_mask]

    H, ransac_mask = cv2.findHomography(cluster_pts2, cluster_pts1, cv2.RANSAC, 3.0)

    return H, cluster_mask.sum()


def transform_point(pt, H):
    """
    Apply a homography H to a single (x, y) point.
    """
    pt_arr = np.array([[pt]], dtype=np.float32)  # shape (1, 1, 2), as cv2.perspectiveTransform expects
    transformed = cv2.perspectiveTransform(pt_arr, H)
    return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])