"""
Automated Satellite Tile Co-Registration
Ensures spatial and geometric alignment between bi-temporal satellite images (T1 & T2).
Utilizes ORB feature detection, FLANN/BF matcher, and RANSAC Homography.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


def coregister_tiles(
    img_t1: np.ndarray,
    img_t2: np.ndarray,
    max_features: int = 2000,
    min_matches: int = 15,
    ransac_thresh: float = 5.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Warps T2 to match the coordinate frame of T1 using robust homography.
    If images are already aligned (e.g. synthetic or pre-rectified orthophotos),
    returns T2 directly with alignment metrics.
    """
    gray1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY) if img_t1.ndim == 3 else img_t1
    gray2 = cv2.cvtColor(img_t2, cv2.COLOR_RGB2GRAY) if img_t2.ndim == 3 else img_t2

    # ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    stats = {
        "aligned": False,
        "num_matches": 0,
        "inliers_count": 0,
        "homography": None
    }

    if des1 is None or des2 is None or len(des1) < min_matches or len(des2) < min_matches:
        # Cannot compute homography; assume already ortho-rectified
        stats["message"] = "Insufficient keypoints; using direct pixel alignment."
        return img_t2, stats

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    stats["num_matches"] = len(matches)

    if len(matches) < min_matches:
        stats["message"] = f"Matches below threshold ({len(matches)} < {min_matches}); skipping warp."
        return img_t2, stats

    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, inlier_mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thresh)

    if H is not None:
        inliers = np.sum(inlier_mask)
        stats["inliers_count"] = int(inliers)
        stats["homography"] = H.tolist()

        # Check if homography is near-identity (e.g., drift < 1 pixel)
        identity = np.eye(3)
        diff = np.max(np.abs(H - identity))
        if diff < 0.05:
            # Virtually identical, avoid unnecessary interpolation blur
            stats["aligned"] = True
            stats["message"] = "Tiles already in pristine geometric co-registration."
            return img_t2, stats

        h, w = img_t1.shape[:2]
        warped_t2 = cv2.warpPerspective(img_t2, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        stats["aligned"] = True
        stats["message"] = f"Successfully co-registered T2 with {inliers} RANSAC inliers."
        return warped_t2, stats

    stats["message"] = "Homography estimation failed; falling back to unwarped image."
    return img_t2, stats
