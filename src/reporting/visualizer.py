"""
High-Contrast Visual Overlays & Comparative Visualizations
Produces tactical intelligence map overlays, composite comparison panels, and heatmaps.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


def create_road_overlay(
    satellite_rgb: np.ndarray,
    road_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.55
) -> np.ndarray:
    """Overlays road mask onto satellite imagery with adjustable opacity."""
    overlay = satellite_rgb.copy()
    mask_bool = road_mask > 0

    colored_mask = np.zeros_like(satellite_rgb)
    colored_mask[mask_bool] = color

    blended = cv2.addWeighted(colored_mask, alpha, overlay, 1.0 - alpha, 0)
    result = np.where(mask_bool[:, :, None], blended, overlay)
    return result


def create_change_detection_composite(
    t1_rgb: np.ndarray,
    t2_rgb: np.ndarray,
    t1_road_mask: np.ndarray,
    t2_road_mask: np.ndarray,
    new_roads_mask: np.ndarray,
    new_skeleton: np.ndarray = None
) -> np.ndarray:
    """
    Creates an intuitive multi-spectral tactical intelligence overlay:
    - Pre-existing T1 roads shown in Cyan (0, 220, 255)
    - Newly constructed T2 roads highlighted in Vibrant Neon Red/Crimson (255, 30, 30)
    - Intersection / skeleton nodes in Bright Green (0, 255, 0)
    """
    canvas = t2_rgb.copy()

    # Pre-existing roads mask (in T1 and still in T2)
    preexisting = (t1_road_mask > 0) & (new_roads_mask == 0)
    # New roads mask
    new_roads = new_roads_mask > 0

    # Apply Cyan for pre-existing roads
    cyan_layer = np.zeros_like(canvas)
    cyan_layer[preexisting] = [0, 210, 230]
    canvas = np.where(preexisting[:, :, None], cv2.addWeighted(cyan_layer, 0.6, canvas, 0.4, 0), canvas)

    # Apply Neon Crimson Red for newly flagged roads
    red_layer = np.zeros_like(canvas)
    red_layer[new_roads] = [255, 35, 35]
    canvas = np.where(new_roads[:, :, None], cv2.addWeighted(red_layer, 0.75, canvas, 0.25, 0), canvas)

    # If skeleton provided, draw bright centerline
    if new_skeleton is not None and np.any(new_skeleton > 0):
        skel_bool = new_skeleton > 0
        canvas[skel_bool] = [255, 255, 0]  # Bright yellow centerline

    return canvas


def create_side_by_side_panel(
    t1_rgb: np.ndarray,
    t2_rgb: np.ndarray,
    t1_mask: np.ndarray,
    t2_mask: np.ndarray,
    change_overlay: np.ndarray
) -> np.ndarray:
    """
    Arranges satellite tiles and road extractions into a standardized 5-pane intelligence audit plate.
    Top row: T1 Satellite, T2 Satellite, Tactical Change Overlay.
    Bottom row: T1 Road Mask, T2 Road Mask, Difference Map.
    """
    h, w, _ = t1_rgb.shape

    # Convert grayscale masks to 3-channel RGB for display
    t1_mask_rgb = cv2.cvtColor(t1_mask, cv2.COLOR_GRAY2RGB)
    t2_mask_rgb = cv2.cvtColor(t2_mask, cv2.COLOR_GRAY2RGB)

    # Colorize difference
    diff_colored = np.zeros_like(t1_rgb)
    # New roads in red
    diff_colored[change_overlay[:, :, 0] > 180] = [230, 40, 40]

    # Combine into top and bottom rows
    top_row = np.hstack([t1_rgb, t2_rgb, change_overlay])
    bottom_row = np.hstack([t1_mask_rgb, t2_mask_rgb, diff_colored])

    panel = np.vstack([top_row, bottom_row])
    return panel
