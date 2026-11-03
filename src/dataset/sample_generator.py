"""
Bi-Temporal Satellite Tile Scenario Generator
Synthesizes paired T1 and T2 satellite imagery with ground truth road networks and newly constructed segments.
Enables offline verification, test execution, and rich demonstrations matching DeepGlobe / MassRoads satellite resolutions.
"""

import os
import cv2
import numpy as np
from typing import Dict, Tuple, List


def _generate_terrain_background(h: int, w: int, terrain_type: str = "agricultural") -> np.ndarray:
    """Creates realistic textured satellite background landscape."""
    np.random.seed(42)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if terrain_type == "agricultural":
        # Greenish-brown crop fields grid
        base_color = np.array([45, 95, 40], dtype=np.float32)  # Lush green / brown
        noise = np.random.normal(0, 15, (h, w, 3))
        canvas = np.clip(base_color + noise, 0, 255).astype(np.uint8)

        # Draw field boundary patterns
        for x in range(0, w, 120):
            cv2.line(canvas, (x, 0), (x, h), (30, 70, 30), 2)
        for y in range(0, h, 140):
            cv2.line(canvas, (0, y), (w, y), (30, 70, 30), 2)

        # Add tree cluster patches
        for _ in range(15):
            cx, cy = np.random.randint(50, w - 50), np.random.randint(50, h - 50)
            radius = np.random.randint(15, 35)
            cv2.circle(canvas, (cx, cy), radius, (20, 55, 15), -1)

    elif terrain_type == "arid_border":
        # Sandy, brownish-yellow desert landscape
        base_color = np.array([195, 175, 140], dtype=np.float32)
        noise = np.random.normal(0, 12, (h, w, 3))
        canvas = np.clip(base_color + noise, 0, 255).astype(np.uint8)

        # Rocky outcrops & dry riverbeds (wadis)
        pts = np.array([[30, 50], [150, 180], [320, 290], [500, 480]], np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=(160, 140, 110), thickness=12)

    else:
        # Mixed suburban / semi-arid
        base_color = np.array([120, 130, 105], dtype=np.float32)
        noise = np.random.normal(0, 14, (h, w, 3))
        canvas = np.clip(base_color + noise, 0, 255).astype(np.uint8)

    return canvas


def _render_road(image: np.ndarray, mask: np.ndarray, waypoints: List[Tuple[int, int]], width: int, color=(70, 70, 75)):
    """Draws a road onto both the visual satellite tile (with borders/textures) and the binary mask."""
    pts = np.array(waypoints, np.int32)

    # Road shoulder / clearing
    cv2.polylines(image, [pts], isClosed=False, color=(90, 85, 80), thickness=width + 4)
    # Road pavement
    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=width)
    # Mask (binary 255)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=width)


def generate_bi_temporal_scenario(
    scenario_name: str = "highway_expansion",
    img_size: int = 512,
    output_dir: str = "./sample_data"
) -> Dict[str, str]:
    """
    Generates a paired T1 (Historical) and T2 (Recent) tile set where:
    - T1 contains baseline pre-existing roads.
    - T2 contains baseline roads + newly constructed roads.
    - Saves:
      - t1_satellite.png
      - t2_satellite.png
      - t1_road_mask.png
      - t2_road_mask.png
      - change_ground_truth.png
    """
    target_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(target_dir, exist_ok=True)

    if scenario_name == "highway_expansion":
        t1_sat = _generate_terrain_background(img_size, img_size, "agricultural")
        t1_mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # Baseline roads at T1
        # Primary arterial running from top-left to bottom-right
        base_road_1 = [(20, 80), (140, 160), (280, 260), (410, 390), (490, 480)]
        _render_road(t1_sat, t1_mask, base_road_1, width=12, color=(65, 65, 70))

        # Secondary cross-road connecting across top
        base_road_2 = [(40, 420), (180, 360), (320, 240), (460, 110)]
        _render_road(t1_sat, t1_mask, base_road_2, width=8, color=(75, 75, 80))

        # T2 tile starts as a duplicate of T1 with realistic temporal changes (slight vegetation seasonal shift)
        t2_sat = t1_sat.copy()
        # Slight seasonal illumination change: add subtle warmth
        t2_sat = np.clip(t2_sat.astype(np.float32) * np.array([1.02, 0.98, 0.95]), 0, 255).astype(np.uint8)
        t2_mask = t1_mask.copy()

        # Newly constructed Highway Bypass connecting across the right quadrant
        new_road_bypass = [(140, 160), (260, 110), (390, 120), (470, 200), (460, 360)]
        _render_road(t2_sat, t2_mask, new_road_bypass, width=14, color=(50, 50, 55))

        # Ground truth change mask = T2 \ T1
        gt_change_mask = np.zeros_like(t1_mask)
        cv2.polylines(gt_change_mask, [np.array(new_road_bypass, np.int32)], isClosed=False, color=255, thickness=14)

    elif scenario_name == "rural_access_road":
        t1_sat = _generate_terrain_background(img_size, img_size, "semi_arid")
        t1_mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # Existing arterial road
        base_road = [(10, 256), (150, 240), (300, 270), (500, 250)]
        _render_road(t1_sat, t1_mask, base_road, width=10, color=(70, 70, 75))

        t2_sat = t1_sat.copy()
        t2_mask = t1_mask.copy()

        # New rural spur road branching northward
        new_spur = [(300, 270), (320, 160), (380, 80), (450, 50)]
        _render_road(t2_sat, t2_mask, new_spur, width=8, color=(60, 60, 65))

        gt_change_mask = np.zeros_like(t1_mask)
        cv2.polylines(gt_change_mask, [np.array(new_spur, np.int32)], isClosed=False, color=255, thickness=8)

    else:  # border_infrastructure
        t1_sat = _generate_terrain_background(img_size, img_size, "arid_border")
        t1_mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # Baseline desert road
        base_road = [(50, 480), (160, 380), (250, 320), (320, 200)]
        _render_road(t1_sat, t1_mask, base_road, width=8, color=(80, 80, 85))

        t2_sat = t1_sat.copy()
        t2_mask = t1_mask.copy()

        # New tactical paved access link to forward checkpoint
        new_border_link = [(320, 200), (360, 120), (430, 60), (490, 40)]
        _render_road(t2_sat, t2_mask, new_border_link, width=10, color=(55, 55, 60))

        gt_change_mask = np.zeros_like(t1_mask)
        cv2.polylines(gt_change_mask, [np.array(new_border_link, np.int32)], isClosed=False, color=255, thickness=10)

    # Save to disk
    paths = {
        "t1_sat": os.path.join(target_dir, "t1_satellite.png"),
        "t2_sat": os.path.join(target_dir, "t2_satellite.png"),
        "t1_mask": os.path.join(target_dir, "t1_road_mask.png"),
        "t2_mask": os.path.join(target_dir, "t2_road_mask.png"),
        "gt_change": os.path.join(target_dir, "change_ground_truth.png"),
    }

    # BGR for OpenCV saving
    cv2.imwrite(paths["t1_sat"], cv2.cvtColor(t1_sat, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["t2_sat"], cv2.cvtColor(t2_sat, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["t1_mask"], t1_mask)
    cv2.imwrite(paths["t2_mask"], t2_mask)
    cv2.imwrite(paths["gt_change"], gt_change_mask)

    return paths


def generate_all_sample_scenarios(output_dir: str = "./sample_data") -> Dict[str, Dict[str, str]]:
    scenarios = ["highway_expansion", "rural_access_road", "border_infrastructure"]
    results = {}
    for sc in scenarios:
        results[sc] = generate_bi_temporal_scenario(sc, img_size=512, output_dir=output_dir)
    return results
