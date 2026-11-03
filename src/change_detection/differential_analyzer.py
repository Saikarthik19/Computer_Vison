"""
Bi-Temporal Road Differential Analyzer
Compares extracted road probability distributions between T1 and T2, isolating genuine new road constructions
while suppressing seasonal vegetation, shadow shifts, and agricultural noise.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from ..inference.skeletonizer import skeletonize, extract_road_graph


class RoadDifferentialAnalyzer:
    def __init__(
        self,
        min_component_area_px: int = 60,
        min_road_length_m: float = 30.0,
        temporal_margin: float = 0.30,
        gsd_meters: float = 0.5
    ):
        self.min_component_area_px = min_component_area_px
        self.min_road_length_m = min_road_length_m
        self.temporal_margin = temporal_margin
        self.gsd_meters = gsd_meters

    def detect_new_roads(
        self,
        t1_result: Dict[str, Any],
        t2_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates differential road changes:
          P_change = ReLU(P(T2) - P(T1) - temporal_margin)
        Filters false positives, closes gaps, and extracts new road centerline graphs.
        """
        prob_t1 = t1_result["probability_map"]
        prob_t2 = t2_result["probability_map"]
        bin_t1 = t1_result["binary_mask"]
        bin_t2 = t2_result["binary_mask"]

        # 1. Soft probability difference
        diff_prob = np.maximum(0.0, prob_t2 - prob_t1)

        # 2. Candidate new road mask
        # Pixels with significant increase in road likelihood
        candidate_mask = (diff_prob > self.temporal_margin) & (bin_t2 > 0)
        # Suppress pixels that were already confident roads at T1
        # Dilate T1 binary mask slightly to prevent boundary misalignment fringe false-positives
        kernel_fringe = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_t1 = cv2.dilate(bin_t1, kernel_fringe)
        candidate_mask = candidate_mask & (dilated_t1 == 0)

        candidate_binary = (candidate_mask.astype(np.uint8)) * 255

        # 3. Morphological closing to bridge small gaps caused by tree overhanging or vehicle occlusion
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed_binary = cv2.morphologyEx(candidate_binary, cv2.MORPH_CLOSE, kernel_close)

        # 4. Connected Component Analysis - filter small specks / non-linear noise
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_binary)

        clean_new_roads_mask = np.zeros_like(closed_binary)
        verified_components = []

        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < self.min_component_area_px:
                continue

            comp_mask = (labels == lbl).astype(np.uint8) * 255

            # Calculate elongation / eccentricity to verify linear road nature
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            cnt = cnts[0]
            if len(cnt) >= 5:
                # Fit ellipse or bounding box to check aspect ratio
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (width, height), angle = rect
                major_axis = max(width, height)
                minor_axis = min(width, height)
                aspect_ratio = major_axis / max(1.0, minor_axis)

                # Road corridors are elongated (aspect ratio > 1.8) or large contiguous stretches
                if aspect_ratio < 1.5 and area < 150:
                    continue  # likely square building roof or pond

            # Add to clean mask
            clean_new_roads_mask[labels == lbl] = 255
            verified_components.append({
                "label": lbl,
                "area_px": int(area),
                "area_m2": round(area * (self.gsd_meters ** 2), 2),
                "bbox": [
                    int(stats[lbl, cv2.CC_STAT_LEFT]),
                    int(stats[lbl, cv2.CC_STAT_TOP]),
                    int(stats[lbl, cv2.CC_STAT_WIDTH]),
                    int(stats[lbl, cv2.CC_STAT_HEIGHT]),
                ],
                "centroid": [float(centroids[lbl][0]), float(centroids[lbl][1])]
            })

        # 5. Extract Skeletons & Graph of strictly NEW roads
        new_skeleton = skeletonize(clean_new_roads_mask)
        new_graph, new_segments = extract_road_graph(new_skeleton, gsd_meters=self.gsd_meters)

        # Filter out new segments shorter than threshold
        valid_segments = [s for s in new_segments if s["total_length_m"] >= self.min_road_length_m]

        total_new_length_m = sum(s["total_length_m"] for s in valid_segments)
        total_new_area_m2 = sum(c["area_m2"] for c in verified_components)

        return {
            "diff_prob_map": diff_prob,
            "raw_candidate_mask": candidate_binary,
            "new_roads_mask": clean_new_roads_mask,
            "new_skeleton": new_skeleton,
            "new_graph": new_graph,
            "new_segments": valid_segments,
            "components": verified_components,
            "total_new_length_km": round(total_new_length_m / 1000.0, 3),
            "total_new_length_m": round(total_new_length_m, 2),
            "total_new_area_m2": round(total_new_area_m2, 2),
            "num_new_segments": len(valid_segments)
        }
