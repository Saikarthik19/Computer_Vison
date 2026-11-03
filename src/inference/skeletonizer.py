"""
Road Network Skeletonization & Vector Graph Extraction
Converts binary road segmentation masks into topological road network graphs (nodes, edges, intersections).
Uses morphological thinning, 8-connectivity neighborhood analysis, and Shapely geometries.
"""

import cv2
import numpy as np
import networkx as nx
from shapely.geometry import LineString, MultiLineString, Polygon, mapping
from typing import Dict, List, Tuple, Any


def skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen morphological thinning algorithm to extract 1-pixel-wide road centerline skeletons.
    """
    mask = (binary_mask > 0).astype(np.uint8)
    prev = np.zeros(mask.shape, np.uint8)
    diff = None

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros(mask.shape, np.uint8)

    # Standard morphological erosion-based skeleton
    temp = np.empty(mask.shape, np.uint8)
    eroded = np.empty(mask.shape, np.uint8)

    while True:
        cv2.erode(mask, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(mask, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        mask = eroded.copy()
        if cv2.countNonZero(mask) == 0:
            break

    return skeleton


def extract_road_graph(skeleton_mask: np.ndarray, gsd_meters: float = 0.5) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
    """
    Constructs a topological NetworkX road graph from a skeletonized binary image.
    Calculates road lengths, identifies intersection nodes and terminal dead-ends.
    """
    G = nx.Graph()
    y_coords, x_coords = np.where(skeleton_mask > 0)
    if len(x_coords) == 0:
        return G, []

    # Map pixel coordinates to graph nodes
    pixels = set(zip(x_coords, y_coords))

    # Add edges between 8-connected neighbors
    for (x, y) in pixels:
        G.add_node((x, y), x=int(x), y=int(y))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in pixels:
                    dist_px = np.sqrt(dx ** 2 + dy ** 2)
                    dist_m = dist_px * gsd_meters
                    G.add_edge((x, y), neighbor, weight=dist_m, weight_px=dist_px)

    # Identify road segments (connected components)
    segments = []
    for i, comp in enumerate(nx.connected_components(G)):
        subgraph = G.subgraph(comp)
        total_length_meters = sum(data['weight'] for u, v, data in subgraph.edges(data=True))

        # Detect degree distribution (nodes with deg > 2 are intersections, deg == 1 are dead ends)
        degrees = dict(subgraph.degree())
        intersections = [node for node, deg in degrees.items() if deg > 2]
        terminals = [node for node, deg in degrees.items() if deg == 1]

        # Extract representative geometry (LineString)
        coords = [list(node) for node in comp]
        if len(coords) >= 2:
            # Sort approximate sequence by minimum spanning tree or coordinate sort
            sorted_coords = sorted(coords, key=lambda c: (c[0], c[1]))
            geom = LineString(sorted_coords)
        else:
            geom = None

        segments.append({
            "segment_id": i + 1,
            "num_nodes": len(comp),
            "num_edges": subgraph.number_of_edges(),
            "total_length_m": round(total_length_meters, 2),
            "total_length_km": round(total_length_meters / 1000.0, 3),
            "intersections_count": len(intersections),
            "terminals_count": len(terminals),
            "geometry": geom
        })

    # Sort segments by descending length
    segments.sort(key=lambda s: s["total_length_m"], reverse=True)
    return G, segments


def mask_to_polygons(binary_mask: np.ndarray, min_area: int = 20) -> List[Polygon]:
    """
    Vectorizes binary road mask into Shapely Polygons for GIS footprint representations.
    """
    contours, hierarchy = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        if len(cnt) >= 3:
            poly_coords = cnt.squeeze(axis=1)
            if len(poly_coords) >= 3:
                try:
                    poly = Polygon(poly_coords)
                    if poly.is_valid and not poly.is_empty:
                        polygons.append(poly)
                except Exception:
                    continue
    return polygons
