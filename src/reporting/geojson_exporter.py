"""
GeoJSON Vector Exporter
Exports extracted road network skeletons, intersections, and new-road change alerts into standard GeoJSON formats.
Compatible with QGIS, ESRI ArcGIS, Google Earth, and web GIS libraries (Leaflet, Mapbox).
"""

import json
from typing import Dict, Any, List
from shapely.geometry import mapping, LineString, Point


def export_change_alerts_geojson(
    alerts_dossier: Dict[str, Any],
    change_analysis: Dict[str, Any],
    output_path: str,
    base_lat: float = 28.6139,
    base_lon: float = 77.2090,
    gsd_meters: float = 0.5
) -> str:
    """
    Exports newly detected road segments as a GeoJSON FeatureCollection.
    Converts pixel coordinates to approximate geographic coordinates based on reference anchor.
    """
    # 1 degree latitude ~ 111,320 meters
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * 0.866  # roughly cos(30 deg)

    features = []
    alerts = alerts_dossier.get("alerts", [])
    segments = change_analysis.get("new_segments", [])

    for i, seg in enumerate(segments):
        alert_info = alerts[i] if i < len(alerts) else {}
        geom = seg.get("geometry")

        if geom is not None and isinstance(geom, LineString) and len(geom.coords) >= 2:
            geo_coords = []
            for px, py in geom.coords:
                # Convert pixel offset to geo coordinates
                d_lat = -(py * gsd_meters) / meters_per_deg_lat
                d_lon = (px * gsd_meters) / meters_per_deg_lon
                geo_coords.append([round(base_lon + d_lon, 6), round(base_lat + d_lat, 6)])

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": geo_coords
                },
                "properties": {
                    "alert_id": alert_info.get("alert_id", f"ALERT-{i+1}"),
                    "severity": alert_info.get("severity", "MEDIUM"),
                    "classification": alert_info.get("type", "New Road"),
                    "length_meters": seg.get("total_length_m", 0.0),
                    "length_km": seg.get("total_length_km", 0.0),
                    "intersections_count": seg.get("intersections_count", 0),
                    "status": alert_info.get("status", "NEW_CONSTRUCTION"),
                    "confidence": alert_info.get("confidence_score", 90.0)
                }
            }
            features.append(feature)

    geojson_doc = {
        "type": "FeatureCollection",
        "metadata": {
            "dossier_id": alerts_dossier.get("dossier_id"),
            "target_sector": alerts_dossier.get("target_sector"),
            "generated_at": alerts_dossier.get("generated_at"),
            "overall_severity": alerts_dossier.get("overall_severity"),
            "crs": "urn:ogc:def:crs:OGC:1.3:CRS84"
        },
        "features": features
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_doc, f, indent=2)

    return output_path
