"""
New-Road Change Alert Engine
Generates structured, actionable intelligence alerts for defense & civil infrastructure monitoring.
Classifies alerts into CRITICAL, HIGH, MEDIUM, and LOW severity with geometric auditing metrics.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List


class RoadAlertEngine:
    def __init__(
        self,
        critical_thresh_m: float = 800.0,
        high_thresh_m: float = 300.0,
        medium_thresh_m: float = 80.0
    ):
        self.critical_thresh_m = critical_thresh_m
        self.high_thresh_m = high_thresh_m
        self.medium_thresh_m = medium_thresh_m

    def generate_alerts(
        self,
        change_analysis: Dict[str, Any],
        tile_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Processes differential analysis results and produces an Alert Dossier.
        """
        metadata = tile_metadata or {}
        scenario = metadata.get("scenario", "Surveillance Sector")
        center_lat = metadata.get("lat", 28.6139)
        center_lon = metadata.get("lon", 77.2090)

        new_segments = change_analysis.get("new_segments", [])
        total_len_m = change_analysis.get("total_new_length_m", 0.0)
        total_len_km = change_analysis.get("total_new_length_km", 0.0)
        total_area_m2 = change_analysis.get("total_new_area_m2", 0.0)

        # Determine overall threat/intelligence priority
        if total_len_m >= self.critical_thresh_m:
            overall_severity = "CRITICAL"
            classification = "Major Strategic Corridor / Arterial Highway Construction"
        elif total_len_m >= self.high_thresh_m:
            overall_severity = "HIGH"
            classification = "Secondary Paved Road Expansion / Bypass Extension"
        elif total_len_m >= self.medium_thresh_m:
            overall_severity = "MEDIUM"
            classification = "Access Spur / Rural Feeder Track Construction"
        elif total_len_m > 0:
            overall_severity = "LOW"
            classification = "Minor Trail / Preliminary Earthwork Clearing"
        else:
            overall_severity = "NONE"
            classification = "No Significant Road Network Modifications Detected"

        alerts_list = []
        for idx, seg in enumerate(new_segments):
            length_m = seg["total_length_m"]

            if length_m >= self.critical_thresh_m:
                seg_severity = "CRITICAL"
                seg_type = "Dual-Carriageway / Primary Highway Segment"
            elif length_m >= self.high_thresh_m:
                seg_severity = "HIGH"
                seg_type = "Paved Collector / Tactical Access Link"
            elif length_m >= self.medium_thresh_m:
                seg_severity = "MEDIUM"
                seg_type = "Secondary Branch Road"
            else:
                seg_severity = "LOW"
                seg_type = "Minor Spur / Track"

            alert_item = {
                "alert_id": f"NTRO-RD-{uuid.uuid4().hex[:8].upper()}",
                "segment_index": idx + 1,
                "severity": seg_severity,
                "type": seg_type,
                "length_meters": length_m,
                "length_km": seg["total_length_km"],
                "intersections": seg["intersections_count"],
                "terminals": seg["terminals_count"],
                "confidence_score": min(98.5, max(85.0, 90.0 + (length_m / 100.0))),
                "status": "UNAUTHORIZED_NEW_CONSTRUCTION" if seg_severity in ["CRITICAL", "HIGH"] else "NEW_ROAD_FLAGGED"
            }
            alerts_list.append(alert_item)

        dossier = {
            "dossier_id": f"DOSSIER-{uuid.uuid4().hex[:6].upper()}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "target_sector": scenario,
            "center_coordinates": f"{center_lat:.4f}° N, {center_lon:.4f}° E",
            "overall_severity": overall_severity,
            "classification": classification,
            "summary_metrics": {
                "total_new_roads_count": len(new_segments),
                "total_new_length_km": total_len_km,
                "total_new_length_meters": total_len_m,
                "total_new_road_area_m2": total_area_m2,
                "critical_segments": sum(1 for a in alerts_list if a["severity"] == "CRITICAL"),
                "high_segments": sum(1 for a in alerts_list if a["severity"] == "HIGH"),
                "medium_segments": sum(1 for a in alerts_list if a["severity"] == "MEDIUM"),
                "low_segments": sum(1 for a in alerts_list if a["severity"] == "LOW"),
            },
            "alerts": alerts_list
        }

        return dossier
