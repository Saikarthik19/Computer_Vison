"""
End-to-End Demonstration Pipeline
SIH 2024 (PS-1564) Satellite Road-Network Extractor with New-Road Change Alerts.
Runs complete extraction, differential analysis, alert creation, and PDF/GeoJSON generation.
"""

import os
import cv2
import json
import argparse
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

from src.dataset import generate_all_sample_scenarios
from src.inference import RoadPredictor
from src.change_detection import coregister_tiles, RoadDifferentialAnalyzer, RoadAlertEngine
from src.reporting import (
    create_change_detection_composite,
    create_side_by_side_panel,
    export_change_alerts_geojson,
    generate_change_alert_pdf
)


def run_pipeline(
    scenario: str = "highway_expansion",
    output_dir: str = "./outputs",
    model_name: str = "DLinkNet34",
    weights_path: str = None
):
    print("=" * 75)
    print(" SIH 2024 · PS-1564 SATELLITE ROAD EXTRACTOR & CHANGE ALERT SYSTEM")
    print(" Inspired by NTRO High-Resolution Road Network Extraction & Monitoring")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    sample_dir = "./sample_data"

    # Step 1: Prepare Bi-Temporal Scenario
    print(f"\n[1/6] Preparing Bi-Temporal Surveillance Scenario: '{scenario}'...")
    all_scenarios = generate_all_sample_scenarios(output_dir=sample_dir)
    scenario_files = all_scenarios[scenario]

    t1_rgb = cv2.cvtColor(cv2.imread(scenario_files["t1_sat"]), cv2.COLOR_BGR2RGB)
    t2_rgb = cv2.cvtColor(cv2.imread(scenario_files["t2_sat"]), cv2.COLOR_BGR2RGB)
    print(f"  --> Loaded T1 Tile: {t1_rgb.shape} | T2 Tile: {t2_rgb.shape}")

    # Step 2: Co-Registration
    print("\n[2/6] Verifying Geometric Co-Registration between T1 and T2...")
    warped_t2, coreg_stats = coregister_tiles(t1_rgb, t2_rgb)
    print(f"  --> Status: {coreg_stats['message']}")

    # Step 3: Road Network Extraction
    print(f"\n[3/6] Running Deep Road-Network Extractor ({model_name})...")
    predictor = RoadPredictor(model_name=model_name, weights_path=weights_path, gsd_meters=0.5)

    print("  --> Extracting Road Network from T1 (Historical Baseline)...")
    res_t1 = predictor.predict_tile(t1_rgb)
    print(f"      Identified: {res_t1['num_segments']} segments | Length: {res_t1['total_road_length_km']} km | Intersections: {res_t1['num_intersections']}")

    print("  --> Extracting Road Network from T2 (Recent Observation)...")
    res_t2 = predictor.predict_tile(warped_t2)
    print(f"      Identified: {res_t2['num_segments']} segments | Length: {res_t2['total_road_length_km']} km | Intersections: {res_t2['num_intersections']}")

    # Step 4: Differential Analysis
    print("\n[4/6] Computing Bi-Temporal Differential Analysis (T2 \\ T1)...")
    analyzer = RoadDifferentialAnalyzer(min_component_area_px=50, min_road_length_m=30.0, gsd_meters=0.5)
    diff_res = analyzer.detect_new_roads(res_t1, res_t2)
    print(f"  --> Flagged {diff_res['num_new_segments']} new road segments!")
    print(f"  --> Cumulative New Road Construction: {diff_res['total_new_length_km']} km ({diff_res['total_new_length_m']} m)")
    print(f"  --> New Pavement Footprint: {diff_res['total_new_area_m2']} sq.m")

    # Step 5: Alert Generation
    print("\n[5/6] Generating Intelligence Alert Dossier...")
    alert_engine = RoadAlertEngine()
    metadata = {
        "scenario": scenario.replace("_", " ").title(),
        "lat": 28.6139,
        "lon": 77.2090
    }
    dossier = alert_engine.generate_alerts(diff_res, tile_metadata=metadata)

    # Step 6: Visualizations & Exports
    print("\n[6/6] Generating Visual Overlays, GeoJSON & Formal PDF Dossier...")
    composite = create_change_detection_composite(
        t1_rgb, warped_t2, res_t1["binary_mask"], res_t2["binary_mask"],
        diff_res["new_roads_mask"], diff_res["new_skeleton"]
    )
    panel = create_side_by_side_panel(
        t1_rgb, warped_t2, res_t1["binary_mask"], res_t2["binary_mask"], composite
    )

    composite_path = os.path.join(output_dir, f"{scenario}_composite_overlay.png")
    panel_path = os.path.join(output_dir, f"{scenario}_audit_panel.png")
    geojson_path = os.path.join(output_dir, f"{scenario}_new_roads.geojson")
    pdf_path = os.path.join(output_dir, f"{scenario}_alert_report.pdf")
    json_path = os.path.join(output_dir, f"{scenario}_dossier.json")

    cv2.imwrite(composite_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    cv2.imwrite(panel_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    # Export GeoJSON
    export_change_alerts_geojson(dossier, diff_res, geojson_path, gsd_meters=0.5)
    print(f"  --> GeoJSON saved: {geojson_path}")

    # Export PDF Report
    generate_change_alert_pdf(dossier, composite_path, pdf_path)
    print(f"  --> Intelligence PDF Dossier saved: {pdf_path}")

    with open(json_path, "w") as f:
        json.dump(dossier, f, indent=2)

    # Print Summary Terminal Table
    print("\n" + "=" * 75)
    print(f" MISSION INTELLIGENCE SUMMARY: {dossier['overall_severity']} ALERT")
    print("=" * 75)
    print(f"Dossier ID        : {dossier['dossier_id']}")
    print(f"Target Sector     : {dossier['target_sector']}")
    print(f"Classification    : {dossier['classification']}")
    print(f"Total New Roads   : {diff_res['num_new_segments']} segments")
    print(f"Total New Length  : {diff_res['total_new_length_km']} km ({diff_res['total_new_length_m']} m)")
    print(f"Total Area Added  : {diff_res['total_new_area_m2']} sq.m")
    print("-" * 75)
    print(f"{'ALERT ID':<18} | {'SEVERITY':<10} | {'LENGTH (m)':<12} | {'CONFIDENCE':<10} | {'CLASSIFICATION'}")
    print("-" * 75)
    for alert in dossier["alerts"]:
        print(f"{alert['alert_id']:<18} | {alert['severity']:<10} | {alert['length_meters']:<12.1f} | {alert['confidence_score']:<10.1f}% | {alert['type']}")
    print("=" * 75)
    print("\n[OK] Pipeline execution finished successfully!")
    print(f"    - Tactical Overlay : {composite_path}")
    print(f"    - Multi-Pane Plate : {panel_path}")
    print(f"    - GeoJSON Vectors  : {geojson_path}")
    print(f"    - Defense PDF Dossier : {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run End-to-End Road Change Alert Pipeline")
    parser.add_argument("--scenario", type=str, default="highway_expansion",
                        choices=["highway_expansion", "rural_access_road", "border_infrastructure"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--model", type=str, default="DLinkNet34")
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    run_pipeline(scenario=args.scenario, output_dir=args.output_dir, model_name=args.model, weights_path=args.weights)
