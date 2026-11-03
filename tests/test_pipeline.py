"""
Automated Unit Tests for Satellite Road Extractor & Change Alert System
"""

import os
import cv2
import numpy as np
import pytest
import torch

from src.models import build_model, CombinedRoadLoss
from src.inference import RoadPredictor, skeletonize, extract_road_graph
from src.change_detection import coregister_tiles, RoadDifferentialAnalyzer, RoadAlertEngine
from src.reporting import export_change_alerts_geojson, generate_change_alert_pdf
from src.dataset import generate_bi_temporal_scenario


def test_model_forward():
    model = build_model("DLinkNet34", num_classes=1, pretrained=False)
    dummy_input = torch.randn(1, 3, 256, 256)
    out = model(dummy_input)
    assert out.shape == (1, 1, 256, 256), f"Unexpected shape {out.shape}"


def test_loss_functions():
    criterion = CombinedRoadLoss()
    logits = torch.randn(2, 1, 128, 128)
    targets = (torch.rand(2, 1, 128, 128) > 0.8).float()
    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_scenario_generation(tmp_path):
    paths = generate_bi_temporal_scenario("highway_expansion", img_size=256, output_dir=str(tmp_path))
    assert os.path.exists(paths["t1_sat"])
    assert os.path.exists(paths["t2_sat"])
    assert os.path.exists(paths["gt_change"])


def test_end_to_end_change_detection(tmp_path):
    # 1. Generate small scenario
    paths = generate_bi_temporal_scenario("highway_expansion", img_size=256, output_dir=str(tmp_path))
    t1_rgb = cv2.cvtColor(cv2.imread(paths["t1_sat"]), cv2.COLOR_BGR2RGB)
    t2_rgb = cv2.cvtColor(cv2.imread(paths["t2_sat"]), cv2.COLOR_BGR2RGB)

    # 2. Co-registration
    warped_t2, stats = coregister_tiles(t1_rgb, t2_rgb)
    assert warped_t2.shape == t1_rgb.shape

    # 3. Predictor
    predictor = RoadPredictor(model_name="DLinkNet34", patch_size=256, gsd_meters=0.5)
    res_t1 = predictor.predict_tile(t1_rgb)
    res_t2 = predictor.predict_tile(warped_t2)

    assert "probability_map" in res_t1
    assert "binary_mask" in res_t2

    # 4. Differential Analyzer
    analyzer = RoadDifferentialAnalyzer(min_component_area_px=20, min_road_length_m=10.0, gsd_meters=0.5)
    diff_res = analyzer.detect_new_roads(res_t1, res_t2)
    assert "new_roads_mask" in diff_res

    # 5. Alert Engine
    engine = RoadAlertEngine()
    dossier = engine.generate_alerts(diff_res, {"scenario": "Test Sector"})
    assert "dossier_id" in dossier
    assert "overall_severity" in dossier

    # 6. GeoJSON export
    geojson_path = os.path.join(tmp_path, "test_alerts.geojson")
    export_change_alerts_geojson(dossier, diff_res, geojson_path, gsd_meters=0.5)
    assert os.path.exists(geojson_path)
    assert os.path.getsize(geojson_path) > 10

    # 7. PDF export
    pdf_path = os.path.join(tmp_path, "test_dossier.pdf")
    temp_img = os.path.join(tmp_path, "temp.png")
    cv2.imwrite(temp_img, t1_rgb)
    generate_change_alert_pdf(dossier, temp_img, pdf_path)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 100
