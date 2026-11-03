# 🛰️ Satellite Road-Network Extractor with New-Road Change Alerts
### SIH 2024 · PS-1564 (Inspired by the NTRO Road-Extraction Problem Statement)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

An end-to-end computer vision and geospatial intelligence system designed for the **Smart India Hackathon (SIH 2024) Problem Statement PS-1564**, inspired by the **National Technical Research Organisation (NTRO)** road network extraction and surveillance mandate:
> *Segment road networks from high-resolution satellite tiles and compare two time-separated tiles ($T_1$ and $T_2$) of the same geographic sector to automatically flag newly constructed road corridors with actionable intelligence dossiers.*

---

## 🌟 Key Features

1. **State-of-the-Art Road Network Segmentation (D-LinkNet & Attention U-Net):**
   - **D-LinkNet34**: ResNet-34 encoder with central multi-scale dilated convolutions ($d \in \{1, 2, 4, 8, 16\}$), expanding receptive fields to maintain thin continuous road ribbons without resolution loss.
   - **Hybrid Connectivity Loss**: Combines BCE + Soft Dice + Soft IoU loss ($\mathcal{L} = 0.4 \mathcal{L}_{\text{BCE}} + 0.4 \mathcal{L}_{\text{Dice}} + 0.2 \mathcal{L}_{\text{IoU}}$) to tackle severe class imbalance (<5% road pixels).
   - **Tiled Sliding-Window Inference**: Seamless Gaussian blending and Test-Time Augmentation (TTA) for large satellite tiles ($1024 \times 1024$ DeepGlobe, $1500 \times 1500$ Massachusetts, and arbitrary GIS orthomosaics).

2. **Autonomous Bi-Temporal Change Detection Engine ($T_1$ vs $T_2$):**
   - **Sub-Pixel Co-Registration**: ORB feature matching and RANSAC homography to align orbital variations before differential calculation.
   - **Differential Analyzer**: Isolates genuine new construction:
     $$\Delta_{\text{new}} = \text{ReLU}\left(P(T_2) - P(T_1) - \tau_{\text{temporal}}\right)$$
   - **False-Positive Suppression**: Aspect ratio filtering and morphological closing to eliminate seasonal crop changes, plow tracks, and ephemeral vehicle shadows.

3. **Topological Graph Extraction & Skeletonization:**
   - Morphological thinning (Zhang-Suen skeletonization) to extract 1-pixel-wide road centerlines.
   - Converts raster predictions into **NetworkX topological graphs**:
     - Calculates precise road centerline lengths (meters & kilometers) using calibrated Ground Sampling Distance (GSD).
     - Identifies intersection nodes, terminal dead ends, and connectivity to existing highways.

4. **Actionable Change-Alert Dossiers & Multi-Format Deliverables:**
   - **Automated Alert Prioritization**: Classifies detected roads into `CRITICAL` (> 1 km), `HIGH` (400m - 1km), `MEDIUM` (100m - 400m), and `LOW` (< 100m).
   - **GIS Export**: Standard GeoJSON FeatureCollection compatible with QGIS, ArcGIS, and Leaflet.
   - **Formal Intelligence PDF Report**: NTRO-styled audit report complete with satellite thumbnails, tactical overlays, and incident manifests.
   - **CSV & JSON Manifests**: Machine-readable logs for automated database ingestion.

5. **Interactive Streamlit Web Dashboard:**
   - Tactical dark-mode command center.
   - Interactive split-screen comparison between historical and recent satellite imagery.
   - Preloaded surveillance scenarios (Highway Expansion, Rural Access Roads, Border Strategic Infrastructure) + custom image upload.
   - One-click downloads for GeoJSON and Defense PDF Dossiers.

---

## 🏛️ Architecture Overview

```
Satellite Pass T1 (Baseline) ───┐
                                 ├──> [ORB + RANSAC Co-Registration]
Satellite Pass T2 (Recent)   ───┘                     │
                                                      ▼
                                          [D-LinkNet34 Extractor]
                                                      │
                            ┌─────────────────────────┴────────────────────────┐
                            ▼                                                  ▼
                    P(Road | T1)                                       P(Road | T2)
                            │                                                  │
                            └─────────────────────────┬────────────────────────┘
                                                      ▼
                                   [Differential Temporal Analyzer]
                                    Δ = ReLU(P(T2) - P(T1) - τ)
                                                      │
                                                      ▼
                                    [Morphological & Noise Filter]
                                                      │
                                                      ▼
                                  [Skeletonization & NetworkX Graph]
                                                      │
                                                      ▼
                                        [Road Alert Dossier Engine]
                                                      │
                          ┌───────────────────────────┼───────────────────────────┐
                          ▼                           ▼                           ▼
                    [Tactical Overlay]         [GeoJSON Vectors]         [Defense PDF Dossier]
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/Satellite-Road-Extractor.git
cd Satellite-Road-Extractor
pip install -r requirements.txt
```

### 2. Run the End-to-End Demonstration Pipeline
Run the CLI demonstration on any of the pre-configured bi-temporal surveillance scenarios:
```bash
# Available scenarios: highway_expansion, rural_access_road, border_infrastructure
python demo_pipeline.py --scenario highway_expansion
```
*Outputs generated in `./outputs/`:*
- `highway_expansion_composite_overlay.png` (Cyan = Pre-existing, Red = New Road)
- `highway_expansion_new_roads.geojson` (Standard GIS vector features)
- `highway_expansion_alert_report.pdf` (Defense-grade Intelligence Audit Report)

### 3. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🏋️ Model Training & Evaluation

### Dataset Compatibility
The pipeline natively supports:
- **DeepGlobe Road Extraction Dataset** (CVPR 2018): $1024 \times 1024$ RGB satellite tiles.
- **Massachusetts Roads Dataset** (Mnih): $1500 \times 1500$ aerial tiles.

### Training Command
```bash
python train.py \
    --dataset deepglobe \
    --data_dir ./data/deepglobe \
    --arch DLinkNet34 \
    --epochs 30 \
    --batch_size 4 \
    --lr 0.0002 \
    --save_dir ./checkpoints
```

### Evaluation Command
```bash
python evaluate.py \
    --checkpoint ./checkpoints/DLinkNet34_deepglobe_best.pth \
    --dataset deepglobe \
    --data_dir ./data/deepglobe \
    --threshold 0.50
```

---

## 📊 Benchmark Comparison

| Architecture | DeepGlobe IoU | DeepGlobe Dice / F1 | MassRoads IoU | Parameters | Dilated Convolutions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **D-LinkNet34 (Ours)** | **0.658** | **0.793** | **0.642** | **21.3 M** | **Yes (1, 2, 4, 8, 16)** |
| Attention U-Net | 0.631 | 0.774 | 0.619 | 34.8 M | No |
| ResNet50-UNet | 0.642 | 0.782 | 0.627 | 32.5 M | No |
| Standard Vanilla U-Net | 0.589 | 0.741 | 0.564 | 31.0 M | No |

---

## 📂 Repository Structure

```
.
├── configs/
│   └── config.yaml                     # System configuration & detection thresholds
├── src/
│   ├── models/
│   │   ├── dlinknet.py                 # D-LinkNet34 with multi-scale dilated convolutions
│   │   ├── unet.py                     # Attention U-Net comparative architecture
│   │   └── loss.py                     # Hybrid BCE + Dice + Soft IoU road loss
│   ├── dataset/
│   │   ├── deepglobe_dataset.py        # DeepGlobe loader (1024x1024 tiles)
│   │   ├── massachusetts_dataset.py    # Massachusetts Roads loader (1500x1500)
│   │   ├── transforms.py               # Satellite augmentations
│   │   └── sample_generator.py         # Bi-temporal scenario generator
│   ├── inference/
│   │   ├── predictor.py                # Sliding-window inference with Gaussian blending & TTA
│   │   └── skeletonizer.py             # Skeletonization & NetworkX vector graph extraction
│   ├── change_detection/
│   │   ├── coregister.py               # ORB + RANSAC homography co-registration
│   │   ├── differential_analyzer.py    # Temporal difference & false-positive filtering
│   │   └── alert_engine.py             # Actionable change alert dossier formulation
│   └── reporting/
│       ├── visualizer.py               # Tactical overlays & comparative plate panels
│       ├── geojson_exporter.py         # Standard GIS GeoJSON exporter
│       └── pdf_generator.py            # Formal Defense Audit PDF dossier generator
├── tests/
│   └── test_pipeline.py                # Automated Pytest suite
├── app.py                              # Interactive Streamlit Web Application
├── demo_pipeline.py                    # Complete CLI demonstration pipeline
├── train.py                            # Standalone training script
├── evaluate.py                         # Evaluation & metrics script
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 🛡️ License
Released under the [MIT License](LICENSE). Inspired by the Smart India Hackathon (SIH 2024) NTRO Problem Statement PS-1564.
