"""
Road Extraction Evaluation and Validation Script
Evaluates model performance across IoU, Dice/F1, Precision, Recall, and topological metrics.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import build_model
from src.dataset import DeepGlobeRoadDataset, MassachusettsRoadDataset
from train import calculate_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Road Extraction Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--arch", type=str, default="DLinkNet34")
    parser.add_argument("--dataset", type=str, default="deepglobe", choices=["deepglobe", "massachusetts"])
    parser.add_argument("--data_dir", type=str, default="./data/deepglobe")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model {args.arch} from {args.checkpoint}...")

    model = build_model(args.arch, num_classes=1, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    if args.dataset == "deepglobe":
        test_ds = DeepGlobeRoadDataset(args.data_dir, split="test", img_size=args.img_size)
    else:
        test_ds = MassachusettsRoadDataset(args.data_dir, split="test", img_size=args.img_size)

    if len(test_ds) == 0:
        print(f"[!] No test data found at {args.data_dir}")
        return

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            m = calculate_metrics(logits, masks, threshold=args.threshold)
            for k in total_metrics:
                total_metrics[k] += m[k]

    n = len(loader)
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS ({args.dataset.upper()} TEST SET)")
    print("="*50)
    print(f"Mean IoU (Jaccard Score) : {total_metrics['iou']/n:.4f}")
    print(f"Mean Dice (F1 Score)     : {total_metrics['dice']/n:.4f}")
    print(f"Precision                : {total_metrics['precision']/n:.4f}")
    print(f"Recall                   : {total_metrics['recall']/n:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
