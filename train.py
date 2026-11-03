"""
Training Pipeline for Satellite Road-Network Extraction
Supports DeepGlobe Road Extraction & Massachusetts Roads Datasets.
Optimizes hybrid BCE + Dice + Soft IoU Loss using AdamW + CosineAnnealingLR.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.models import build_model, CombinedRoadLoss
from src.dataset import DeepGlobeRoadDataset, MassachusettsRoadDataset, get_train_transforms, get_val_transforms


def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """Computes IoU (Jaccard), Dice (F1), Precision, and Recall."""
    probs = torch.sigmoid(preds)
    bin_preds = (probs > threshold).float()
    targets = (targets > 0.5).float()

    intersection = (bin_preds * targets).sum().item()
    total_preds = bin_preds.sum().item()
    total_targets = targets.sum().item()

    union = total_preds + total_targets - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2.0 * intersection + 1e-6) / (total_preds + total_targets + 1e-6)
    precision = (intersection + 1e-6) / (total_preds + 1e-6)
    recall = (intersection + 1e-6) / (total_targets + 1e-6)

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, len(dataloader))


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics_sum = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            m = calculate_metrics(logits, masks)
            for k in metrics_sum:
                metrics_sum[k] += m[k]

    n = max(1, len(dataloader))
    avg_metrics = {k: v / n for k, v in metrics_sum.items()}
    avg_metrics["loss"] = total_loss / n
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Road Extraction Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default="deepglobe", choices=["deepglobe", "massachusetts"])
    parser.add_argument("--data_dir", type=str, default="./data/deepglobe")
    parser.add_argument("--arch", type=str, default="DLinkNet34", choices=["DLinkNet34", "AttentionUNet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training {args.arch} on {args.dataset} dataset | Device: {device}")

    # Build Model
    model = build_model(args.arch, num_classes=1, pretrained=True).to(device)
    criterion = CombinedRoadLoss(bce_weight=0.4, dice_weight=0.4, iou_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Dataset & Dataloaders
    if args.dataset == "deepglobe":
        train_ds = DeepGlobeRoadDataset(args.data_dir, split="train", img_size=args.img_size)
        val_ds = DeepGlobeRoadDataset(args.data_dir, split="val", img_size=args.img_size)
    else:
        train_ds = MassachusettsRoadDataset(args.data_dir, split="train", img_size=args.img_size)
        val_ds = MassachusettsRoadDataset(args.data_dir, split="val", img_size=args.img_size)

    if len(train_ds) == 0:
        print(f"[!] Warning: No images found at {args.data_dir}. Check dataset folder structure.")
        print("[*] To test pipeline execution without full dataset, run: python demo_pipeline.py")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_res = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_res['loss']:.4f} | "
              f"IoU: {val_res['iou']:.4f} | Dice: {val_res['dice']:.4f} | "
              f"Prec: {val_res['precision']:.4f} | Rec: {val_res['recall']:.4f}")

        # Save Best Checkpoint
        if val_res["iou"] > best_iou:
            best_iou = val_res["iou"]
            ckpt_path = os.path.join(args.save_dir, f"{args.arch}_{args.dataset}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "arch": args.arch
            }, ckpt_path)
            print(f"  --> Saved new best checkpoint with IoU {best_iou:.4f} to {ckpt_path}")


if __name__ == "__main__":
    main()
