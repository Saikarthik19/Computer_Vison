"""
Massachusetts Roads Dataset Loader
Reference: Volodymyr Mnih, PhD Thesis (2013) "Machine Learning for Aerial Image Labeling"
Image tiles: 1500x1500px, 1m/pixel GSD.
Directory structure:
  root_dir/
    train/
    train_labels/
    val/
    val_labels/
    test/
    test_labels/
"""

import os
import glob
from typing import Optional, List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import get_train_transforms, get_val_transforms


class MassachusettsRoadDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        img_size: int = 512,
        max_samples: Optional[int] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform or (get_train_transforms(img_size) if split == "train" else get_val_transforms(img_size))

        self.samples = self._load_samples()
        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    def _load_samples(self) -> List[Tuple[str, Optional[str]]]:
        samples = []
        img_dir = os.path.join(self.root_dir, self.split)
        label_dir = os.path.join(self.root_dir, f"{self.split}_labels")

        if not os.path.exists(img_dir):
            return samples

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.tiff")) +
                           glob.glob(os.path.join(img_dir, "*.tif")) +
                           glob.glob(os.path.join(img_dir, "*.png")) +
                           glob.glob(os.path.join(img_dir, "*.jpg")))

        for img_path in img_files:
            fname = os.path.basename(img_path)
            # Massachusetts dataset label files have matching base names (often .tif or .png)
            base_no_ext = os.path.splitext(fname)[0]
            label_candidates = [
                os.path.join(label_dir, f"{base_no_ext}.tif"),
                os.path.join(label_dir, f"{base_no_ext}.tiff"),
                os.path.join(label_dir, f"{base_no_ext}.png")
            ]
            mask_path = None
            for cand in label_candidates:
                if os.path.exists(cand):
                    mask_path = cand
                    break

            samples.append((img_path, mask_path))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 128).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "path": img_path
        }
