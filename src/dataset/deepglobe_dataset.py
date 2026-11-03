"""
DeepGlobe Road Extraction Dataset Loader
Reference: CVPR 2018 DeepGlobe Challenge
Each sample consists of:
  - Image: {id}_sat.jpg (1024x1024 RGB satellite imagery)
  - Mask:  {id}_mask.png (1024x1024 RGB where white (255, 255, 255) indicates road)
"""

import os
import glob
from typing import Optional, List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import get_train_transforms, get_val_transforms


class DeepGlobeRoadDataset(Dataset):
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
        if not os.path.exists(self.root_dir):
            return samples

        # Look for *_sat.jpg or standard image naming
        sat_images = sorted(glob.glob(os.path.join(self.root_dir, "*_sat.jpg")))
        if not sat_images:
            sat_images = sorted(glob.glob(os.path.join(self.root_dir, "images", "*.png")) +
                                glob.glob(os.path.join(self.root_dir, "images", "*.jpg")))

        for sat_path in sat_images:
            base_dir = os.path.dirname(sat_path)
            fname = os.path.basename(sat_path)

            if "_sat.jpg" in fname:
                mask_name = fname.replace("_sat.jpg", "_mask.png")
                mask_path = os.path.join(base_dir, mask_name)
            else:
                mask_path = os.path.join(base_dir.replace("images", "masks"), fname.replace(".jpg", ".png"))

            if os.path.exists(mask_path):
                samples.append((sat_path, mask_path))
            else:
                samples.append((sat_path, None))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sat_path, mask_path = self.samples[idx]

        # Load RGB satellite image
        image = cv2.imread(sat_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read satellite image at {sat_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load Mask if available
        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 128).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Apply Albumentations
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
            "path": sat_path
        }
