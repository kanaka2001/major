r"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Module: src/data/dataset.py

Production PyTorch Dataset, RGBA-Safe Preprocessing Pipeline,
and Temperature-Smoothed Long-Tail Sampler.

Features:
- RGBA alpha-channel composition on neutral white background
- Grayscale / Palette to 3-channel RGB expansion
- Botanical domain-specific data augmentations (affine rotation, flips, color jitter)
- WeightedRandomSampler with temperature-smoothed class weights P(c) \propto n_c^0.5
- Multi-attribute batch yielding (images, species_targets, part_targets, image_paths)

Author: Lead Deep Learning Architect
"""

import os
import csv
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def safe_load_rgb_image(image_path: str) -> Image.Image:
    """
    Safely opens and converts an image to a 3-channel RGB PIL Image:
    - RGBA images (169 files in dataset) are composited onto a neutral white background.
    - Grayscale ('L') or Palette ('P') images are expanded to 3 identical RGB channels.
    - CMYK images are converted to RGB.
    """
    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            # Neutral white background composite
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg
        elif img.mode != "RGB":
            return img.convert("RGB")
        else:
            return img.copy()

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Botanical domain-specific training augmentation pipeline:
    - RandomResizedCrop: simulates varying field viewing distances
    - RandomHorizontalFlip & RandomVerticalFlip: isotropic leaf orientations
    - RandomRotation (+-180 deg): plant foliage has no canonical 'up'
    - ColorJitter: simulates diurnal sunlight, overcast, and shaded environments
    - ToTensor & ImageNet normalization
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic validation / test / inference transformation pipeline:
    - Resize to (image_size, image_size)
    - ToTensor & ImageNet normalization
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

class MedicinalPlantDataset(Dataset):
    """
    Multi-Attribute PyTorch Dataset for Medicinal Plant Classification.
    Yields:
      - images: torch.FloatTensor of shape [3, 224, 224]
      - species_targets: torch.LongTensor scalar index [0 .. 1202]
      - part_targets: torch.LongTensor scalar index [0 .. 4]
      - image_paths: relative/absolute path string
    """
    def __init__(
        self,
        manifest_path: str,
        class_mapping_path: str,
        split: str = "train",
        workspace_root: str = None,
        transform: transforms.Compose = None,
        image_size: int = 224
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.workspace_root = workspace_root or os.path.abspath(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        self.image_size = image_size

        if transform is None:
            self.transform = (
                get_train_transforms(image_size)
                if split == "train"
                else get_eval_transforms(image_size)
            )
        else:
            self.transform = transform

        # Load class mappings
        with open(class_mapping_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.species_to_idx = self.metadata["species_to_idx"]
        self.idx_to_species = {int(k): v for k, v in self.metadata["idx_to_species"].items()}
        self.part_to_idx = self.metadata["part_to_idx"]
        self.idx_to_part = {int(k): v for k, v in self.metadata["idx_to_part"].items()}
        self.num_species = len(self.species_to_idx)
        self.num_parts = len(self.part_to_idx)

        # Load samples for this split from manifest CSV
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.samples.append({
                        "rel_path": row["rel_path"],
                        "species": row["species"],
                        "species_idx": int(row["species_idx"]),
                        "part": row["part"],
                        "part_idx": int(row["part_idx"])
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        full_path = os.path.join(self.workspace_root, item["rel_path"])
        
        try:
            pil_img = safe_load_rgb_image(full_path)
            img_tensor = self.transform(pil_img)
        except Exception as e:
            # Fallback zero tensor in rare I/O edge cases
            img_tensor = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

        species_target = torch.tensor(item["species_idx"], dtype=torch.long)
        part_target = torch.tensor(item["part_idx"], dtype=torch.long)
        image_path = item["rel_path"]

        return img_tensor, species_target, part_target, image_path

def build_temperature_smoothed_sampler(
    dataset: MedicinalPlantDataset,
    temperature: float = 0.5
) -> WeightedRandomSampler:
    r"""
    Builds a WeightedRandomSampler for the training split where class sampling
    probability is smoothed by exponent (temperature): P(c) \propto n_c^{temperature}.
    Each image i of class c receives sample weight:
      w_i = (n_c^{temperature - 1})
    When temperature = 0.5:
      w_i = 1 / sqrt(n_c)
    This boosts rare classes (e.g. Ginger, Ashwagandha) without starving majority classes (Neem, Tulsi).
    """
    # Count occurrences in the dataset
    class_counts = np.zeros(dataset.num_species, dtype=np.float64)
    for sample in dataset.samples:
        class_counts[sample["species_idx"]] += 1.0

    # Avoid zero division
    class_counts = np.maximum(class_counts, 1.0)

    # Per-class sampling weight for individual samples
    # w(c) = (n_c ** (temperature - 1))
    class_sample_weights = class_counts ** (temperature - 1.0)

    # Compute sample-level weights
    sample_weights = [
        class_sample_weights[sample["species_idx"]]
        for sample in dataset.samples
    ]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def get_dataloaders(
    workspace_root: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    temperature: float = 0.5,
    image_size: int = 224,
    pin_memory: bool = False
):
    """
    Factory function producing PyTorch DataLoaders for train, val, and test splits.
    """
    if workspace_root is None:
        workspace_root = os.path.abspath(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )

    manifest_path = os.path.join(workspace_root, "dataset", "splits_manifest.csv")
    class_mapping_path = os.path.join(workspace_root, "dataset", "class_mapping.json")

    train_ds = MedicinalPlantDataset(
        manifest_path=manifest_path,
        class_mapping_path=class_mapping_path,
        split="train",
        workspace_root=workspace_root,
        image_size=image_size
    )

    val_ds = MedicinalPlantDataset(
        manifest_path=manifest_path,
        class_mapping_path=class_mapping_path,
        split="val",
        workspace_root=workspace_root,
        image_size=image_size
    )

    test_ds = MedicinalPlantDataset(
        manifest_path=manifest_path,
        class_mapping_path=class_mapping_path,
        split="test",
        workspace_root=workspace_root,
        image_size=image_size
    )

    if use_weighted_sampler:
        train_sampler = build_temperature_smoothed_sampler(train_ds, temperature=temperature)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "metadata": train_ds.metadata
    }

if __name__ == "__main__":
    # Quick sanity check
    print("Testing MedicinalPlantDataset...")
    loaders = get_dataloaders(batch_size=4, num_workers=0)
    print(f"Train samples: {len(loaders['train'].dataset):,}")
    print(f"Val samples:   {len(loaders['val'].dataset):,}")
    print(f"Test samples:  {len(loaders['test'].dataset):,}")
    
    images, sp_targets, p_targets, paths = next(iter(loaders["train"]))
    print(f"Batch images shape:          {images.shape}")
    print(f"Batch species_targets shape: {sp_targets.shape}")
    print(f"Batch part_targets shape:    {p_targets.shape}")
    print(f"Sample paths:                {paths[:2]}")
