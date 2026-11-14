#!/usr/bin/env python3
"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Script: scripts/train.py

Production CLI Training Entrypoint for Hybrid CNN–ViT Medicinal Plant System:
Supports:
  python scripts/train.py --epochs 30 --batch-size auto --amp --patience 7
  python scripts/train.py --smoke-test (for fast 1-epoch pilot validation)
  python scripts/train.py --resume checkpoints/last_checkpoint.pth

Author: Lead Deep Learning Architect
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.data.dataset import get_dataloaders
from src.models.hybrid_cnn_vit import HybridCNNViT
from src.training.losses import MultiAttributeMedicinalLoss
from src.training.trainer import MedicinalPlantTrainer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Hybrid CNN–ViT Medicinal Plant Recognition System"
    )
    # Training Loop
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs (default: 30)")
    parser.add_argument("--batch-size", type=str, default="auto", help="Batch size or 'auto' (default: auto)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate (default: 3e-4)")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate (default: 1e-6)")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs (default: 3)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (default: True)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience epochs (default: 7)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: 'auto', 'cuda', 'cpu', 'mps'")

    # Model Architecture
    parser.add_argument("--cnn-backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "convnext_tiny"], help="CNN micro-texture backbone")
    parser.add_argument("--vit-embed-dim", type=int, default=384, help="ViT embedding dimension (default: 384)")
    parser.add_argument("--vit-depth", type=int, default=6, help="ViT transformer depth (default: 6)")
    parser.add_argument("--fusion-dim", type=int, default=512, help="Cross-attention fusion dimension (default: 512)")
    parser.add_argument("--drop-rate", type=float, default=0.2, help="Dropout probability (default: 0.2)")

    # Data & Sampling
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature P(c) proportional to n_c^T (default: 0.5)")
    parser.add_argument("--num-workers", type=str, default="auto", help="DataLoader num_workers or 'auto'")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution (default: 224)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Execution Modes & Resumption
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch pilot validation test with limited batches")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Limit training batches per epoch")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="Limit validation batches per epoch")

    return parser.parse_args()

def auto_detect_batch_size(device: torch.device) -> int:
    """
    Selects optimal batch size based on available VRAM / RAM:
    - >= 12 GB VRAM: 64
    - >= 8 GB VRAM: 32
    - < 8 GB VRAM or CPU / MPS: 16
    """
    if device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_vram_gb >= 12.0:
            return 64
        elif total_vram_gb >= 8.0:
            return 32
        else:
            return 16
    else:
        # CPU / MPS
        return 16

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("VTU Major Project: Hybrid CNN–ViT Production Training Engine (Phase 7)")
    print("=" * 80)

    # 1. Device Resolution & Hardware Census
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            device_desc = f"CUDA ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/(1024**3):.2f} GB VRAM)"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_desc = "Apple Silicon MPS"
        else:
            device = torch.device("cpu")
            device_desc = f"CPU ({os.cpu_count()} logical cores)"
    else:
        device = torch.device(args.device)
        device_desc = str(device)

    print(f"Hardware Detected: {device_desc}")

    # 2. Batch Size & Workers Resolution
    if args.batch_size.lower() == "auto":
        batch_size = auto_detect_batch_size(device)
        print(f"  -> Batch size auto-configured: {batch_size}")
    else:
        batch_size = int(args.batch_size)

    if args.num_workers.lower() == "auto":
        cpu_cores = os.cpu_count() or 2
        num_workers = min(8, max(2, cpu_cores // 2))
        if device.type == "cpu":
            num_workers = 2 # prevent CPU core over-subscription during CPU training
    else:
        num_workers = int(args.num_workers)

    pin_memory = (device.type == "cuda")

    # Smoke Test Adjustments
    if args.smoke_test:
        print("\n>>> SMOKE TEST MODE ACTIVATED (1-Epoch Pilot Health Check) <<<")
        args.epochs = 1
        args.warmup_epochs = 1
        if args.limit_train_batches is None:
            args.limit_train_batches = 10
        if args.limit_val_batches is None:
            args.limit_val_batches = 5

    # 3. DataLoaders
    print("\n[Step 1/4] Initializing DataLoaders & Class Mapping...")
    loaders = get_dataloaders(
        workspace_root=WORKSPACE_ROOT,
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=True,
        temperature=args.temperature,
        image_size=args.image_size,
        pin_memory=pin_memory
    )
    metadata = loaders["metadata"]
    num_species = metadata["num_species"]
    num_parts = metadata["num_parts"]
    idx_to_species = {int(k): v for k, v in metadata["idx_to_species"].items()}

    print(f"  -> Total Species Classes: {num_species:,}")
    print(f"  -> Total Organ Classes:   {num_parts}")
    print(f"  -> Train Samples:         {len(loaders['train'].dataset):,}")
    print(f"  -> Val Samples:           {len(loaders['val'].dataset):,}")

    # 4. Model Scaffolding
    print("\n[Step 2/4] Instantiating Hybrid CNN–ViT Architecture from Scratch...")
    model = HybridCNNViT(
        num_species=num_species,
        num_parts=num_parts,
        cnn_backbone=args.cnn_backbone,
        vit_embed_dim=args.vit_embed_dim,
        vit_depth=args.vit_depth,
        vit_heads=6,
        fusion_dim=args.fusion_dim,
        drop_rate=args.drop_rate
    )
    summary = model.get_model_summary()
    print(f"  -> Model Parameters:      {summary['total_parameters']:,} (100% trainable)")
    print(f"  -> CNN Backbone:          {summary['cnn_backbone']}")
    print(f"  -> ViT Embed Dim:         {summary['vit_embed_dim']} (6 blocks, 196 patches)")

    # 5. Multi-Attribute Class-Balanced Loss
    print("\n[Step 3/4] Configuring Class-Balanced Focal Loss (beta=0.9999, gamma=2.0)...")
    training_counts = [metadata["class_counts_train"][idx_to_species[i]] for i in range(num_species)]
    loss_fn = MultiAttributeMedicinalLoss(
        species_counts=training_counts,
        num_species=num_species,
        num_parts=num_parts,
        beta=0.9999,
        gamma=2.0,
        organ_loss_weight=0.2
    )

    # 6. Production Trainer Initialization & Execution
    trainer_config = {
        "device": device.type,
        "epochs": args.epochs,
        "base_lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip,
        "amp": args.amp,
        "patience": args.patience,
        "checkpoint_dir": args.checkpoint_dir,
        "log_interval": 5 if args.smoke_test else 50,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "batch_size": batch_size,
        "hardware_desc": device_desc
    }

    print("\n[Step 4/4] Launching Production Trainer Engine...")
    trainer = MedicinalPlantTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        loss_fn=loss_fn,
        config=trainer_config,
        class_mapping=metadata
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    results = trainer.train()

    print("\nExecution complete.")
    return results

if __name__ == "__main__":
    main()
