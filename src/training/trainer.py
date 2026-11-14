"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Module: src/training/trainer.py

Production-Grade, Crash-Resilient MedicinalPlantTrainer:
- Dynamic hardware & compute auto-configuration (CUDA FP16 AMP / MPS / CPU fallback)
- AdamW with decoupled weight decay (excluding biases & LayerNorm/BatchNorm affine params)
- 3-Epoch Linear Warmup + Cosine Annealing learning rate schedule
- Gradient clipping (max_norm = 1.0) for long-tail focal loss backpropagation stability
- Multi-task loss: Class-Balanced Focal Loss (species) + 0.2 * CrossEntropy (5 organs)
- Exact Top-1, Top-5, Organ accuracy, and Macro F1 evaluation across all 1,203 species
- Resilient checkpointing (best_model.pth, last_checkpoint.pth) & Early Stopping
- Persistent CSV metric tracking (checkpoints/training_metrics.csv)

Author: Lead Deep Learning Architect
"""

import os
import sys
import csv
import time
import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicinalPlantTrainer:
    """
    Industrial-Grade Production Trainer for Hybrid CNN–ViT Medicinal Plant Architecture.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        config: Dict[str, Any],
        class_mapping: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.class_mapping = class_mapping or {}
        
        # 1. Device & Compute Setup
        self.device = self._setup_device(config.get("device", "auto"))
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 2. Mixed Precision (AMP)
        self.use_amp = (self.device.type == "cuda" and config.get("amp", True))
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
            print("  -> Enabled Automatic Mixed Precision (AMP) via torch.amp.GradScaler('cuda').")
        else:
            self.scaler = None
            if self.device.type == "cuda":
                print("  -> AMP disabled by config; running in standard FP32 mode.")

        # 3. Optimizer with Parameter Group Decoupled Weight Decay
        self.base_lr = float(config.get("base_lr", 3e-4))
        self.min_lr = float(config.get("min_lr", 1e-6))
        self.weight_decay = float(config.get("weight_decay", 0.01))
        self.optimizer = self._build_optimizer()

        # 4. Schedule & Hyperparameters
        self.epochs = int(config.get("epochs", 30))
        self.warmup_epochs = int(config.get("warmup_epochs", 3))
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
        self.patience = int(config.get("patience", 7))
        self.log_interval = int(config.get("log_interval", 50))
        self.limit_train_batches = config.get("limit_train_batches", None)
        self.limit_val_batches = config.get("limit_val_batches", None)

        # 5. Checkpointing & State Tracking
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.last_checkpoint_path = os.path.join(self.checkpoint_dir, "last_checkpoint.pth")
        self.csv_log_path = os.path.join(self.checkpoint_dir, "training_metrics.csv")

        self.start_epoch = 1
        self.best_val_f1 = 0.0
        self.best_val_top1 = 0.0
        self.patience_counter = 0
        self.history = []

        self._init_csv_logger()

    def _setup_device(self, device_pref: str) -> torch.device:
        if device_pref == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                torch.backends.cudnn.benchmark = True
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_pref)
        return device

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Decouples weight decay: applies weight decay (0.01) only to 2D/4D weight matrices
        (Linear and Conv layers), excluding 1D biases and LayerNorm/BatchNorm affine params.
        """
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower() or "pos_embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.base_lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        return optimizer

    def get_current_lr(self, epoch: int) -> float:
        """
        Computes learning rate for current epoch:
        - Epoch 1..warmup_epochs: Linear scaling from min_lr (1e-6) to base_lr (3e-4)
        - Epoch warmup_epochs+1..epochs: Cosine Annealing down to min_lr
        """
        if epoch <= self.warmup_epochs:
            # Linear warmup
            progress = epoch / max(1, self.warmup_epochs)
            lr = self.min_lr + progress * (self.base_lr - self.min_lr)
        else:
            # Cosine decay
            total_decay_epochs = max(1, self.epochs - self.warmup_epochs)
            current_decay_epoch = epoch - self.warmup_epochs
            progress = current_decay_epoch / total_decay_epochs
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + cosine_factor * (self.base_lr - self.min_lr)
        return lr

    def _update_lr(self, epoch: int) -> float:
        lr = self.get_current_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _init_csv_logger(self):
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss_total",
                    "train_loss_species",
                    "train_loss_organ",
                    "val_loss_total",
                    "val_loss_species",
                    "val_loss_organ",
                    "val_top1_acc",
                    "val_top5_acc",
                    "val_organ_acc",
                    "val_macro_f1",
                    "learning_rate",
                    "elapsed_seconds"
                ])

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        current_lr = self._update_lr(epoch)

        total_loss_sum = 0.0
        species_loss_sum = 0.0
        organ_loss_sum = 0.0
        batches_processed = 0

        t0 = time.time()
        num_batches = len(self.train_loader)
        if self.limit_train_batches is not None:
            num_batches = min(num_batches, self.limit_train_batches)

        print(f"\n--- Epoch [{epoch}/{self.epochs}] Training (LR: {current_lr:.6e}) ---")

        for batch_idx, (images, sp_targets, organ_targets, _) in enumerate(self.train_loader, 1):
            if self.limit_train_batches is not None and batch_idx > self.limit_train_batches:
                break

            images = images.to(self.device, non_blocking=True)
            sp_targets = sp_targets.to(self.device, non_blocking=True)
            organ_targets = organ_targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    out = self.model(images)
                    losses = self.loss_fn(
                        out["species_logits"], sp_targets,
                        out["organ_logits"], organ_targets
                    )
                    loss = losses["loss_total"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(images)
                losses = self.loss_fn(
                    out["species_logits"], sp_targets,
                    out["organ_logits"], organ_targets
                )
                loss = losses["loss_total"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()

            total_loss_sum += loss.item()
            species_loss_sum += losses["loss_species"].item()
            organ_loss_sum += losses["loss_organ"].item()
            batches_processed += 1

            if batch_idx % self.log_interval == 0 or batch_idx == num_batches:
                elapsed = time.time() - t0
                batches_per_sec = batches_processed / max(elapsed, 1e-4)
                print(
                    f"  Batch [{batch_idx:5d}/{num_batches:5d}] | "
                    f"Loss: {loss.item():.4f} (Species: {losses['loss_species'].item():.4f}, Organ: {losses['loss_organ'].item():.4f}) | "
                    f"Speed: {batches_per_sec:.2f} batch/s"
                )

        avg_total = total_loss_sum / max(batches_processed, 1)
        avg_species = species_loss_sum / max(batches_processed, 1)
        avg_organ = organ_loss_sum / max(batches_processed, 1)

        return {
            "train_loss_total": avg_total,
            "train_loss_species": avg_species,
            "train_loss_organ": avg_organ,
            "learning_rate": current_lr,
            "train_time_sec": time.time() - t0
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()

        total_loss_sum = 0.0
        species_loss_sum = 0.0
        organ_loss_sum = 0.0
        batches_processed = 0

        correct_top1 = 0
        correct_top5 = 0
        correct_organ = 0
        total_samples = 0

        num_species = self.model.num_species
        # Matrix for macro F1: confusion matrix [C, C]
        conf_mat = torch.zeros((num_species, num_species), dtype=torch.int64, device=self.device)

        t0 = time.time()
        num_batches = len(self.val_loader)
        if self.limit_val_batches is not None:
            num_batches = min(num_batches, self.limit_val_batches)

        print(f"--- Epoch [{epoch}/{self.epochs}] Validation ---")

        with torch.no_grad():
            for batch_idx, (images, sp_targets, organ_targets, _) in enumerate(self.val_loader, 1):
                if self.limit_val_batches is not None and batch_idx > self.limit_val_batches:
                    break

                images = images.to(self.device, non_blocking=True)
                sp_targets = sp_targets.to(self.device, non_blocking=True)
                organ_targets = organ_targets.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        out = self.model(images)
                        losses = self.loss_fn(
                            out["species_logits"], sp_targets,
                            out["organ_logits"], organ_targets
                        )
                else:
                    out = self.model(images)
                    losses = self.loss_fn(
                        out["species_logits"], sp_targets,
                        out["organ_logits"], organ_targets
                    )

                loss = losses["loss_total"]
                total_loss_sum += loss.item()
                species_loss_sum += losses["loss_species"].item()
                organ_loss_sum += losses["loss_organ"].item()
                batches_processed += 1

                B = sp_targets.size(0)
                total_samples += B

                # Top-1 & Top-5 Species Accuracies
                sp_logits = out["species_logits"]
                _, pred_top5 = torch.topk(sp_logits, k=min(5, num_species), dim=-1)
                pred_top1 = pred_top5[:, 0]

                correct_top1 += (pred_top1 == sp_targets).sum().item()
                correct_top5 += (pred_top5 == sp_targets.unsqueeze(1)).sum().item()

                # Organ Accuracy
                organ_logits = out["organ_logits"]
                pred_organ = organ_logits.argmax(dim=-1)
                correct_organ += (pred_organ == organ_targets).sum().item()

                # Vectorized Confusion Matrix for Macro F1
                flat_idx = sp_targets * num_species + pred_top1
                bincount = torch.bincount(flat_idx, minlength=num_species * num_species)
                conf_mat += bincount.reshape(num_species, num_species)

        # Compute Metrics
        val_top1_acc = (correct_top1 / max(total_samples, 1)) * 100.0
        val_top5_acc = (correct_top5 / max(total_samples, 1)) * 100.0
        val_organ_acc = (correct_organ / max(total_samples, 1)) * 100.0

        # Vectorized Macro F1
        tp = torch.diag(conf_mat).float()
        fp = (conf_mat.sum(dim=0) - tp).float()
        fn = (conf_mat.sum(dim=1) - tp).float()
        denom = 2 * tp + fp + fn
        f1_per_class = torch.where(denom > 0, (2.0 * tp) / denom, torch.zeros_like(denom))
        # Macro F1 across classes present in ground truth or predictions
        present_mask = (conf_mat.sum(dim=1) + conf_mat.sum(dim=0)) > 0
        if present_mask.sum() > 0:
            macro_f1 = f1_per_class[present_mask].mean().item()
        else:
            macro_f1 = f1_per_class.mean().item()

        avg_total = total_loss_sum / max(batches_processed, 1)
        avg_species = species_loss_sum / max(batches_processed, 1)
        avg_organ = organ_loss_sum / max(batches_processed, 1)

        val_metrics = {
            "val_loss_total": avg_total,
            "val_loss_species": avg_species,
            "val_loss_organ": avg_organ,
            "val_top1_acc": val_top1_acc,
            "val_top5_acc": val_top5_acc,
            "val_organ_acc": val_organ_acc,
            "val_macro_f1": macro_f1,
            "val_samples": total_samples,
            "val_time_sec": time.time() - t0
        }

        print(
            f"  Validation Loss: {avg_total:.4f} | "
            f"Top-1 Acc: {val_top1_acc:.2f}% | "
            f"Top-5 Acc: {val_top5_acc:.2f}% | "
            f"Organ Acc: {val_organ_acc:.2f}% | "
            f"Macro F1: {macro_f1:.4f}"
        )
        return val_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "metrics": metrics,
            "class_mapping": self.class_mapping,
            "config": self.config
        }
        # Always save last checkpoint
        torch.save(state, self.last_checkpoint_path)

        if is_best:
            torch.save(state, self.best_checkpoint_path)
            print(f"  ★ New Global Best Model Saved: Macro F1 = {metrics['val_macro_f1']:.4f} (Top-1 = {metrics['val_top1_acc']:.2f}%)")

    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from '{checkpoint_path}'...")
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scaler and state.get("scaler_state_dict"):
            self.scaler.load_state_dict(state["scaler_state_dict"])
        self.start_epoch = state["epoch"] + 1
        self.best_val_f1 = state.get("metrics", {}).get("val_macro_f1", 0.0)
        self.best_val_top1 = state.get("metrics", {}).get("val_top1_acc", 0.0)
        print(f"  -> Resumed at Epoch {self.start_epoch} (Best F1: {self.best_val_f1:.4f}, Best Top-1: {self.best_val_top1:.2f}%)")

    def _log_metrics_to_csv(self, epoch: int, train_m: Dict[str, float], val_m: Dict[str, float]):
        elapsed = train_m.get("train_time_sec", 0.0) + val_m.get("val_time_sec", 0.0)
        with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(train_m["train_loss_total"], 5),
                round(train_m["train_loss_species"], 5),
                round(train_m["train_loss_organ"], 5),
                round(val_m["val_loss_total"], 5),
                round(val_m["val_loss_species"], 5),
                round(val_m["val_loss_organ"], 5),
                round(val_m["val_top1_acc"], 2),
                round(val_m["val_top5_acc"], 2),
                round(val_m["val_organ_acc"], 2),
                round(val_m["val_macro_f1"], 5),
                round(train_m["learning_rate"], 8),
                round(elapsed, 2)
            ])

    def train(self) -> Dict[str, Any]:
        """
        Executes full multi-epoch training loop with validation, checkpointing, and early stopping.
        """
        print("=" * 80)
        print(f"Starting Multi-Epoch Training Pipeline on {self.device} (Epochs: {self.epochs})")
        print("=" * 80)

        training_start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            # Check if best (always save initial model at epoch 1)
            current_f1 = val_metrics["val_macro_f1"]
            is_best = (current_f1 > self.best_val_f1) or (epoch == 1 and not os.path.exists(self.best_checkpoint_path))
            if is_best:
                self.best_val_f1 = max(self.best_val_f1, current_f1)
                self.best_val_top1 = max(self.best_val_top1, val_metrics["val_top1_acc"])
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoints & CSV log
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            self._log_metrics_to_csv(epoch, train_metrics, val_metrics)

            combined_epoch_record = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "is_best": is_best
            }
            self.history.append(combined_epoch_record)

            # Early Stopping Check
            if self.patience_counter >= self.patience:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation Macro F1 has not improved for {self.patience} epochs.")
                print(f"Best Macro F1: {self.best_val_f1:.4f} | Best Top-1 Acc: {self.best_val_top1:.2f}%")
                break

        total_training_time = time.time() - training_start_time
        print("\n" + "=" * 80)
        print(f"Training Pipeline Complete in {total_training_time / 60:.2f} minutes.")
        print(f"Best Validation Macro F1: {self.best_val_f1:.4f}")
        print(f"Best Validation Top-1:   {self.best_val_top1:.2f}%")
        print(f"Best Weights Checkpoint:  {self.best_checkpoint_path}")
        print(f"Metrics CSV Log:          {self.csv_log_path}")
        print("=" * 80)

        return {
            "best_val_f1": self.best_val_f1,
            "best_val_top1": self.best_val_top1,
            "history": self.history,
            "total_time_seconds": total_training_time
        }
