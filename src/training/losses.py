"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Module: src/training/losses.py

Class-Balanced Focal Loss with Multi-Attribute Auxiliary Loss:
- Effective Number of Samples weighting (Cui et al., CVPR 2019):
  W_c = (1 - beta) / (1 - beta^{n_c}) with beta = 0.9999
- Focal Loss modulation:
  FL(p_t) = - W_y * (1 - p_t)^gamma * log(p_t) with gamma = 2.0
- Supports hard scalar targets and soft one-hot targets (MixUp / CutMix compatible)
- Multi-Task loss integration:
  L_total = L_species_CB_Focal + 0.2 * L_organ_CE

Author: Lead Deep Learning Architect
"""

from typing import Dict, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss for Extreme Long-Tail Botanical Classification.
    """
    def __init__(
        self,
        samples_per_class: Union[np.ndarray, list, torch.Tensor],
        num_classes: int = 1203,
        beta: float = 0.9999,
        gamma: float = 2.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        # Compute effective number of samples weights:
        # E_n = (1 - beta^n) / (1 - beta)
        # W_c = 1 / E_n
        counts = np.array(samples_per_class, dtype=np.float64)
        counts = np.maximum(counts, 1.0) # avoid division by zero
        
        effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        weights = 1.0 / np.maximum(effective_num, 1e-8)
        
        # Normalize so that sum(weights) = num_classes (mean weight is 1.0)
        weights = weights / np.sum(weights) * num_classes
        
        # Register as persistent buffer
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] raw model predictions
            targets: [B] class indices or [B, num_classes] one-hot/soft probabilities
        """
        B, C = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        if targets.dim() == 1:
            # Hard scalar targets [B]
            if self.label_smoothing > 0.0:
                smooth_targets = torch.full_like(logits, self.label_smoothing / (C - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
                target_probs = smooth_targets
                focal_mod = (1.0 - probs) ** self.gamma
                weighted_loss = -self.class_weights.unsqueeze(0) * target_probs * focal_mod * log_probs
                return weighted_loss.sum(dim=-1).mean()
            else:
                target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                sample_weights = self.class_weights.gather(0, targets)
                focal_mod = (1.0 - target_probs) ** self.gamma
                loss = -sample_weights * focal_mod * target_log_probs
                return loss.mean()
        else:
            # Soft / one-hot targets [B, C] (MixUp / CutMix)
            focal_mod = (1.0 - probs) ** self.gamma
            weighted_loss = -self.class_weights.unsqueeze(0) * targets * focal_mod * log_probs
            return weighted_loss.sum(dim=-1).mean()

class MultiAttributeMedicinalLoss(nn.Module):
    """
    Joint Multi-Task Loss combining Primary Species Class-Balanced Focal Loss
    and Auxiliary Organ Cross-Entropy Loss:
      L_total = L_species_CB_Focal + organ_weight * L_organ_CE
    """
    def __init__(
        self,
        species_counts: Union[np.ndarray, list, torch.Tensor],
        num_species: int = 1203,
        num_parts: int = 5,
        beta: float = 0.9999,
        gamma: float = 2.0,
        organ_loss_weight: float = 0.2,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.species_loss_fn = ClassBalancedFocalLoss(
            samples_per_class=species_counts,
            num_classes=num_species,
            beta=beta,
            gamma=gamma,
            label_smoothing=label_smoothing
        )
        self.organ_loss_fn = nn.CrossEntropyLoss()
        self.organ_loss_weight = organ_loss_weight

    def forward(
        self,
        species_logits: torch.Tensor,
        species_targets: torch.Tensor,
        organ_logits: torch.Tensor,
        organ_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        loss_species = self.species_loss_fn(species_logits, species_targets)
        loss_organ = self.organ_loss_fn(organ_logits, organ_targets)
        loss_total = loss_species + self.organ_loss_weight * loss_organ

        return {
            "loss_total": loss_total,
            "loss_species": loss_species,
            "loss_organ": loss_organ
        }

if __name__ == "__main__":
    print("Testing ClassBalancedFocalLoss and MultiAttributeMedicinalLoss...")
    # Mock class counts with extreme imbalance
    mock_counts = [5000] * 5 + [100] * 50 + [10] * 1000 + [1] * 148 # 1203 classes
    loss_fn = MultiAttributeMedicinalLoss(species_counts=mock_counts, num_species=1203, num_parts=5)

    dummy_species_logits = torch.randn(4, 1203, requires_grad=True)
    dummy_species_targets = torch.tensor([0, 10, 500, 1202], dtype=torch.long)
    dummy_organ_logits = torch.randn(4, 5, requires_grad=True)
    dummy_organ_targets = torch.tensor([0, 1, 2, 4], dtype=torch.long)

    losses = loss_fn(dummy_species_logits, dummy_species_targets, dummy_organ_logits, dummy_organ_targets)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    losses["loss_total"].backward()
    print(f"Species logits grad norm: {dummy_species_logits.grad.norm().item():.4f}")
    print(f"Organ logits grad norm:   {dummy_organ_logits.grad.norm().item():.4f}")

