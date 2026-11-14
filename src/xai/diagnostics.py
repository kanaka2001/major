"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Module: src/xai/diagnostics.py

Explainable AI (XAI) Diagnostic Engine:
- CNN Micro-Texture Grad-CAM on terminal convolutional layer (model.cnn_target_layer)
- ViT Macro-Morphology Attention Rollout across all 6 Transformer encoder blocks
- Hybrid Composite Saliency (0.5 * Grad-CAM + 0.5 * Attention Rollout)
- Anatomical Focus Score (AFS) calculating percentage of saliency on plant tissue vs background
- 4-Panel publication-quality diagnostic figure generator (reports/xai_diagnostics/<filename>_xai.png)

Author: Lead AI Diagnostics & Inference Engineer
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import math
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src.models.hybrid_cnn_vit import HybridCNNViT
from src.data.dataset import safe_load_rgb_image, IMAGENET_MEAN, IMAGENET_STD

def otsu_threshold(gray_img: np.ndarray) -> float:
    """
    Computes Otsu's optimal binarization threshold for foreground separation.
    """
    pixel_counts, bin_edges = np.histogram(gray_img, bins=256, range=(0.0, 1.0))
    total_pixels = gray_img.size
    
    current_max = 0.0
    threshold = 0.5
    weight_bg = 0.0
    sum_bg = 0.0
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total_mean = np.sum(bin_centers * pixel_counts) / total_pixels
    
    for i in range(256):
        weight_bg += pixel_counts[i]
        if weight_bg == 0:
            continue
        weight_fg = total_pixels - weight_bg
        if weight_fg == 0:
            break
            
        sum_bg += bin_centers[i] * pixel_counts[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (total_mean * total_pixels - sum_bg) / weight_fg
        
        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_var > current_max:
            current_max = between_var
            threshold = bin_centers[i]
            
    return float(threshold)

def compute_anatomical_focus_score(saliency_map: np.ndarray, rgb_img: np.ndarray) -> float:
    """
    Calculates Anatomical Focus Score (AFS):
    Percentage of saliency energy concentrated on the plant foreground versus background clutter.
    """
    # Convert RGB to grayscale
    gray = np.dot(rgb_img[..., :3], [0.2989, 0.5870, 0.1140])
    gray = gray / 255.0
    
    thresh = otsu_threshold(gray)
    
    # Determine whether plant tissue is darker or brighter than background
    # (Plant images on white sheets are dark foreground; dark backgrounds are bright foreground)
    # Check corner pixels to determine background tone
    corners = np.concatenate([gray[:10, :10], gray[:10, -10:], gray[-10:, :10], gray[-10:, -10:]])
    bg_mean = np.mean(corners)
    
    if bg_mean > 0.5:
        # Light background -> plant foreground is darker
        fg_mask = (gray < thresh).astype(np.float32)
    else:
        # Dark background -> plant foreground is brighter
        fg_mask = (gray > thresh).astype(np.float32)
        
    # If mask is degenerate, use center weighted mask
    if fg_mask.sum() == 0 or fg_mask.sum() == fg_mask.size:
        h, w = gray.shape
        y, x = np.ogrid[:h, :w]
        center_dist = np.sqrt((x - w/2)**2 + (y - h/2)**2)
        fg_mask = (center_dist < (min(h, w) * 0.45)).astype(np.float32)

    fg_energy = np.sum(saliency_map * fg_mask)
    total_energy = np.sum(saliency_map) + 1e-12
    afs = float((fg_energy / total_energy) * 100.0)
    return max(0.0, min(100.0, afs))

class PlantXAIExplainer:
    """
    Production-Grade Visual Explainability Suite for Hybrid CNN–ViT Medicinal Plant System.
    """
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        class_mapping_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Load Metadata
        if class_mapping_path is None:
            workspace_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            class_mapping_path = os.path.join(workspace_root, "dataset", "class_mapping.json")

        import json
        with open(class_mapping_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.num_species = self.metadata["num_species"]
        self.num_parts = self.metadata["num_parts"]
        self.idx_to_species = {int(k): v for k, v in self.metadata["idx_to_species"].items()}
        self.idx_to_part = {int(k): v for k, v in self.metadata["idx_to_part"].items()}

        # 2. Load Model
        if model is not None:
            self.model = model.to(self.device)
        else:
            if checkpoint_path is None:
                workspace_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                checkpoint_path = os.path.join(workspace_root, "checkpoints", "best_model.pth")
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = os.path.join(workspace_root, "checkpoints", "last_checkpoint.pth")

            self.model = HybridCNNViT(
                num_species=self.num_species,
                num_parts=self.num_parts,
                cnn_backbone="mobilenet_v2",
                vit_embed_dim=384,
                vit_depth=6,
                vit_heads=6,
                fusion_dim=512
            ).to(self.device)

            if os.path.exists(checkpoint_path):
                print(f"PlantXAIExplainer: Loading weights from '{checkpoint_path}'...")
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                print("PlantXAIExplainer: Warning — checkpoint not found, initialized with base weights.")

        self.model.eval()

        # 3. Setup Preprocessing Transforms
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        # Hook storage for Grad-CAM
        self.activations = []
        self.gradients = []

    def _forward_hook(self, module, inp, out):
        self.activations.append(out)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0])

    def explain_image(
        self,
        image_input: Union[str, Image.Image, torch.Tensor],
        target_species_idx: Optional[int] = None,
        ground_truth_species: Optional[str] = None,
        ground_truth_organ: Optional[str] = None,
        save_plot: bool = True,
        output_dir: str = "reports/xai_diagnostics",
        file_prefix: str = "sample"
    ) -> Dict[str, Any]:
        """
        Executes complete multimodal XAI pipeline on a single image.
        Returns:
          - Saliency maps (Grad-CAM, ViT Attention Rollout, Hybrid Composite)
          - Model predictions (Top-1 species, organ, probabilities)
          - Anatomical Focus Score (AFS)
          - Saved 4-panel diagnostic figure path
        """
        # 1. Process Input into [1, 3, 224, 224] Tensor & Displayable RGB
        if isinstance(image_input, str):
            pil_rgb = safe_load_rgb_image(image_input)
            resized_pil = pil_rgb.resize((224, 224), Image.Resampling.BILINEAR)
            rgb_display = np.array(resized_pil)
            input_tensor = self.eval_transform(pil_rgb).unsqueeze(0).to(self.device)
        elif isinstance(image_input, Image.Image):
            pil_rgb = image_input.convert("RGB")
            resized_pil = pil_rgb.resize((224, 224), Image.Resampling.BILINEAR)
            rgb_display = np.array(resized_pil)
            input_tensor = self.eval_transform(pil_rgb).unsqueeze(0).to(self.device)
        elif isinstance(image_input, torch.Tensor):
            if image_input.dim() == 3:
                input_tensor = image_input.unsqueeze(0).to(self.device)
            else:
                input_tensor = image_input.to(self.device)
            # Reconstruct RGB display from normalized tensor
            inv_img = input_tensor[0].detach().cpu().clone()
            for c in range(3):
                inv_img[c] = inv_img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
            inv_img = inv_img.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
            rgb_display = (inv_img * 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        input_tensor.requires_grad_(True)

        # 2. Register Hooks on CNN Target Layer
        self.activations.clear()
        self.gradients.clear()
        f_handle = self.model.cnn_target_layer.register_forward_hook(self._forward_hook)
        b_handle = self.model.cnn_target_layer.register_full_backward_hook(self._backward_hook)

        # 3. Model Forward Pass
        self.model.zero_grad()
        out = self.model(input_tensor, return_features=True, return_attns=True)

        species_logits = out["species_logits"]
        organ_logits = out["organ_logits"]

        species_probs = F.softmax(species_logits, dim=-1)
        organ_probs = F.softmax(organ_logits, dim=-1)

        pred_species_idx = int(species_logits.argmax(dim=-1).item())
        pred_organ_idx = int(organ_logits.argmax(dim=-1).item())

        chosen_species_idx = target_species_idx if target_species_idx is not None else pred_species_idx
        species_conf = float(species_probs[0, chosen_species_idx].item())
        organ_conf = float(organ_probs[0, pred_organ_idx].item())

        species_name = self.idx_to_species.get(chosen_species_idx, f"Species_{chosen_species_idx}")
        organ_name = self.idx_to_part.get(pred_organ_idx, f"Part_{pred_organ_idx}")

        # 4. CNN Grad-CAM Backward Pass
        target_score = species_logits[0, chosen_species_idx]
        target_score.backward(retain_graph=True)

        act = self.activations[0] # [1, 1280, 7, 7]
        grad = self.gradients[0]  # [1, 1280, 7, 7]

        # Remove hooks
        f_handle.remove()
        b_handle.remove()

        # Global average pool gradients along spatial dimensions to get importance weights
        weights = grad.mean(dim=(2, 3), keepdim=True) # [1, 1280, 1, 1]
        grad_cam = F.relu((weights * act).sum(dim=1, keepdim=True)) # [1, 1, 7, 7]
        grad_cam = F.interpolate(grad_cam, size=(224, 224), mode="bilinear", align_corners=False)
        grad_cam_np = grad_cam[0, 0].detach().cpu().numpy()
        # Min-Max normalize
        cam_min, cam_max = grad_cam_np.min(), grad_cam_np.max()
        grad_cam_norm = (grad_cam_np - cam_min) / (cam_max - cam_min + 1e-8)

        # 5. ViT Attention Rollout
        vit_rollout = self.model.get_attention_rollout(input_tensor) # [1, 14, 14]
        vit_rollout_interp = F.interpolate(
            vit_rollout.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False
        )[0, 0].detach().cpu().numpy()
        roll_min, roll_max = vit_rollout_interp.min(), vit_rollout_interp.max()
        vit_rollout_norm = (vit_rollout_interp - roll_min) / (roll_max - roll_min + 1e-8)

        # 6. Hybrid Fused Saliency & Anatomical Focus Score
        hybrid_saliency = 0.5 * grad_cam_norm + 0.5 * vit_rollout_norm
        hyb_min, hyb_max = hybrid_saliency.min(), hybrid_saliency.max()
        hybrid_saliency_norm = (hybrid_saliency - hyb_min) / (hyb_max - hyb_min + 1e-8)

        afs = compute_anatomical_focus_score(hybrid_saliency_norm, rgb_display)

        # 7. Generate Publication-Quality 4-Panel Visualization
        save_path = None
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{file_prefix}_xai_diagnostic.png")
            self._render_diagnostic_figure(
                rgb_display=rgb_display,
                grad_cam_norm=grad_cam_norm,
                vit_rollout_norm=vit_rollout_norm,
                hybrid_saliency_norm=hybrid_saliency_norm,
                species_name=species_name,
                species_conf=species_conf,
                organ_name=organ_name,
                organ_conf=organ_conf,
                ground_truth_species=ground_truth_species,
                ground_truth_organ=ground_truth_organ,
                afs=afs,
                save_path=save_path
            )

        return {
            "species_idx": chosen_species_idx,
            "species_name": species_name,
            "species_confidence": round(species_conf, 4),
            "organ_idx": pred_organ_idx,
            "organ_name": organ_name,
            "organ_confidence": round(organ_conf, 4),
            "ground_truth_species": ground_truth_species,
            "ground_truth_organ": ground_truth_organ,
            "anatomical_focus_score": round(afs, 2),
            "grad_cam_map": grad_cam_norm,
            "vit_rollout_map": vit_rollout_norm,
            "hybrid_saliency_map": hybrid_saliency_norm,
            "diagnostic_plot_path": save_path
        }

    def _render_diagnostic_figure(
        self,
        rgb_display: np.ndarray,
        grad_cam_norm: np.ndarray,
        vit_rollout_norm: np.ndarray,
        hybrid_saliency_norm: np.ndarray,
        species_name: str,
        species_conf: float,
        organ_name: str,
        organ_conf: float,
        ground_truth_species: Optional[str],
        ground_truth_organ: Optional[str],
        afs: float,
        save_path: str
    ):
        """
        Renders a side-by-side 4-panel visual interpretability figure:
        [1] Original Image | [2] CNN Grad-CAM | [3] ViT Attention Rollout | [4] Hybrid Composite Overlay
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), dpi=200)

        # Panel 1: Original Image
        gt_text = ""
        if ground_truth_species or ground_truth_organ:
            gt_text = f"GT: {ground_truth_species or 'N/A'} ({ground_truth_organ or 'N/A'})\n"

        axes[0].imshow(rgb_display)
        axes[0].set_title(
            f"{gt_text}Pred: {species_name} ({species_conf*100:.2f}%)\nOrgan: {organ_name} ({organ_conf*100:.2f}%)",
            fontsize=10, fontweight="bold", pad=8
        )
        axes[0].axis("off")

        # Panel 2: CNN Grad-CAM Heatmap
        cam_im = axes[1].imshow(grad_cam_norm, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1].set_title(
            "CNN Grad-CAM (Micro-Texture)\nTarget: Terminal Conv_1 Stage",
            fontsize=11, fontweight="bold", pad=8
        )
        axes[1].axis("off")
        plt.colorbar(cam_im, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: ViT Attention Rollout Heatmap
        vit_im = axes[2].imshow(vit_rollout_norm, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[2].set_title(
            "ViT Attention Rollout (Macro)\nLayers: 6 | Heads: 6 | Tokens: 197",
            fontsize=11, fontweight="bold", pad=8
        )
        axes[2].axis("off")
        plt.colorbar(vit_im, ax=axes[2], fraction=0.046, pad=0.04)

        # Panel 4: Superimposed Hybrid Composite Overlay
        heatmap_colored = cm.jet(hybrid_saliency_norm)[..., :3] # RGB in [0, 1]
        rgb_norm = rgb_display.astype(np.float32) / 255.0
        overlay = 0.52 * rgb_norm + 0.48 * heatmap_colored
        overlay = np.clip(overlay, 0.0, 1.0)

        axes[3].imshow(overlay)
        axes[3].set_title(
            f"Hybrid Composite Overlay\nAnatomical Focus Score: {afs:.1f}%",
            fontsize=11, fontweight="bold", pad=8
        )
        axes[3].axis("off")

        plt.suptitle(
            f"VTU Major Project: Multimodal Explainable AI (XAI) Diagnostic Report — {species_name}",
            fontsize=13, fontweight="heavy", y=1.02
        )
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
