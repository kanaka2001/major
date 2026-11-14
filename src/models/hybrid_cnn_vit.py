"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Module: src/models/hybrid_cnn_vit.py

Dual-Branch Hybrid CNN–ViT Architecture trained from scratch:
- Branch 1 (CNN): Local micro-texture extraction (veins, margin serrations, bark/seed texture)
- Branch 2 (ViT): Global macro-morphology attention (leaf geometry, organ spatial context)
- Feature Fusion: Spatial Cross-Attention + Gated Fusion
- Multi-Attribute Output: Primary Species Head (1,203 classes) + Auxiliary Organ Head (5 classes)
- Explainability (XAI): Explicit hook points for Grad-CAM (CNN) and Attention Rollout (ViT)

Author: Lead Deep Learning Architect
"""

import math
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# -----------------------------------------------------------------------------
# 1. Vision Transformer Components (Self-Attention & Transformer Encoder)
# -----------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module that explicitly saves and returns
    attention weight matrices [B, num_heads, N, N] for Attention Rollout.
    """
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Buffer for latest attention map (used by XAI Attention Rollout)
        self.latest_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = x.shape
        # qkv shape: [B, N, 3, num_heads, head_dim] -> permute to 3 x [B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        self.latest_attn_weights = attn.detach()
        attn_out = self.attn_drop(attn)

        # Context output: [B, num_heads, N, head_dim] -> [B, N, C]
        out = (attn_out @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attn:
            return out, attn
        return out, None

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Encoder Block with GELU MLP.
    """
    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 4.0, drop: float = 0.1, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x, return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

class ViTBranch(nn.Module):
    """
    Vision Transformer Branch for Global Macro-Morphology Modeling.
    - Patch size 16x16: 14x14 = 196 patches from 224x224 input
    - Prepend [CLS] token + learnable position embeddings
    - Stack of Transformer Blocks supporting Attention Rollout
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2 # 196
        self.embed_dim = embed_dim

        # Exposed patch projection layer
        self.vit_patch_conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.vit_patch_conv.weight)
        if self.vit_patch_conv.bias is not None:
            nn.init.zeros_(self.vit_patch_conv.bias)

    def forward(self, x: torch.Tensor, return_all_attns: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        B = x.shape[0]
        # Patch embedding: [B, 3, 224, 224] -> [B, embed_dim, 14, 14] -> [B, 196, embed_dim]
        patch_tokens = self.vit_patch_conv(x).flatten(2).transpose(1, 2)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, patch_tokens), dim=1) # [B, 197, embed_dim]
        x = self.pos_drop(x + self.pos_embed)

        all_attns = []
        for blk in self.blocks:
            x, attn = blk(x, return_attn=return_all_attns)
            if return_all_attns and attn is not None:
                all_attns.append(attn)

        x = self.norm(x)
        cls_out = x[:, 0]        # [B, embed_dim]
        patch_out = x[:, 1:]     # [B, 196, embed_dim]
        return cls_out, patch_out, all_attns


# -----------------------------------------------------------------------------
# 2. Local Micro-Texture CNN Branch
# -----------------------------------------------------------------------------

class CNNBranch(nn.Module):
    """
    Convolutional Branch for Local Botanical Micro-Texture Extraction.
    Configurable between MobileNetV2 and ConvNeXt-Tiny (trained from scratch).
    Terminal layer is explicitly exposed via cnn_target_layer for Grad-CAM hooks.
    """
    def __init__(self, backbone_type: str = "mobilenet_v2"):
        super().__init__()
        self.backbone_type = backbone_type.lower()

        if self.backbone_type == "mobilenet_v2":
            base = models.mobilenet_v2(weights=None)
            # base.features outputs [B, 1280, 7, 7] for 224x224 input
            self.features = base.features
            self.out_channels = 1280
            # Explicit Grad-CAM target layer: last inverted residual block
            self.cnn_target_layer = self.features[-1]
        elif self.backbone_type == "convnext_tiny":
            base = models.convnext_tiny(weights=None)
            # base.features outputs [B, 768, 7, 7]
            self.features = base.features
            self.out_channels = 768
            # Explicit Grad-CAM target layer: last stage block
            self.cnn_target_layer = self.features[-1][-1]
        else:
            raise ValueError(f"Unsupported CNN backbone: {backbone_type}. Choose 'mobilenet_v2' or 'convnext_tiny'.")

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Spatial feature map: [B, out_channels, 7, 7]
        spatial_map = self.features(x)
        # Pooled vector: [B, out_channels]
        pooled = self.global_pool(spatial_map).flatten(1)
        return pooled, spatial_map


# -----------------------------------------------------------------------------
# 3. Spatial Cross-Attention & Gated Feature Fusion
# -----------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Fuses Global ViT representations with Local CNN Micro-Texture maps.
    ViT [CLS] token acts as Query; CNN spatial cells act as Key & Value.
    """
    def __init__(self, cnn_dim: int, vit_dim: int, fusion_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.head_dim = fusion_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.proj_cnn = nn.Linear(cnn_dim, fusion_dim)
        self.proj_vit = nn.Linear(vit_dim, fusion_dim)

        # Cross-attention: Query from ViT [CLS], Key/Value from CNN spatial cells (49 cells)
        self.q_proj = nn.Linear(vit_dim, fusion_dim)
        self.k_proj = nn.Linear(cnn_dim, fusion_dim)
        self.v_proj = nn.Linear(cnn_dim, fusion_dim)
        self.out_proj = nn.Linear(fusion_dim, fusion_dim)

        # Gated fusion gate
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, cnn_vec: torch.Tensor, cnn_spatial: torch.Tensor, vit_cls: torch.Tensor) -> torch.Tensor:
        B, C_cnn, H, W = cnn_spatial.shape
        cnn_tokens = cnn_spatial.flatten(2).transpose(1, 2) # [B, 49, C_cnn]

        # 1. Linear projections
        f_cnn = self.proj_cnn(cnn_vec) # [B, fusion_dim]
        f_vit = self.proj_vit(vit_cls) # [B, fusion_dim]

        # 2. Multi-Head Cross Attention
        # Q: [B, 1, fusion_dim]
        q = self.q_proj(vit_cls).unsqueeze(1).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V: [B, 49, fusion_dim]
        k = self.k_proj(cnn_tokens).reshape(B, 49, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(cnn_tokens).reshape(B, 49, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-attn weights: [B, num_heads, 1, 49]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        f_cross = (attn @ v).transpose(1, 2).reshape(B, self.fusion_dim) # [B, fusion_dim]
        f_cross = self.out_proj(f_cross)

        # 3. Gated Fusion
        concat_feats = torch.cat([f_cnn, f_vit, f_cross], dim=-1) # [B, fusion_dim * 3]
        g = self.gate(concat_feats) # [B, fusion_dim]
        f_fused = g * f_cnn + (1.0 - g) * (f_vit + f_cross)
        f_fused = self.norm(self.dropout(f_fused))
        return f_fused


# -----------------------------------------------------------------------------
# 4. Master Hybrid CNN–ViT Model
# -----------------------------------------------------------------------------

class HybridCNNViT(nn.Module):
    """
    Dual-Branch Hybrid CNN–ViT Architecture for Intelligent Medicinal Plant Recognition.
    
    Architecture:
    - Input: [B, 3, 224, 224]
    - CNN Branch: Micro-texture feature extraction -> [B, C_cnn, 7, 7]
    - ViT Branch: Macro-morphology patch self-attention -> [B, 197, D_vit]
    - Fusion: Cross-Attention + Gated Fusion -> [B, fusion_dim]
    - Primary Head: [B, num_species] (1,203 species)
    - Auxiliary Head: [B, num_parts] (5 anatomical parts: leaf, whole_plant, general, seed, bark)
    """
    def __init__(
        self,
        num_species: int = 1203,
        num_parts: int = 5,
        cnn_backbone: str = "mobilenet_v2",
        vit_embed_dim: int = 384,
        vit_depth: int = 6,
        vit_heads: int = 6,
        fusion_dim: int = 512,
        drop_rate: float = 0.2
    ):
        super().__init__()
        self.num_species = num_species
        self.num_parts = num_parts

        # 1. Dual Branches
        self.cnn_branch = CNNBranch(backbone_type=cnn_backbone)
        self.vit_branch = ViTBranch(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            drop_rate=drop_rate
        )

        # Expose hook targets for XAI
        self.cnn_target_layer = self.cnn_branch.cnn_target_layer
        self.vit_patch_conv = self.vit_branch.vit_patch_conv

        # 2. Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(
            cnn_dim=self.cnn_branch.out_channels,
            vit_dim=vit_embed_dim,
            fusion_dim=fusion_dim,
            num_heads=4
        )

        # 3. Multi-Attribute Output Heads
        self.species_classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(fusion_dim, num_species)
        )

        self.organ_classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(fusion_dim // 2, num_parts)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attns: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass yielding multi-task predictions and optional feature representations.
        """
        # CNN Branch
        cnn_vec, cnn_spatial = self.cnn_branch(x)

        # ViT Branch
        vit_cls, vit_patches, attns = self.vit_branch(x, return_all_attns=return_attns)

        # Cross-Attention Fusion
        fused = self.fusion(cnn_vec, cnn_spatial, vit_cls)

        # Classifiers
        species_logits = self.species_classifier(fused)
        organ_logits = self.organ_classifier(fused)

        out = {
            "species_logits": species_logits,
            "organ_logits": organ_logits
        }

        if return_features:
            out["fused_features"] = fused
            out["cnn_spatial_map"] = cnn_spatial
            out["vit_cls_token"] = vit_cls
            out["vit_patch_tokens"] = vit_patches

        if return_attns:
            out["vit_attentions"] = attns

        return out

    def get_attention_rollout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes Attention Rollout (Abnar & Zuidema, 2020) across all ViT transformer blocks:
        Recursively multiplies attention matrices across layers with identity matrix addition:
          A_hat = 0.5 * A + 0.5 * I
          R_{l} = A_hat_l @ R_{l-1}
        Returns the spatial attention heatmap from [CLS] to all 196 image patches: [B, 14, 14]
        """
        with torch.no_grad():
            _, _, attns = self.vit_branch(x, return_all_attns=True)
            B = x.shape[0]
            num_tokens = attns[0].shape[-1] # 197
            
            # Start with identity matrix
            eye = torch.eye(num_tokens, device=x.device).unsqueeze(0).expand(B, -1, -1)
            rollout = eye.clone()

            for layer_attn in attns:
                # Average attention across heads: [B, num_heads, 197, 197] -> [B, 197, 197]
                mean_attn = layer_attn.mean(dim=1)
                # Add residual identity connection
                a_hat = 0.5 * mean_attn + 0.5 * eye
                # Re-normalize rows
                a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)
                rollout = a_hat @ rollout

            # Mask from [CLS] token (index 0) to all patch tokens (1 .. 196)
            cls_rollout = rollout[:, 0, 1:] # [B, 196]
            grid_size = int(math.sqrt(cls_rollout.shape[-1])) # 14
            heatmap = cls_rollout.reshape(B, grid_size, grid_size)
            # Min-max normalize per sample
            heatmap_min = heatmap.amin(dim=(-2, -1), keepdim=True)
            heatmap_max = heatmap.amax(dim=(-2, -1), keepdim=True)
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
            return heatmap

    def predict_species(
        self,
        x: torch.Tensor,
        idx_to_species: Dict[int, str],
        temperature: float = 1.0,
        top_k: int = 5
    ) -> List[List[Dict[str, float]]]:
        """
        Runs inference and returns top-k species predictions with calibrated probabilities.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            scaled_logits = out["species_logits"] / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

            results = []
            for b in range(x.shape[0]):
                sample_preds = []
                for k_idx in range(top_k):
                    idx = int(top_indices[b, k_idx].item())
                    prob = float(top_probs[b, k_idx].item())
                    sp_name = idx_to_species.get(idx, f"Species_{idx}")
                    sample_preds.append({
                        "species": sp_name,
                        "species_idx": idx,
                        "probability": round(prob, 5),
                        "percentage": f"{prob * 100:.2f}%"
                    })
                results.append(sample_preds)
            return results

    def get_model_summary(self) -> Dict[str, any]:
        """
        Computes parameter counts and architecture layer breakdown.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_params = sum(p.numel() for p in self.cnn_branch.parameters())
        vit_params = sum(p.numel() for p in self.vit_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        head_params = sum(p.numel() for p in self.species_classifier.parameters()) + sum(p.numel() for p in self.organ_classifier.parameters())

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "cnn_branch_parameters": cnn_params,
            "vit_branch_parameters": vit_params,
            "fusion_parameters": fusion_params,
            "classifier_head_parameters": head_params,
            "num_species_classes": self.num_species,
            "num_organ_classes": self.num_parts,
            "cnn_backbone": self.cnn_branch.backbone_type,
            "vit_embed_dim": self.vit_branch.embed_dim,
            "vit_num_patches": self.vit_branch.num_patches
        }

if __name__ == "__main__":
    print("Testing HybridCNNViT architecture...")
    model = HybridCNNViT(num_species=1203, num_parts=5, cnn_backbone="mobilenet_v2")
    summary = model.get_model_summary()
    for k, v in summary.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input, return_features=True, return_attns=True)
    print(f"Output species_logits shape: {out['species_logits'].shape}")
    print(f"Output organ_logits shape:   {out['organ_logits'].shape}")
    print(f"Fused features shape:        {out['fused_features'].shape}")
    print(f"CNN spatial map shape:       {out['cnn_spatial_map'].shape}")
    print(f"ViT patch tokens shape:      {out['vit_patch_tokens'].shape}")
    print(f"ViT attention layers count:  {len(out['vit_attentions'])}")
    print(f"Layer 0 attention shape:     {out['vit_attentions'][0].shape}")

    rollout = model.get_attention_rollout(dummy_input)
    print(f"Attention rollout heatmap:   {rollout.shape}")

