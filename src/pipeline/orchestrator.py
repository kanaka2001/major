"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Module: src/pipeline/orchestrator.py

Unified End-to-End Orchestration Pipeline:
- Ingestion & RGBA/Grayscale-safe Preprocessing
- Dual-Branch Hybrid CNN–ViT Model Inference
- Dual-Threshold Uncertainty Gating & Pl@ntNet Fallback
- Multimodal Explainable AI Diagnostics (Grad-CAM + Attention Rollout + AFS)
- Authoritative Pharmacopoeial Knowledge Retrieval (WHO, IMPPAT 3.0, API, NCCIH)
- Subsystem Latency Profiling & Telemetry

Author: Lead Full-Stack AI Solutions & Systems Architect
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
from PIL import Image

# Ensure workspace root is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.data.dataset import safe_load_rgb_image
from src.knowledge.engine import MedicinalKnowledgeEngine
from src.inference.uncertainty import UncertaintyRouter
from src.xai.diagnostics import PlantXAIExplainer

class MedicinalPlantPipeline:
    """
    Singleton Pipeline Manager ensuring models and databases are initialized once
    and efficiently shared across inference cycles.
    """
    _instance = None

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        class_mapping_path: Optional[str] = None,
        knowledge_db_path: Optional[str] = None
    ):
        print("[Pipeline] Initializing unified subsystems...")
        self.workspace_root = WORKSPACE_ROOT
        
        # 1. Knowledge Engine
        self.knowledge_engine = MedicinalKnowledgeEngine(db_path=knowledge_db_path)
        
        # 2. Uncertainty Router (Dual-branch Hybrid CNN-ViT Model)
        self.uncertainty_router = UncertaintyRouter(
            checkpoint_path=checkpoint_path,
            class_mapping_path=class_mapping_path
        )
        
        # 3. Explainable AI Engine (Shares loaded model instance)
        self.xai_explainer = PlantXAIExplainer(
            model=self.uncertainty_router.model,
            class_mapping_path=class_mapping_path,
            device=self.uncertainty_router.device
        )
        print("[Pipeline] All subsystems initialized and ready.")

    @classmethod
    def get_instance(cls) -> "MedicinalPlantPipeline":
        if cls._instance is None:
            cls._instance = MedicinalPlantPipeline()
        return cls._instance

def analyze_medicinal_plant(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    organ_hint: Optional[str] = None,
    confidence_threshold: float = 0.60,
    entropy_threshold: float = 0.45,
    generate_xai: bool = True,
    output_dir: str = "reports/pipeline_runs"
) -> Dict[str, Any]:
    """
    Master end-to-end analysis pipeline linking vision, XAI, uncertainty detection,
    and pharmacopoeial knowledge retrieval.

    Args:
        image_input: File path, Path object, PIL Image, or NumPy array representing a plant photo.
        organ_hint: Optional plant organ (leaf, flower, fruit, bark, whole_plant, general).
        confidence_threshold: Top-1 probability threshold for uncertainty gating (default: 0.60).
        entropy_threshold: Normalized Shannon entropy threshold for uncertainty gating (default: 0.45).
        generate_xai: Whether to generate Grad-CAM and ViT Attention Rollout plots (default: True).
        output_dir: Output directory for diagnostic plots and run telemetry.

    Returns:
        Structured dictionary containing prediction, uncertainty metrics, XAI diagnostics,
        complete pharmacopoeial profile, and latency metrics.
    """
    pipeline = MedicinalPlantPipeline.get_instance()
    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. Preprocessing Stage
    # -------------------------------------------------------------------------
    t0 = time.time()
    temp_image_path = None
    orig_filename = "query_plant"

    if isinstance(image_input, (str, Path)):
        resolved_path = str(image_input)
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(WORKSPACE_ROOT, resolved_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Image not found at path: '{image_input}'")
        pil_image = safe_load_rgb_image(resolved_path)
        temp_image_path = resolved_path
        orig_filename = Path(resolved_path).stem
    elif isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")
        os.makedirs(output_dir, exist_ok=True)
        temp_image_path = os.path.join(output_dir, f"temp_{int(time.time()*1000)}.jpg")
        pil_image.save(temp_image_path, "JPEG", quality=95)
    elif isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(image_input).convert("RGB")
        os.makedirs(output_dir, exist_ok=True)
        temp_image_path = os.path.join(output_dir, f"temp_{int(time.time()*1000)}.jpg")
        pil_image.save(temp_image_path, "JPEG", quality=95)
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    t_preprocess = time.time() - t0

    # -------------------------------------------------------------------------
    # 2. Vision Inference & Uncertainty Gating Stage
    # -------------------------------------------------------------------------
    t1 = time.time()
    # Temporarily set thresholds on router
    old_conf_thresh = pipeline.uncertainty_router.confidence_threshold
    old_ent_thresh = pipeline.uncertainty_router.entropy_threshold
    pipeline.uncertainty_router.confidence_threshold = float(confidence_threshold)
    pipeline.uncertainty_router.entropy_threshold = float(entropy_threshold)

    route_res = pipeline.uncertainty_router.route(temp_image_path, organ_hint=organ_hint)

    # Restore default thresholds
    pipeline.uncertainty_router.confidence_threshold = old_conf_thresh
    pipeline.uncertainty_router.entropy_threshold = old_ent_thresh

    t_inference = time.time() - t1

    local_pred = route_res.get("prediction") or route_res.get("local_prediction") or {}
    detected_species = local_pred.get("species", "Unknown")
    detected_organ = local_pred.get("organ", organ_hint or "general")
    confidence = float(local_pred.get("confidence", 0.0))
    is_uncertain = bool(route_res.get("is_uncertain", False))
    routing_source = route_res.get("source", "unknown")
    routing_reason = route_res.get("routing_reason", "")
    uncertainty_metrics = route_res.get("uncertainty_metrics", {})
    fallback_payload = route_res.get("fallback_payload")

    # If routed to fallback, determine candidate species from fallback
    if is_uncertain and fallback_payload:
        best_match = fallback_payload.get("bestMatch")
        if best_match and fallback_payload.get("status") in ("success", "simulated_fallback"):
            detected_species = best_match

    # -------------------------------------------------------------------------
    # 3. Multimodal Explainable AI Diagnostics (Grad-CAM + Rollout + AFS)
    # -------------------------------------------------------------------------
    t2 = time.time()
    xai_info: Dict[str, Any] = {}
    if generate_xai:
        os.makedirs(output_dir, exist_ok=True)
        file_prefix = f"pipe_{orig_filename}_{int(time.time())}"
        
        target_idx = None
        if detected_species:
            clean_sp = detected_species.replace(" ", "_")
            target_idx = pipeline.xai_explainer.metadata.get("species_to_idx", {}).get(clean_sp)

        xai_res = pipeline.xai_explainer.explain_image(
            image_input=pil_image,
            target_species_idx=target_idx,
            ground_truth_species=detected_species,
            ground_truth_organ=detected_organ,
            save_plot=True,
            output_dir=output_dir,
            file_prefix=file_prefix
        )

        plot_path = xai_res.get("diagnostic_plot_path")
        rel_plot_path = os.path.relpath(plot_path, WORKSPACE_ROOT) if plot_path else None

        xai_info = {
            "anatomical_focus_score": round(xai_res.get("anatomical_focus_score", 0.0), 2),
            "afs_percentage": f"{xai_res.get('anatomical_focus_score', 0.0):.2f}%",
            "diagnostic_plot_path": rel_plot_path,
            "diagnostic_plot_absolute_path": plot_path,
            "interpretability_narrative": {
                "cnn_grad_cam": "Visualizes fine-grained micro-textures (leaf venation, margin dentation, trichomes) activating terminal conv layer.",
                "vit_attention_rollout": "Visualizes macroscopic organ morphology and spatial global contours via 6-layer Transformer attention rollout.",
                "hybrid_composite": "Blends micro-texture and macro-morphology (50/50) overlayed onto original input.",
                "anatomical_focus_metric": f"AFS of {xai_res.get('anatomical_focus_score', 0.0):.1f}% represents energy focused directly on authenticated plant tissue."
            }
        }
    else:
        xai_info = {"status": "XAI generation skipped by user"}

    t_xai = time.time() - t2

    # -------------------------------------------------------------------------
    # 4. Pharmacopoeial & Clinical Knowledge Retrieval
    # -------------------------------------------------------------------------
    t3 = time.time()
    pharmacopoeia_profile = pipeline.knowledge_engine.get_species_profile(detected_species)
    t_knowledge = time.time() - t3

    t_total = time.time() - t_start

    # -------------------------------------------------------------------------
    # 5. Master Output Payload Synthesis
    # -------------------------------------------------------------------------
    taxonomy = pharmacopoeia_profile.get("taxonomy", {})
    traditional = pharmacopoeia_profile.get("traditional_systems", {})
    clinical = pharmacopoeia_profile.get("clinical_evidence", {})
    phytochemistry = pharmacopoeia_profile.get("phytochemistry", {})
    safety = pharmacopoeia_profile.get("safety_and_contraindications", {})

    # Generate high-level clinical summary
    primary_bioactive_names = [c.get("compound", "") for c in phytochemistry.get("primary_bioactives", [])[:3]]
    bioactive_str = ", ".join([b for b in primary_bioactive_names if b]) or "Secondary metabolites"
    
    ayur = traditional.get("ayurveda", {})
    virya_str = ayur.get("virya", "Unspecified")
    prabhava_str = ayur.get("prabhava", "General adaptogenic")

    if is_uncertain:
        if fallback_payload and fallback_payload.get("provider") == "plantnet_v2_live_api":
            status_banner = f"🌐 Identified via Live Pl@ntNet v2 API ({detected_species})"
        elif fallback_payload and fallback_payload.get("status") == "api_key_required":
            status_banner = f"⚠️ Ambiguous / Low-Confidence Input (Confidence: {confidence * 100:.2f}%) — Set PLANTNET_API_KEY for Live Verification"
        else:
            status_banner = "⚠️ Ambiguous / Out-of-Distribution Input — Routed to Botanical Fallback"
    else:
        status_banner = "✅ High-Confidence Local Recognition (Hybrid CNN–ViT)"

    return {
        "status": "SUCCESS",
        "status_banner": status_banner,
        "identification": {
            "detected_species": detected_species,
            "scientific_name": taxonomy.get("scientific_name", detected_species),
            "botanical_family": taxonomy.get("botanical_family", "Unknown"),
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "detected_organ": detected_organ,
            "source_engine": routing_source,
            "is_uncertain": is_uncertain,
            "routing_reason": routing_reason,
            "top5_candidates": local_pred.get("top5_candidates", [])
        },
        "uncertainty_metrics": {
            "top1_confidence": uncertainty_metrics.get("top1_confidence", round(confidence, 4)),
            "normalized_entropy": uncertainty_metrics.get("normalized_entropy", 0.0),
            "confidence_threshold": confidence_threshold,
            "entropy_threshold": entropy_threshold,
            "is_low_confidence": uncertainty_metrics.get("is_low_confidence", False),
            "is_high_entropy": uncertainty_metrics.get("is_high_entropy", False)
        },
        "explainable_ai": xai_info,
        "pharmacopoeial_knowledge": {
            "lookup_status": pharmacopoeia_profile.get("lookup_status", "unknown"),
            "taxonomy": taxonomy,
            "morphology": pharmacopoeia_profile.get("morphology", {}),
            "traditional_systems": traditional,
            "clinical_evidence": clinical,
            "phytochemistry": phytochemistry,
            "safety_and_contraindications": safety
        },
        "executive_clinical_summary": {
            "species": f"{detected_species} ({taxonomy.get('scientific_name', detected_species)})",
            "family": taxonomy.get("botanical_family", "Unknown"),
            "ayurvedic_energetics": f"Virya: {virya_str} | Prabhava: {prabhava_str}",
            "primary_bioactives": bioactive_str,
            "clinical_evidence_grade": clinical.get("evidence_grading", "Uncatalogued in human trials"),
            "safety_advisory": safety.get("pregnancy_lactation_precautions", "Consult a qualified medical practitioner before therapeutic administration.")
        },
        "latency_profile": {
            "preprocessing_ms": round(t_preprocess * 1000, 2),
            "model_inference_ms": round(t_inference * 1000, 2),
            "xai_generation_ms": round(t_xai * 1000, 2),
            "knowledge_retrieval_ms": round(t_knowledge * 1000, 2),
            "total_latency_ms": round(t_total * 1000, 2),
            "total_latency_seconds": round(t_total, 3)
        }
    }
