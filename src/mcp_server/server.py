"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Module: src/mcp_server/server.py

Model Context Protocol (MCP) Standard Knowledge Server:
- Universal JSON-RPC 2.0 transport over Standard I/O (stdio)
- Tool 1: identify_medicinal_plant(image_path: str, organ_hint: Optional[str])
- Tool 2: explain_plant_prediction(image_path: str, target_class: Optional[str], output_dir: Optional[str])
- Tool 3: get_plant_information(species_name: str)
- Tool 4: get_medicinal_uses(species_name: str, system: Optional[str])

Integrates Hybrid CNN–ViT inference, XAI visual diagnostics, and clinical/Ayurvedic knowledge.

Author: Lead Systems & AI Agent Architect
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, Any, Optional, List, Union

# Ensure workspace root is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Configure internal logger to stderr to preserve stdout exclusively for JSON-RPC messages
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(asctime)s] [MCP-Server] [%(levelname)s] %(message)s"
)
logger = logging.getLogger("mcp_server")

# Tool JSON Schema Definitions (Draft-07 compliant)
TOOLS_REGISTRY = [
    {
        "name": "identify_medicinal_plant",
        "description": (
            "Identifies a medicinal plant from a photograph using our 1,203-class Hybrid CNN–ViT model. "
            "Applies dual-threshold uncertainty gating (confidence < 0.60 or entropy > 0.45) "
            "to automatically route out-of-distribution or ambiguous inputs to the Pl@ntNet API fallback."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute or relative filesystem path to the query plant image (.jpg, .jpeg, .png)."
                },
                "organ_hint": {
                    "type": "string",
                    "enum": ["leaf", "flower", "fruit", "bark", "whole_plant", "general"],
                    "description": "Optional prior knowledge of the plant organ depicted in the image."
                }
            },
            "required": ["image_path"]
        }
    },
    {
        "name": "explain_plant_prediction",
        "description": (
            "Generates multimodal Explainable AI (XAI) visual diagnostics for a medicinal plant image. "
            "Computes CNN micro-texture Grad-CAM on the terminal convolution layer, "
            "ViT macroscopic Attention Rollout across all 6 Transformer encoder blocks, "
            "a fused composite saliency map, and the Anatomical Focus Score (AFS). "
            "Saves a publication-quality 4-panel diagnostic figure to disk."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute or relative filesystem path to the query plant image."
                },
                "target_class": {
                    "type": "string",
                    "description": "Optional target species name (common or binomial) for targeted Grad-CAM saliency. If omitted, targets the model's top prediction."
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory where the 4-panel diagnostic plot should be saved. Defaults to 'reports/xai_diagnostics'."
                }
            },
            "required": ["image_path"]
        }
    },
    {
        "name": "get_plant_information",
        "description": (
            "Retrieves an exhaustive botanical, taxonomic, phytochemical, and pharmacopoeial profile "
            "for a given medicinal plant from our curated WHO/IMPPAT/NCCIH knowledge database. "
            "Accepts common vernacular names (e.g. 'Tulsi', 'Aloe Vera', 'Holy Basil') or scientific binomials."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "species_name": {
                    "type": "string",
                    "description": "Common vernacular name or binomial scientific name of the medicinal plant."
                }
            },
            "required": ["species_name"]
        }
    },
    {
        "name": "get_medicinal_uses",
        "description": (
            "Queries traditional medical systems and clinical trial evidence for a specific plant. "
            "Retrieves Ayurvedic pharmacodynamics (Rasa, Guna, Virya, Vipaka, Prabhava, Dosha Karma), "
            "Siddha/Unani indications, peer-reviewed clinical trials (CTRI, NCT, IRCT), bioactives, "
            "or safety/contraindications."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "species_name": {
                    "type": "string",
                    "description": "Common or scientific name of the medicinal plant."
                },
                "system": {
                    "type": "string",
                    "enum": ["all", "ayurveda", "siddha_unani", "clinical", "phytochemistry", "safety"],
                    "default": "all",
                    "description": "Specific medical framework or evidence category to filter by."
                }
            },
            "required": ["species_name"]
        }
    }
]

class MedicinalPlantsMCPServer:
    """
    Production-grade Model Context Protocol Server for Medicinal Plant AI & Pharmacopoeial Knowledge.
    """
    def __init__(self):
        self.workspace_root = WORKSPACE_ROOT
        self._knowledge_engine = None
        self._uncertainty_router = None
        self._xai_explainer = None

    @property
    def knowledge_engine(self):
        """Lazy-instantiated knowledge engine."""
        if self._knowledge_engine is None:
            logger.info("Initializing MedicinalKnowledgeEngine...")
            from src.knowledge.engine import MedicinalKnowledgeEngine
            self._knowledge_engine = MedicinalKnowledgeEngine()
        return self._knowledge_engine

    @property
    def uncertainty_router(self):
        """Lazy-instantiated uncertainty router."""
        if self._uncertainty_router is None:
            logger.info("Initializing UncertaintyRouter...")
            from src.inference.uncertainty import UncertaintyRouter
            self._uncertainty_router = UncertaintyRouter()
        return self._uncertainty_router

    @property
    def xai_explainer(self):
        """Lazy-instantiated XAI diagnostic engine (shares model with uncertainty router)."""
        if self._xai_explainer is None:
            logger.info("Initializing PlantXAIExplainer...")
            from src.xai.diagnostics import PlantXAIExplainer
            router = self.uncertainty_router
            self._xai_explainer = PlantXAIExplainer(
                model=router.model,
                device=router.device
            )
        return self._xai_explainer

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    def tool_identify_medicinal_plant(self, image_path: str, organ_hint: Optional[str] = None) -> Dict[str, Any]:
        """Tool 1: Identifies plant and evaluates uncertainty routing."""
        resolved_path = image_path
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(self.workspace_root, image_path)

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Plant image not found at specified path: '{image_path}'")

        route_res = self.uncertainty_router.route(resolved_path, organ_hint=organ_hint)

        pred_info = route_res.get("prediction") or route_res.get("local_prediction") or {}
        top_species = pred_info.get("species", "Unknown")
        confidence = pred_info.get("confidence", 0.0)
        organ = pred_info.get("organ", organ_hint or "general")

        # Provide immediate link to knowledge base if recognized
        knowledge_summary = self.knowledge_engine.resolve_species_key(top_species)

        return {
            "source": route_res.get("source"),
            "is_uncertain": route_res.get("is_uncertain"),
            "routing_reason": route_res.get("routing_reason"),
            "detected_species": top_species,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "detected_organ": organ,
            "uncertainty_metrics": route_res.get("uncertainty_metrics"),
            "top5_candidates": pred_info.get("top5_candidates", []),
            "fallback_payload": route_res.get("fallback_payload"),
            "curated_knowledge_available": knowledge_summary is not None,
            "curated_species_key": knowledge_summary
        }

    def tool_explain_plant_prediction(
        self,
        image_path: str,
        target_class: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Tool 2: Generates multimodal XAI visual explanation and 4-panel figure."""
        resolved_path = image_path
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(self.workspace_root, image_path)

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Plant image not found at specified path: '{image_path}'")

        out_directory = output_dir or os.path.join(self.workspace_root, "reports", "xai_diagnostics")
        os.makedirs(out_directory, exist_ok=True)

        target_idx = None
        if target_class:
            clean_target = target_class.replace(" ", "_")
            target_idx = self.xai_explainer.metadata.get("species_to_idx", {}).get(clean_target)

        base_stem = os.path.splitext(os.path.basename(resolved_path))[0]
        file_prefix = f"mcp_{base_stem}"

        diag_res = self.xai_explainer.explain_image(
            image_input=resolved_path,
            target_species_idx=target_idx,
            save_plot=True,
            output_dir=out_directory,
            file_prefix=file_prefix
        )

        plot_path = diag_res.get("diagnostic_plot_path")
        rel_plot_path = os.path.relpath(plot_path, self.workspace_root) if plot_path else None

        return {
            "predicted_species": diag_res.get("species_name"),
            "species_confidence": f"{diag_res.get('species_confidence', 0.0) * 100:.2f}%",
            "predicted_organ": diag_res.get("organ_name"),
            "organ_confidence": f"{diag_res.get('organ_confidence', 0.0) * 100:.2f}%",
            "anatomical_focus_score_afs": f"{diag_res.get('anatomical_focus_score', 0.0):.2f}%",
            "diagnostic_figure_path": rel_plot_path,
            "diagnostic_figure_absolute_path": plot_path,
            "xai_interpretability_summary": {
                "cnn_grad_cam": "Highlights micro-textural epidermal markers (venation, trichomes, dentation) activating terminal conv layer.",
                "vit_attention_rollout": "Highlights macroscopic morphological boundaries and spatial organ contour attended by 6-layer Transformer.",
                "fused_composite": "Blended 50/50 saliency map overlayed at 48% opacity directly on the RGB plant photograph.",
                "clinical_relevance": f"Anatomical Focus Score of {diag_res.get('anatomical_focus_score', 0.0):.1f}% verifies proportion of attention resting on authentic plant tissue."
            }
        }

    def tool_get_plant_information(self, species_name: str) -> Dict[str, Any]:
        """Tool 3: Retrieves exhaustive botanical and clinical profile."""
        return self.knowledge_engine.get_species_profile(species_name)

    def tool_get_medicinal_uses(self, species_name: str, system: Optional[str] = "all") -> Dict[str, Any]:
        """Tool 4: Retrieves pharmacodynamics and clinical evidence filtered by system."""
        return self.knowledge_engine.get_medicinal_uses(species_name, system=system)

    # -------------------------------------------------------------------------
    # Dispatcher & JSON-RPC Protocol Layer
    # -------------------------------------------------------------------------

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an MCP tool call and returns result."""
        if name == "identify_medicinal_plant":
            return self.tool_identify_medicinal_plant(
                image_path=arguments.get("image_path"),
                organ_hint=arguments.get("organ_hint")
            )
        elif name == "explain_plant_prediction":
            return self.tool_explain_plant_prediction(
                image_path=arguments.get("image_path"),
                target_class=arguments.get("target_class"),
                output_dir=arguments.get("output_dir")
            )
        elif name == "get_plant_information":
            return self.tool_get_plant_information(
                species_name=arguments.get("species_name")
            )
        elif name == "get_medicinal_uses":
            return self.tool_get_medicinal_uses(
                species_name=arguments.get("species_name"),
                system=arguments.get("system", "all")
            )
        else:
            raise ValueError(f"Unknown MCP tool requested: '{name}'")

    def handle_request(self, request_data: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Handles a single JSON-RPC 2.0 request and returns corresponding response."""
        if isinstance(request_data, str):
            try:
                request = json.loads(request_data)
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
                }
        else:
            request = request_data

        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        # Handle notifications (e.g. notifications/initialized) which have no id
        if req_id is None and method:
            logger.info(f"Received notification: {method}")
            return None

        try:
            # 1. MCP initialize handshake
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {
                                "listChanged": False
                            }
                        },
                        "serverInfo": {
                            "name": "mcp-server-medicinal-plants",
                            "version": "1.0.0",
                            "description": "VTU Major Project: Hybrid CNN-ViT, XAI & Pharmacopoeial Knowledge MCP Server"
                        }
                    }
                }

            # 2. Ping
            elif method == "ping":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {}
                }

            # 3. List tools
            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": TOOLS_REGISTRY
                    }
                }

            # 4. Call tool
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})

                if not tool_name:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Missing 'name' in tools/call parameters"}
                    }

                tool_result = self.call_tool(tool_name, tool_args)
                formatted_text = json.dumps(tool_result, indent=2, ensure_ascii=False)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": formatted_text
                            }
                        ],
                        "isError": False
                    }
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method '{method}' not found"}
                }

        except Exception as e:
            logger.error(f"Error handling method '{method}': {str(e)}\n{traceback.format_exc()}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": str(e),
                    "data": {"traceback": traceback.format_exc()}
                }
            }

    def run_stdio(self):
        """Runs the MCP server reading line-delimited JSON-RPC from standard input."""
        logger.info("Medicinal Plants MCP Server started listening on stdio...")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            response = self.handle_request(line)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

if __name__ == "__main__":
    server = MedicinalPlantsMCPServer()
    server.run_stdio()

