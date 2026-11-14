"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Module: src/ui/app.py

Interactive Web Dashboard & Demonstration Suite:
- Gradio Blocks implementation (build_gradio_blocks) with Rich HTML cards
- Zero-dependency Standalone Web Server fallback (launch_standalone_web_dashboard)
- Interactive visual diagnostic plots (Grad-CAM, ViT Attention Rollout, Composite)
- Anatomical Focus Score (AFS) visualization
- Multi-Tab Pharmacopoeial Knowledge Navigator with Rich Human-Readable UI:
    * 📖 Taxonomy: Botanical identity, regional vernacular names, synonyms, organ morphology
    * 🧘 Ayurveda: Dravya Guna energetics (Rasa, Guna, Virya, Vipaka, Prabhava), Dosha karma
    * 🏥 Clinical Trials: Evaluated therapeutic indications, WHO/CTRI/NCT human clinical trial cards
    * 🧪 Phytochemistry & Safety: Bioactive compounds table, adverse effects, pregnancy & drug interactions
    * ⚙️ Developer JSON: Full raw JSON telemetry, latency profile, and logit distribution
- Full developer telemetry and JSON inspector

Author: Lead Full-Stack AI Solutions & Systems Architect
"""

import os
import sys
import json
import base64
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.pipeline.orchestrator import analyze_medicinal_plant

# -----------------------------------------------------------------------------
# Rich Python HTML Formatters (for Gradio Blocks & Server-Side Rendering)
# -----------------------------------------------------------------------------

def format_taxonomy_html(pk: Dict[str, Any]) -> str:
    if not pk or not pk.get("taxonomy"):
        return '<div style="padding: 16px; color: #94a3b8;">No taxonomy data available.</div>'
    tax = pk.get("taxonomy", {})
    morph = pk.get("morphology", {})
    vern = tax.get("vernacular_names", {})
    syns = tax.get("synonyms", [])

    flags = {
        "english": "🇬🇧 English", "hindi": "🇮🇳 Hindi", "sanskrit": "🕉️ Sanskrit",
        "kannada": "🟡 Kannada", "tamil": "🔴 Tamil", "telugu": "🔵 Telugu",
        "malayalam": "🟢 Malayalam", "marathi": "🟠 Marathi", "bengali": "🟣 Bengali"
    }

    vern_items = "".join([
        f'<div style="background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px;">'
        f'<div style="font-size: 11px; text-transform: uppercase; color: #94a3b8;">{flags.get(k.lower(), k.upper())}</div>'
        f'<div style="font-size: 13px; font-weight: 600; color: #f8fafc;">{v}</div></div>'
        for k, v in vern.items()
    ])

    syn_badges = "".join([
        f'<span style="display: inline-block; background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 4px 10px; border-radius: 9999px; font-size: 11px; margin: 2px 4px 2px 0;"><em>{s}</em></span>'
        for s in syns
    ])

    morph_icons = {
        "leaves": "🍃 Foliage & Leaves",
        "stem": "🌿 Stem & Bark Habit",
        "flowers": "🌸 Inflorescence & Flowers",
        "seeds_fruits": "🌰 Fruits & Seeds"
    }
    morph_items = "".join([
        f'<div style="background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">'
        f'<div style="font-size: 12px; font-weight: 700; color: #38bdf8; margin-bottom: 4px;">{morph_icons.get(k, k.upper())}</div>'
        f'<div style="font-size: 13px; color: #cbd5e1; line-height: 1.4;">{v}</div></div>'
        for k, v in morph.items()
    ])

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f8fafc;">
        <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
                <div>
                    <h3 style="margin: 0 0 4px 0; font-size: 18px; color: #34d399;">
                        {tax.get('canonical_name', 'Botanical Specimen')} 
                        <span style="font-size: 14px; font-weight: 400; color: #94a3b8; font-style: italic;">({vern.get('english', tax.get('canonical_name', ''))})</span>
                    </h3>
                    <p style="margin: 0; font-size: 14px; color: #cbd5e1;">
                        <em>{tax.get('scientific_name', '')}</em> <span style="color: #94a3b8;">{tax.get('author', '')}</span>
                    </p>
                </div>
                <div>
                    <span style="display: inline-block; background: #065f4630; color: #34d399; border: 1px solid #05966950; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600;">Family: {tax.get('botanical_family', 'N/A')}</span>
                    {f'<span style="display: inline-block; background: #0284c725; color: #38bdf8; border: 1px solid #0284c750; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600; margin-left: 6px;">GBIF ID: {tax.get("taxonomic_serial_number_gbif")}</span>' if tax.get('taxonomic_serial_number_gbif') else ''}
                </div>
            </div>
        </div>

        {f'<div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 14px;"><div style="font-size: 13px; font-weight: 700; color: #34d399; margin-bottom: 10px;">🌐 Vernacular & Multilingual Regional Names</div><div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px;">{vern_items}</div></div>' if vern_items else ''}

        {f'<div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 14px;"><div style="font-size: 13px; font-weight: 700; color: #34d399; margin-bottom: 6px;">🏷️ Botanical & Taxonomic Synonyms</div><div>{syn_badges}</div></div>' if syn_badges else ''}

        {f'<div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px;"><div style="font-size: 13px; font-weight: 700; color: #34d399; margin-bottom: 10px;">🔬 Diagnostic Organ Morphology</div><div>{morph_items}</div></div>' if morph_items else ''}
    </div>
    """

def format_ayurveda_html(pk: Dict[str, Any]) -> str:
    if not pk or not pk.get("traditional_systems"):
        return '<div style="padding: 16px; color: #94a3b8;">No traditional systems data available.</div>'
    trad = pk.get("traditional_systems", {})
    ayur = trad.get("ayurveda", {})
    sid = trad.get("siddha", {})
    unani = trad.get("unani", {})

    rasa_badges = "".join([f'<span style="display: inline-block; background: #065f4630; color: #34d399; border: 1px solid #05966950; padding: 4px 10px; border-radius: 9999px; font-size: 11px; margin-right: 6px;">{r}</span>' for r in ayur.get("rasa", [])])
    guna_badges = "".join([f'<span style="display: inline-block; background: #0284c725; color: #38bdf8; border: 1px solid #0284c750; padding: 4px 10px; border-radius: 9999px; font-size: 11px; margin-right: 6px;">{g}</span>' for g in ayur.get("guna", [])])

    sid_indications = "".join([f'<li>{ind}</li>' for ind in sid.get("therapeutic_indications", [])])
    unani_actions = "".join([f'<span style="display: inline-block; background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 4px 10px; border-radius: 9999px; font-size: 11px; margin-right: 6px;">{act}</span>' for act in unani.get("actions", [])])

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f8fafc;">
        <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 14px;">
            <div style="font-size: 14px; font-weight: 700; color: #34d399; margin-bottom: 12px;">🧘 Ayurvedic Dravya Guna Pharmacodynamics</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-bottom: 12px;">
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px 14px;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8;">🔥 Virya (Therapeutic Potency)</div>
                    <div style="font-size: 13px; font-weight: 700; color: #f59e0b;">{ayur.get('virya', 'N/A')}</div>
                </div>
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px 14px;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8;">⚡ Vipaka (Post-Digestive Effect)</div>
                    <div style="font-size: 13px; font-weight: 700; color: #38bdf8;">{ayur.get('vipaka', 'N/A')}</div>
                </div>
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px 14px; grid-column: 1 / -1;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8;">⭐ Prabhava (Special Action)</div>
                    <div style="font-size: 13px; font-weight: 700; color: #34d399;">{ayur.get('prabhava', 'N/A')}</div>
                </div>
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px 14px; grid-column: 1 / -1;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8;">⚖️ Dosha Karma (Tridosha Impact)</div>
                    <div style="font-size: 13px; color: #f8fafc;">{ayur.get('dosha_karma', 'N/A')}</div>
                </div>
            </div>

            <div style="display: flex; flex-wrap: wrap; gap: 16px; margin-top: 10px;">
                <div><div style="font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 4px;">Rasa (Taste Profile):</div><div>{rasa_badges or '<span style=\"color:#64748b;\">N/A</span>'}</div></div>
                <div><div style="font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 4px;">Guna (Qualities):</div><div>{guna_badges or '<span style=\"color:#64748b;\">N/A</span>'}</div></div>
            </div>
        </div>

        {f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px;">'
          f'<div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px;">'
          f'<div style="font-size: 13px; font-weight: 700; color: #34d399; margin-bottom: 8px;">🌿 Siddha System</div>'
          f'<div style="margin-bottom: 6px;"><strong style="font-size: 12px; color: #94a3b8;">Quality:</strong> <span style="display: inline-block; background: #b4530925; color: #fbbf24; border: 1px solid #b4530950; padding: 2px 8px; border-radius: 9999px; font-size: 11px;">{sid.get("humoral_quality", "N/A")}</span></div>'
          f'<div style="font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 4px;">Indications:</div><ul style="margin: 0; padding-left: 18px; font-size: 13px; color: #cbd5e1;">{sid_indications}</ul></div>'
          if sid else ''}

        {f'<div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px;">'
          f'<div style="font-size: 13px; font-weight: 700; color: #34d399; margin-bottom: 8px;">📜 Unani Tibb System</div>'
          f'<div style="margin-bottom: 6px;"><strong style="font-size: 12px; color: #94a3b8;">Mizaj:</strong> <span style="display: inline-block; background: #0284c725; color: #38bdf8; border: 1px solid #0284c750; padding: 2px 8px; border-radius: 9999px; font-size: 11px;">{unani.get("mizaj", "N/A")}</span></div>'
          f'<div style="font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 4px;">Actions:</div><div>{unani_actions}</div></div></div>'
          if unani else ''}
    </div>
    """

def format_clinical_html(pk: Dict[str, Any]) -> str:
    if not pk or not pk.get("clinical_evidence"):
        return '<div style="padding: 16px; color: #94a3b8;">No clinical evidence data available.</div>'
    clin = pk.get("clinical_evidence", {})
    indications = clin.get("evaluated_indications", [])
    trials = clin.get("trial_registries", [])

    ind_chips = "".join([
        f'<div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; font-size: 13px; color: #e2e8f0; display: flex; align-items: center; gap: 8px;">'
        f'<span style="color: #34d399; font-weight: 700;">✓</span> {ind}</div>'
        for ind in indications
    ])

    trial_cards = "".join([
        f'<div style="background: #0f172a; border-left: 3px solid #38bdf8; border: 1px solid #334155; border-left-width: 4px; border-radius: 6px; padding: 14px; margin-bottom: 12px;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
        f'<span style="background: #0284c725; color: #38bdf8; border: 1px solid #0284c750; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700;">Registry: {t.get("registry_id")}</span>'
        f'<span style="font-size: 12px; color: #94a3b8;">Human Cohort Study</span></div>'
        f'<div style="font-size: 13px; margin-bottom: 4px;"><strong style="color: #f8fafc;">Study Design:</strong> <span style="color: #cbd5e1;">{t.get("study_design")}</span></div>'
        f'<div style="font-size: 13px; margin-bottom: 6px;"><strong style="color: #f8fafc;">Dosage Form:</strong> <span style="color: #cbd5e1;">{t.get("dosage_form")}</span></div>'
        f'<div style="background: #1e293b; border-left: 3px solid #34d399; padding: 8px 12px; border-radius: 4px; font-size: 12.5px; color: #a7f3d0; line-height: 1.4;">'
        f'<strong>Clinical Outcomes:</strong> {t.get("outcomes")}</div></div>'
        for t in trials
    ])

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f8fafc;">
        <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 14px; font-weight: 700; color: #34d399;">🏥 Clinical Evidence Consensus</div>
                <span style="background: #065f4630; color: #34d399; border: 1px solid #05966950; padding: 4px 12px; border-radius: 9999px; font-size: 11px; font-weight: 700;">{clin.get('evidence_grading', 'Evaluated').split('(')[0].strip()}</span>
            </div>
            <p style="margin: 0; font-size: 13px; color: #cbd5e1; line-height: 1.5;">
                {clin.get('evidence_grading', 'Peer-reviewed human clinical evidence indexed from WHO Monographs and Clinical Trial Registries.')}
            </p>
        </div>

        {f'<div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 14px;"><div style="font-size: 13px; font-weight: 700; color: #34d399; margin-bottom: 10px;">🎯 Evaluated Therapeutic Indications</div><div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 8px;">{ind_chips}</div></div>' if ind_chips else ''}

        {f'<div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px;"><div style="font-size: 13px; font-weight: 700; color: #34d399; margin-bottom: 12px;">📑 Peer-Reviewed Human Clinical Trial Registries</div><div>{trial_cards}</div></div>' if trial_cards else ''}
    </div>
    """

def format_phyto_html(pk: Dict[str, Any]) -> str:
    if not pk:
        return '<div style="padding: 16px; color: #94a3b8;">No phytochemistry data available.</div>'
    phyto = pk.get("phytochemistry", {})
    safety = pk.get("safety_and_contraindications", {})
    bioactives = phyto.get("primary_bioactives", [])
    interactions = safety.get("herb_drug_interactions", [])

    table_rows = "".join([
        f'<tr><td style="padding: 10px 12px; border-bottom: 1px solid #33415550;"><strong style="color: #38bdf8;">{b.get("compound")}</strong></td>'
        f'<td style="padding: 10px 12px; border-bottom: 1px solid #33415550;"><span style="background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 2px 8px; border-radius: 9999px; font-size: 11px;">{b.get("class")}</span></td>'
        f'<td style="padding: 10px 12px; border-bottom: 1px solid #33415550; color: #94a3b8;">{b.get("fraction")}</td></tr>'
        for b in bioactives
    ])

    inter_list = "".join([f'<li>{inter}</li>' for inter in interactions])

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f8fafc;">
        {f'<div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 14px;"><div style="font-size: 13px; font-weight: 700; color: #34d399; margin-bottom: 10px;">🧪 Primary Bioactive Phytochemicals</div><table style="width: 100%; border-collapse: collapse; font-size: 13px;"><thead><tr style="text-align: left; background: #1e293b; color: #94a3b8; font-size: 11px; text-transform: uppercase;"><th style="padding: 8px 12px;">Compound</th><th style="padding: 8px 12px;">Chemical Class</th><th style="padding: 8px 12px;">Botanical Fraction</th></tr></thead><tbody>{table_rows}</tbody></table></div>' if table_rows else ''}

        <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px;">
            <div style="font-size: 13px; font-weight: 700; color: #34d399; margin-bottom: 12px;">⚠️ Clinical Safety, Toxicity & Contraindications</div>

            {f'<div style="background: #78350f20; border-left: 4px solid #f59e0b; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; font-size: 13px; color: #fde68a;"><strong>Adverse Effects & Toxicity:</strong> {safety.get("adverse_effects")}</div>' if safety.get("adverse_effects") else ''}

            {f'<div style="background: #7f1d1d20; border-left: 4px solid #ef4444; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; font-size: 13px; color: #fca5a5;"><strong>Pregnancy & Lactation Advisory:</strong> {safety.get("pregnancy_lactation_precautions")}</div>' if safety.get("pregnancy_lactation_precautions") else ''}

            {f'<div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 12px; margin-bottom: 12px;"><div style="font-size: 13px; font-weight: 700; color: #fbbf24; margin-bottom: 6px;">💊 Herb-Drug Interactions:</div><ul style="margin: 0; padding-left: 18px; font-size: 12.5px; color: #cbd5e1; line-height: 1.4;">{inter_list}</ul></div>' if inter_list else ''}

            {f'<div style="background: #0c4a6e20; border-left: 4px solid #0284c7; border-radius: 6px; padding: 12px 16px; font-size: 13px; color: #bae6fd;"><strong>Evidence Limitations:</strong> {safety.get("clinical_evidence_limitations")}</div>' if safety.get("clinical_evidence_limitations") else ''}
        </div>
    </div>
    """

# -----------------------------------------------------------------------------
# Gradio Blocks Application Definition
# -----------------------------------------------------------------------------

def build_gradio_blocks():
    """Builds and returns the Gradio Blocks application."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is not installed in the current environment.")

    def run_gradio_pipeline(image, organ_hint, conf_thresh, ent_thresh, gen_xai):
        if image is None:
            return (
                "⚠️ Please upload a plant image or select an example below.",
                None,
                "0.0%",
                "<p style='color: #94a3b8;'>No image analyzed.</p>",
                "<p style='color: #94a3b8;'>No image analyzed.</p>",
                "<p style='color: #94a3b8;'>No image analyzed.</p>",
                "<p style='color: #94a3b8;'>No image analyzed.</p>",
                "{}"
            )

        organ_param = None if organ_hint == "Auto-Detect / General" else organ_hint.lower().replace(" ", "_")
        res = analyze_medicinal_plant(
            image_input=image,
            organ_hint=organ_param,
            confidence_threshold=float(conf_thresh),
            entropy_threshold=float(ent_thresh),
            generate_xai=bool(gen_xai),
            output_dir=os.path.join(WORKSPACE_ROOT, "reports", "pipeline_runs")
        )

        # 1. Status Banner
        banner_color = "#10b981" if not res["identification"]["is_uncertain"] else "#f59e0b"
        banner_html = f"""
        <div style="background-color: {banner_color}18; border-left: 5px solid {banner_color}; padding: 14px 18px; border-radius: 6px; margin-bottom: 12px;">
            <h3 style="margin: 0 0 6px 0; color: {banner_color}; font-size: 16px;">{res['status_banner']}</h3>
            <p style="margin: 0; font-size: 14px; color: #374151;">
                <strong>Detected:</strong> {res['identification']['detected_species']} ({res['identification']['scientific_name']}) &nbsp;|&nbsp; 
                <strong>Confidence:</strong> {res['identification']['confidence_percentage']} &nbsp;|&nbsp; 
                <strong>Organ:</strong> {res['identification']['detected_organ']} &nbsp;|&nbsp; 
                <strong>Latency:</strong> {res['latency_profile']['total_latency_ms']} ms
            </p>
        </div>
        """

        # 2. XAI Diagnostic Image
        plot_path = None
        afs_str = "N/A"
        if res.get("explainable_ai", {}).get("diagnostic_plot_absolute_path"):
            plot_path = res["explainable_ai"]["diagnostic_plot_absolute_path"]
            afs_str = res["explainable_ai"].get("afs_percentage", "N/A")

        # 3. Rich Tab Contents
        pk = res.get("pharmacopoeial_knowledge", {})
        tax_html = format_taxonomy_html(pk)
        ayur_html = format_ayurveda_html(pk)
        clin_html = format_clinical_html(pk)
        phyto_html = format_phyto_html(pk)
        raw_json = json.dumps(res, indent=2)

        return banner_html, plot_path, afs_str, tax_html, ayur_html, clin_html, phyto_html, raw_json

    # Pre-configure sample images if they exist
    sample_tulsi = os.path.join(WORKSPACE_ROOT, "assets/sample_images/tulsi.jpg")
    sample_aloe = os.path.join(WORKSPACE_ROOT, "assets/sample_images/aloe_vera.jpg")
    sample_cath = os.path.join(WORKSPACE_ROOT, "assets/sample_images/periwinkle.jpg")
    sample_tomato = os.path.join(WORKSPACE_ROOT, "assets/sample_images/tomato.jpg")

    examples_list = []
    for s_path, hint in [(sample_tulsi, "General"), (sample_aloe, "General"), (sample_cath, "Leaf"), (sample_tomato, "General")]:
        if os.path.exists(s_path):
            examples_list.append([s_path, hint, 0.60, 0.45, True])

    theme = gr.themes.Soft(
        primary_hue="emerald",
        neutral_hue="slate"
    )

    with gr.Blocks(theme=theme, title="VTU Medicinal Plant AI") as demo:
        gr.Markdown(
            """
            # 🌿 Intelligent Medicinal Plant Recognition System
            ### VTU Major Project: Hybrid CNN–ViT Architecture, Explainable AI & MCP Knowledge Server
            *Dual-branch computer vision (MobileNetV2 + 6-Block ViT), Grad-CAM & Attention Rollout diagnostics, and Ayurvedic pharmacopoeial knowledge integration.*
            """
        )

        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=4):
                input_image = gr.Image(type="filepath", label="Upload Medicinal Plant Image", sources=["upload", "webcam"])
                organ_hint = gr.Dropdown(
                    choices=["Auto-Detect / General", "Leaf", "Flower", "Fruit", "Bark", "Whole Plant"],
                    value="Auto-Detect / General",
                    label="Botanical Organ Hint (Optional)"
                )
                with gr.Accordion("Advanced Uncertainty & XAI Settings", open=False):
                    conf_slider = gr.Slider(minimum=0.10, maximum=0.95, value=0.60, step=0.05, label="Confidence Threshold (tau_conf)")
                    ent_slider = gr.Slider(minimum=0.05, maximum=0.95, value=0.45, step=0.05, label="Normalized Entropy Threshold (tau_entropy)")
                    xai_toggle = gr.Checkbox(value=True, label="Generate Explainable AI Heatmaps (Grad-CAM + ViT Rollout)")

                analyze_btn = gr.Button("🔍 Analyze Medicinal Plant", variant="primary", size="lg")

                if examples_list:
                    gr.Examples(
                        examples=examples_list,
                        inputs=[input_image, organ_hint, conf_slider, ent_slider, xai_toggle],
                        label="Pre-loaded Test Partition Samples"
                    )

            # Right Column: Results & Knowledge
            with gr.Column(scale=6):
                status_banner = gr.HTML(
                    """
                    <div style="background-color: #f3f4f6; padding: 14px 18px; border-radius: 6px; border: 1px dashed #d1d5db;">
                        <p style="margin: 0; color: #6b7280; font-size: 14px;">Upload a plant image and click <strong>Analyze Medicinal Plant</strong> to begin inference.</p>
                    </div>
                    """
                )

                with gr.Group():
                    gr.Markdown("### 🔬 Multimodal Explainable AI Diagnostics")
                    xai_plot = gr.Image(label="4-Panel Diagnostic: [1] Input | [2] CNN Grad-CAM | [3] ViT Attention Rollout | [4] Composite Overlay", interactive=False)
                    with gr.Row():
                        afs_display = gr.Textbox(label="Anatomical Focus Score (AFS)", value="N/A", scale=2)
                        gr.Markdown("*Measures percentage of visual saliency energy focused directly on authenticated plant anatomy.*", scale=8)

                with gr.Tabs():
                    with gr.TabItem("📖 Taxonomy & Morphology"):
                        out_tax = gr.HTML(label="Taxonomy Profile")
                    with gr.TabItem("🧘 Ayurvedic Energetics"):
                        out_ayur = gr.HTML(label="Ayurvedic & Traditional Systems")
                    with gr.TabItem("🏥 Clinical Trial Evidence"):
                        out_clin = gr.HTML(label="Peer-Reviewed Clinical Registries (WHO/CTRI/NCCIH)")
                    with gr.TabItem("🧪 Phytochemistry & Safety"):
                        out_phyto = gr.HTML(label="Bioactive Compounds & Contraindications")
                    with gr.TabItem("⚙️ Raw Developer JSON"):
                        out_raw = gr.Code(language="json", label="Full Telemetry & Latency Breakdown")

        analyze_btn.click(
            fn=run_gradio_pipeline,
            inputs=[input_image, organ_hint, conf_slider, ent_slider, xai_toggle],
            outputs=[status_banner, xai_plot, afs_display, out_tax, out_ayur, out_clin, out_phyto, out_raw]
        )

    return demo

# -----------------------------------------------------------------------------
# Embedded Standalone Web Dashboard (Zero-Dependency Modernized UI)
# -----------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VTU Major Project — Intelligent Medicinal Plant AI</title>
  <style>
    :root {
      --primary: #059669;
      --primary-hover: #047857;
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }
    .container { max-width: 1280px; margin: 0 auto; }
    header { margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
    h1 { font-size: 24px; font-weight: 700; color: #34d399; margin-bottom: 6px; }
    .subtitle { font-size: 14px; color: var(--text-muted); }
    .grid { display: grid; grid-template-columns: 420px 1fr; gap: 24px; }
    @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
    .upload-zone { border: 2px dashed #475569; border-radius: 8px; padding: 32px 16px; text-align: center; cursor: pointer; transition: all 0.2s; background: #0f172a50; margin-bottom: 16px; }
    .upload-zone:hover { border-color: var(--primary); background: #05966910; }
    .btn { display: inline-block; width: 100%; background: var(--primary); color: white; border: none; padding: 12px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: background 0.2s; font-size: 15px; }
    .btn:hover { background: var(--primary-hover); }
    .samples { display: flex; gap: 8px; margin-top: 12px; }
    .sample-btn { flex: 1; padding: 8px; background: #334155; border: 1px solid #475569; border-radius: 6px; color: var(--text); font-size: 12px; cursor: pointer; }
    .sample-btn:hover { background: #475569; }
    .form-group { margin-bottom: 14px; }
    label { display: block; font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; }
    select, input[type="range"] { width: 100%; background: #0f172a; border: 1px solid var(--border); color: white; padding: 8px 12px; border-radius: 6px; font-size: 14px; }
    .slider-val { float: right; font-weight: bold; color: var(--accent); }
    .banner { padding: 14px 18px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; display: none; }
    .banner-confident { background: #065f4625; border-left: 4px solid #10b981; color: #a7f3d0; }
    .banner-uncertain { background: #78350f25; border-left: 4px solid #f59e0b; color: #fde68a; }
    .xai-preview { width: 100%; border-radius: 8px; margin-top: 10px; border: 1px solid var(--border); display: none; }
    .tabs { display: flex; border-bottom: 1px solid var(--border); margin: 20px 0 14px 0; gap: 8px; }
    .tab-btn { background: none; border: none; color: var(--text-muted); padding: 8px 14px; font-size: 13px; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; }
    .tab-btn.active { color: #34d399; border-bottom-color: #34d399; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    pre { background: #0f172a; border: 1px solid var(--border); padding: 14px; border-radius: 8px; overflow-x: auto; font-size: 12px; max-height: 450px; color: #cbd5e1; }
    .metric-chip { display: inline-block; background: #0f172a; border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; font-size: 12px; margin-right: 8px; margin-bottom: 8px; }
    .metric-chip strong { color: var(--accent); }
    #spinner { display: none; text-align: center; padding: 20px; font-weight: bold; color: var(--accent); }

    /* Rich Card Styling for Human-Readable Knowledge UI */
    .section-card { background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 14px; }
    .section-title { font-size: 14px; font-weight: 700; color: #34d399; margin-bottom: 10px; display: flex; align-items: center; gap: 6px; }
    .badge { display: inline-block; padding: 3px 9px; border-radius: 9999px; font-size: 11px; font-weight: 600; margin: 2px 4px 2px 0; }
    .badge-primary { background: #065f4630; color: #34d399; border: 1px solid #05966950; }
    .badge-secondary { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; }
    .badge-accent { background: #0284c725; color: #38bdf8; border: 1px solid #0284c750; }
    .badge-warning { background: #b4530925; color: #fbbf24; border: 1px solid #b4530950; }
    .badge-danger { background: #991b1b25; color: #f87171; border: 1px solid #991b1b50; }
    .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
    .info-item { background: #1e293b; border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; }
    .info-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); margin-bottom: 4px; }
    .info-value { font-size: 13px; font-weight: 600; color: #f8fafc; }
    .clinical-trial-card { background: #0f172a; border: 1px solid #334155; border-left: 4px solid #38bdf8; border-radius: 6px; padding: 14px; margin-bottom: 12px; }
    .table-custom { width: 100%; border-collapse: collapse; font-size: 13px; }
    .table-custom th { text-align: left; padding: 8px 12px; background: #1e293b; color: var(--text-muted); border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; }
    .table-custom td { padding: 10px 12px; border-bottom: 1px solid #33415550; color: #cbd5e1; }
    .alert-box { border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; font-size: 13px; }
    .alert-warning { background: #78350f20; border-left: 4px solid #f59e0b; color: #fde68a; }
    .alert-danger { background: #7f1d1d20; border-left: 4px solid #ef4444; color: #fca5a5; }
    .alert-info { background: #0c4a6e20; border-left: 4px solid #0284c7; color: #bae6fd; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🌿 Intelligent Medicinal Plant Recognition & Diagnostic Dashboard</h1>
      <p class="subtitle">VTU Major Project | Hybrid CNN–ViT Backbone (1,203 Species) | Multimodal XAI (Grad-CAM & Rollout) | Model Context Protocol Knowledge Engine</p>
    </header>

    <div class="grid">
      <!-- Input Panel -->
      <div class="card">
        <h2 style="font-size: 16px; margin-bottom: 14px; color: #38bdf8;">Query Input & Control Parameters</h2>
        <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
          <p id="upload-label">📸 Click or Drag & Drop Plant Image</p>
          <input type="file" id="file-input" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
        </div>

        <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 6px;">Or test with verified partition samples:</p>
        <div class="samples">
          <button class="sample-btn" onclick="loadSample('assets/sample_images/tulsi.jpg', 'Tulsi')">Tulsi (Sample 1)</button>
          <button class="sample-btn" onclick="loadSample('assets/sample_images/aloe_vera.jpg', 'Aloe Vera')">Aloe Vera (Sample 2)</button>
          <button class="sample-btn" onclick="loadSample('assets/sample_images/periwinkle.jpg', 'Catharanthus')">Periwinkle (Sample 3)</button>
          <button class="sample-btn" onclick="loadSample('assets/sample_images/tomato.jpg', 'Tomato')">Tomato (Sample 4)</button>
        </div>

        <div class="form-group" style="margin-top: 16px;">
          <label>Botanical Organ Hint</label>
          <select id="organ-select">
            <option value="">Auto-Detect / General</option>
            <option value="leaf">Leaf</option>
            <option value="flower">Flower</option>
            <option value="fruit">Fruit / Seed</option>
            <option value="bark">Bark / Stem</option>
            <option value="whole_plant">Whole Plant / Habit</option>
          </select>
        </div>

        <div class="form-group">
          <label>Confidence Threshold (&tau;<sub>conf</sub>): <span class="slider-val" id="conf-val">0.60</span></label>
          <input type="range" id="conf-slider" min="0.10" max="0.95" step="0.05" value="0.60" oninput="document.getElementById('conf-val').innerText = this.value">
        </div>

        <div class="form-group">
          <label>Normalized Entropy Threshold (&tau;<sub>entropy</sub>): <span class="slider-val" id="ent-val">0.45</span></label>
          <input type="range" id="ent-slider" min="0.05" max="0.95" step="0.05" value="0.45" oninput="document.getElementById('ent-val').innerText = this.value">
        </div>

        <div class="form-group">
          <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
            <input type="checkbox" id="xai-check" checked style="width: auto;">
            Generate Multimodal Explainable AI Diagnostics
          </label>
        </div>

        <button class="btn" id="run-btn" onclick="executeInference()">🔍 Analyze Medicinal Plant</button>
      </div>

      <!-- Results Panel -->
      <div class="card">
        <h2 style="font-size: 16px; margin-bottom: 14px; color: #34d399;">Diagnostic Synthesis & Pharmacopoeial Knowledge</h2>
        <div id="spinner">⚙️ Running Hybrid CNN–ViT Inference & Saliency Rollout...</div>
        <div id="status-banner" class="banner"></div>

        <div id="metrics-chips" style="margin-bottom: 12px;"></div>

        <!-- 4-Panel XAI Diagnostic Plot -->
        <div id="xai-container" style="display: none; margin-bottom: 16px;">
          <h3 style="font-size: 14px; color: #38bdf8; margin-bottom: 6px;">Multimodal XAI Side-by-Side Diagnostics</h3>
          <img id="xai-img" class="xai-preview" alt="XAI Diagnostic Plot">
          <p id="afs-caption" style="font-size: 12px; color: var(--text-muted); margin-top: 6px;"></p>
        </div>

        <!-- Knowledge Tabs -->
        <div class="tabs">
          <button class="tab-btn active" onclick="switchTab(0)">📖 Taxonomy</button>
          <button class="tab-btn" onclick="switchTab(1)">🧘 Ayurveda</button>
          <button class="tab-btn" onclick="switchTab(2)">🏥 Clinical Trials</button>
          <button class="tab-btn" onclick="switchTab(3)">🧪 Phytochemistry & Safety</button>
          <button class="tab-btn" onclick="switchTab(4)">⚙️ Developer JSON</button>
        </div>

        <div id="tab-0" class="tab-content active">
          <div id="tax-container">
            <div style="text-align: center; padding: 36px 12px; color: #64748b;">
              <p style="font-size: 14px;">Select a plant image and click <strong>🔍 Analyze Medicinal Plant</strong> to view structured botanical taxonomy, morphology, and multilingual names.</p>
            </div>
          </div>
        </div>

        <div id="tab-1" class="tab-content">
          <div id="ayur-container">
            <div style="text-align: center; padding: 36px 12px; color: #64748b;">
              <p style="font-size: 14px;">Ayurvedic Dravya Guna energetics (Virya, Vipaka, Prabhava, Rasa, Guna) will render here.</p>
            </div>
          </div>
        </div>

        <div id="tab-2" class="tab-content">
          <div id="clin-container">
            <div style="text-align: center; padding: 36px 12px; color: #64748b;">
              <p style="font-size: 14px;">Peer-reviewed clinical trial registries (WHO Monographs, CTRI, NCT) will render here.</p>
            </div>
          </div>
        </div>

        <div id="tab-3" class="tab-content">
          <div id="phyto-container">
            <div style="text-align: center; padding: 36px 12px; color: #64748b;">
              <p style="font-size: 14px;">Bioactive phytochemical fractions, adverse reactions, and herb-drug contraindications will render here.</p>
            </div>
          </div>
        </div>

        <div id="tab-4" class="tab-content">
          <pre id="raw-pre">// Full JSON telemetry with subcomponent latency breakdown will render here</pre>
        </div>
      </div>
    </div>
  </div>

  <script>
    let selectedImageFile = null;
    let selectedSamplePath = "assets/sample_images/tulsi.jpg";

    function handleFileSelect(evt) {
      const file = evt.target.files[0];
      if (file) {
        selectedImageFile = file;
        selectedSamplePath = null;
        document.getElementById('upload-label').innerText = '✅ ' + file.name;
      }
    }

    function loadSample(path, label) {
      selectedSamplePath = path;
      selectedImageFile = null;
      document.getElementById('upload-label').innerText = 'Selected: ' + label + ' (' + path.split('/').pop() + ')';
    }

    function switchTab(index) {
      const btns = document.querySelectorAll('.tab-btn');
      const tabs = document.querySelectorAll('.tab-content');
      btns.forEach((b, i) => b.classList.toggle('active', i === index));
      tabs.forEach((t, i) => t.classList.toggle('active', i === index));
    }

    async function executeInference() {
      const spinner = document.getElementById('spinner');
      const banner = document.getElementById('status-banner');
      const btn = document.getElementById('run-btn');
      spinner.style.display = 'block';
      btn.disabled = true;

      try {
        let payload;
        if (selectedImageFile) {
          const reader = new FileReader();
          const base64Promise = new Promise(resolve => {
            reader.onload = () => resolve(reader.result);
            reader.readAsDataURL(selectedImageFile);
          });
          const b64 = await base64Promise;
          payload = {
            image_data: b64,
            organ_hint: document.getElementById('organ-select').value || null,
            confidence_threshold: parseFloat(document.getElementById('conf-slider').value),
            entropy_threshold: parseFloat(document.getElementById('ent-slider').value),
            generate_xai: document.getElementById('xai-check').checked
          };
        } else {
          payload = {
            image_path: selectedSamplePath,
            organ_hint: document.getElementById('organ-select').value || null,
            confidence_threshold: parseFloat(document.getElementById('conf-slider').value),
            entropy_threshold: parseFloat(document.getElementById('ent-slider').value),
            generate_xai: document.getElementById('xai-check').checked
          };
        }

        const resp = await fetch('/api/analyze', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });

        const data = await resp.json();
        renderResults(data);
      } catch (err) {
        alert('Error executing inference: ' + err);
      } finally {
        spinner.style.display = 'none';
        btn.disabled = false;
      }
    }

    function renderTaxonomy(pk) {
      if (!pk || !pk.taxonomy) return '<p style="color:#94a3b8; padding:16px;">No taxonomy data available.</p>';
      const tax = pk.taxonomy;
      const morph = pk.morphology || {};
      const vern = tax.vernacular_names || {};
      const syns = tax.synonyms || [];

      let html = `
        <div class="section-card">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
            <div>
              <h3 style="font-size:18px; font-weight:700; color:#34d399; margin-bottom:4px;">
                ${tax.canonical_name || 'Botanical Specimen'} 
                <span style="font-size:14px; font-weight:400; color:#94a3b8; font-style:italic;">
                  (${vern.english || tax.canonical_name || ''})
                </span>
              </h3>
              <p style="font-size:14px; color:#cbd5e1; margin-bottom:8px;">
                <em>${tax.scientific_name || ''}</em> ${tax.author ? '<span style="color:#94a3b8;">' + tax.author + '</span>' : ''}
              </p>
            </div>
            <div>
              <span class="badge badge-primary">Family: ${tax.botanical_family || 'N/A'}</span>
              ${tax.taxonomic_serial_number_gbif ? `<span class="badge badge-accent">GBIF ID: ${tax.taxonomic_serial_number_gbif}</span>` : ''}
            </div>
          </div>
        </div>
      `;

      if (Object.keys(vern).length > 0) {
        const flags = {
          english: '🇬🇧 English', hindi: '🇮🇳 Hindi', sanskrit: '🕉️ Sanskrit',
          kannada: '🟡 Kannada', tamil: '🔴 Tamil', telugu: '🔵 Telugu',
          malayalam: '🟢 Malayalam', marathi: '🟠 Marathi', bengali: '🟣 Bengali'
        };
        html += `
          <div class="section-card">
            <div class="section-title">🌐 Vernacular & Multilingual Regional Names</div>
            <div class="info-grid">
              ${Object.entries(vern).map(([lang, val]) => `
                <div class="info-item">
                  <div class="info-label">${flags[lang.toLowerCase()] || lang.toUpperCase()}</div>
                  <div class="info-value">${val}</div>
                </div>
              `).join('')}
            </div>
          </div>
        `;
      }

      if (syns.length > 0) {
        html += `
          <div class="section-card">
            <div class="section-title">🏷️ Botanical & Taxonomic Synonyms</div>
            <div style="margin-top: 6px;">
              ${syns.map(s => `<span class="badge badge-secondary"><em>${s}</em></span>`).join(' ')}
            </div>
          </div>
        `;
      }

      if (Object.keys(morph).length > 0) {
        const morphIcons = {
          leaves: '🍃 Foliage & Leaves',
          stem: '🌿 Stem & Bark Habit',
          flowers: '🌸 Inflorescence & Flowers',
          seeds_fruits: '🌰 Fruits & Seeds'
        };
        html += `
          <div class="section-card">
            <div class="section-title">🔬 Diagnostic Organ Morphology</div>
            <div>
              ${Object.entries(morph).map(([part, desc]) => `
                <div class="info-item" style="margin-bottom:8px;">
                  <div class="info-label" style="color:#38bdf8;">${morphIcons[part] || part.toUpperCase()}</div>
                  <div style="font-size:13px; color:#cbd5e1; line-height:1.4;">${desc}</div>
                </div>
              `).join('')}
            </div>
          </div>
        `;
      }

      return html;
    }

    function renderAyurveda(pk) {
      if (!pk || !pk.traditional_systems) return '<p style="color:#94a3b8; padding:16px;">No traditional systems data available.</p>';
      const trad = pk.traditional_systems;
      const ayur = trad.ayurveda || {};
      const sid = trad.siddha || {};
      const unani = trad.unani || {};

      let html = '';

      if (Object.keys(ayur).length > 0) {
        html += `
          <div class="section-card">
            <div class="section-title">🧘 Ayurvedic Dravya Guna Pharmacodynamics</div>
            <div class="info-grid" style="grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 12px;">
              <div class="info-item">
                <div class="info-label">🔥 Virya (Therapeutic Potency)</div>
                <div class="info-value" style="color:#f59e0b;">${ayur.virya || 'N/A'}</div>
              </div>
              <div class="info-item">
                <div class="info-label">⚡ Vipaka (Post-Digestive Transformation)</div>
                <div class="info-value" style="color:#38bdf8;">${ayur.vipaka || 'N/A'}</div>
              </div>
              <div class="info-item" style="grid-column: 1 / -1;">
                <div class="info-label">⭐ Prabhava (Specific Therapeutic Action)</div>
                <div class="info-value" style="color:#34d399;">${ayur.prabhava || 'N/A'}</div>
              </div>
              <div class="info-item" style="grid-column: 1 / -1;">
                <div class="info-label">⚖️ Dosha Karma (Tridoshic Impact)</div>
                <div class="info-value">${ayur.dosha_karma || 'N/A'}</div>
              </div>
            </div>

            <div style="display: flex; flex-wrap: wrap; gap: 16px;">
              <div>
                <div style="font-size:12px; font-weight:600; color:#94a3b8; margin-bottom:4px;">Rasa (Taste Profile):</div>
                <div>${(ayur.rasa || []).map(r => `<span class="badge badge-primary">${r}</span>`).join(' ') || '<span style="color:#64748b;">N/A</span>'}</div>
              </div>
              <div>
                <div style="font-size:12px; font-weight:600; color:#94a3b8; margin-bottom:4px;">Guna (Qualities):</div>
                <div>${(ayur.guna || []).map(g => `<span class="badge badge-accent">${g}</span>`).join(' ') || '<span style="color:#64748b;">N/A</span>'}</div>
              </div>
            </div>
          </div>
        `;
      }

      if (Object.keys(sid).length > 0 || Object.keys(unani).length > 0) {
        html += `
          <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px;">
            ${Object.keys(sid).length > 0 ? `
              <div class="section-card">
                <div class="section-title">🌿 Siddha System of Medicine</div>
                <div style="margin-bottom:8px;"><strong style="color:#94a3b8; font-size:12px;">Quality:</strong> <span class="badge badge-warning">${sid.humoral_quality || 'N/A'}</span></div>
                <div style="font-size:12px; font-weight:600; color:#94a3b8; margin-bottom:4px;">Indications:</div>
                <ul style="margin:0; padding-left:18px; font-size:13px; color:#cbd5e1;">
                  ${(sid.therapeutic_indications || []).map(ind => `<li>${ind}</li>`).join('')}
                </ul>
              </div>
            ` : ''}

            ${Object.keys(unani).length > 0 ? `
              <div class="section-card">
                <div class="section-title">📜 Unani Tibb System</div>
                <div style="margin-bottom:8px;"><strong style="color:#94a3b8; font-size:12px;">Mizaj:</strong> <span class="badge badge-accent">${unani.mizaj || 'N/A'}</span></div>
                <div style="font-size:12px; font-weight:600; color:#94a3b8; margin-bottom:4px;">Actions:</div>
                <div>
                  ${(unani.actions || []).map(act => `<span class="badge badge-secondary">${act}</span>`).join(' ')}
                </div>
              </div>
            ` : ''}
          </div>
        `;
      }

      return html || '<p style="color:#94a3b8; padding:16px;">No traditional systems records found.</p>';
    }

    function renderClinical(pk) {
      if (!pk || !pk.clinical_evidence) return '<p style="color:#94a3b8; padding:16px;">No clinical evidence data available.</p>';
      const clin = pk.clinical_evidence;
      const indications = clin.evaluated_indications || [];
      const trials = clin.trial_registries || [];

      let html = `
        <div class="section-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class="section-title" style="margin-bottom:0;">🏥 Clinical Evidence Consensus</div>
            <span class="badge badge-primary" style="font-size:11px; padding:4px 12px;">${clin.evidence_grading ? clin.evidence_grading.split('(')[0].trim() : 'Evaluated'}</span>
          </div>
          <p style="font-size:13px; color:#cbd5e1; line-height:1.4;">
            ${clin.evidence_grading || 'Peer-reviewed human clinical evidence indexed from WHO Monographs and Clinical Trial Registries.'}
          </p>
        </div>
      `;

      if (indications.length > 0) {
        html += `
          <div class="section-card">
            <div class="section-title">🎯 Evaluated Therapeutic Indications</div>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 8px;">
              ${indications.map(ind => `
                <div style="background:#1e293b; border:1px solid #334155; border-radius:6px; padding:8px 12px; font-size:13px; color:#e2e8f0; display:flex; align-items:center; gap:8px;">
                  <span style="color:#34d399; font-weight:700;">✓</span> ${ind}
                </div>
              `).join('')}
            </div>
          </div>
        `;
      }

      if (trials.length > 0) {
        html += `
          <div class="section-card">
            <div class="section-title">📑 Peer-Reviewed Human Clinical Trial Registries</div>
            ${trials.map(t => `
              <div class="clinical-trial-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                  <span class="badge badge-accent" style="font-size:12px;">Registry ID: ${t.registry_id}</span>
                  <span style="font-size:12px; color:#94a3b8;">Human Cohort Study</span>
                </div>
                <div style="font-size:13px; margin-bottom:4px;">
                  <strong style="color:#f8fafc;">Study Design:</strong> <span style="color:#cbd5e1;">${t.study_design}</span>
                </div>
                <div style="font-size:13px; margin-bottom:6px;">
                  <strong style="color:#f8fafc;">Formulation / Dose:</strong> <span style="color:#cbd5e1;">${t.dosage_form}</span>
                </div>
                <div style="background:#1e293b; border-left:3px solid #34d399; padding:8px 12px; border-radius:4px; font-size:12.5px; color:#a7f3d0; line-height:1.4;">
                  <strong>Clinical Outcomes:</strong> ${t.outcomes}
                </div>
              </div>
            `).join('')}
          </div>
        `;
      }

      return html;
    }

    function renderPhyto(pk) {
      if (!pk) return '<p style="color:#94a3b8; padding:16px;">No phytochemistry data available.</p>';
      const phyto = pk.phytochemistry || {};
      const safety = pk.safety_and_contraindications || {};
      const bioactives = phyto.primary_bioactives || [];
      const interactions = safety.herb_drug_interactions || [];

      let html = '';

      if (bioactives.length > 0) {
        html += `
          <div class="section-card">
            <div class="section-title">🧪 Primary Bioactive Phytochemicals</div>
            <table class="table-custom">
              <thead>
                <tr>
                  <th>Compound</th>
                  <th>Chemical Class</th>
                  <th>Fraction / Plant Part</th>
                </tr>
              </thead>
              <tbody>
                ${bioactives.map(b => `
                  <tr>
                    <td><strong style="color:#38bdf8;">${b.compound}</strong></td>
                    <td><span class="badge badge-secondary">${b.class}</span></td>
                    <td style="color:#94a3b8;">${b.fraction}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        `;
      }

      html += `
        <div class="section-card">
          <div class="section-title">⚠️ Clinical Safety, Toxicity & Contraindications</div>
          
          ${safety.adverse_effects ? `
            <div class="alert-box alert-warning">
              <strong>Adverse Effects & Toxicity:</strong> ${safety.adverse_effects}
            </div>
          ` : ''}

          ${safety.pregnancy_lactation_precautions ? `
            <div class="alert-box alert-danger">
              <strong>Pregnancy & Lactation Advisory:</strong> ${safety.pregnancy_lactation_precautions}
            </div>
          ` : ''}

          ${interactions.length > 0 ? `
            <div style="background:#1e293b; border:1px solid #334155; border-radius:6px; padding:12px; margin-bottom:12px;">
              <div style="font-size:13px; font-weight:700; color:#fbbf24; margin-bottom:6px;">💊 Herb-Drug Interactions:</div>
              <ul style="margin:0; padding-left:18px; font-size:12.5px; color:#cbd5e1; line-height:1.4;">
                ${interactions.map(inter => `<li>${inter}</li>`).join('')}
              </ul>
            </div>
          ` : ''}

          ${safety.clinical_evidence_limitations ? `
            <div class="alert-box alert-info">
              <strong>Evidence Limitations:</strong> ${safety.clinical_evidence_limitations}
            </div>
          ` : ''}
        </div>
      `;

      return html;
    }

    function renderResults(data) {
      const banner = document.getElementById('status-banner');
      banner.style.display = 'block';
      const isUncertain = data.identification.is_uncertain;
      banner.className = 'banner ' + (isUncertain ? 'banner-uncertain' : 'banner-confident');
      banner.innerHTML = `<strong>${data.status_banner}</strong><br>` +
        `Detected: <strong>${data.identification.detected_species}</strong> (${data.identification.scientific_name}) &bull; ` +
        `Confidence: <strong>${data.identification.confidence_percentage}</strong> &bull; ` +
        `Organ: <strong>${data.identification.detected_organ}</strong> &bull; ` +
        `Latency: <strong>${data.latency_profile.total_latency_ms} ms</strong>`;

      // Metrics Chips
      const chips = document.getElementById('metrics-chips');
      chips.innerHTML = `
        <span class="metric-chip">Source: <strong>${data.identification.source_engine}</strong></span>
        <span class="metric-chip">Top-1 Conf: <strong>${data.uncertainty_metrics.top1_confidence}</strong></span>
        <span class="metric-chip">Norm. Entropy: <strong>${data.uncertainty_metrics.normalized_entropy}</strong></span>
        <span class="metric-chip">Inference: <strong>${data.latency_profile.model_inference_ms} ms</strong></span>
      `;

      // XAI Plot
      const xaiContainer = document.getElementById('xai-container');
      const xaiImg = document.getElementById('xai-img');
      const afsCaption = document.getElementById('afs-caption');
      if (data.explainable_ai && data.explainable_ai.diagnostic_plot_path) {
        xaiImg.src = '/' + data.explainable_ai.diagnostic_plot_path + '?t=' + Date.now();
        xaiImg.style.display = 'block';
        xaiContainer.style.display = 'block';
        afsCaption.innerHTML = `<strong>Anatomical Focus Score (AFS):</strong> ${data.explainable_ai.afs_percentage} &bull; ${data.explainable_ai.interpretability_narrative.anatomical_focus_metric}`;
      } else {
        xaiContainer.style.display = 'none';
      }

      // Render Rich Human-Readable Cards into Tabs
      const pk = data.pharmacopoeial_knowledge || {};
      document.getElementById('tax-container').innerHTML = renderTaxonomy(pk);
      document.getElementById('ayur-container').innerHTML = renderAyurveda(pk);
      document.getElementById('clin-container').innerHTML = renderClinical(pk);
      document.getElementById('phyto-container').innerHTML = renderPhyto(pk);
      document.getElementById('raw-pre').innerText = JSON.stringify(data, null, 2);
    }
  </script>
</body>
</html>
"""

class StandaloneWebHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving the interactive web dashboard and /api/analyze endpoint."""
    
    def log_message(self, format, *args):
        # Override to suppress standard console noise
        pass

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        clean_path = url.path.lstrip("/")

        # Serve Dashboard Root
        if clean_path in ["", "index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            return

        # Serve static generated plots or dataset images
        file_disk_path = os.path.join(WORKSPACE_ROOT, clean_path)
        if os.path.exists(file_disk_path) and os.path.isfile(file_disk_path):
            mime_type = mimetypes.guess_type(file_disk_path)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.end_headers()
            with open(file_disk_path, "rb") as f:
                self.wfile.write(f.read())
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/analyze":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            
            try:
                body = json.loads(body_bytes.decode("utf-8"))
                image_input = None

                # Handle base64 image or path
                if "image_data" in body and body["image_data"]:
                    b64_str = body["image_data"]
                    if "," in b64_str:
                        b64_str = b64_str.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64_str)
                    
                    runs_dir = os.path.join(WORKSPACE_ROOT, "reports", "pipeline_runs")
                    os.makedirs(runs_dir, exist_ok=True)
                    temp_path = os.path.join(runs_dir, f"upload_{int(os.getpid())}.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(img_bytes)
                    image_input = temp_path
                else:
                    image_input = body.get("image_path", "dataset/Tulsi/general/Tulsi_general_00053.jpg")

                res = analyze_medicinal_plant(
                    image_input=image_input,
                    organ_hint=body.get("organ_hint"),
                    confidence_threshold=float(body.get("confidence_threshold", 0.60)),
                    entropy_threshold=float(body.get("entropy_threshold", 0.45)),
                    generate_xai=bool(body.get("generate_xai", True)),
                    output_dir=os.path.join(WORKSPACE_ROOT, "reports", "pipeline_runs")
                )

                resp_bytes = json.dumps(res, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(resp_bytes)
            except Exception as e:
                err_bytes = json.dumps({"status": "ERROR", "error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(err_bytes)
            return

        self.send_response(404)
        self.end_headers()

def launch_standalone_web_dashboard(port: int = 7860, host: str = "0.0.0.0"):
    """Starts the standalone zero-dependency interactive dashboard."""
    server = HTTPServer((host, port), StandaloneWebHandler)
    print(f"\n" + "=" * 75)
    print(f"🌿 VTU MEDICINAL PLANT AI — INTERACTIVE WEB DASHBOARD LAUNCHED")
    print(f"   Local Access URL : http://localhost:{port}")
    print(f"   Network Access   : http://{host}:{port}")
    print(f"=" * 75 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard] Server shut down cleanly.")
        server.server_close()

def launch_app(port: int = 7860, host: str = "0.0.0.0", share: bool = False):
    """
    Launches Gradio app if Gradio is installed, otherwise launches
    the standalone embedded web dashboard seamlessly.
    """
    try:
        demo = build_gradio_blocks()
        print(f"\n[Dashboard] Launching official Gradio Blocks Interface on port {port}...")
        demo.launch(server_name=host, server_port=port, share=share)
    except ImportError:
        print(f"[Dashboard] Note: 'gradio' package not detected in current environment.")
        print(f"[Dashboard] Engaging zero-dependency Standalone Web Dashboard...")
        launch_standalone_web_dashboard(port=port, host=host)

if __name__ == "__main__":
    launch_app()
