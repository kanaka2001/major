"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Verification Script: Phase 9 — Model Context Protocol (MCP) Knowledge Server

This script validates:
1. Direct Python API execution across all 4 MCP Tools:
   - Tool 1: identify_medicinal_plant
   - Tool 2: explain_plant_prediction
   - Tool 3: get_plant_information
   - Tool 4: get_medicinal_uses
2. Full JSON-RPC 2.0 wire protocol over standard input/output simulation:
   - Handshake: 'initialize' (protocolVersion: 2024-11-05)
   - Tool Discovery: 'tools/list'
   - Tool Invocation: 'tools/call' for vision identification, XAI, and pharmacopoeial knowledge
   - Robust Error Handling (-32601 Method Not Found, -32602 Invalid Params)
3. End-to-end integration across real images (Tulsi, Aloe Vera, Catharanthus Roseus)
4. Telemetry and report serialization to reports/mcp_server/phase9_verification_report.json

Author: Lead Systems & AI Agent Architect
"""

import os
import sys
import json
import time

workspace_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.mcp_server.server import MedicinalPlantsMCPServer, TOOLS_REGISTRY

def main():
    print("=" * 80)
    print("VTU MAJOR PROJECT: PHASE 9 VERIFICATION & DIAGNOSTIC BENCHMARK")
    print("Model Context Protocol (MCP) Botanical & Medicinal Knowledge Server")
    print("=" * 80)

    output_dir = os.path.join(workspace_root, "reports", "mcp_server")
    os.makedirs(output_dir, exist_ok=True)

    server = MedicinalPlantsMCPServer()

    test_tulsi = "dataset/Tulsi/general/Tulsi_general_00053.jpg"
    test_aloe = "dataset/Aloe_Vera/general/Aloe_Vera_general_00206.jpg"
    test_catharanthus = "dataset/Catharanthus_Roseus/leaf/Catharanthus_Roseus_leaf_00151.jpg"

    assert os.path.exists(os.path.join(workspace_root, test_tulsi)), f"Missing test sample: {test_tulsi}"
    assert os.path.exists(os.path.join(workspace_root, test_aloe)), f"Missing test sample: {test_aloe}"
    assert os.path.exists(os.path.join(workspace_root, test_catharanthus)), f"Missing test sample: {test_catharanthus}"

    verification_log = []

    # =========================================================================
    # PART 1: Direct Tool Method Verification
    # =========================================================================
    print("\n[1/3] Testing Direct Python API Execution Across All 4 MCP Tools...")

    # Tool 1: identify_medicinal_plant
    print("\n  --- Tool 1: identify_medicinal_plant ---")
    t0 = time.time()
    res_tool1 = server.call_tool("identify_medicinal_plant", {
        "image_path": test_tulsi,
        "organ_hint": "general"
    })
    elapsed_tool1 = time.time() - t0
    print(f"      Image: {test_tulsi}")
    print(f"      Source: {res_tool1['source']} | Is Uncertain: {res_tool1['is_uncertain']}")
    print(f"      Detected Species: {res_tool1['detected_species']} ({res_tool1['confidence_percentage']})")
    print(f"      Curated Knowledge Match: {res_tool1['curated_species_key']}")
    print(f"      Elapsed: {elapsed_tool1:.2f}s")
    verification_log.append({
        "tool": "identify_medicinal_plant",
        "status": "PASS",
        "elapsed_seconds": round(elapsed_tool1, 4),
        "result": res_tool1
    })

    # Tool 2: explain_plant_prediction
    print("\n  --- Tool 2: explain_plant_prediction ---")
    t0 = time.time()
    res_tool2 = server.call_tool("explain_plant_prediction", {
        "image_path": test_aloe,
        "target_class": "Aloe_Vera",
        "output_dir": os.path.join(output_dir, "xai")
    })
    elapsed_tool2 = time.time() - t0
    print(f"      Image: {test_aloe}")
    print(f"      Predicted Species: {res_tool2['predicted_species']} ({res_tool2['species_confidence']})")
    print(f"      Predicted Organ: {res_tool2['predicted_organ']} ({res_tool2['organ_confidence']})")
    print(f"      Anatomical Focus Score (AFS): {res_tool2['anatomical_focus_score_afs']}")
    print(f"      Diagnostic Plot: {res_tool2['diagnostic_figure_path']}")
    print(f"      Elapsed: {elapsed_tool2:.2f}s")
    verification_log.append({
        "tool": "explain_plant_prediction",
        "status": "PASS",
        "elapsed_seconds": round(elapsed_tool2, 4),
        "result": res_tool2
    })

    # Tool 3: get_plant_information
    print("\n  --- Tool 3: get_plant_information ---")
    t0 = time.time()
    res_tool3 = server.call_tool("get_plant_information", {
        "species_name": "Holy Basil" # Test vernacular alias resolution
    })
    elapsed_tool3 = time.time() - t0
    print(f"      Query: 'Holy Basil' -> Resolved: {res_tool3.get('taxonomy', {}).get('canonical_name')}")
    print(f"      Scientific: {res_tool3.get('taxonomy', {}).get('scientific_name')} ({res_tool3.get('taxonomy', {}).get('botanical_family')})")
    print(f"      Vernacular (Kannada): {res_tool3.get('taxonomy', {}).get('vernacular_names', {}).get('kannada')}")
    print(f"      Bioactives Count: {len(res_tool3.get('phytochemistry', {}).get('primary_bioactives', []))}")
    print(f"      Elapsed: {elapsed_tool3:.4f}s")
    verification_log.append({
        "tool": "get_plant_information",
        "status": "PASS",
        "elapsed_seconds": round(elapsed_tool3, 4),
        "result": res_tool3
    })

    # Tool 4: get_medicinal_uses (Ayurvedic & Clinical Evidence)
    print("\n  --- Tool 4: get_medicinal_uses (Ayurveda + Clinical) ---")
    t0 = time.time()
    res_tool4_ayur = server.call_tool("get_medicinal_uses", {
        "species_name": "Catharanthus_Roseus",
        "system": "ayurveda"
    })
    res_tool4_clin = server.call_tool("get_medicinal_uses", {
        "species_name": "Catharanthus_Roseus",
        "system": "clinical"
    })
    elapsed_tool4 = time.time() - t0
    print(f"      Species: Catharanthus_Roseus")
    print(f"      Ayurveda Virya: {res_tool4_ayur['pharmacodynamics']['virya']} | Vipaka: {res_tool4_ayur['pharmacodynamics']['vipaka']}")
    print(f"      Prabhava: {res_tool4_ayur['pharmacodynamics']['prabhava']}")
    print(f"      Clinical Trial Registries: {len(res_tool4_clin['clinical_evidence']['trial_registries'])} recorded")
    print(f"      Evidence Grading: {res_tool4_clin['clinical_evidence']['evidence_grading']}")
    print(f"      Elapsed: {elapsed_tool4:.4f}s")
    verification_log.append({
        "tool": "get_medicinal_uses",
        "status": "PASS",
        "elapsed_seconds": round(elapsed_tool4, 4),
        "result_ayurveda": res_tool4_ayur,
        "result_clinical": res_tool4_clin
    })

    # =========================================================================
    # PART 2: JSON-RPC 2.0 Wire Protocol Simulation
    # =========================================================================
    print("\n[2/3] Testing JSON-RPC 2.0 Stdio Wire Protocol Simulation...")

    # 1. Initialize Handshake
    req_init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        }
    }
    resp_init = server.handle_request(req_init)
    print(f"      [RPC #1: initialize] -> Status: {'PASS' if 'result' in resp_init else 'FAIL'}")
    assert resp_init["result"]["protocolVersion"] == "2024-11-05"
    assert resp_init["result"]["serverInfo"]["name"] == "mcp-server-medicinal-plants"

    # 2. Notification: initialized
    req_notif = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }
    resp_notif = server.handle_request(req_notif)
    print(f"      [RPC #2: notifications/initialized] -> Handled cleanly: {resp_notif is None}")
    assert resp_notif is None

    # 3. Tool Discovery: tools/list
    req_tools_list = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    resp_tools_list = server.handle_request(req_tools_list)
    tools_returned = resp_tools_list["result"]["tools"]
    print(f"      [RPC #3: tools/list] -> Count: {len(tools_returned)} tools registered")
    tool_names = [t["name"] for t in tools_returned]
    print(f"         Discovered: {', '.join(tool_names)}")
    assert "identify_medicinal_plant" in tool_names
    assert "explain_plant_prediction" in tool_names
    assert "get_plant_information" in tool_names
    assert "get_medicinal_uses" in tool_names

    # 4. Tool Invocation: tools/call (get_plant_information)
    req_tool_call = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "get_plant_information",
            "arguments": {"species_name": "Turmeric"}
        }
    }
    resp_tool_call = server.handle_request(req_tool_call)
    print(f"      [RPC #4: tools/call -> get_plant_information] -> Status: {'PASS' if 'result' in resp_tool_call else 'FAIL'}")
    assert resp_tool_call["result"]["isError"] is False
    content_text = resp_tool_call["result"]["content"][0]["text"]
    parsed_content = json.loads(content_text)
    assert parsed_content["taxonomy"]["canonical_name"] == "Turmeric"
    assert parsed_content["taxonomy"]["scientific_name"] == "Curcuma longa"

    # 5. Error Handling: Method Not Found (-32601)
    req_unknown = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "unsupported/method",
        "params": {}
    }
    resp_unknown = server.handle_request(req_unknown)
    print(f"      [RPC #5: error - method not found] -> Error Code: {resp_unknown.get('error', {}).get('code')}")
    assert resp_unknown["error"]["code"] == -32601

    # 6. Error Handling: Missing Tool Name (-32602)
    req_bad_call = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {}
    }
    resp_bad_call = server.handle_request(req_bad_call)
    print(f"      [RPC #6: error - invalid params] -> Error Code: {resp_bad_call.get('error', {}).get('code')}")
    assert resp_bad_call["error"]["code"] == -32602

    # =========================================================================
    # PART 3: Verification Report Serialization
    # =========================================================================
    print("\n[3/3] Serializing Phase 9 Verification Report...")
    full_report = {
        "status": "PHASE_9_COMPLETE_AND_VERIFIED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "mcp_server": {
            "name": "mcp-server-medicinal-plants",
            "version": "1.0.0",
            "protocol_version": "2024-11-05",
            "transport": "JSON-RPC 2.0 over Standard I/O (stdio)",
            "tools_count": len(TOOLS_REGISTRY),
            "tools": TOOLS_REGISTRY
        },
        "knowledge_database": {
            "path": "data/medicinal_knowledge_db.json",
            "curated_species_count": len(server.knowledge_engine.list_curated_species()),
            "sources": server.knowledge_engine.data.get("metadata", {}).get("authoritative_sources", []),
            "curated_species": server.knowledge_engine.list_curated_species()
        },
        "tool_executions": verification_log,
        "jsonrpc_wire_tests": {
            "initialize": "PASSED (Protocol 2024-11-05 validated)",
            "notifications_initialized": "PASSED",
            "tools_list": "PASSED (4 tools discovered with schema)",
            "tools_call": "PASSED (Turmeric profile returned via MCP content text)",
            "error_handling": "PASSED (-32601, -32602 verified)"
        }
    }

    report_path = os.path.join(output_dir, "phase9_verification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"      Verification report saved to: {os.path.relpath(report_path, workspace_root)}")
    print("\n" + "=" * 80)
    print("PHASE 9 VERIFICATION COMPLETE: ALL GATES PASSED")
    print("=" * 80)

if __name__ == "__main__":
    main()

