"""SOP-Grounded Amazon Bedrock RMA & Maintenance Copilot.

Synthesizes structured data-center technician repair tickets and SPRT canary halt briefs
via Amazon Bedrock (`us.amazon.nova-lite-v1:0`) strictly verified against deterministic
capacity-floor (`N-k`), host-drain, Lockout/Tagout (`LOTO`), and entity-grounding guardrails.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

SOP_RUNBOOK_CATALOG: List[Dict[str, Any]] = [
    {
        "sop_id": "SOP-HBM-ECC-01",
        "title": "Accelerator HBM Uncorrectable ECC Blade Replacement",
        "keywords": ["hbm", "ecc", "uncorrectable", "memory", "sram", "blade", "rma"],
        "fru_part": "FRU-ACCEL-MOD-64G",
        "required_loto": "Verify host state == DC_RMA_DISPATCH, attach ESD wrist strap, open sled 48V hot-swap latch, wait 30s discharge LED off.",
        "burn_in_gate": "Run 4-pass HBM row/column pattern sweep + local 8-rank AllReduce loopback with 0 ECC events.",
    },
    {
        "sop_id": "SOP-OPTIC-400G-02",
        "title": "400G OSFP Optical Transceiver Low Rx Power & EFA Flap",
        "keywords": ["optic", "400g", "rx", "crc", "efa", "neuronlink", "transceiver", "link", "flap"],
        "fru_part": "FRU-OSFP-400G-DR4",
        "required_loto": "Confirm host cordoned/drained, engage Class-1M laser eye protection, inspect MPO-12 ferrule with fiber scope before seating.",
        "burn_in_gate": "Verify Rx optical power > -4.5 dBm and post-FEC BER < 1e-12 across 500-step EFA stress test.",
    },
    {
        "sop_id": "SOP-PDU-PHASE-03",
        "title": "Rack ATS / PDU Branch Rail Fault & Power Redundancy Check",
        "keywords": ["pdu", "psu", "power", "rail", "ats", "breaker", "phase", "voltage"],
        "fru_part": "FRU-PSU-3000W-TITANIUM",
        "required_loto": "Mandatory dual-person electrical LOTO padlock on Feed-B breaker; verify N-k rack healthy capacity >= 6/8 before pulling PSU.",
        "burn_in_gate": "Confirm dual-feed A+B load balance within 5% delta and zero BMC undervoltage alerts.",
    },
    {
        "sop_id": "SOP-CANARY-SPRT-04",
        "title": "Staged Firmware Wave SPRT Regression Halt & Atomic Rollback",
        "keywords": ["canary", "sprt", "firmware", "wave", "rollback", "halt", "regression", "efa-nic-fw", "bmc"],
        "fru_part": "N/A-SOFTWARE-ROLLBACK",
        "required_loto": "Freeze promotion pipeline immediately; pin next-boot slot to last-known-good firmware digest; respect rack N-k reboot concurrency.",
        "burn_in_gate": "Re-run Canary_1Pct qualification stream until Wald LLR <= B (-3.8918) before resuming Wave_10Pct.",
    },
]

HOST_ID_RE = re.compile(r"\busv-r0[1-9]-h0[0-9]\b")
SOP_ID_RE = re.compile(r"\bSOP-[A-Z0-9-]+\b")


def retrieve_matching_sop(query_text: str) -> Dict[str, Any]:
    """Deterministic keyword/token overlap retrieval over hardware & canary SOP runbooks."""
    tokens = set(re.findall(r"[a-z0-9_-]+", query_text.lower()))
    best_sop = SOP_RUNBOOK_CATALOG[0]
    best_score = -1
    for sop in SOP_RUNBOOK_CATALOG:
        score = sum(3 for kw in sop["keywords"] if kw in tokens or kw in query_text.lower())
        if sop["sop_id"].lower() in query_text.lower():
            score += 10
        if score > best_score:
            best_score = score
            best_sop = sop
    return best_sop


def evaluate_deterministic_guardrails(
    incident_type: str,
    host_state: str,
    rack_healthy_hosts: int,
    rack_total_hosts: int,
    max_unavailable_k: int,
    loto_verified: bool,
) -> Dict[str, Any]:
    """Hard capacity-floor and physical-touch safety gates evaluated outside the LLM."""
    min_healthy_floor = rack_total_hosts - max_unavailable_k
    rack_floor_ok = rack_healthy_hosts >= min_healthy_floor

    if incident_type == "canary_halt":
        return {
            "incident_type": "canary_halt",
            "host_drain_verified": True,
            "rack_capacity_floor_ok": rack_floor_ok,
            "min_healthy_floor_n_minus_k": f"{rack_healthy_hosts}/{rack_total_hosts} (floor {min_healthy_floor})",
            "loto_verified": True,
            "authorized_for_action": rack_floor_ok,
            "guardrail_verdict": "AUTHORIZED_ROLLBACK" if rack_floor_ok else "BLOCKED_RACK_CAPACITY_FLOOR",
            "blocking_reasons": [] if rack_floor_ok else [
                f"Rack healthy count ({rack_healthy_hosts}) below N-k floor ({min_healthy_floor})"
            ],
        }

    drain_ok = host_state == "DC_RMA_DISPATCH"
    reasons: List[str] = []
    if not drain_ok:
        reasons.append(f"Host state is {host_state} (must reach DC_RMA_DISPATCH with workloads drained before physical touch)")
    if not rack_floor_ok:
        reasons.append(f"Rack healthy hosts ({rack_healthy_hosts}/{rack_total_hosts}) breaches N-k floor ({min_healthy_floor})")
    if not loto_verified:
        reasons.append("Mandatory Lockout/Tagout (loto_verified) sign-off is missing")

    authorized = drain_ok and rack_floor_ok and bool(loto_verified)
    return {
        "incident_type": "host_rma",
        "host_drain_verified": drain_ok,
        "rack_capacity_floor_ok": rack_floor_ok,
        "min_healthy_floor_n_minus_k": f"{rack_healthy_hosts}/{rack_total_hosts} (floor {min_healthy_floor})",
        "loto_verified": bool(loto_verified),
        "authorized_for_action": authorized,
        "guardrail_verdict": "AUTHORIZED_DC_RMA_TOUCH" if authorized else "HOLD_GUARDRAIL_BLOCKED",
        "blocking_reasons": reasons,
    }


def verify_entity_grounding(
    generated_text: str,
    allowed_hosts: Set[str],
    allowed_sops: Set[str],
) -> Dict[str, Any]:
    """Verify every host ID and SOP code cited in Bedrock output exists in the grounded context."""
    cited_hosts = set(HOST_ID_RE.findall(generated_text))
    cited_sops = set(SOP_ID_RE.findall(generated_text))
    unknown_hosts = sorted(cited_hosts - allowed_hosts)
    unknown_sops = sorted(cited_sops - allowed_sops)
    grounded = (len(unknown_hosts) == 0) and (len(unknown_sops) == 0)
    return {
        "grounded": grounded,
        "cited_hosts": sorted(cited_hosts),
        "cited_sops": sorted(cited_sops),
        "hallucinated_entities": unknown_hosts + unknown_sops,
    }


class RMACopilot:
    """Amazon Bedrock (`us.amazon.nova-lite-v1:0`) SOP-grounded RMA & Canary Halt Copilot."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or os.environ.get("NITRO_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0")
        self.region_name = (
            region_name
            or os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("AWS_REGION")
            or "us-west-2"
        )
        self._client: Any = None

    def _get_bedrock_client(self) -> Any:
        if self._client is None:
            import subprocess
            import boto3
            from botocore.exceptions import BotoCoreError

            try:
                client = boto3.client("bedrock-runtime", region_name=self.region_name)
                # Trigger credential resolution eagerly so local aws-login profiles without botocore[crt] bridge cleanly
                if os.environ.get("AWS_EXECUTION_ENV") is None:
                    _ = boto3.Session().get_credentials()
                self._client = client
            except BotoCoreError:
                proc = subprocess.run(
                    ["aws", "configure", "export-credentials", "--format", "process"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=8.0,
                )
                creds = json.loads(proc.stdout)
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region_name,
                    aws_access_key_id=creds["AccessKeyId"],
                    aws_secret_access_key=creds["SecretAccessKey"],
                    aws_session_token=creds.get("SessionToken"),
                )
        return self._client

    def synthesize_ticket(
        self,
        incident_type: str = "host_rma",
        host_id: str = "usv-r02-h05",
        rack_id: str = "rack-02",
        location: str = "Aisle-A · Row 06 · U29",
        host_state: str = "DC_RMA_DISPATCH",
        syndrome_code: str = "HBM_UNCORRECTABLE_ECC",
        rack_healthy_hosts: int = 7,
        rack_total_hosts: int = 8,
        max_unavailable_k: int = 2,
        loto_verified: bool = True,
        canary_context: Optional[Dict[str, Any]] = None,
        operator_note: str = "",
        offline_mode: bool = False,
    ) -> Dict[str, Any]:
        """Generate an SOP-grounded RMA work order or canary halt brief and verify all guardrails."""
        query_blob = f"{incident_type} {syndrome_code} {operator_note} {(canary_context or {}).get('build_id', '')}"
        matched_sop = retrieve_matching_sop(query_blob)
        guardrails = evaluate_deterministic_guardrails(
            incident_type=incident_type,
            host_state=host_state,
            rack_healthy_hosts=rack_healthy_hosts,
            rack_total_hosts=rack_total_hosts,
            max_unavailable_k=max_unavailable_k,
            loto_verified=loto_verified,
        )

        allowed_hosts = {host_id}
        allowed_sops = {s["sop_id"] for s in SOP_RUNBOOK_CATALOG}

        system_prompt = (
            "You are an AWS Data Center & EC2 UltraServer Operations Copilot. "
            "Write a concise, factual 4-bullet technician action brief strictly grounded in the provided JSON facts and SOP. "
            "Rules: (1) Cite the exact sop_id and host_id or build_id provided; never invent other host IDs or SOP codes. "
            "(2) Respect guardrail_verdict: if HOLD_GUARDRAIL_BLOCKED, explicitly instruct the technician NOT to unseat hardware until blocking_reasons are cleared. "
            "(3) Never use em-dashes. Keep total length under 130 words."
        )

        context_payload = {
            "incident_type": incident_type,
            "host_id": host_id,
            "rack_id": rack_id,
            "location": location,
            "host_state": host_state,
            "syndrome_code": syndrome_code,
            "matched_sop": {
                "sop_id": matched_sop["sop_id"],
                "title": matched_sop["title"],
                "fru_part": matched_sop["fru_part"],
                "required_loto": matched_sop["required_loto"],
                "burn_in_gate": matched_sop["burn_in_gate"],
            },
            "guardrails": guardrails,
            "canary_context": canary_context or {},
            "operator_note": operator_note,
        }

        if offline_mode:
            t0 = time.perf_counter()
            action_line = (
                f"Proceed with {matched_sop['fru_part']} replacement at {location} under {matched_sop['sop_id']}."
                if guardrails["authorized_for_action"]
                else f"HOLD physical touch on {host_id}: {'; '.join(guardrails['blocking_reasons'])}."
            )
            text = (
                f"- Incident: {host_id} in {rack_id} reported {syndrome_code} (state: {host_state}).\n"
                f"- Guardrail Verdict: {guardrails['guardrail_verdict']} ({guardrails['min_healthy_floor_n_minus_k']}).\n"
                f"- Action & LOTO ({matched_sop['sop_id']}): {action_line} {matched_sop['required_loto']}\n"
                f"- Post-Action Gate: {matched_sop['burn_in_gate']}"
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            tokens_in, tokens_out = 0, 0
            engine_used = "deterministic-offline-test"
        else:
            client = self._get_bedrock_client()
            t0 = time.perf_counter()
            resp = client.converse(
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": "Incident & Guardrail Facts:\n" + json.dumps(context_payload, indent=2)}],
                    }
                ],
                inferenceConfig={"maxTokens": 260, "temperature": 0.1},
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            blocks = resp.get("output", {}).get("message", {}).get("content", [])
            text = "\n".join(b.get("text", "") for b in blocks if "text" in b).strip()
            # Sanitize any accidental em-dash from model text
            text = text.replace("\u2014", "-").replace("\u2013", "-")
            usage = resp.get("usage", {})
            tokens_in = int(usage.get("inputTokens", 0))
            tokens_out = int(usage.get("outputTokens", 0))
            engine_used = self.model_id

        grounding = verify_entity_grounding(text, allowed_hosts=allowed_hosts, allowed_sops=allowed_sops)

        return {
            "incident_type": incident_type,
            "host_id": host_id,
            "rack_id": rack_id,
            "location": location,
            "syndrome_code": syndrome_code,
            "matched_sop": matched_sop,
            "guardrails": guardrails,
            "grounding_check": grounding,
            "copilot_brief": text,
            "bedrock_model": engine_used,
            "bedrock_latency_ms": round(lat_ms, 2),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        }
