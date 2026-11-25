"""Local HTTP Server & AWS CloudFront `/api/*` Lambda Origin Handler for NitroUltraFleet."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

from .repair_state_machine import (
    BlastRadiusBudgetExceededError,
    CASVersionConflictError,
    FleetRepairStateMachine,
    InvalidStateTransitionError,
)
from .rma_copilot import RMACopilot
from .sprt_canary import PRESET_ROLLOUT_SCENARIOS, SPRTCanaryEngine
from .topology_scheduler import (
    analyze_collective_trace,
    generate_collective_trace,
    schedule_workload_queue,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"
BENCHMARK_JSON_PATH = ROOT_DIR / "benchmarks" / "latest_results.json"

STATE_MACHINE = FleetRepairStateMachine()
COPILOT = RMACopilot()

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def _load_benchmarks() -> Dict[str, Any]:
    if BENCHMARK_JSON_PATH.is_file():
        return json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
    return {}


def _build_state_snapshot() -> Dict[str, Any]:
    hosts = STATE_MACHINE.list_hosts()
    racks = STATE_MACHINE.get_rack_budget_summary()
    chain = STATE_MACHINE.verify_audit_chain()
    transitions = STATE_MACHINE.list_recent_transitions(limit=20)
    work_orders = STATE_MACHINE.plan_aisle_work_orders()
    return {
        "hosts": hosts,
        "rack_budgets": racks,
        "audit_chain": chain,
        "recent_transitions": transitions,
        "aisle_work_orders": work_orders,
        "benchmarks": _load_benchmarks(),
    }


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    """Pure router shared by local HTTP server and AWS Lambda Function URL behind CloudFront."""
    try:
        if method == "GET" and path == "/api/health":
            hosts = STATE_MACHINE.list_hosts()
            chain = STATE_MACHINE.verify_audit_chain()
            healthy_cnt = sum(1 for h in hosts if h["state"] == "HEALTHY")
            return 200, {
                "status": "ok",
                "service": "nitro-ultraserver-fleet",
                "delivery": os.environ.get("NITRO_DELIVERY", "cloudfront-s3-oac + lambda-api-origin"),
                "aws_region": os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION", "us-west-2"),
                "bedrock_model": os.environ.get("NITRO_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"),
                "python_version": platform.python_version(),
                "request_id": request_id,
                "fleet_hosts_total": len(hosts),
                "fleet_hosts_healthy": healthy_cnt,
                "sqlite_wal_chain_valid": chain["valid"],
                "sqlite_wal_rows": chain["checked_rows"],
            }

        if method == "GET" and path == "/api/benchmarks":
            data = _load_benchmarks()
            data["request_id"] = request_id
            data["live_audit_chain"] = STATE_MACHINE.verify_audit_chain()
            return 200, data

        if method == "GET" and path == "/api/state":
            snap = _build_state_snapshot()
            snap["request_id"] = request_id
            return 200, snap

        if method == "POST" and path == "/api/canary/sprt":
            scenario_id = str(body.get("scenario_id") or "").strip()
            preset = PRESET_ROLLOUT_SCENARIOS.get(scenario_id)
            p0 = float(body.get("p0", preset["p0"] if preset else 0.01))
            p1 = float(body.get("p1", preset["p1"] if preset else 0.06))
            alpha = float(body.get("alpha", preset["alpha"] if preset else 0.02))
            beta = float(body.get("beta", preset["beta"] if preset else 0.02))
            build_id = str(body.get("build_id") or (preset["build_id"] if preset else "custom-fw-candidate"))
            seed = int(body.get("seed", preset["seed"] if preset else 4201))

            engine = SPRTCanaryEngine(p0=p0, p1=p1, alpha=alpha, beta=beta)
            wave_rates = body.get("wave_rates") or (preset["wave_rates"] if preset else None)
            obs_rate = body.get("observed_failure_rate")
            if obs_rate is not None:
                wave_rates = None
                obs_rate = float(obs_rate)

            out = engine.run_staged_rollout(
                build_id=build_id,
                wave_rates=wave_rates,
                observed_failure_rate=obs_rate,
                explicit_wave_outcomes=body.get("explicit_wave_outcomes"),
                seed=seed,
            )
            if preset:
                out["scenario_title"] = preset["title"]
                out["component"] = preset["component"]
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/repair/transition":
            action = str(body.get("action") or "transition").strip()
            if action == "simulate_cas_race":
                # Execute a real 4-thread concurrent CAS race on the chosen host
                host_id = str(body.get("host_id") or "usv-r01-h00").strip()
                host_info = STATE_MACHINE.get_host(host_id)
                stale_ver = int(host_info["version"])
                target_state = str(body.get("target_state") or "SUSPECT_CORDONED").strip()
                syndrome = str(body.get("syndrome_code") or "NEURONLINK_CRC_FLAP").strip()

                def _worker(idx: int) -> Dict[str, Any]:
                    try:
                        res = STATE_MACHINE.transition_host(
                            host_id=host_id,
                            target_state=target_state,
                            expected_version=stale_ver,
                            syndrome_code=syndrome,
                            actor=f"worker-thread-{idx}",
                        )
                        return {"worker": f"worker-thread-{idx}", "outcome": "COMMITTED", "detail": res}
                    except CASVersionConflictError as exc:
                        return {"worker": f"worker-thread-{idx}", "outcome": "CAS_CONFLICT_REJECTED", "error": str(exc)}
                    except (BlastRadiusBudgetExceededError, InvalidStateTransitionError) as exc:
                        return {"worker": f"worker-thread-{idx}", "outcome": "GUARDRAIL_BLOCKED", "error": str(exc)}

                with ThreadPoolExecutor(max_workers=4) as pool:
                    results: List[Dict[str, Any]] = list(pool.map(_worker, range(1, 5)))

                snap = _build_state_snapshot()
                return 200, {
                    "action": "simulate_cas_race",
                    "host_id": host_id,
                    "contested_version": stale_ver,
                    "winners_count": sum(1 for r in results if r["outcome"] == "COMMITTED"),
                    "conflicts_rejected": sum(1 for r in results if r["outcome"] == "CAS_CONFLICT_REJECTED"),
                    "thread_results": results,
                    "state_snapshot": snap,
                    "request_id": request_id,
                }

            host_id = str(body.get("host_id") or "usv-r01-h03").strip()
            target_state = str(body.get("target_state") or "").strip()
            if not target_state:
                return 400, {"error": "target_state is required"}

            exp_ver = body.get("expected_version")
            res = STATE_MACHINE.transition_host(
                host_id=host_id,
                target_state=target_state,
                expected_version=int(exp_ver) if exp_ver is not None else None,
                syndrome_code=body.get("syndrome_code"),
                actor=str(body.get("actor") or "operator-console"),
                loto_verified=body.get("loto_verified"),
                burn_in_passed=body.get("burn_in_passed"),
            )
            snap = _build_state_snapshot()
            return 200, {
                "status": "transition_committed",
                "transition": res,
                "state_snapshot": snap,
                "request_id": request_id,
            }

        if method == "POST" and path == "/api/scheduler/allocate":
            trace_preset = str(body.get("trace_preset") or "rank11_nic_straggler").strip()
            workload_preset = str(body.get("workload_preset") or "mixed_training_and_ci").strip()

            custom_trace = body.get("trace")
            trace_obj = custom_trace if isinstance(custom_trace, dict) else generate_collective_trace(preset=trace_preset)
            straggler_analysis = analyze_collective_trace(trace_obj)

            # Pull live non-healthy hosts from SQLite unless caller overrides cordoned_hosts
            if "cordoned_hosts" in body and isinstance(body["cordoned_hosts"], list):
                cordoned = [str(x) for x in body["cordoned_hosts"]]
            else:
                cordoned = [h["host_id"] for h in STATE_MACHINE.list_hosts() if h["state"] != "HEALTHY"]

            schedule_out = schedule_workload_queue(
                preset_name=workload_preset,
                jobs=body.get("jobs"),
                cordoned_hosts=cordoned,
            )
            return 200, {
                "straggler_detector": straggler_analysis,
                "scheduler": schedule_out,
                "request_id": request_id,
            }

        if method == "POST" and path == "/api/copilot/rma":
            incident_type = str(body.get("incident_type") or "host_rma").strip()
            host_id = str(body.get("host_id") or "usv-r02-h05").strip()
            offline_mode = bool(body.get("offline_mode", False))

            try:
                host = STATE_MACHINE.get_host(host_id)
            except KeyError:
                host = {
                    "host_id": host_id,
                    "rack_id": "rack-02",
                    "aisle": "Aisle-A",
                    "row_num": 6,
                    "rack_unit": 29,
                    "state": "DC_RMA_DISPATCH",
                    "syndrome_code": "HBM_UNCORRECTABLE_ECC",
                    "loto_verified": 1,
                }

            rack_id = str(body.get("rack_id") or host["rack_id"])
            rack_summaries = {r["rack_id"]: r for r in STATE_MACHINE.get_rack_budget_summary()}
            rack_info = rack_summaries.get(rack_id, {"healthy_hosts": 7, "total_hosts": 8})

            host_state = str(body.get("host_state") or host["state"])
            syndrome_code = str(body.get("syndrome_code") or host["syndrome_code"])
            loto_flag = bool(host["loto_verified"]) if "loto_verified" not in body else bool(body["loto_verified"])
            loc_str = f"{host['aisle']} · Row {int(host['row_num']):02d} · U{int(host['rack_unit']):02d}"

            out = COPILOT.synthesize_ticket(
                incident_type=incident_type,
                host_id=host_id,
                rack_id=rack_id,
                location=loc_str,
                host_state=host_state,
                syndrome_code=syndrome_code,
                rack_healthy_hosts=int(body.get("rack_healthy_hosts", rack_info["healthy_hosts"])),
                rack_total_hosts=int(body.get("rack_total_hosts", rack_info["total_hosts"])),
                max_unavailable_k=STATE_MACHINE.max_unavailable_per_rack,
                loto_verified=loto_flag,
                canary_context=body.get("canary_context"),
                operator_note=str(body.get("operator_note") or ""),
                offline_mode=offline_mode,
            )
            out["aisle_route"] = STATE_MACHINE.plan_aisle_work_orders()
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/reset":
            STATE_MACHINE.reset(include_demo_faults=True)
            snap = _build_state_snapshot()
            return 200, {
                "status": "reset_complete",
                "service": "nitro-ultraserver-fleet",
                "state_snapshot": snap,
                "request_id": request_id,
            }

        return 404, {"error": f"Unknown API route: {method} {path}"}
    except CASVersionConflictError as exc:
        return 409, {"error": str(exc), "error_code": "CAS_VERSION_CONFLICT"}
    except BlastRadiusBudgetExceededError as exc:
        return 409, {"error": str(exc), "error_code": "N_MINUS_K_RACK_GUARD_BLOCKED"}
    except InvalidStateTransitionError as exc:
        return 400, {"error": str(exc), "error_code": "INVALID_STATE_TRANSITION"}
    except KeyError as exc:
        return 404, {"error": str(exc)}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda Function URL handler mounted behind Amazon CloudFront `/api/*` origin."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    query_params = {k: str(v) for k, v in (event.get("queryStringParameters") or {}).items() if v is not None}
    req_id = getattr(context, "aws_request_id", "cloudfront-api")

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if raw_path.startswith("/api/"):
        body: Dict[str, Any] = {}
        raw_body = event.get("body")
        if raw_body:
            if event.get("isBase64Encoded") and isinstance(raw_body, str):
                raw_body = base64.b64decode(raw_body).decode("utf-8")
            body = json.loads(raw_body) if isinstance(raw_body, str) else dict(raw_body)
        status, payload = route_api_request(method, raw_path, query_params, body, request_id=req_id)
        return {
            "statusCode": status,
            "headers": {
                **CORS_HEADERS,
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
            "body": json.dumps(payload, default=str),
        }

    target = WEB_DIR / ("index.html" if raw_path in ("/", "") else raw_path.lstrip("/"))
    if not target.is_file() and raw_path.endswith(".svg"):
        target = DOCS_DIR / Path(raw_path).name
    if target.is_file():
        ctype = mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8"
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": ctype},
            "body": target.read_text(encoding="utf-8"),
        }
    return {
        "statusCode": 404,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps({"error": "Not found", "path": raw_path}),
    }


class LocalNitroFleetHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}

        if path.startswith("/api/"):
            status, payload = route_api_request("GET", path, query, {}, request_id="local-dev")
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        candidate = (WEB_DIR / rel).resolve()
        if not candidate.is_file() and path.endswith(".svg"):
            candidate = (DOCS_DIR / Path(path).name).resolve()
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"

        data = candidate.read_bytes()
        ctype = mimetypes.guess_type(str(candidate))[0] or "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, {}, body, request_id="local-dev")
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_local_server(port: int = 3099) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), LocalNitroFleetHandler)
    print(f"NitroUltraFleet Control Plane running at http://localhost:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "3099"))
    run_local_server(port=port_num)
