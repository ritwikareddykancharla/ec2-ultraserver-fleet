"""Optimistic-Locked SQLite WAL Host Repair & Rack Commissioning Lifecycle.

Enforces deterministic host transitions:
    HEALTHY -> SUSPECT_CORDONED -> SOFT_RESET_TRIAL -> DC_RMA_DISPATCH -> BURN_IN_QUALIFICATION -> HEALTHY
with Compare-And-Swap (CAS) version locking, N-k rack/PDU fault-domain blast-radius constraints,
chronic flapping ("bad apple") soft-reset escalation, SHA-256 hash-chained SQLite WAL audit history,
and serpentine aisle-ordered data-center technician work-order routing.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

VALID_STATES: Tuple[str, ...] = (
    "HEALTHY",
    "SUSPECT_CORDONED",
    "SOFT_RESET_TRIAL",
    "DC_RMA_DISPATCH",
    "BURN_IN_QUALIFICATION",
)

ALLOWED_TRANSITIONS: Dict[str, Set[str]] = {
    "HEALTHY": {"SUSPECT_CORDONED"},
    "SUSPECT_CORDONED": {"SOFT_RESET_TRIAL", "DC_RMA_DISPATCH"},
    "SOFT_RESET_TRIAL": {"BURN_IN_QUALIFICATION", "DC_RMA_DISPATCH"},
    "DC_RMA_DISPATCH": {"BURN_IN_QUALIFICATION"},
    "BURN_IN_QUALIFICATION": {"HEALTHY", "DC_RMA_DISPATCH"},
}

HARD_FAULT_SYNDROMES: Set[str] = {
    "HBM_UNCORRECTABLE_ECC",
    "PSU_RAIL_PHASE_FAULT",
    "NVME_NAMESPACE_DEAD",
}

GENESIS_HASH = "GENESIS_NITRO_ULTRAFLEET_WAL_0000"


class RepairLifecycleError(Exception):
    """Base exception for deterministic state-machine violations."""


class InvalidStateTransitionError(RepairLifecycleError):
    """Raised when attempting an illegal host lifecycle transition or missing pre-condition."""


class CASVersionConflictError(RepairLifecycleError):
    """Raised when optimistic Compare-And-Swap version check fails under concurrent writes."""


class BlastRadiusBudgetExceededError(RepairLifecycleError):
    """Raised when cordoning a host violates the rack or PDU N-k minimum healthy capacity budget."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _canonical_sha256(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FleetRepairStateMachine:
    """SQLite WAL state machine with CAS optimistic concurrency and N-k fault-domain guardrails."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_unavailable_per_rack: int = 2,
        max_unavailable_per_pdu: int = 3,
        seed_demo_faults: bool = True,
    ) -> None:
        base_dir = Path("/tmp/nitroultrafleet-runtime")
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else (base_dir / "fleet_repair.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_unavailable_per_rack = int(max_unavailable_per_rack)
        self.max_unavailable_per_pdu = int(max_unavailable_per_pdu)
        self.seed_demo_faults = bool(seed_demo_faults)
        self._init_lock = threading.Lock()
        self._ensure_schema_and_seed()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=8.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _ensure_schema_and_seed(self) -> None:
        with self._init_lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fleet_hosts (
                        host_id TEXT PRIMARY KEY,
                        rack_id TEXT NOT NULL,
                        aisle TEXT NOT NULL,
                        row_num INTEGER NOT NULL,
                        rack_unit INTEGER NOT NULL,
                        pdu_domain TEXT NOT NULL,
                        tor_switch TEXT NOT NULL,
                        state TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        syndrome_code TEXT NOT NULL,
                        flap_count_72h INTEGER NOT NULL DEFAULT 0,
                        soft_reset_attempts INTEGER NOT NULL DEFAULT 0,
                        loto_verified INTEGER NOT NULL DEFAULT 0,
                        burn_in_passed INTEGER NOT NULL DEFAULT 1,
                        updated_at_utc TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS state_transitions_log (
                        seq_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        transition_id TEXT NOT NULL UNIQUE,
                        host_id TEXT NOT NULL,
                        rack_id TEXT NOT NULL,
                        from_state TEXT NOT NULL,
                        to_state TEXT NOT NULL,
                        expected_version INTEGER NOT NULL,
                        new_version INTEGER NOT NULL,
                        syndrome_code TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        guardrail_note TEXT NOT NULL,
                        duration_ms REAL NOT NULL,
                        prev_hash TEXT NOT NULL,
                        row_hash TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL
                    );
                    """
                )
                count = conn.execute("SELECT COUNT(*) AS cnt FROM fleet_hosts;").fetchone()["cnt"]
                if count == 0:
                    self._seed_hosts_tx(conn, include_demo_faults=self.seed_demo_faults)
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise
            finally:
                conn.close()

    def _seed_hosts_tx(self, conn: sqlite3.Connection, include_demo_faults: bool) -> None:
        now = _utc_now_iso()
        # 4 racks x 8 hosts = 32-host UltraServer pod across 2 data-hall aisles and 4 PDU domains
        rack_specs = [
            ("rack-01", "Aisle-A", 2, "pdu-A1", "tor-sw-01"),
            ("rack-02", "Aisle-A", 6, "pdu-A2", "tor-sw-02"),
            ("rack-03", "Aisle-B", 6, "pdu-B1", "tor-sw-03"),
            ("rack-04", "Aisle-B", 2, "pdu-B2", "tor-sw-04"),
        ]
        demo_overrides: Dict[str, Tuple[str, str, int, int, int]] = {}
        if include_demo_faults:
            # (state, syndrome_code, flap_count_72h, loto_verified, burn_in_passed)
            demo_overrides = {
                "usv-r01-h03": ("SUSPECT_CORDONED", "NEURONLINK_CRC_FLAP", 0, 0, 0),
                "usv-r02-h05": ("DC_RMA_DISPATCH", "HBM_UNCORRECTABLE_ECC", 1, 1, 0),
                "usv-r03-h02": ("DC_RMA_DISPATCH", "OPTIC_400G_RX_LOW", 2, 1, 0),
                "usv-r04-h06": ("BURN_IN_QUALIFICATION", "POST_RMA_MEM_SWEEP", 0, 1, 1),
            }

        prev_hash = GENESIS_HASH
        seq_counter = 1
        for r_idx, (rack_id, aisle, row_num, pdu_domain, tor_sw) in enumerate(rack_specs, start=1):
            for slot in range(8):
                host_id = f"usv-r0{r_idx}-h0{slot}"
                rack_unit = 4 + slot * 5
                state, syndrome, flaps, loto, burn_in = demo_overrides.get(
                    host_id, ("HEALTHY", "NONE", 0, 0, 1)
                )
                conn.execute(
                    """
                    INSERT INTO fleet_hosts (
                        host_id, rack_id, aisle, row_num, rack_unit, pdu_domain, tor_switch,
                        state, version, syndrome_code, flap_count_72h, soft_reset_attempts,
                        loto_verified, burn_in_passed, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?);
                    """,
                    (
                        host_id,
                        rack_id,
                        aisle,
                        row_num,
                        rack_unit,
                        pdu_domain,
                        tor_sw,
                        state,
                        1,
                        syndrome,
                        flaps,
                        loto,
                        burn_in,
                        now,
                    ),
                )
                if state != "HEALTHY":
                    tid = f"TRN-{seq_counter:05d}"
                    seq_counter += 1
                    payload = {
                        "transition_id": tid,
                        "host_id": host_id,
                        "rack_id": rack_id,
                        "from_state": "HEALTHY",
                        "to_state": state,
                        "expected_version": 0,
                        "new_version": 1,
                        "syndrome_code": syndrome,
                        "actor": "fleet-seed-controller",
                        "prev_hash": prev_hash,
                    }
                    row_hash = _canonical_sha256(payload)
                    conn.execute(
                        """
                        INSERT INTO state_transitions_log (
                            transition_id, host_id, rack_id, from_state, to_state,
                            expected_version, new_version, syndrome_code, actor,
                            guardrail_note, duration_ms, prev_hash, row_hash, created_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            tid,
                            host_id,
                            rack_id,
                            "HEALTHY",
                            state,
                            0,
                            1,
                            syndrome,
                            "fleet-seed-controller",
                            "Initial seeded fleet impairment",
                            0.12,
                            prev_hash,
                            row_hash,
                            now,
                        ),
                    )
                    prev_hash = row_hash

    def reset(self, include_demo_faults: Optional[bool] = None) -> None:
        use_faults = self.seed_demo_faults if include_demo_faults is None else bool(include_demo_faults)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute("DELETE FROM fleet_hosts;")
            conn.execute("DELETE FROM state_transitions_log;")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='state_transitions_log';")
            self._seed_hosts_tx(conn, include_demo_faults=use_faults)
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def list_hosts(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM fleet_hosts ORDER BY rack_id ASC, host_id ASC;"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_host(self, host_id: str) -> Dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM fleet_hosts WHERE host_id = ?;", (host_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown host_id: {host_id}")
            return dict(row)
        finally:
            conn.close()

    def get_rack_budget_summary(self) -> List[Dict[str, Any]]:
        hosts = self.list_hosts()
        by_rack: Dict[str, List[Dict[str, Any]]] = {}
        for h in hosts:
            by_rack.setdefault(h["rack_id"], []).append(h)

        summaries: List[Dict[str, Any]] = []
        for rack_id in sorted(by_rack.keys()):
            r_hosts = by_rack[rack_id]
            total = len(r_hosts)
            healthy = sum(1 for h in r_hosts if h["state"] == "HEALTHY")
            unavailable = total - healthy
            min_healthy_floor = total - self.max_unavailable_per_rack
            summaries.append(
                {
                    "rack_id": rack_id,
                    "aisle": r_hosts[0]["aisle"],
                    "row_num": r_hosts[0]["row_num"],
                    "pdu_domain": r_hosts[0]["pdu_domain"],
                    "tor_switch": r_hosts[0]["tor_switch"],
                    "total_hosts": total,
                    "healthy_hosts": healthy,
                    "unavailable_hosts": unavailable,
                    "max_unavailable_budget_k": self.max_unavailable_per_rack,
                    "min_healthy_floor_n_minus_k": min_healthy_floor,
                    "remaining_cordon_budget": max(0, self.max_unavailable_per_rack - unavailable),
                    "cordoned_host_ids": [h["host_id"] for h in r_hosts if h["state"] != "HEALTHY"],
                }
            )
        return summaries

    def transition_host(
        self,
        host_id: str,
        target_state: str,
        expected_version: Optional[int] = None,
        syndrome_code: Optional[str] = None,
        actor: str = "nitro-repair-controller",
        loto_verified: Optional[bool] = None,
        burn_in_passed: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Atomically transition a host with SQLite WAL CAS version locking and N-k rack guards."""
        t0 = time.perf_counter()
        if target_state not in VALID_STATES:
            raise InvalidStateTransitionError(f"Unsupported target state: {target_state}")

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute(
                "SELECT * FROM fleet_hosts WHERE host_id = ?;", (host_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown host_id: {host_id}")

            cur_state = row["state"]
            cur_version = int(row["version"])
            rack_id = row["rack_id"]
            pdu_domain = row["pdu_domain"]
            flaps = int(row["flap_count_72h"])
            soft_attempts = int(row["soft_reset_attempts"])
            eff_syndrome = (syndrome_code if syndrome_code is not None else row["syndrome_code"]).strip() or "NONE"

            if expected_version is not None and int(expected_version) != cur_version:
                raise CASVersionConflictError(
                    f"CAS conflict on {host_id}: expected version {expected_version}, found version {cur_version} (state={cur_state})"
                )
            check_version = cur_version if expected_version is None else int(expected_version)

            allowed_next = ALLOWED_TRANSITIONS.get(cur_state, set())
            if target_state not in allowed_next:
                raise InvalidStateTransitionError(
                    f"Illegal state transition for {host_id}: {cur_state} -> {target_state} (allowed: {sorted(allowed_next)})"
                )

            guardrail_notes: List[str] = []

            # 1. N-k Rack & PDU Blast-Radius Guard when cordoning a healthy host
            if cur_state == "HEALTHY" and target_state == "SUSPECT_CORDONED":
                rack_unavail = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM fleet_hosts WHERE rack_id = ? AND state != 'HEALTHY';",
                    (rack_id,),
                ).fetchone()["cnt"]
                if rack_unavail >= self.max_unavailable_per_rack:
                    raise BlastRadiusBudgetExceededError(
                        f"N-k rack budget exceeded on {rack_id}: {rack_unavail}/{self.max_unavailable_per_rack} hosts already non-healthy"
                    )
                pdu_unavail = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM fleet_hosts WHERE pdu_domain = ? AND state != 'HEALTHY';",
                    (pdu_domain,),
                ).fetchone()["cnt"]
                if pdu_unavail >= self.max_unavailable_per_pdu:
                    raise BlastRadiusBudgetExceededError(
                        f"N-k PDU budget exceeded on {pdu_domain}: {pdu_unavail}/{self.max_unavailable_per_pdu} hosts already non-healthy"
                    )
                guardrail_notes.append(
                    f"N-k rack check passed ({rack_unavail + 1}/{self.max_unavailable_per_rack} unavailable in {rack_id})"
                )

            # 2. Chronic Flapping ("Bad Apple") & Hard-Syndrome Soft-Reset Guard
            if cur_state == "SUSPECT_CORDONED" and target_state == "SOFT_RESET_TRIAL":
                if eff_syndrome in HARD_FAULT_SYNDROMES:
                    raise InvalidStateTransitionError(
                        f"Hard hardware syndrome {eff_syndrome} cannot use SOFT_RESET_TRIAL; dispatch to DC_RMA_DISPATCH"
                    )
                if flaps >= 2:
                    raise InvalidStateTransitionError(
                        f"Chronic flapping host {host_id} (flap_count_72h={flaps}) blocked from soft reset; escalate to DC_RMA_DISPATCH"
                    )
                soft_attempts += 1
                guardrail_notes.append(f"Soft reset trial #{soft_attempts} permitted (flap_count_72h={flaps})")

            # 3. LOTO / Drain verification before exiting DC_RMA_DISPATCH into BURN_IN_QUALIFICATION
            eff_loto = int(row["loto_verified"]) if loto_verified is None else (1 if loto_verified else 0)
            if target_state == "DC_RMA_DISPATCH" and loto_verified is None:
                eff_loto = 1
            if cur_state == "DC_RMA_DISPATCH" and target_state == "BURN_IN_QUALIFICATION":
                if not eff_loto:
                    raise InvalidStateTransitionError(
                        f"Cannot advance {host_id} from DC_RMA_DISPATCH without Lockout/Tagout (loto_verified=True)"
                    )
                guardrail_notes.append("Physical FRU replacement + LOTO verification confirmed")

            # 4. Post-Repair Burn-In Qualification Gate before returning to HEALTHY
            eff_burn_in = int(row["burn_in_passed"]) if burn_in_passed is None else (1 if burn_in_passed else 0)
            if target_state == "BURN_IN_QUALIFICATION" and burn_in_passed is None:
                eff_burn_in = 1
            if cur_state == "BURN_IN_QUALIFICATION" and target_state == "HEALTHY":
                if not eff_burn_in:
                    raise InvalidStateTransitionError(
                        f"Host {host_id} failed burn-in qualification; cannot return to HEALTHY capacity pool"
                    )
                eff_syndrome = "NONE"
                flaps = 0
                soft_attempts = 0
                guardrail_notes.append("Burn-in qualification passed; capacity restored to HEALTHY pool")

            now = _utc_now_iso()
            upd = conn.execute(
                """
                UPDATE fleet_hosts
                   SET state = ?,
                       version = version + 1,
                       syndrome_code = ?,
                       flap_count_72h = ?,
                       soft_reset_attempts = ?,
                       loto_verified = ?,
                       burn_in_passed = ?,
                       updated_at_utc = ?
                 WHERE host_id = ? AND version = ?;
                """,
                (
                    target_state,
                    eff_syndrome,
                    flaps,
                    soft_attempts,
                    eff_loto,
                    eff_burn_in,
                    now,
                    host_id,
                    check_version,
                ),
            )
            if upd.rowcount != 1:
                raise CASVersionConflictError(f"Concurrent CAS update lost race on {host_id}")

            new_version = check_version + 1
            last_log = conn.execute(
                "SELECT seq_id, row_hash FROM state_transitions_log ORDER BY seq_id DESC LIMIT 1;"
            ).fetchone()
            prev_hash = last_log["row_hash"] if last_log else GENESIS_HASH
            next_seq = (int(last_log["seq_id"]) + 1) if last_log else 1
            tid = f"TRN-{next_seq:05d}"

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            note_str = "; ".join(guardrail_notes) if guardrail_notes else f"{cur_state} -> {target_state}"
            hash_payload = {
                "transition_id": tid,
                "host_id": host_id,
                "rack_id": rack_id,
                "from_state": cur_state,
                "to_state": target_state,
                "expected_version": check_version,
                "new_version": new_version,
                "syndrome_code": eff_syndrome,
                "actor": actor,
                "prev_hash": prev_hash,
            }
            row_hash = _canonical_sha256(hash_payload)

            conn.execute(
                """
                INSERT INTO state_transitions_log (
                    transition_id, host_id, rack_id, from_state, to_state,
                    expected_version, new_version, syndrome_code, actor,
                    guardrail_note, duration_ms, prev_hash, row_hash, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    tid,
                    host_id,
                    rack_id,
                    cur_state,
                    target_state,
                    check_version,
                    new_version,
                    eff_syndrome,
                    actor,
                    note_str,
                    round(elapsed_ms, 3),
                    prev_hash,
                    row_hash,
                    now,
                ),
            )
            conn.execute("COMMIT;")

            return {
                "transition_id": tid,
                "host_id": host_id,
                "rack_id": rack_id,
                "from_state": cur_state,
                "to_state": target_state,
                "previous_version": check_version,
                "new_version": new_version,
                "syndrome_code": eff_syndrome,
                "guardrail_note": note_str,
                "duration_ms": round(elapsed_ms, 3),
                "row_hash": row_hash,
            }
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def verify_audit_chain(self) -> Dict[str, Any]:
        """Verify SHA-256 hash links across all recorded SQLite WAL state transitions."""
        t0 = time.perf_counter()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM state_transitions_log ORDER BY seq_id ASC;"
            ).fetchall()
        finally:
            conn.close()

        expected_prev = GENESIS_HASH
        for r in rows:
            if r["prev_hash"] != expected_prev:
                return {
                    "valid": False,
                    "checked_rows": len(rows),
                    "broken_at_transition": r["transition_id"],
                    "verify_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }
            recomputed = _canonical_sha256(
                {
                    "transition_id": r["transition_id"],
                    "host_id": r["host_id"],
                    "rack_id": r["rack_id"],
                    "from_state": r["from_state"],
                    "to_state": r["to_state"],
                    "expected_version": int(r["expected_version"]),
                    "new_version": int(r["new_version"]),
                    "syndrome_code": r["syndrome_code"],
                    "actor": r["actor"],
                    "prev_hash": r["prev_hash"],
                }
            )
            if recomputed != r["row_hash"]:
                return {
                    "valid": False,
                    "checked_rows": len(rows),
                    "broken_at_transition": r["transition_id"],
                    "verify_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }
            expected_prev = r["row_hash"]

        return {
            "valid": True,
            "checked_rows": len(rows),
            "head_hash": expected_prev,
            "verify_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    def list_recent_transitions(self, limit: int = 25) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM state_transitions_log ORDER BY seq_id DESC LIMIT ?;",
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def plan_aisle_work_orders(
        self,
        extra_stops: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compute serpentine S-shape aisle walk route for DC_RMA_DISPATCH tickets vs. ticket-FIFO order."""
        hosts = [h for h in self.list_hosts() if h["state"] == "DC_RMA_DISPATCH"]
        if extra_stops:
            hosts.extend(extra_stops)

        def _coords(item: Dict[str, Any]) -> Tuple[float, float]:
            # Parts cage at (x=0m, y=0m). Aisles spaced 12m apart; rows spaced 3.5m along aisle.
            aisle_name = str(item.get("aisle", "Aisle-A"))
            aisle_idx = 1 if aisle_name.endswith("A") else (2 if aisle_name.endswith("B") else 3)
            x_m = aisle_idx * 12.0
            y_m = float(item.get("row_num", 1)) * 3.5 + float(item.get("rack_unit", 1)) * 0.045
            return x_m, y_m

        def _path_meters(seq: List[Dict[str, Any]]) -> float:
            if not seq:
                return 0.0
            cur_x, cur_y = 0.0, 0.0  # Parts cage origin
            dist = 0.0
            for st in seq:
                nx, ny = _coords(st)
                dist += abs(nx - cur_x) + abs(ny - cur_y)
                cur_x, cur_y = nx, ny
            dist += abs(cur_x) + abs(cur_y)  # Return to parts cage
            return dist

        fifo_order = list(hosts)
        fifo_m = _path_meters(fifo_order)

        # Serpentine S-shape sort: odd aisles ascending by row/unit, even aisles descending by row/unit
        def _serpentine_key(item: Dict[str, Any]) -> Tuple[int, float]:
            aisle_name = str(item.get("aisle", "Aisle-A"))
            aisle_idx = 1 if aisle_name.endswith("A") else (2 if aisle_name.endswith("B") else 3)
            pos = float(item.get("row_num", 1)) * 100.0 + float(item.get("rack_unit", 1))
            return (aisle_idx, pos if aisle_idx % 2 == 1 else -pos)

        serpentine_order = sorted(hosts, key=_serpentine_key)
        serpentine_m = _path_meters(serpentine_order)
        saved_pct = ((fifo_m - serpentine_m) / fifo_m * 100.0) if fifo_m > 0 else 0.0

        return {
            "parts_cage_origin": "Hall-C · Parts Cage (0m, 0m)",
            "work_order_count": len(serpentine_order),
            "fifo_walk_meters": round(fifo_m, 2),
            "aisle_serpentine_walk_meters": round(serpentine_m, 2),
            "walk_distance_saved_pct": round(max(0.0, saved_pct), 2),
            "ordered_work_orders": [
                {
                    "stop_index": idx + 1,
                    "host_id": h["host_id"],
                    "rack_id": h["rack_id"],
                    "location": f"{h['aisle']} · Row {h['row_num']:02d} · U{h['rack_unit']:02d}",
                    "syndrome_code": h["syndrome_code"],
                    "pdu_domain": h["pdu_domain"],
                }
                for idx, h in enumerate(serpentine_order)
            ],
        }
