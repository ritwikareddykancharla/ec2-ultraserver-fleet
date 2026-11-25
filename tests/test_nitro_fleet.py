"""Unit test suite for NitroUltraFleet control-plane modules and Lambda API router."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nitro_fleet.repair_state_machine import (
    BlastRadiusBudgetExceededError,
    CASVersionConflictError,
    FleetRepairStateMachine,
    InvalidStateTransitionError,
)
from nitro_fleet.rma_copilot import RMACopilot, evaluate_deterministic_guardrails
from nitro_fleet.server import lambda_handler, route_api_request
from nitro_fleet.sprt_canary import (
    SPRTCanaryEngine,
    compute_fixed_horizon_samples,
    compute_wald_boundaries,
    run_monte_carlo_sprt_benchmark,
)
from nitro_fleet.topology_scheduler import (
    analyze_collective_trace,
    generate_collective_trace,
    schedule_workload_queue,
)


class TestNitroUltraFleet(unittest.TestCase):
    def test_01_wald_sprt_boundaries_and_fixed_horizon_math(self) -> None:
        upper_a, lower_b = compute_wald_boundaries(alpha=0.02, beta=0.02)
        self.assertAlmostEqual(upper_a, 3.89182, places=4)
        self.assertAlmostEqual(lower_b, -3.89182, places=4)
        fixed_n = compute_fixed_horizon_samples(p0=0.01, p1=0.06, alpha=0.02, beta=0.02)
        self.assertEqual(fixed_n, 192)

    def test_02_sprt_halts_regressed_firmware_in_first_canary_wave(self) -> None:
        engine = SPRTCanaryEngine(p0=0.01, p1=0.06, alpha=0.02, beta=0.02)
        out = engine.run_staged_rollout(
            build_id="efa-nic-fw-3.8.0-rc2",
            observed_failure_rate=0.09,
            seed=4201,
        )
        self.assertEqual(out["overall_verdict"], "HALTED_AND_ROLLED_BACK")
        self.assertEqual(out["halted_at_wave"], "Canary_1Pct")
        self.assertEqual(out["skipped_waves"], ["Wave_10Pct", "Wave_50Pct", "Fleet_100Pct"])
        self.assertLess(out["summary_metrics"]["total_samples_evaluated"], 192)

    def test_03_sprt_promotes_clean_firmware_across_all_four_waves(self) -> None:
        engine = SPRTCanaryEngine(p0=0.01, p1=0.06, alpha=0.02, beta=0.02)
        out = engine.run_staged_rollout(
            build_id="nitro-card-fw-5.14.2",
            observed_failure_rate=0.005,
            seed=4208,
        )
        self.assertEqual(out["overall_verdict"], "PROMOTED_TO_FLEET_100PCT")
        self.assertIsNone(out["halted_at_wave"])
        self.assertEqual(len(out["promoted_waves"]), 4)

    def test_04_sprt_monte_carlo_respects_error_bounds_and_saves_samples(self) -> None:
        mc = run_monte_carlo_sprt_benchmark(num_trials=200, p0=0.01, p1=0.06, alpha=0.02, beta=0.02, seed=77)
        self.assertLessEqual(mc["false_halt_rate_pct"], 4.0)
        self.assertLessEqual(mc["false_promotion_rate_pct"], 4.0)
        self.assertGreater(mc["overall_sample_reduction_pct"], 40.0)

    def test_05_repair_lifecycle_full_walk_and_illegal_skip_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sm = FleetRepairStateMachine(db_path=str(Path(tmp) / "test.db"), seed_demo_faults=False)
            r1 = sm.transition_host("usv-r01-h00", "SUSPECT_CORDONED", expected_version=1, syndrome_code="OPTIC_400G_RX_LOW")
            self.assertEqual(r1["new_version"], 2)

            # Skipping directly from SUSPECT_CORDONED to HEALTHY must fail
            with self.assertRaises(InvalidStateTransitionError):
                sm.transition_host("usv-r01-h00", "HEALTHY", expected_version=2)

            r2 = sm.transition_host("usv-r01-h00", "SOFT_RESET_TRIAL", expected_version=2)
            r3 = sm.transition_host("usv-r01-h00", "DC_RMA_DISPATCH", expected_version=r2["new_version"])
            r4 = sm.transition_host(
                "usv-r01-h00", "BURN_IN_QUALIFICATION", expected_version=r3["new_version"], loto_verified=True
            )
            r5 = sm.transition_host(
                "usv-r01-h00", "HEALTHY", expected_version=r4["new_version"], burn_in_passed=True
            )
            self.assertEqual(r5["to_state"], "HEALTHY")
            self.assertTrue(sm.verify_audit_chain()["valid"])

    def test_06_sqlite_cas_optimistic_concurrency_allows_single_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sm = FleetRepairStateMachine(db_path=str(Path(tmp) / "cas.db"), seed_demo_faults=False)

            def attempt(worker_id: int) -> str:
                try:
                    sm.transition_host(
                        "usv-r02-h01",
                        "SUSPECT_CORDONED",
                        expected_version=1,
                        syndrome_code="NEURONLINK_CRC_FLAP",
                        actor=f"w-{worker_id}",
                    )
                    return "WON"
                except CASVersionConflictError:
                    return "CONFLICT"

            with ThreadPoolExecutor(max_workers=6) as pool:
                outcomes = list(pool.map(attempt, range(6)))
            self.assertEqual(outcomes.count("WON"), 1)
            self.assertEqual(outcomes.count("CONFLICT"), 5)

    def test_07_n_minus_k_rack_blast_radius_budget_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sm = FleetRepairStateMachine(
                db_path=str(Path(tmp) / "budget.db"),
                max_unavailable_per_rack=2,
                seed_demo_faults=False,
            )
            sm.transition_host("usv-r03-h00", "SUSPECT_CORDONED", syndrome_code="LINK_FLAP")
            sm.transition_host("usv-r03-h01", "SUSPECT_CORDONED", syndrome_code="LINK_FLAP")
            with self.assertRaises(BlastRadiusBudgetExceededError):
                sm.transition_host("usv-r03-h02", "SUSPECT_CORDONED", syndrome_code="LINK_FLAP")

    def test_08_sqlite_sha256_audit_chain_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_file = Path(tmp) / "tamper.db"
            sm = FleetRepairStateMachine(db_path=str(db_file), seed_demo_faults=True)
            sm.transition_host("usv-r01-h03", "SOFT_RESET_TRIAL")
            self.assertTrue(sm.verify_audit_chain()["valid"])

            conn = sqlite3.connect(str(db_file))
            conn.execute("UPDATE state_transitions_log SET actor = 'forged-actor' WHERE seq_id = 1;")
            conn.commit()
            conn.close()
            self.assertFalse(sm.verify_audit_chain()["valid"])

    def test_09_aisle_serpentine_routing_saves_floor_walk_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sm = FleetRepairStateMachine(db_path=str(Path(tmp) / "route.db"), seed_demo_faults=False)
            # Alternating aisles in FIFO arrival order vs serpentine aisle order
            stops = [
                {"host_id": "usv-r01-h01", "rack_id": "rack-01", "aisle": "Aisle-A", "row_num": 2, "rack_unit": 9, "syndrome_code": "ECC", "pdu_domain": "pdu-A1"},
                {"host_id": "usv-r03-h06", "rack_id": "rack-03", "aisle": "Aisle-B", "row_num": 6, "rack_unit": 34, "syndrome_code": "OPTIC", "pdu_domain": "pdu-B1"},
                {"host_id": "usv-r02-h07", "rack_id": "rack-02", "aisle": "Aisle-A", "row_num": 6, "rack_unit": 39, "syndrome_code": "ECC", "pdu_domain": "pdu-A2"},
                {"host_id": "usv-r04-h01", "rack_id": "rack-04", "aisle": "Aisle-B", "row_num": 2, "rack_unit": 9, "syndrome_code": "PSU", "pdu_domain": "pdu-B2"},
            ]
            plan = sm.plan_aisle_work_orders(extra_stops=stops)
            self.assertLess(plan["aisle_serpentine_walk_meters"], plan["fifo_walk_meters"])
            self.assertGreater(plan["walk_distance_saved_pct"], 10.0)

    def test_10_cusum_and_wait_for_isolates_straggler_and_deadlock_cycle(self) -> None:
        tr_strag = generate_collective_trace(preset="rank11_nic_straggler")
        res_strag = analyze_collective_trace(tr_strag)
        self.assertEqual(res_strag["diagnosis_type"], "SINGLE_RANK_STRAGGLER")
        self.assertEqual(res_strag["isolated_culprits"], [11])
        self.assertGreater(res_strag["lead_steps_before_timeout"], 10)

        tr_cycle = generate_collective_trace(preset="dual_rank_deadlock_cycle")
        res_cycle = analyze_collective_trace(tr_cycle)
        self.assertEqual(res_cycle["diagnosis_type"], "COLLECTIVE_DEADLOCK_CYCLE")
        self.assertEqual(res_cycle["isolated_culprits"], [3, 7, 12])

    def test_11_buddy_subring_and_easy_backfill_zero_fragmentation(self) -> None:
        out = schedule_workload_queue(
            preset_name="mixed_training_and_ci",
            cordoned_hosts=["usv-r01-h03", "usv-r02-h05"],
        )
        cmp_m = out["comparison"]
        self.assertEqual(cmp_m["buddy_cross_spine_fragmentations"], 0)
        self.assertGreater(cmp_m["easy_backfilled_jobs"], 0)
        self.assertGreater(cmp_m["buddy_easy_utilization_pct"], cmp_m["naive_fifo_utilization_pct"])

    def test_12_rma_guardrails_and_lambda_api_router(self) -> None:
        blocked = evaluate_deterministic_guardrails(
            incident_type="host_rma",
            host_state="SUSPECT_CORDONED",
            rack_healthy_hosts=7,
            rack_total_hosts=8,
            max_unavailable_k=2,
            loto_verified=False,
        )
        self.assertFalse(blocked["authorized_for_action"])
        self.assertEqual(blocked["guardrail_verdict"], "HOLD_GUARDRAIL_BLOCKED")

        copilot = RMACopilot()
        offline_ticket = copilot.synthesize_ticket(
            incident_type="host_rma",
            host_id="usv-r02-h05",
            host_state="DC_RMA_DISPATCH",
            syndrome_code="HBM_UNCORRECTABLE_ECC",
            loto_verified=True,
            offline_mode=True,
        )
        self.assertTrue(offline_ticket["grounding_check"]["grounded"])

        resp_health = lambda_handler({"rawPath": "/api/health", "requestContext": {"http": {"method": "GET"}}})
        self.assertEqual(resp_health["statusCode"], 200)
        health_json = json.loads(resp_health["body"])
        self.assertEqual(health_json["status"], "ok")

        code_sprt, sprt_json = route_api_request(
            "POST", "/api/canary/sprt", {}, {"scenario_id": "efa_crc_regression"}
        )
        self.assertEqual(code_sprt, 200)
        self.assertEqual(sprt_json["halted_at_wave"], "Canary_1Pct")


if __name__ == "__main__":
    unittest.main()
