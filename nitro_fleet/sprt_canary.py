"""Wald Sequential Probability Ratio Test (SPRT) Firmware Canary & Qualification Engine.

Evaluates staged host qualification and firmware rollout waves:
    Canary_1Pct -> Wave_10Pct -> Wave_50Pct -> Fleet_100Pct
using exact Wald log-likelihood ratio boundaries:
    A = ln((1 - beta) / alpha)   [Upper threshold: Halt & Rollback]
    B = ln(beta / (1 - alpha))   [Lower threshold: Promote Wave]
"""
from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROLLOUT_WAVES: Tuple[str, ...] = (
    "Canary_1Pct",
    "Wave_10Pct",
    "Wave_50Pct",
    "Fleet_100Pct",
)

WAVE_FLEET_EXPOSURE_PCT: Dict[str, float] = {
    "Canary_1Pct": 1.0,
    "Wave_10Pct": 10.0,
    "Wave_50Pct": 50.0,
    "Fleet_100Pct": 100.0,
}

PRESET_ROLLOUT_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "efa_crc_regression": {
        "scenario_id": "efa_crc_regression",
        "build_id": "efa-nic-fw-3.8.0-rc2",
        "title": "Regressed 400G EFA Firmware (8.5% Link CRC Rate)",
        "component": "EFA Adapter Firmware",
        "p0": 0.01,
        "p1": 0.06,
        "alpha": 0.02,
        "beta": 0.02,
        "wave_rates": {
            "Canary_1Pct": 0.085,
            "Wave_10Pct": 0.085,
            "Wave_50Pct": 0.085,
            "Fleet_100Pct": 0.085,
        },
        "seed": 4201,
    },
    "clean_nitro_fw": {
        "scenario_id": "clean_nitro_fw",
        "build_id": "nitro-card-fw-5.14.2",
        "title": "Clean Nitro Hypervisor Update (0.8% Baseline Noise)",
        "component": "Nitro Security & VPC Card",
        "p0": 0.01,
        "p1": 0.06,
        "alpha": 0.02,
        "beta": 0.02,
        "wave_rates": {
            "Canary_1Pct": 0.008,
            "Wave_10Pct": 0.008,
            "Wave_50Pct": 0.008,
            "Fleet_100Pct": 0.008,
        },
        "seed": 4208,
    },
    "wave2_rev_b_psu_bug": {
        "scenario_id": "wave2_rev_b_psu_bug",
        "build_id": "bmc-power-ctl-4.2.1-rc1",
        "title": "Rev-B Rack Bug Caught in Wave_10Pct (9.2% Failure)",
        "component": "Rack BMC Power Controller",
        "p0": 0.01,
        "p1": 0.06,
        "alpha": 0.02,
        "beta": 0.02,
        "wave_rates": {
            "Canary_1Pct": 0.006,
            "Wave_10Pct": 0.092,
            "Wave_50Pct": 0.092,
            "Fleet_100Pct": 0.092,
        },
        "seed": 4219,
    },
    "neuron_driver_borderline": {
        "scenario_id": "neuron_driver_borderline",
        "build_id": "neuron-rt-2.21.4-rc3",
        "title": "Moderate Collective Init Regression (6.2% Failure)",
        "component": "Accelerator Runtime Driver",
        "p0": 0.01,
        "p1": 0.06,
        "alpha": 0.02,
        "beta": 0.02,
        "wave_rates": {
            "Canary_1Pct": 0.062,
            "Wave_10Pct": 0.062,
            "Wave_50Pct": 0.062,
            "Fleet_100Pct": 0.062,
        },
        "seed": 4231,
    },
}


def compute_wald_boundaries(alpha: float, beta: float) -> Tuple[float, float]:
    """Return exact Wald log-likelihood boundaries (A_halt, B_promote)."""
    if not (0.0 < alpha < 0.5) or not (0.0 < beta < 0.5):
        raise ValueError("alpha and beta must be in (0.0, 0.5)")
    upper_a = math.log((1.0 - beta) / alpha)
    lower_b = math.log(beta / (1.0 - alpha))
    return upper_a, lower_b


def compute_fixed_horizon_samples(
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
) -> int:
    """Classical fixed-sample binomial normal-approximation test size matching (p0, p1, alpha, beta)."""
    if not (0.0 < p0 < p1 < 1.0):
        raise ValueError("Require 0 < p0 < p1 < 1")
    norm = statistics.NormalDist()
    z_alpha = norm.inv_cdf(1.0 - alpha)
    z_beta = norm.inv_cdf(1.0 - beta)
    num = z_alpha * math.sqrt(p0 * (1.0 - p0)) + z_beta * math.sqrt(p1 * (1.0 - p1))
    den = p1 - p0
    return int(math.ceil((num / den) ** 2))


@dataclass
class WaveEvaluation:
    wave_name: str
    decision: str
    samples_evaluated: int
    failures_observed: int
    empirical_failure_rate: float
    final_llr: float
    upper_bound_a: float
    lower_bound_b: float
    fixed_horizon_samples: int
    sample_reduction_pct: float
    trajectory: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wave_name": self.wave_name,
            "decision": self.decision,
            "samples_evaluated": self.samples_evaluated,
            "failures_observed": self.failures_observed,
            "empirical_failure_rate": round(self.empirical_failure_rate, 4),
            "final_llr": round(self.final_llr, 4),
            "upper_bound_a": round(self.upper_bound_a, 4),
            "lower_bound_b": round(self.lower_bound_b, 4),
            "fixed_horizon_samples": self.fixed_horizon_samples,
            "sample_reduction_pct": round(self.sample_reduction_pct, 2),
            "trajectory": self.trajectory,
        }


class SPRTCanaryEngine:
    """Sequential Probability Ratio Test controller for multi-wave firmware rollouts."""

    def __init__(
        self,
        p0: float = 0.01,
        p1: float = 0.06,
        alpha: float = 0.02,
        beta: float = 0.02,
        max_samples_per_wave: int = 400,
    ) -> None:
        if not (0.0 < p0 < p1 < 1.0):
            raise ValueError("Require 0 < p0 < p1 < 1")
        self.p0 = float(p0)
        self.p1 = float(p1)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_samples_per_wave = int(max_samples_per_wave)
        self.upper_a, self.lower_b = compute_wald_boundaries(self.alpha, self.beta)
        self.step_fail_llr = math.log(self.p1 / self.p0)
        self.step_pass_llr = math.log((1.0 - self.p1) / (1.0 - self.p0))
        self.fixed_horizon_n = compute_fixed_horizon_samples(self.p0, self.p1, self.alpha, self.beta)

    def evaluate_single_wave(
        self,
        wave_name: str,
        outcomes: Sequence[int],
        record_trajectory: bool = True,
    ) -> WaveEvaluation:
        """Stream binary qualification check outcomes (1=failure, 0=pass) through Wald SPRT."""
        llr = 0.0
        failures = 0
        samples = 0
        decision = "TRUNCATED_HOLD"
        trajectory: List[Dict[str, float]] = [{"step": 0, "llr": 0.0, "failures": 0}] if record_trajectory else []

        limit = min(len(outcomes), self.max_samples_per_wave)
        for idx in range(limit):
            is_fail = 1 if outcomes[idx] else 0
            samples += 1
            failures += is_fail
            llr += self.step_fail_llr if is_fail else self.step_pass_llr

            if record_trajectory:
                trajectory.append(
                    {
                        "step": samples,
                        "llr": round(llr, 4),
                        "failures": failures,
                    }
                )

            if llr >= self.upper_a:
                decision = "HALT_ROLLBACK"
                break
            if llr <= self.lower_b:
                decision = "PROMOTE_WAVE"
                break
        else:
            # Truncated decision at max_samples_per_wave cap
            decision = "HALT_ROLLBACK" if llr > 0.0 else "PROMOTE_WAVE"

        emp_rate = failures / max(1, samples)
        reduction = (1.0 - (samples / max(1, self.fixed_horizon_n))) * 100.0
        return WaveEvaluation(
            wave_name=wave_name,
            decision=decision,
            samples_evaluated=samples,
            failures_observed=failures,
            empirical_failure_rate=emp_rate,
            final_llr=llr,
            upper_bound_a=self.upper_a,
            lower_bound_b=self.lower_b,
            fixed_horizon_samples=self.fixed_horizon_n,
            sample_reduction_pct=reduction,
            trajectory=trajectory,
        )

    def run_staged_rollout(
        self,
        build_id: str = "efa-nic-fw-3.8.0-rc2",
        wave_rates: Optional[Dict[str, float]] = None,
        observed_failure_rate: Optional[float] = None,
        explicit_wave_outcomes: Optional[Dict[str, Sequence[int]]] = None,
        seed: int = 4201,
    ) -> Dict[str, Any]:
        """Run sequential gating across Canary_1Pct -> Wave_10Pct -> Wave_50Pct -> Fleet_100Pct."""
        t0 = time.perf_counter()
        rng = random.Random(seed)
        default_rate = observed_failure_rate if observed_failure_rate is not None else self.p0
        rates = wave_rates or {w: default_rate for w in ROLLOUT_WAVES}

        wave_reports: List[Dict[str, Any]] = []
        skipped_waves: List[str] = []
        halted_at_wave: Optional[str] = None
        total_samples = 0
        total_failures = 0

        for idx, wave in enumerate(ROLLOUT_WAVES):
            if explicit_wave_outcomes and wave in explicit_wave_outcomes:
                stream = list(explicit_wave_outcomes[wave])
            else:
                rate = float(rates.get(wave, default_rate))
                stream = [1 if rng.random() < rate else 0 for _ in range(self.max_samples_per_wave)]

            ev = self.evaluate_single_wave(wave_name=wave, outcomes=stream, record_trajectory=True)
            wave_reports.append(ev.to_dict())
            total_samples += ev.samples_evaluated
            total_failures += ev.failures_observed

            if ev.decision == "HALT_ROLLBACK":
                halted_at_wave = wave
                skipped_waves = list(ROLLOUT_WAVES[idx + 1 :])
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        overall_verdict = "HALTED_AND_ROLLED_BACK" if halted_at_wave else "PROMOTED_TO_FLEET_100PCT"
        max_exposure_pct = WAVE_FLEET_EXPOSURE_PCT[halted_at_wave] if halted_at_wave else 100.0
        protected_fleet_pct = 100.0 - max_exposure_pct if halted_at_wave else 100.0
        fixed_total_for_evaluated_waves = self.fixed_horizon_n * len(wave_reports)
        fixed_full_four_wave_samples = self.fixed_horizon_n * len(ROLLOUT_WAVES)

        return {
            "build_id": build_id,
            "overall_verdict": overall_verdict,
            "halted_at_wave": halted_at_wave,
            "promoted_waves": [w["wave_name"] for w in wave_reports if w["decision"] == "PROMOTE_WAVE"],
            "skipped_waves": skipped_waves,
            "parameters": {
                "p0_baseline_rate": self.p0,
                "p1_regression_rate": self.p1,
                "alpha_type1_bound": self.alpha,
                "beta_type2_bound": self.beta,
                "upper_bound_a": round(self.upper_a, 4),
                "lower_bound_b": round(self.lower_b, 4),
                "step_fail_llr": round(self.step_fail_llr, 4),
                "step_pass_llr": round(self.step_pass_llr, 4),
            },
            "summary_metrics": {
                "total_samples_evaluated": total_samples,
                "total_failures_observed": total_failures,
                "fixed_horizon_per_wave": self.fixed_horizon_n,
                "fixed_horizon_evaluated_waves": fixed_total_for_evaluated_waves,
                "fixed_horizon_four_waves": fixed_full_four_wave_samples,
                "sample_reduction_vs_evaluated_fixed_pct": round(
                    (1.0 - total_samples / max(1, fixed_total_for_evaluated_waves)) * 100.0, 2
                ),
                "sample_reduction_vs_full_rollout_pct": round(
                    (1.0 - total_samples / max(1, fixed_full_four_wave_samples)) * 100.0, 2
                ),
                "max_fleet_exposure_pct": max_exposure_pct,
                "protected_fleet_pct": protected_fleet_pct,
                "eval_ms": round(elapsed_ms, 3),
            },
            "wave_results": wave_reports,
        }


def run_monte_carlo_sprt_benchmark(
    num_trials: int = 500,
    p0: float = 0.01,
    p1: float = 0.06,
    alpha: float = 0.02,
    beta: float = 0.02,
    seed: int = 20261125,
) -> Dict[str, Any]:
    """Execute 500 Monte Carlo SPRT trials (250 clean H0 builds + 250 regressed H1 builds) vs. fixed Z-test."""
    t0 = time.perf_counter()
    engine = SPRTCanaryEngine(p0=p0, p1=p1, alpha=alpha, beta=beta, max_samples_per_wave=450)
    rng = random.Random(seed)

    clean_trials = num_trials // 2
    regressed_trials = num_trials - clean_trials

    false_halts = 0
    clean_samples: List[int] = []
    for _ in range(clean_trials):
        outcomes = [1 if rng.random() < p0 else 0 for _ in range(engine.max_samples_per_wave)]
        res = engine.evaluate_single_wave("Canary_1Pct", outcomes, record_trajectory=False)
        clean_samples.append(res.samples_evaluated)
        if res.decision == "HALT_ROLLBACK":
            false_halts += 1

    false_promotions = 0
    regressed_samples: List[int] = []
    for _ in range(regressed_trials):
        outcomes = [1 if rng.random() < p1 else 0 for _ in range(engine.max_samples_per_wave)]
        res = engine.evaluate_single_wave("Canary_1Pct", outcomes, record_trajectory=False)
        regressed_samples.append(res.samples_evaluated)
        if res.decision == "PROMOTE_WAVE":
            false_promotions += 1

    all_samples = clean_samples + regressed_samples
    mean_sprt_samples = statistics.mean(all_samples)
    mean_clean_samples = statistics.mean(clean_samples)
    mean_regressed_samples = statistics.mean(regressed_samples)
    fixed_n = engine.fixed_horizon_n

    # Also compare fixed-horizon Z-test empirical errors on the exact same (p0, p1)
    # Critical failure count k_crit for one-sided binomial/normal test at alpha:
    z_alpha = statistics.NormalDist().inv_cdf(1.0 - alpha)
    k_crit = math.ceil(fixed_n * p0 + z_alpha * math.sqrt(fixed_n * p0 * (1.0 - p0)))

    fixed_false_halts = 0
    for _ in range(clean_trials):
        fails = sum(1 for _ in range(fixed_n) if rng.random() < p0)
        if fails >= k_crit:
            fixed_false_halts += 1

    fixed_false_promotions = 0
    for _ in range(regressed_trials):
        fails = sum(1 for _ in range(fixed_n) if rng.random() < p1)
        if fails < k_crit:
            fixed_false_promotions += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    sample_reduction_pct = (1.0 - (mean_sprt_samples / fixed_n)) * 100.0
    regressed_sample_reduction_pct = (1.0 - (mean_regressed_samples / fixed_n)) * 100.0

    return {
        "num_trials": num_trials,
        "clean_trials_h0": clean_trials,
        "regressed_trials_h1": regressed_trials,
        "p0": p0,
        "p1": p1,
        "alpha_bound": alpha,
        "beta_bound": beta,
        "upper_bound_a": round(engine.upper_a, 4),
        "lower_bound_b": round(engine.lower_b, 4),
        "false_halt_count": false_halts,
        "false_halt_rate_pct": round((false_halts / clean_trials) * 100.0, 2),
        "false_promotion_count": false_promotions,
        "false_promotion_rate_pct": round((false_promotions / regressed_trials) * 100.0, 2),
        "mean_sprt_samples": round(mean_sprt_samples, 2),
        "mean_clean_samples": round(mean_clean_samples, 2),
        "mean_regressed_samples": round(mean_regressed_samples, 2),
        "fixed_horizon_z_samples": fixed_n,
        "fixed_z_critical_failures": k_crit,
        "fixed_z_false_halt_rate_pct": round((fixed_false_halts / clean_trials) * 100.0, 2),
        "fixed_z_false_promotion_rate_pct": round((fixed_false_promotions / regressed_trials) * 100.0, 2),
        "overall_sample_reduction_pct": round(sample_reduction_pct, 2),
        "regressed_halt_sample_reduction_pct": round(regressed_sample_reduction_pct, 2),
        "benchmark_duration_ms": round(elapsed_ms, 2),
    }
