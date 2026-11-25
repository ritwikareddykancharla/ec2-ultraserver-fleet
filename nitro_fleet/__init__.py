"""NitroUltraFleet control-plane package for staged canary rollouts, host repair, and sub-ring scheduling."""

from .sprt_canary import ROLLOUT_WAVES, SPRTCanaryEngine, compute_fixed_horizon_samples
from .repair_state_machine import (
    BlastRadiusBudgetExceededError,
    CASVersionConflictError,
    FleetRepairStateMachine,
    InvalidStateTransitionError,
)
from .topology_scheduler import (
    analyze_collective_trace,
    schedule_workload_queue,
)
from .rma_copilot import RMACopilot

__all__ = [
    "ROLLOUT_WAVES",
    "SPRTCanaryEngine",
    "compute_fixed_horizon_samples",
    "FleetRepairStateMachine",
    "CASVersionConflictError",
    "BlastRadiusBudgetExceededError",
    "InvalidStateTransitionError",
    "analyze_collective_trace",
    "schedule_workload_queue",
    "RMACopilot",
]
