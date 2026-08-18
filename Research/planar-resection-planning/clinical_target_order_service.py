"""Lazy web-inference service for the frozen v10.4 target-order BC model.

The service deliberately reproduces the Gate B rollout configuration: macro
targets, deterministic shortest-path transfer, automatic fixed 15/5 clamp
cycles, candidate_count=6, and CPU inference.  The checkpoint is exposed for
research simulation even though its frozen Gate B decision is NO-GO.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from clinical_target_order_policy import TargetOrderScorer, make_selector
from clinical_window_environment import _cell_list
from clinical_window_evaluation import rollout_clinical_policy
from planner import boundary_cells, is_connected


CURRENT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CURRENT_DIR / "results" / "clinical_window_v10_4_target_order"
CHECKPOINT = MODEL_ROOT / "runs" / "target_order_bc.pt"
FEATURE_SCALES = MODEL_ROOT / "teacher" / "feature_scales.json"
EVALUATION = MODEL_ROOT / "runs" / "gate_b_evaluation.json"
POLICY_ID = "clinical_v104_target_order_bc"
CANDIDATE_COUNT = 6
MAX_ROWS = 30
MAX_COLS = 40

GATE_CLINICAL_CONFIG: Mapping[str, Any] = {
    "early_end_mode": "disabled",
    "early_end_minutes": 0.0,
    "bleeding_probability": 1.0,
    "max_steps_multiplier": 8.0,
}


class ClinicalTargetOrderService:
    """Load the frozen scorer once and run deterministic, CPU-only rollouts."""

    def __init__(self) -> None:
        self._model: Optional[TargetOrderScorer] = None
        self._selector = None
        self._checkpoint_sha256: Optional[str] = None
        self._load_lock = threading.Lock()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _gate_decision() -> str:
        if not EVALUATION.is_file():
            return "unknown"
        payload = json.loads(EVALUATION.read_text(encoding="utf-8"))
        return str(payload.get("go_no_go", {}).get("decision", "unknown"))

    def status(self) -> Dict[str, object]:
        available = CHECKPOINT.is_file() and FEATURE_SCALES.is_file()
        return {
            "policy_id": POLICY_ID,
            "label": "v10.4 target-order behavior-cloning policy",
            "available": available,
            "loaded": self._selector is not None,
            "checkpoint": str(CHECKPOINT.relative_to(CURRENT_DIR)),
            "checkpoint_sha256": self._checkpoint_sha256,
            "gate_b_decision": self._gate_decision(),
            "scope": "connected planar domains up to 30x40 with interior vessel cells",
            "control_mode": "macro target with deterministic transfer",
            "clamp_schedule": "automatic fixed 15 min clamped / 5 min unclamped",
            "candidate_count": CANDIDATE_COUNT,
            "quality_note": (
                "Research simulation only. This checkpoint received Gate B NO-GO because "
                "of rare but severe simulated blood-loss failures."
            ),
        }

    def load(self) -> Dict[str, object]:
        if not CHECKPOINT.is_file():
            raise ValueError(f"v10.4 checkpoint is missing: {CHECKPOINT}")
        if not FEATURE_SCALES.is_file():
            raise ValueError(f"v10.4 feature scales are missing: {FEATURE_SCALES}")
        with self._load_lock:
            if self._selector is None:
                import torch

                scales = json.loads(FEATURE_SCALES.read_text(encoding="utf-8"))
                model = TargetOrderScorer()
                try:
                    state_dict = torch.load(
                        CHECKPOINT, map_location="cpu", weights_only=True
                    )
                except TypeError:  # Compatibility with older PyTorch runtimes.
                    state_dict = torch.load(CHECKPOINT, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()
                self._model = model
                self._selector = make_selector(
                    model, scales, candidate_count=CANDIDATE_COUNT
                )
                self._checkpoint_sha256 = self._sha256(CHECKPOINT)
        return self.status()

    @staticmethod
    def _validate(payload: Mapping[str, Any]) -> Dict[str, object]:
        rows, cols = int(payload["rows"]), int(payload["cols"])
        if not 1 <= rows <= MAX_ROWS or not 1 <= cols <= MAX_COLS:
            raise ValueError(
                f"The v10.4 model accepts at most {MAX_ROWS} rows and {MAX_COLS} columns"
            )
        domain = {tuple(map(int, cell)) for cell in payload["domain_cells"]}
        vessels = {
            tuple(map(int, cell)) for cell in payload.get("obstacle_cells", ())
        }
        start = tuple(map(int, payload["start_cell"]))
        canvas = {(row, col) for row in range(rows) for col in range(cols)}
        if not domain or not domain <= canvas or not is_connected(domain):
            raise ValueError(
                "domain_cells must be a non-empty connected region inside the declared canvas"
            )
        if not vessels <= domain:
            raise ValueError("All vessel cells must lie inside domain_cells")
        boundary = boundary_cells(domain)
        if vessels & boundary:
            raise ValueError("Vessel cells must be interior cells of the domain")
        if start not in boundary or start in vessels:
            raise ValueError("start_cell must be a non-vessel domain boundary cell")
        return {
            "scenario_id": payload.get("scenario_id", "web-user-scenario"),
            "rows": rows,
            "cols": cols,
            "domain_cells": _cell_list(domain),
            "obstacle_cells": _cell_list(vessels),
            "start_cell": list(start),
            "cell_size_mm": 4.0,
        }

    def plan(self, payload: Mapping[str, Any]) -> Dict[str, object]:
        scenario = self._validate(payload)
        status = self.load()
        assert self._selector is not None
        record = rollout_clinical_policy(
            scenario,
            self._selector,
            clinical_config=GATE_CLINICAL_CONFIG,
            include_replay=True,
            include_step_trace=True,
            mechanics_update_interval=0,
            control_mode="macro",
        )
        replay = record.pop("replay")
        events = list(replay["events"])
        domain = {tuple(cell) for cell in scenario["domain_cells"]}
        cut = {tuple(cell) for cell in record["cut_cells"]}
        return {
            "status": record["status"],
            "policy": status["label"],
            "policy_id": POLICY_ID,
            "gate_b_decision": status["gate_b_decision"],
            "quality_note": status["quality_note"],
            "rows": scenario["rows"],
            "cols": scenario["cols"],
            "domain_cells": scenario["domain_cells"],
            "boundary_cells": _cell_list(boundary_cells(domain)),
            "obstacle_cells": scenario["obstacle_cells"],
            "start_cell": scenario["start_cell"],
            "components": record["components"],
            "events": events,
            "event_count": len(events),
            "cut_count": len(cut),
            "transfer_count": record["transfer_count"],
            "release_count": sum(
                event.get("action") == "expose_vessel" for event in events
            ),
            "sealed_vessel_count": record["sealed_vessel_count"],
            "coverage": record["coverage"],
            "uncovered_cells": _cell_list(domain - cut),
            "active_component_ids": record["hidden_component_ids"],
            "released_component_ids": record["sealed_component_ids"],
            "failure_reason": record["failure_reason"],
            "elapsed_minutes": record["elapsed_minutes"],
            "expected_blood_loss_ml": record["expected_blood_loss_ml"],
            "peak_expected_bleeding_rate_ml_per_min": record[
                "peak_expected_bleeding_rate_ml_per_min"
            ],
            "total_clamped_minutes": record["total_clamped_minutes"],
            "total_unclamped_minutes": record["total_unclamped_minutes"],
            "clamp_cycle_count": record["clamp_cycle_count"],
            "total_reward": record["total_reward"],
            "reward_components": record["reward_components"],
            "reward_trace": record["reward_trace"],
            "reward_trace_kind": "single_scenario_rollout",
            "reward_trace_note": (
                "Macro-step environment rewards recomputed during this simulation; "
                "this is not a historical BC training-reward curve."
            ),
        }


clinical_target_order_service = ClinicalTargetOrderService()
