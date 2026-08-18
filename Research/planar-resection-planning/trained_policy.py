"""Safe simulator-side loading and execution of a trained Maskable PPO policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from environment import (
    PlanarResectionEnv,
    _cell_list,
    local_grid_action_masks,
    local_grid_observation,
    local_to_canvas_action,
)
from planner import boundary_cells, is_connected

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"

POLICY_SPECS = {
    "toy5_plain": {
        "path": RESULTS_DIR / "ppo_toy5_plain_seed2026072901_100k" / "final_model.zip",
        "rows": 5,
        "cols": 5,
        "obstacles": set(),
        "label": "masked-ppo-toy5-plain",
        "scope": "5x5 complete square domain, no vessel cells, boundary start",
        "quality_note": "First-round control policy; use only within this toy scope.",
    },
    "toy7_vessel_diagnostic": {
        "path": RESULTS_DIR / "ppo_toy7_vessel_seed2026072901_100k" / "final_model.zip",
        "rows": 7,
        "cols": 7,
        "obstacles": {(3, 3)},
        "label": "masked-ppo-toy7-vessel-diagnostic",
        "scope": "7x7 complete square domain, one central releasable vessel, boundary start",
        "quality_note": "Diagnostic model: completes its toy task but did not converge on transfer efficiency; not a recommended planner.",
        "representation": "fixed_canvas",
    },
    "toy7_vessel_spatial_v2": {
        "path": RESULTS_DIR / "ppo_spatial_v2_stable_toy7_vessel_seed2026072901_100k"
        / "checkpoints" / "ppo_spatial_49992_steps.zip",
        "rows": 7,
        "cols": 7,
        "obstacles": {(3, 3)},
        "label": "masked-ppo-toy7-vessel-spatial-v2",
        "scope": "7x7 complete square domain, one central releasable vessel, boundary start",
        "quality_note": "Selected 50k spatial-v2 checkpoint; validated transfer overhead 0.2015 on the frozen toy validation set.",
        "representation": "local_grid",
    },
    "toy7_vessel_spatial_v3": {
        "path": RESULTS_DIR / "ppo_spatial_v3_mixed_random_teacher_seed2026072901_100k_retry2" / "final_model.zip",
        "rows": 7,
        "cols": 7,
        "obstacles": {(3, 3)},
        "label": "masked-ppo-toy7-vessel-spatial-v3",
        "scope": "7x7 connected domain, arbitrary interior vessel layout, boundary start",
        "quality_note": "Mixed-teacher/random-layout model; passed 120 Test + 80 Stress frozen generalization evaluation with 0.3005 Test transfer overhead. Custom vessel layouts are supported within the 7x7 scope but remain research-only.",
        "representation": "local_grid",
    },
    "variable_c_stage_c": {
        "path": RESULTS_DIR / "variable_spatial_stage_c_from_b_seed2026073301_75k" / "final_model.zip",
        "label": "variable-ppo-stage-c",
        "scope": "variable-size curriculum stage C (17-24 x 17-32) on padded 30x40 canvas, 1200 actions",
        "quality_note": "Stage-C curriculum PPO (seed 2026073301, 75,776 steps). 100% completion/legal on A/B/C frozen sets; transfer within 3-4% of the S-form baseline; avoids planner-scale degradation. Research-only.",
        "representation": "variable_padded",
    },
}
DEFAULT_POLICY_ID = "variable_c_stage_c"
MAX_ML_ROWS = 30
MAX_ML_COLS = 40


class TrainedPolicyService:
    """Lazy, CPU-only model service so simulator testing cannot occupy training GPU."""

    def __init__(self) -> None:
        self._model = None
        self._loaded_path: Optional[Path] = None
        self._loaded_policy_id: Optional[str] = None

    @staticmethod
    def _policy_spec(policy_id: str) -> Mapping[str, Any]:
        try:
            return POLICY_SPECS[policy_id]
        except KeyError as exc:
            raise ValueError(f"Unknown policy_id: {policy_id}") from exc

    @staticmethod
    def _resolve_model(path: Optional[str] = None, *, policy_id: str = DEFAULT_POLICY_ID) -> Path:
        candidate = Path(path).expanduser() if path else Path(TrainedPolicyService._policy_spec(policy_id)["path"])
        if candidate.suffix != ".zip":
            candidate = candidate.with_suffix(".zip")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(RESULTS_DIR.resolve())
        except ValueError as exc:
            raise ValueError("Model path must be located inside planar_simulator/results") from exc
        return resolved

    def status(self, policy_id: str = DEFAULT_POLICY_ID) -> Dict[str, object]:
        spec = self._policy_spec(policy_id)
        path = self._resolve_model(policy_id=policy_id)
        return {
            "policy_id": policy_id,
            "model_path": str(path.relative_to(CURRENT_DIR)),
            "available": path.is_file(),
            "loaded": self._model is not None and self._loaded_path == path and self._loaded_policy_id == policy_id,
            "scope": spec["scope"],
            "quality_note": spec["quality_note"],
            "available_policies": [
                {"policy_id": key, "available": Path(value["path"]).is_file(), "scope": value["scope"],
                 "quality_note": value["quality_note"]}
                for key, value in POLICY_SPECS.items()
            ],
        }

    def load(self, path: Optional[str] = None, *, policy_id: str = DEFAULT_POLICY_ID) -> Dict[str, object]:
        resolved = self._resolve_model(path, policy_id=policy_id)
        if not resolved.is_file():
            raise ValueError(f"Trained model is not available yet: {resolved.relative_to(CURRENT_DIR)}")
        if self._model is None or self._loaded_path != resolved:
            try:
                from sb3_contrib import MaskablePPO
                import torch
            except ImportError as exc:
                raise RuntimeError("sb3-contrib is required to load the trained policy") from exc
            # On CPU, PyTorch may reject an otherwise valid masked categorical
            # distribution solely because its float32 probabilities sum to
            # 1±tiny-rounding-error.  Masks are validated independently by the
            # environment; disabling this redundant strict simplex assertion
            # matches SB3's normal inference semantics on the training GPU.
            torch.distributions.Distribution.set_default_validate_args(False)
            # Variable-size models register a custom policy class; importing the
            # module is what makes MaskablePPO.load() resolve it.
            try:
                import variable_policy  # noqa: F401
            except ImportError:
                pass
            self._model = MaskablePPO.load(str(resolved), device="cpu")
            self._loaded_path = resolved
            self._loaded_policy_id = policy_id
        return self.status(policy_id)

    @staticmethod
    def _validate_scope(
        payload: Mapping[str, Any], policy_id: str = DEFAULT_POLICY_ID,
    ) -> None:
        spec = TrainedPolicyService._policy_spec(policy_id)
        rows, cols = int(payload["rows"]), int(payload["cols"])
        domain = {tuple(cell) for cell in payload["domain_cells"]}
        obstacles = {tuple(cell) for cell in payload.get("obstacle_cells", ())}
        start = tuple(payload["start_cell"])
        if spec.get("representation") == "variable_padded":
            if not 1 <= rows <= MAX_ML_ROWS or not 1 <= cols <= MAX_ML_COLS:
                raise ValueError(
                    f"The current ML policy accepts at most {MAX_ML_ROWS} rows and {MAX_ML_COLS} columns"
                )
            if any(
                not 0 <= row < rows or not 0 <= col < cols
                for row, col in domain | obstacles | {start}
            ):
                raise ValueError("ML scenario cells must lie inside the declared canvas")
            # Variable-size policies accept connected domains inside the padded
            # 30x40 canvas; only the common legality checks below apply.
            if not domain or not is_connected(domain):
                raise ValueError("Variable-padded domains must be non-empty and connected")
            boundary = boundary_cells(domain)
            if not obstacles <= domain or any(cell in boundary for cell in obstacles):
                raise ValueError("Variable-padded vessel cells must be interior cells of the domain")
        else:
            expected = {(row, col) for row in range(int(spec["rows"])) for col in range(int(spec["cols"]))}
            is_v3_custom = policy_id == "toy7_vessel_spatial_v3"
            if (rows, cols) != (int(spec["rows"]), int(spec["cols"])):
                raise ValueError(
                    f"This policy is validated only for {spec['scope']}. Use the matching simulator ML test preset."
                )
            if is_v3_custom:
                if not domain or not domain <= expected or not is_connected(domain):
                    raise ValueError("Custom v3 domains must be non-empty connected subsets of the 7x7 grid")
                boundary = boundary_cells(domain)
                if not obstacles <= domain or any(cell in boundary for cell in obstacles):
                    raise ValueError("Custom v3 vessel cells must be interior cells of the 7x7 domain")
            elif obstacles != spec["obstacles"]:
                raise ValueError(
                    f"This policy is validated only for {spec['scope']}. Use the matching simulator ML test preset."
                )
        if start not in boundary_cells(domain):
            raise ValueError("The policy test start must lie on the trained-grid boundary")
        if start in obstacles:
            raise ValueError("The policy test start cannot overlap a vessel cell")

    def plan(self, payload: Mapping[str, Any]) -> Dict[str, object]:
        policy_id = DEFAULT_POLICY_ID
        self._validate_scope(payload, policy_id)
        self.load(policy_id=policy_id)
        assert self._model is not None
        spec = self._policy_spec(policy_id)
        scenario = {
            "rows": int(payload["rows"]), "cols": int(payload["cols"]),
            "domain_cells": payload["domain_cells"], "obstacle_cells": payload.get("obstacle_cells", ()),
            "start_cell": payload["start_cell"],
        }
        representation = spec.get("representation")
        if representation == "variable_padded":
            from environment import VariableGridScenarioPoolEnv
            env = VariableGridScenarioPoolEnv([scenario], seed=0)
            state_env = env.base_env
        else:
            env = PlanarResectionEnv(scenario=scenario)
            state_env = env
        observation, _ = env.reset()
        while not state_env.terminated and not state_env.truncated:
            if representation == "local_grid":
                observation = local_grid_observation(env, int(spec["rows"]))
                action_masks = local_grid_action_masks(env, int(spec["rows"]))
            else:
                action_masks = env.action_masks()
            action, _ = self._model.predict(
                observation, deterministic=True, action_masks=action_masks,
            )
            canvas_action = (
                local_to_canvas_action(int(action), int(spec["rows"]))
                if representation == "local_grid" else int(action)
            )
            observation, _, _, _, _ = env.step(canvas_action)
        return {
            "status": "ok" if state_env.cut == state_env.domain and state_env.failure_reason is None else "partial",
            "policy": spec["label"], "policy_id": policy_id, "quality_note": spec["quality_note"],
            "rows": state_env.rows, "cols": state_env.cols,
            "domain_cells": _cell_list(state_env.domain), "boundary_cells": _cell_list(boundary_cells(state_env.domain)),
            "obstacle_cells": _cell_list(state_env.obstacles), "start_cell": list(state_env.start), "components": state_env.components,
            "events": state_env.events, "event_count": len(state_env.events), "cut_count": len(state_env.cut),
            "transfer_count": sum(item["action"] == "transfer" for item in state_env.events),
            "release_count": sum(item["action"] == "release" for item in state_env.events), "coverage": round(len(state_env.cut) / len(state_env.domain), 8),
            "uncovered_cells": _cell_list(state_env.domain - state_env.cut), "active_component_ids": [],
            "released_component_ids": [], "failure_reason": state_env.failure_reason,
        }


trained_policy_service = TrainedPolicyService()
