"""Independent validation and path metrics for pilot experiments."""

from __future__ import annotations

import hashlib
import json
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Set, Tuple

from planner import Cell, neighbors4


def _cell(value: Iterable[int]) -> Cell:
    row, col = value
    return int(row), int(col)


def cut_perimeter(cut: Set[Cell]) -> int:
    return sum(1 for cell in cut for nxt in neighbors4(cell) if nxt not in cut)


def event_digest(events: List[Mapping[str, object]]) -> str:
    payload = json.dumps(events, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_and_measure(result: Mapping[str, object]) -> Dict[str, object]:
    """Validate planner events without using its internal state and compute metrics."""
    components = {int(item["id"]): item for item in result["components"]}  # type: ignore[index]
    domain = {_cell(value) for value in result["domain_cells"]}  # type: ignore[index]
    events = result["events"]  # type: ignore[index]
    cut: Set[Cell] = set()
    released: Set[int] = set()
    errors: List[str] = []
    compactness: List[float] = []
    transfer_count = 0

    for position, event in enumerate(events):
        action = event["action"]
        if action == "cut":
            cell = _cell(event["cell"])
            if cell not in domain:
                errors.append(f"event {position}: cut outside domain")
            elif cell in cut:
                errors.append(f"event {position}: duplicate cut")
            elif cut and not any(nxt in cut for nxt in neighbors4(cell)):
                errors.append(f"event {position}: cut is not adjacent to prior cut")
            cut.add(cell)
            area = len(cut)
            compactness.append((cut_perimeter(cut) ** 2) / area)
        elif action == "transfer":
            transfer_count += 1
            cell = _cell(event["cell"])
            if cell not in cut:
                errors.append(f"event {position}: transfer outside cut region")
        elif action == "release":
            component_id = int(event["component_id"])
            component = components.get(component_id)
            if component is None:
                errors.append(f"event {position}: unknown released component")
            elif component_id in released:
                errors.append(f"event {position}: duplicate release")
            else:
                ring = {_cell(cell) for cell in component["ring"]}  # type: ignore[index]
                if not ring <= cut:
                    errors.append(f"event {position}: release before complete ring")
                released.add(component_id)
        else:
            errors.append(f"event {position}: unknown action {action}")

    cut_count = len(cut)
    coverage = len(cut) / len(domain) if domain else 0.0
    expected_coverage = float(result["coverage"])
    if abs(coverage - expected_coverage) > 1e-8:
        errors.append("reported coverage differs from event reconstruction")
    if int(result["cut_count"]) != cut_count:
        errors.append("reported cut count differs from event reconstruction")
    return {
        "hard_valid": not errors and result["status"] == "ok" and coverage == 1.0,
        "event_valid": not errors,
        "validation_errors": errors,
        "coverage": coverage,
        "cut_count": cut_count,
        "transfer_count": transfer_count,
        "transfer_overhead": transfer_count / cut_count if cut_count else float("inf"),
        "mean_compactness": mean(compactness) if compactness else float("inf"),
        "max_compactness": max(compactness) if compactness else float("inf"),
        "release_count": len(released),
        "event_digest": event_digest(events),
    }


def normalized_objective(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> float | None:
    """Equal-weight external objective; ``None`` denotes a hard-constraint failure."""
    if not candidate["hard_valid"] or not baseline["hard_valid"]:
        return None

    def ratio(value: float, reference: float) -> float:
        if reference == 0:
            return 1.0 if value == 0 else float("inf")
        return value / reference

    transfer = ratio(float(candidate["transfer_overhead"]), float(baseline["transfer_overhead"]))
    compactness = ratio(float(candidate["mean_compactness"]), float(baseline["mean_compactness"]))
    return 0.5 * transfer + 0.5 * compactness
