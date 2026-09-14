from skills.builtin.plan_resection.reward_function.candidate_reward import (
    select_candidate_index,
)


def _score(
    score,
    family,
    ratio,
    *,
    recall=1.0,
    vessel_ratio=0.0,
    anatomy=0.0,
    detachable=0.8,
):
    return {
        "score": float(score),
        "candidate_family": family,
        "pred_ratio_sample": float(ratio),
        "tumor_recall_sample": float(recall),
        "vessel_ratio_sample": float(vessel_ratio),
        "anatomy_support": float(anatomy),
        "detachable_support": float(detachable),
    }


def test_local_protect_uses_maximum_base_score_inside_local_cap():
    scored = [
        _score(9.0, "plane", 0.12),
        _score(2.0, "local", 0.10),
        _score(3.0, "corridor", 0.14),
        _score(8.0, "local", 0.20),
    ]
    context = {
        "centrality": 0.20,
        "tumor_liver_ratio": 0.003,
        "n_tumor_components": 1,
        "vessel_signal": 0.0,
        "tumor_total_volume_mm3": 2000.0,
        "tumor_max_volume_mm3": 2000.0,
    }

    index, details = select_candidate_index(scored, "major", context)

    assert index == 2
    assert details["surgical_mode"] == "local_protect"
    assert details["selection_policy"] == "mode_eligible_max_base_score_v1"
    assert details["selection_fallback"] is False
    assert details["mode_eligible_count"] == 2


def test_high_burden_mode_applies_extent_and_detachability_before_base_score():
    scored = [
        _score(10.0, "plane", 0.04, detachable=0.9),
        _score(4.0, "plane", 0.12, detachable=0.7),
        _score(5.0, "portal_vessel", 0.16, detachable=0.7),
        _score(9.0, "plane", 0.20, detachable=0.2),
    ]
    context = {
        "centrality": 0.30,
        "tumor_liver_ratio": 0.020,
        "n_tumor_components": 2,
        "vessel_signal": 0.0,
        "extent_signal": 0.2,
        "burden_signal": 0.5,
        "tumor_total_volume_mm3": 30000.0,
        "tumor_max_volume_mm3": 16000.0,
    }

    index, details = select_candidate_index(scored, "segmental", context)

    assert index == 2
    assert details["surgical_mode"] == "high_burden_anatomic"
    assert details["selection_fallback"] is False
    assert scored[index]["pred_ratio_sample"] >= details["anatomic_mode_ratio_floor"]


def test_central_anatomic_mode_rejects_insufficient_extent():
    scored = [
        _score(9.0, "plane", 0.20, detachable=0.9),
        _score(4.0, "plane", 0.34, detachable=0.9),
        _score(5.0, "hepatic_vessel", 0.38, detachable=0.9),
    ]
    context = {
        "centrality": 0.70,
        "tumor_liver_ratio": 0.010,
        "n_tumor_components": 3,
        "tumor_spread_mm": 120.0,
        "vessel_signal": 0.20,
        "extent_signal": 0.3,
        "burden_signal": 0.2,
    }

    index, details = select_candidate_index(scored, "intermediate_local", context)

    assert index == 2
    assert details["surgical_mode"] == "central_anatomic"
    assert details["anatomic_large_pressure"] == 1.0


def test_regional_segmental_mode_uses_maximum_base_score_above_floor():
    scored = [
        _score(8.0, "plane", 0.04),
        _score(3.0, "liver_pca", 0.18),
        _score(4.0, "lobar_axis", 0.22),
    ]
    context = {
        "centrality": 0.30,
        "tumor_liver_ratio": 0.008,
        "n_tumor_components": 1,
        "vessel_signal": 0.0,
        "extent_signal": 0.2,
        "burden_signal": 0.0,
    }

    index, details = select_candidate_index(scored, "segmental", context)

    assert index == 2
    assert details["surgical_mode"] == "regional_segmental"
    assert details["selection_fallback"] is False


def test_compact_mode_uses_base_score_inside_non_extreme_pareto_front():
    scored = [
        _score(1.0, "local", 0.10, recall=0.994, anatomy=0.0),
        _score(2.0, "corridor", 0.05, recall=0.990, anatomy=0.2),
        _score(9.0, "plane", 0.30, recall=0.994, anatomy=0.3),
    ]
    context = {
        "centrality": 0.20,
        "tumor_liver_ratio": 0.003,
        "n_tumor_components": 1,
        "vessel_signal": 0.0,
    }

    index, details = select_candidate_index(scored, "local", context)

    assert index == 1
    assert details["surgical_mode"] == "compact"
    assert details["pareto_front_count"] == 2
    assert details["selected_base_score"] == 2.0


def test_empty_mode_set_falls_back_only_to_coverage_eligible_candidates():
    scored = [
        _score(5.0, "plane", 0.10, recall=1.0),
        _score(20.0, "local", 0.10, recall=0.80),
    ]
    context = {
        "centrality": 0.20,
        "tumor_liver_ratio": 0.003,
        "n_tumor_components": 1,
        "vessel_signal": 0.0,
        "tumor_total_volume_mm3": 2000.0,
        "tumor_max_volume_mm3": 2000.0,
    }

    index, details = select_candidate_index(scored, "major", context)

    assert index == 0
    assert details["selection_policy"] == "mode_fallback_max_base_score_v1"
    assert details["selection_fallback"] is True
    assert details["requires_user_review"] is True
    assert details["selected_pool_count"] == 1


def test_extent_floor_relaxation_does_not_relax_coverage_gate():
    scored = [
        _score(2.0, "local", 0.001, recall=1.0),
        _score(100.0, "local", 0.20, recall=0.80),
    ]

    index, details = select_candidate_index(scored, "local", {})

    assert index == 0
    assert details["eligibility_floor_fallback"] is True
    assert details["requires_user_review"] is True
