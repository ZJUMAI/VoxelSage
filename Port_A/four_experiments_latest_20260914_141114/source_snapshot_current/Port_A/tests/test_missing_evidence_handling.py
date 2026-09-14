import json
import re

from core.reflection_module import ReflectionEngine
from core.server import _has_successful_skill_evidence, _unavailable_answer_for_contract


def test_case_not_found_is_non_retryable_and_has_no_alternative():
    analysis = ReflectionEngine.analyze_failure(
        "liver_analysis",
        "PORT_B_HTTP_404",
        "Case 'missing' not found",
        {"target_case_id": "missing"},
    )

    assert analysis["category"] == "missing_case"
    assert analysis["retryable"] is False
    strategy = ReflectionEngine.suggest_recovery_strategy({}, "liver_analysis", analysis)
    assert strategy["action"] == "skip"
    assert strategy["alternative_skills"] == []


def test_unavailable_contract_shape_used_by_server():
    query = 'End with one JSON object: {"value": null, "unit": "count"}.'
    match = re.search(r'["\']unit["\']\s*:\s*["\']([^"\']+)["\']', query)
    payload = json.dumps({"value": None, "unit": match.group(1)}, ensure_ascii=False)

    assert json.loads(payload) == {"value": None, "unit": "count"}


def test_ungrounded_measurement_is_replaced_with_null():
    session = {"tool_store": {"missing": {"segmentation": {"status": "reference_masks"}}}}
    assert _has_successful_skill_evidence(session) is False
    answer = _unavailable_answer_for_contract(
        'End with {"value": 0, "unit": "cm"}.',
        "No tool evidence",
    )
    assert json.loads(answer.splitlines()[-1]) == {"value": None, "unit": "cm"}
