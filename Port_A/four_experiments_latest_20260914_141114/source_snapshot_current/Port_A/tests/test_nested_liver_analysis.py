import asyncio

import core.server as server
from core.tool_optimizer import ToolOptimizer
from core.medical_knowledge_base import MedicalKnowledgeBase
from core.server import format_tool_context


def nested_result():
    return {
        "liver_volume_cm3": 1772.49,
        "tumor_results": {
            "tumor_1": {
                "diameter": {"max_diameter_mm": 51.69, "method": "convex_hull"},
                "volume_cm3": 22.91,
            }
        },
    }


def test_cached_first_case_does_not_hide_tools_needed_by_second_case():
    tools = [{"function": {"name": "liver_analysis"}}]
    session = {"_active_case_ids": ["a", "b"], "tool_store": {"a": {"liver_analysis": nested_result()}}}
    available, _ = ToolOptimizer.filter_redundant_tools(session, tools)
    assert available == tools
    session["tool_store"]["b"] = {"liver_analysis": nested_result()}
    available, _ = ToolOptimizer.filter_redundant_tools(session, tools)
    assert available == []


def test_partial_parent_keeps_missing_measurement_tools():
    names = ["tumor_diameter", "vessel_volume", "tumor_vessel_distance"]
    tools = [{"function": {"name": name}} for name in names]
    session = {"tool_store": {"a": {"liver_analysis": {"liver_volume_cm3": 1400}}}}
    assert ToolOptimizer.filter_redundant_tools(session, tools)[0] == tools
    session["tool_store"]["a"]["liver_analysis"] = nested_result()
    available, _ = ToolOptimizer.filter_redundant_tools(session, tools)
    assert [t["function"]["name"] for t in available] == names[1:]


def test_failed_or_empty_result_does_not_hide_retry_tool():
    tools = [{"function": {"name": "liver_analysis"}}]
    for result in [{}, {"_error": "timeout"}, {"status": "error"}]:
        session = {"tool_store": {"a": {"liver_analysis": result}}}
        assert ToolOptimizer.filter_redundant_tools(session, tools)[0] == tools


def test_oracle_aggregate_does_not_fabricate_lesion_identity():
    result = {"tumor_count": 80, "tumor_results": {}, "largest_lesion_diameter_mm": 15.5}
    context = format_tool_context({"tool_store": {"a": {"liver_analysis": result}}, "_active_case_ids": ["a"]})
    assert "15.5 mm" in context
    assert "80" in context
    assert "tumor_1" not in context
    assert not ToolOptimizer.parent_has_measurement(result, "tumor_diameter")


def test_missing_tumor_measurements_are_not_zero_or_a_crash():
    result = {"liver_volume_cm3": 1400, "tumor_results": None}
    context = format_tool_context({"tool_store": {"a": {"liver_analysis": result}}, "_active_case_ids": ["a"]})
    assert "1400" in context
    assert "肿瘤数量: 0" not in context


def test_nested_diameter_is_exposed_to_model_context():
    session = {
        "tool_store": {"case_1": {"liver_analysis": nested_result()}},
        "_active_case_ids": ["case_1"],
    }
    context = format_tool_context(session)
    assert "51.7mm" in context
    assert "22.91cm³" in context


def test_nested_diameter_passes_consistency_and_summary_extraction():
    result = nested_result()
    consistency = MedicalKnowledgeBase.validate_result_consistency(result)
    assert not any("max_diameter_mm" in issue["message"] for issue in consistency["issues"])
    summary = MedicalKnowledgeBase.extract_analysis_summary(
        {"case_1": {"liver_analysis": result}}, "case_1"
    )
    assert summary["max_tumor_diameter"] == 51.69


def test_execute_tool_calls_does_not_duplicate_a_single_call(monkeypatch):
    seen = []

    async def fake_execute(websocket, session, round_index, call, semaphore):
        seen.append(call["id"])
        return call, {"status": "ok", "result": {}}

    monkeypatch.setattr(server, "execute_one_tool_call", fake_execute)
    session = {"skill_call_history": [], "_active_case_ids": ["case_1"]}
    call = {
        "id": "call_1",
        "function": {
            "name": "liver_analysis",
            "arguments": '{"target_case_id":"case_1"}',
        },
    }
    results = asyncio.run(server.execute_tool_calls(None, session, 1, [call]))
    assert len(results) == 1
    assert seen == ["call_1"]
