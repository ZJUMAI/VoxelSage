"""Evaluate Port A + live Port B against independent Maximum3DDiameter references."""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.environ.get("DIAMETER_EVAL_ROOT", "E:/DeepTumour_agent_eval/diameter_validation"))
DATA_ROOT = Path("E:/DeepTumour_agent_eval/live_portb_formal_v1")
load_dotenv("//wsl.localhost/Ubuntu-22.04/home/voxelsage/VoxelSage/.env")
os.environ["PORT_B_INTERNAL"] = "http://127.0.0.1:8875"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = os.environ["NO_PROXY"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core.server as server


class Sink:
    async def send_json(self, payload):
        pass


def answer_value(text):
    matches = re.findall(r'\{[^{}]*"value"\s*:[^{}]*\}', text or "")
    if not matches:
        return None
    try:
        return json.loads(matches[-1]).get("value")
    except ValueError:
        return None


async def main():
    assert not server.llm_configuration_errors()
    skills = await server.port_b_list_skills(force_refresh=True)
    cases = {x["case"]: x for x in json.loads((DATA_ROOT / "live_cases.json").read_text(encoding="utf-8"))}
    refs = json.loads((ROOT / "pyradiomics_bodymaps_reference.json").read_text(encoding="utf-8"))
    sem = asyncio.Semaphore(2)
    rows = []

    async def run(case_id, expected_mm):
        async with sem:
            source_id = case_id.removeprefix("LIVE_")
            port_b_case = cases[source_id]["port_b_case"]
            session = server.create_empty_session("diameter_standard_" + source_id)
            session.update({
                "case_ids": [port_b_case],
                "_active_case_ids": [port_b_case],
                "_active_volumes": [port_b_case + ".nii.gz"],
                "available_skills": skills["skills"],
                "available_tools": server.inject_target_case_id(skills["tools"], [port_b_case]),
                "max_new_tokens": 400,
                "segmentation_status": "completed",
                "original_question": "What is the maximum 3D diameter of the largest liver lesion in cm?",
            })
            session["tool_store"][port_b_case] = {
                "segmentation": {"status": "reference_masks", "case_id": port_b_case}
            }
            session["current_user_query"] = (
                "What is the maximum 3D diameter of the largest liver lesion in cm? "
                "Use the available case and a measurement skill. End with exactly one JSON object: "
                '{"value": <number>, "unit": "cm"}.'
            )
            started = time.monotonic()
            try:
                answer = await asyncio.wait_for(server.run_agent_loop(Sink(), session, max_rounds=4), 180)
                actual_cm = answer_value(answer)
                expected_cm = expected_mm / 10.0
                error_mm = abs(actual_cm * 10.0 - expected_mm) if isinstance(actual_cm, (int, float)) else None
                row = {
                    "case": source_id,
                    "expected_pyradiomics_mm": expected_mm,
                    "actual_agent_cm": actual_cm,
                    "absolute_error_mm": error_mm,
                    "correct_within_0_01_mm": error_mm is not None and error_mm <= 0.01,
                    "answer": answer,
                    "calls": list(session["skill_call_history"]),
                }
            except Exception as exc:
                row = {"case": source_id, "error": type(exc).__name__, "calls": list(session["skill_call_history"])}
            row["seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            (ROOT / "agent_diameter_results.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("PROGRESS", len(rows), source_id, row.get("correct_within_0_01_mm"), flush=True)

    await asyncio.gather(*(run(case_id, value) for case_id, value in refs.items()))
    valid = [x for x in rows if x.get("absolute_error_mm") is not None]
    summary = {
        "n": len(rows),
        "numeric_coverage": len(valid),
        "within_0_01_mm": sum(x["correct_within_0_01_mm"] for x in rows),
        "exceptions": sum("error" in x for x in rows),
        "mean_absolute_error_mm": sum(x["absolute_error_mm"] for x in valid) / len(valid) if valid else None,
        "max_absolute_error_mm": max((x["absolute_error_mm"] for x in valid), default=None),
        "single_tool_call": sum(len(x.get("calls", [])) == 1 for x in rows),
    }
    (ROOT / "agent_diameter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
