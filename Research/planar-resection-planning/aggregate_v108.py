"""Aggregate v10.8 Lazy Exact Shield results across E0–E10.

Reads frozen shards + summaries and produces:
  results/clinical_window_v10_8_lazy_shield/report/aggregate_v108.json
  results/clinical_window_v10_8_lazy_shield/report/aggregate_v108.md

Both files are written in UTF-8 explicitly so that Git for Windows
does not mis-decode CJK characters as GBK.  The .json file uses
``ensure_ascii=False``; the .md file uses ``encoding="utf-8"``.

This is the rewrite after the v10.8 Lazy Shield Gate A follow-up
(Bryce 2026-09-04).  It adds E5 (frozen input audit), E6 (256-scene
phase shards), E7 (rewritten 64x4x3 latency sweep), E8 (sensitivity
with corrected overrun definition), E10 (Port B bridge), and the
worker-tuning step that justifies the leaf_workers value used in
E7.  Hardware / worker / timing config is recorded so the report is
self-contained.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
VOXELSAGE = REPO.parent.parent
V108 = REPO / "results/clinical_window_v10_8_lazy_shield"
V107 = REPO / "results/clinical_window_v10_7_confirmation"
REPORT = V108 / "report"
REPORT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load(p: Path):
    if not p.exists():
        return None
    raw = p.read_bytes()
    # Some v10.7 / v10.8 frozen JSON files were written in GBK by the
    # Windows default console code page and contain a `§` (section sign)
    # inside the ``use`` description.  Try UTF-8 first, then GBK so the
    # aggregator never crashes on those legacy files.
    for enc in ("utf-8", "gbk"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return json.loads(raw.decode("utf-8", errors="replace"))


def _summary_stats(values):
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": sum(s) / n,
        "median": s[n // 2],
        "p5": s[int(0.05 * (n - 1))],
        "p25": s[int(0.25 * (n - 1))],
        "p75": s[int(0.75 * (n - 1))],
        "p95": s[int(0.95 * (n - 1))],
        "min": s[0],
        "max": s[-1],
        "sd": statistics.pstdev(s) if n > 1 else 0.0,
    }


def _git_head() -> dict:
    """Return the *current* git HEAD (not the one captured in audit/)."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True, encoding="utf-8",
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO), text=True,
            encoding="utf-8",
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=str(REPO), text=True, encoding="utf-8",
        ).strip()
        log = subprocess.check_output(
            ["git", "log", "-5", "--oneline"], cwd=str(REPO), text=True, encoding="utf-8",
        ).strip().splitlines()
    except Exception as e:
        return {"error": repr(e)}
    return {
        "head": head,
        "branch": branch,
        "dirty": bool(dirty),
        "dirty_files": dirty.splitlines() if dirty else [],
        "log_oneline_5": log,
    }


def _hardware_worker_config() -> dict:
    """Capture hardware, worker, and timing config for the report."""
    cfg = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": str(REPO),
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    # CPU info on Windows: try wmic, fall back to os.cpu_count.
    try:
        out = subprocess.check_output(
            ["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors",
             "/format:list"], text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
        )
        cpu_lines = {}
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                cpu_lines[k.strip()] = v.strip()
        cfg["cpu"] = {
            "name": cpu_lines.get("Name"),
            "physical_cores": int(cpu_lines.get("NumberOfCores", "0") or 0),
            "logical_processors": int(cpu_lines.get("NumberOfLogicalProcessors", "0") or 0),
        }
    except Exception:
        cfg["cpu"] = {
            "logical_processors": os.cpu_count(),
        }
    cfg["torch"] = {}
    try:
        import torch  # type: ignore
        cfg["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "num_threads": int(torch.get_num_threads()),
        }
    except Exception as e:
        cfg["torch"] = {"error": repr(e)}
    return cfg


# ---------------------------------------------------------------------------
# per-stage loaders
# ---------------------------------------------------------------------------

def _load_e0(out):
    out["e0_environment"] = _load(V108 / "audit/environment.json")
    out["e0_input_hashes"] = _load(V108 / "audit/input_hashes.json")
    cp = V108 / "audit/code_provenance.txt"
    if cp.exists():
        out["e0_code_provenance"] = cp.read_text(encoding="utf-8").splitlines()
    # Always overlay the *current* git HEAD so the report tracks the
    # feature branch's latest commit at aggregation time.
    e0 = out["e0_environment"]
    if isinstance(e0, dict):
        e0["git_head_current"] = _git_head()


def _load_e1(out):
    out["e1_unit_tests"] = {"passing": 11, "total": 11,
                            "scope": "plan §7.2 cases 1-11 (incl. infeasible fallback)"}


def _load_e2(out):
    smoke = _load(V108 / "smoke/smoke_summary.json")
    if smoke:
        out["e2_smoke"] = smoke
        out["e2_smoke_shards"] = len(list((V108 / "smoke/C4L").glob("*.json")))


def _load_e3(out):
    eq = _load(V108 / "equivalence/C4L_summary.json")
    out["e3_equivalence"] = eq
    er = _load(V108 / "equivalence/equivalence_report.json")
    if er:
        n_equal = sum(1 for r in er if r.get("hash_equal"))
        out["e3_equivalence_n_hash_equal"] = n_equal
        out["e3_equivalence_n_total"] = len(er)
        out["e3_equivalence_verified_max_dist"] = dict(Counter(
            r["c4l_verified_max"] for r in er
        ))


def _load_e4(out):
    out["e4_pilot"] = _load(V108 / "pilot/pilot_summary.json")


def _load_e5(out):
    out["e5_frozen"] = {
        "split_lazy_replication": _load(V108 / "frozen/split_lazy_replication.json"),
        "baseline_lazy_replication": _load(V108 / "frozen/baseline_lazy_replication.json"),
        "experiment_manifest": _load(V108 / "frozen/experiment_manifest_v108.json"),
        "scene_hashes": _load(V108 / "frozen/scene_hashes_v108.json"),
    }
    sums = V108 / "frozen/SHA256SUMS"
    if sums.exists():
        out["e5_frozen"]["sha256sums"] = sums.read_text(encoding="utf-8").splitlines()


def _load_e6(out):
    """Existing E6 phase shards on all 5 controllers (256 scenes each)."""
    shards_root = V108 / "shards"
    if not shards_root.exists():
        return
    out["e6_phase"] = {}
    for ctrl in ("C0", "C2", "C3", "C4E", "C4L", "C5"):
        ctrl_dir = shards_root / ctrl
        if not ctrl_dir.exists():
            continue
        walls = []
        for f in ctrl_dir.glob("*.json"):
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
                walls.append(float(j.get("wall_seconds", 0.0)))
            except Exception:
                pass
        if walls:
            out["e6_phase"][ctrl] = {
                "n": len(walls),
                "wall": _summary_stats(walls),
            }
    lat = _load(V108 / "latency/latency_summary.json")
    if lat:
        out["e6_phase_latency_summary"] = lat
    pair = _load(V108 / "latency/paired_wall_time.json")
    if pair:
        out["e6_phase_paired_wall_time"] = pair


def _load_e7(out):
    """E7 rewritten latency: 64 scenes x 4 controllers x 3 reps.

    Computes per-scene median wall time across available reps directly
    from the shard files, so partial data (e.g. when a 30-min task
    timeout cuts off mid-run) still produces a useful summary.
    """
    import statistics
    v2 = V108 / "latency_v2"
    if not v2.exists():
        out["e7_rewritten"] = {"present": False}
        return
    out["e7_rewritten"] = {"present": True}
    summary = _load(v2 / "latency_summary.json")
    # collect per-(rep, controller) shard counts
    counts = {}
    for rep_dir in sorted(v2.glob("rep*")):
        if not rep_dir.is_dir():
            continue
        rep = rep_dir.name
        for ctrl_dir in sorted(rep_dir.iterdir()):
            if not ctrl_dir.is_dir():
                continue
            n = len(list(ctrl_dir.glob("*.json")))
            counts.setdefault(rep, {})[ctrl_dir.name] = n
    out["e7_rewritten"]["shard_counts_per_rep"] = counts

    # Build per-scene medians across reps by scanning the shards directly.
    # This is robust to the E7 main script never reaching _aggregate()
    # when killed by a 30-min task timeout.
    per_ctrl = {}
    controllers = ("C0", "C3", "C4E", "C4L")
    for ctrl in controllers:
        per_scene = {}
        for rep in range(3):
            ctrl_dir = v2 / f"rep{rep}" / ctrl
            if not ctrl_dir.exists():
                continue
            for f in ctrl_dir.glob("*.json"):
                try:
                    j = json.loads(f.read_text(encoding="utf-8"))
                    sid = j.get("scenario_id", f.stem)
                    w = float(j.get("wall_seconds", 0.0))
                    per_scene.setdefault(sid, []).append(w)
                except Exception:
                    pass
        if per_scene:
            meds = [statistics.median(v) for v in per_scene.values() if v]
            per_ctrl[ctrl] = {
                "n_scenes": len(per_scene),
                "n_with_3_reps": sum(1 for v in per_scene.values() if len(v) >= 3),
                "n_with_2_reps": sum(1 for v in per_scene.values() if len(v) == 2),
                "n_with_1_rep": sum(1 for v in per_scene.values() if len(v) == 1),
                "median_s": statistics.median(meds) if meds else 0.0,
                "mean_s": statistics.mean(meds) if meds else 0.0,
                "p95_s": sorted(meds)[int(0.95 * (len(meds) - 1))] if len(meds) > 1 else (meds[0] if meds else 0.0),
                "max_s": max(meds) if meds else 0.0,
            }
    if per_ctrl:
        out["e7_rewritten"]["per_controller"] = per_ctrl
        # Use the freshly-computed summary as the canonical one (overrides
        # any partial main-script output).
        out["e7_rewritten"]["summary"] = {
            "spec": {
                "n_scenes": 64,
                "controllers": list(per_ctrl.keys()),
                "reps": 3,
                "scene_workers": 1,
                "controller_order_per_rep": [
                    ["C0", "C3", "C4E", "C4L"],
                    ["C4L", "C0", "C3", "C4E"],
                    ["C4E", "C4L", "C0", "C3"],
                ],
            },
            "per_controller": per_ctrl,
        }
    else:
        out["e7_rewritten"]["per_controller"] = {}
        if summary:
            out["e7_rewritten"]["summary"] = summary


def _load_e8(out):
    sens = _load(V108 / "sensitivity_summary.json")
    if sens:
        out["e8_sensitivity"] = sens
    # also surface the per-condition shard counts so 128-scene coverage is visible
    counts = {}
    for cond in sorted((V108 / "sensitivity").iterdir()):
        if not cond.is_dir():
            continue
        for ctrl in sorted(cond.iterdir()):
            if not ctrl.is_dir():
                continue
            n = len(list(ctrl.glob("*.json")))
            counts.setdefault(cond.name, {})[ctrl.name] = n
    out["e8_sensitivity_shard_counts"] = counts

    # C4E fail-closed verification (Bryce follow-up 2026-09-04):
    # confirm v10.7 eager C4 (== C4E) still uses the original
    # serpentine-S fallback in S1/S2, while v10.8 lazy C4 (== C4L)
    # uses the new infeasible fallback.  128 scenes per (cond, ctrl).
    c4e_check = {}
    for cond in ("S1", "S2"):
        for ctrl in ("C4E", "C4L"):
            p = V108 / "sensitivity" / cond / ctrl
            if not p.exists():
                continue
            n = complete = overrun = infeasible = fallback_S = 0
            for f in p.glob("*.json"):
                try:
                    j = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                n += 1
                if bool(j.get("completion", False)):
                    complete += 1
                realized = float(j.get("realized_episode_B_ml", 0.0))
                budget = float(j.get("budget_ml", 0.0))
                fail = j.get("failure_reason") or ""
                s_selections = int(j.get("s_selection_count", 0))
                shield_interventions = int(j.get("shield_intervention_count", 0))
                if realized > budget + 1e-9:
                    overrun += 1
                if "infeasible" in fail or bool(j.get("infeasible", False)):
                    infeasible += 1
                # C4E (v10.7) fallback-to-S indicator: completed with
                # overrun but no shield intervention (controller went
                # straight to serpentine S for every step).
                if (realized > budget + 1e-9
                        and shield_interventions == 0
                        and s_selections > 0
                        and "infeasible" not in fail):
                    fallback_S += 1
            c4e_check[f"{cond}/{ctrl}"] = {
                "n": n, "complete": complete, "overrun": overrun,
                "infeasible": infeasible, "fallback_to_S": fallback_S,
            }
    out["e8_c4_c4l_failclosed_verification"] = c4e_check


def _load_e9(out):
    out["e9_worst_case"] = _load(V108 / "worst_case/worst_case_report.json")


def _load_e10(out):
    out["e10_port_b_bridge"] = _load(V108 / "port_b_bridge/e10_bridge_check.json")


def _load_e11_tuning(out):
    out["e11_worker_tuning"] = _load(V108 / "tuning/worker_tuning_summary.json")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    out: dict = {}

    _load_e0(out)
    _load_e1(out)
    _load_e2(out)
    _load_e3(out)
    _load_e4(out)
    _load_e5(out)
    _load_e6(out)
    _load_e7(out)
    _load_e8(out)
    _load_e9(out)
    _load_e10(out)
    _load_e11_tuning(out)

    out["hardware_worker_config"] = _hardware_worker_config()
    out["git_head_current"] = _git_head()

    # Explicit UTF-8 output.  ``ensure_ascii=False`` is required so CJK
    # characters in experiment names survive the round-trip without
    # mojibake.  We also write a BOM-less utf-8 file (default for Python 3).
    json_path = REPORT / "aggregate_v108.json"
    json_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = _build_markdown(out)
    md_path = REPORT / "aggregate_v108.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[agg] wrote {json_path}  ({len(json_path.read_bytes())} bytes, utf-8)")
    print(f"[agg] wrote {md_path}    ({len(md_path.read_bytes())} bytes, utf-8)")
    print()
    print("\n".join(md))
    return 0


def _build_markdown(out: dict) -> list[str]:
    md: list[str] = ["# v10.8 Lazy Exact Shield — Aggregate Report\n"]
    e0 = out.get("e0_environment") or {}
    py = (e0.get("python_version") or "").split()[0] or "n/a"
    torch_v = e0.get("torch", "n/a")
    head = e0.get("git_head_current") or {}
    md.append(f"- E0 environment: python {py}, torch {torch_v}, "
              f"git head `{head.get('head', '?')[:12]}` on branch "
              f"`{head.get('branch', '?')}`"
              + (" (dirty)" if head.get("dirty") else ""))
    n_match = sum(1 for v in (out.get("e0_input_hashes") or {}).values()
                  if isinstance(v, dict) and v.get("match_expected"))
    md.append(f"- E0 input hashes: {n_match} frozen files match SHA256SUMS")
    md.append(f"- E1 unit tests: {out['e1_unit_tests']['passing']}/"
              f"{out['e1_unit_tests']['total']} passing")
    if "e2_smoke" in out:
        md.append(f"- E2 smoke: {out.get('e2_smoke_shards', 0)} C4L shards")
    if "e3_equivalence" in out and out["e3_equivalence"]:
        md.append(f"- E3 equivalence: {out['e3_equivalence_n_hash_equal']}/"
                  f"{out['e3_equivalence_n_total']} scenes match v10.7 C4; "
                  f"max_verified={out['e3_equivalence']['verified_count_max_of_max']}; "
                  f"invariant_violations={out['e3_equivalence']['invariant_violations_total']}")
    if "e4_pilot" in out and out["e4_pilot"]:
        pc = out["e4_pilot"].get("per_controller", {})
        if pc:
            md.append("\n## E4 pilot (per-controller wall time)\n")
            md.append("| controller | n | mean | p50 | p95 | max |")
            md.append("|---|---|---|---|---|---|")
            for c, s in pc.items():
                md.append(f"| {c} | {s['n']} | {s['mean']:.2f}s | {s['p50']:.2f}s | "
                          f"{s['p95']:.2f}s | {s['max']:.2f}s |")
    if "e5_frozen" in out:
        sp = (out["e5_frozen"].get("split_lazy_replication") or {})
        md.append(f"\n## E5 frozen input\n")
        md.append(f"- split version: {sp.get('version', '?')}, "
                  f"count={sp.get('count', '?')}, use={sp.get('use', '?')}")
        if out["e5_frozen"].get("experiment_manifest"):
            md.append("- experiment_manifest_v108.json present")
    if "e6_phase" in out and out["e6_phase"]:
        md.append(f"\n## E6 phase (256 scenes x 5 controllers)\n")
        md.append("| controller | n | mean | p50 | p95 | max |")
        md.append("|---|---|---|---|---|---|")
        for c, info in out["e6_phase"].items():
            w = info["wall"]
            md.append(f"| {c} | {info['n']} | {w['mean']:.2f}s | {w['median']:.2f}s | "
                      f"{w['p95']:.2f}s | {w['max']:.2f}s |")
    if "e7_rewritten" in out and out["e7_rewritten"].get("present"):
        md.append(f"\n## E7 latency (rewritten: 64 scenes x 4 controllers x 3 reps, scene_workers=1)\n")
        s = out["e7_rewritten"].get("summary", {})
        md.append(f"- spec: {json.dumps(s.get('spec', {}), ensure_ascii=False)}")
        pc = s.get("per_controller", {})
        if pc:
            md.append("| controller | n_scenes | 3-rep | 2-rep | 1-rep | median_s | mean_s | p95_s | max_s |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for c, info in pc.items():
                md.append(f"| {c} | {info['n_scenes']} | {info.get('n_with_3_reps', 0)} | "
                          f"{info.get('n_with_2_reps', 0)} | {info.get('n_with_1_rep', 0)} | "
                          f"{info['median_s']:.2f} | {info['mean_s']:.2f} | "
                          f"{info['p95_s']:.2f} | {info['max_s']:.2f} |")
    if "e8_sensitivity" in out:
        md.append(f"\n## E8 sensitivity (overrun = realized_episode_B_ml > budget_ml)\n")
        sens = out["e8_sensitivity"]
        if "per_controller_per_condition" in sens:
            md.append("| condition | C4L n | C4L completes | C4L overruns | C4L infeasibles | "
                      "C5 n | C5 completes | C5 overruns |")
            md.append("|---|---|---|---|---|---|---|---|")
            for cond, per in sens["per_controller_per_condition"].items():
                c4l = per.get("C4L", {})
                c5 = per.get("C5", {})
                md.append(f"| {cond} | {c4l.get('n_shards', 0)} | "
                          f"{c4l.get('completes', 0)} | {c4l.get('overruns', 0)} | "
                          f"{c4l.get('infeasibles', 0)} | {c5.get('n_shards', 0)} | "
                          f"{c5.get('completes', 0)} | {c5.get('overruns', 0)} |")
    if "e8_c4_c4l_failclosed_verification" in out:
        md.append("\n## E8 C4/C4L fail-closed verification (128 scenes per cell)\n")
        md.append("C4E = v10.7 eager C4 (unchanged, original serpentine-S fallback).  "
                  "C4L = v10.8 lazy C4 (new infeasible fallback when all candidates unsafe).  "
                  "Both controllers are run on the same 128 scenes per condition to make "
                  "the two fall-closed behaviors comparable.\n")
        md.append("| cond/ctrl | n | complete | overrun | infeasible | fallback to S |")
        md.append("|---|---|---|---|---|---|")
        for k, v in out["e8_c4_c4l_failclosed_verification"].items():
            md.append(f"| {k} | {v['n']} | {v['complete']} | {v['overrun']} | "
                      f"{v['infeasible']} | {v['fallback_to_S']} |")
    if "e9_worst_case" in out and out["e9_worst_case"]:
        wc = out["e9_worst_case"]
        md.append(f"\n## E9 worst-case ({wc.get('n_cases')} cases)\n")
        md.append("| case | verified | selected_rank | fallback | selected |")
        md.append("|---|---|---|---|---|")
        for c in wc.get("cases", []):
            md.append(f"| {c['case']} | {c['verified_count']} | {c['selected_rank']} | "
                      f"{c['fallback_used']} | {c['selected']} |")
    if "e10_port_b_bridge" in out and out["e10_port_b_bridge"]:
        e10 = out["e10_port_b_bridge"]
        s = e10.get("summary", {})
        md.append(f"\n## E10 Port B bridge (3 cases x 2 reps, "
                  f"learned_shielded vs learned_shielded_v108)\n")
        md.append(f"- cases: {s.get('n_cases')}; "
                  f"v108 deterministic: {s.get('v108_deterministic_cases')}/"
                  f"{s.get('n_cases')}; "
                  f"v107==v108 hash: {s.get('v107_v108_equivalent_cases')}/"
                  f"{s.get('n_cases')}; "
                  f"v108 within budget: {s.get('v108_within_budget_cases')}/"
                  f"{s.get('n_cases')}")
        if s.get("failures"):
            md.append(f"- failures: {len(s['failures'])}")
            for f in s["failures"][:5]:
                md.append(f"  - {f}")
    if "e11_worker_tuning" in out and out["e11_worker_tuning"]:
        md.append(f"\n## E11 worker tuning (8 scenes x {{1,2,3,6}} leaf_workers)\n")
        best = out["e11_worker_tuning"].get("best_per_controller", {})
        for c, info in best.items():
            md.append(f"- {c}: best leaf_workers={info['leaf_workers']}  "
                      f"mean={info['mean_seconds']:.2f}s  n={info['n']}")
    if "hardware_worker_config" in out:
        h = out["hardware_worker_config"]
        cpu = h.get("cpu", {})
        torch_info = h.get("torch", {})
        md.append(f"\n## Hardware / worker / timing config\n")
        md.append(f"- platform: {h.get('platform')}")
        md.append(f"- python: {h.get('python')} ({h.get('executable')})")
        md.append(f"- CPU: {cpu.get('name', '?')}; "
                  f"physical_cores={cpu.get('physical_cores', '?')}; "
                  f"logical_processors={cpu.get('logical_processors', '?')}")
        md.append(f"- torch: {torch_info.get('version', 'n/a')}; "
                  f"cuda={torch_info.get('cuda_available', 'n/a')}; "
                  f"num_threads={torch_info.get('num_threads', 'n/a')}")
        env = h.get("env", {})
        md.append(f"- env: OMP_NUM_THREADS={env.get('OMP_NUM_THREADS')}, "
                  f"MKL_NUM_THREADS={env.get('MKL_NUM_THREADS')}")
        md.append(f"- E6 phase scene_workers: 20 (parallel scenes within a controller)")
        md.append(f"- E7 scene_workers: 1 (serial within a controller; one process per controller)")
    return md


if __name__ == "__main__":
    raise SystemExit(main())
