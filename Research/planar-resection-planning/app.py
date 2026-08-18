#!/usr/bin/env python3
"""One-command development server for the planar resection simulator."""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from planner import generate_domain, plan_resection, serpentine_priority_resection  # noqa: E402
from mechanics import solve_tension  # noqa: E402
from trained_policy import trained_policy_service  # noqa: E402
from clinical_target_order_service import clinical_target_order_service  # noqa: E402

app = FastAPI(title="Planar Resection Simulator", version="0.4.0")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(CURRENT_DIR / "static" / "index.html", media_type="text/html")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "planar-resection-simulator"}


@app.get("/api/policy/clinical-v104/status")
def api_clinical_v104_status() -> Dict[str, object]:
    return clinical_target_order_service.status()


@app.post("/api/policy/clinical-v104/plan")
def api_clinical_v104_plan(payload: Dict[str, Any]) -> Dict[str, object]:
    try:
        return clinical_target_order_service.plan(payload)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/generate")
def api_generate(payload: Dict[str, Any]) -> Dict[str, object]:
    try:
        return generate_domain(
            seed=payload.get("seed"),
            rows=payload.get("rows"),
            cols=payload.get("cols"),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/plan")
def api_plan(payload: Dict[str, Any]) -> Dict[str, object]:
    try:
        return plan_resection(
            rows=payload["rows"],
            cols=payload["cols"],
            domain_cells=payload["domain_cells"],
            obstacle_cells=payload.get("obstacle_cells", []),
            start_cell=payload["start_cell"],
            weights=payload.get("weights"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/plan/serpentine")
def api_plan_serpentine(payload: Dict[str, Any]) -> Dict[str, object]:
    """Run the deterministic S-form priority baseline on a scenario.

    Mirrors the rule-planner response shape so the frontend can replay either
    algorithm with the same presentPlan() renderer.
    """
    try:
        return serpentine_priority_resection(
            rows=payload["rows"],
            cols=payload["cols"],
            domain_cells=payload["domain_cells"],
            obstacle_cells=payload.get("obstacle_cells", ()),
            start_cell=payload["start_cell"],
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tension")
def api_tension(payload: Dict[str, Any]) -> Dict[str, object]:
    try:
        return solve_tension(
            rows=payload["rows"],
            cols=payload["cols"],
            domain_cells=payload["domain_cells"],
            vessel_cells=payload.get("vessel_cells", []),
            cut_cells=payload.get("cut_cells", []),
            parameters=payload.get("parameters"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/policy/status")
def api_policy_status() -> Dict[str, object]:
    return trained_policy_service.status()


@app.post("/api/policy/load")
def api_policy_load(payload: Dict[str, Any]) -> Dict[str, object]:
    try:
        return trained_policy_service.load()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/policy/plan")
def api_policy_plan(payload: Dict[str, Any]) -> Dict[str, object]:
    try:
        return trained_policy_service.plan(payload)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _lan_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("10.255.255.255", 1))
            return str(sock.getsockname()[0])
        finally:
            sock.close()
    except OSError:
        return "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the planar resection simulator")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8910)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    import uvicorn

    print("Planar resection simulator")
    print(f"  Local:   http://127.0.0.1:{args.port}/")
    print(f"  Network: http://{_lan_ip()}:{args.port}/")
    print("  Standalone planar simulator (default port 8910).")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
