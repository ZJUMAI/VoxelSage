"""Resumable fixed-round v10.6 BC on policy_train teacher labels only."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_target_order_policy_v106 import TargetOrderScorerV106, unpack_spatial

BASE = SIM / "results/clinical_window_v10_6_shielded_learning"
TEACHER = BASE / "teacher"
FROZEN = BASE / "frozen"


def best_safe_indices(t_total: np.ndarray, b_total: np.ndarray, safe: np.ndarray, valid: np.ndarray) -> np.ndarray:
    best = np.zeros(len(t_total), dtype=np.int64)
    for i in range(len(t_total)):
        choices = np.flatnonzero(safe[i] & valid[i])
        if not len(choices):
            raise RuntimeError(f"state {i} has no exact-safe candidate")
        best[i] = min(choices, key=lambda j: (round(float(t_total[i, j]), 6),
                                               round(float(b_total[i, j]), 6), int(j)))
    return best


def normalize_candidate_batch(features: np.ndarray, scales: dict) -> np.ndarray:
    out = features.astype(np.float32, copy=True)
    mean = np.asarray(scales["mean"], np.float32)
    std = np.asarray(scales["std"], np.float32)
    idx = np.asarray(scales["scaled_indices"], dtype=int)
    out[..., idx] = (out[..., idx] - mean[idx]) / std[idx]
    return out


def tensor_batch(data, indices, feature_scales, time_scale, blood_scale, device):
    grid = unpack_spatial(data["grid_bits"][indices], data["transfer_q"][indices])
    feature = normalize_candidate_batch(data["features"][indices], feature_scales)
    return {
        "grid": torch.from_numpy(grid).to(device),
        "feature": torch.from_numpy(feature).to(device),
        "global": torch.from_numpy(data["global_context"][indices].astype(np.float32)).to(device),
        "targets": torch.from_numpy(data["targets"][indices].astype(np.int64)).to(device),
        "valid": torch.from_numpy(data["valid"][indices]).to(device),
        "safe": torch.from_numpy(data["safe_exact"][indices]).to(device),
        "completion": torch.from_numpy(data["completion"][indices].astype(np.float32)).to(device),
        "T_total": torch.from_numpy((data["T_total"][indices] / time_scale).astype(np.float32)).to(device),
        "B_tail": torch.from_numpy((data["B_tail"][indices] / blood_scale).astype(np.float32)).to(device),
        "B_total": torch.from_numpy((data["B_total"][indices] / blood_scale).astype(np.float32)).to(device),
    }


def losses(output, batch, best_index, weights):
    valid = batch["valid"]
    safe_valid = valid & batch["safe"]
    masked_score = output["score"].masked_fill(~safe_valid, -1e9)
    rank = F.cross_entropy(masked_score, best_index)
    def mse(name):
        # Invalid candidate slots contain NaN supervision. Multiplying by a
        # zero mask is insufficient because IEEE NaN * 0 remains NaN.
        return F.mse_loss(output[name][valid], batch[name][valid])
    completion = F.binary_cross_entropy_with_logits(
        output["completion_logit"][valid], batch["completion"][valid]
    )
    safe = F.binary_cross_entropy_with_logits(
        output["safe_logit"][valid], batch["safe"][valid].float()
    )
    components = {
        "rank": rank, "T_total": mse("T_total"), "B_tail": mse("B_tail"),
        "B_total": mse("B_total"), "completion": completion, "safe": safe,
    }
    total = sum(float(weights[name]) * value for name, value in components.items())
    return total, components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--spatial", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026081601)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=BASE / "runs/bc/config_00_seed_2026081601")
    parser.add_argument("--limit-states", type=int, default=None)
    parser.add_argument(
        "--resume-checkpoint", type=Path, default=None,
        help="Continue from an audited epoch checkpoint; --epochs is the number of additional rounds.",
    )
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; do not run formal BC on CPU")
    device = torch.device(args.device)
    archive = np.load(TEACHER / "teacher_rankings_v106.npz")
    data = {key: archive[key] for key in archive.files}
    archive.close()
    n = len(data["global_context"])
    if args.limit_states: n = min(n, args.limit_states)
    feature_scales = json.loads((TEACHER / "feature_scales_v106.json").read_text(encoding="utf-8"))
    frozen_scales = json.loads((FROZEN / "scales_v10_6.json").read_text(encoding="utf-8"))
    time_scale = float(frozen_scales["time_scale_minutes"])
    blood_scale = float(frozen_scales["blood_scale_ml"])
    best = best_safe_indices(data["T_total"][:n], data["B_total"][:n],
                             data["safe_exact"][:n], data["valid"][:n])
    weights = {"rank": 1.0, "T_total": 0.2, "B_tail": 0.3,
               "B_total": 0.3, "completion": 0.1, "safe": 0.2}
    model = TargetOrderScorerV106(hidden=args.hidden, spatial=args.spatial).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    history = []
    start_epoch = 0
    if args.resume_checkpoint is not None:
        try:
            resume = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        except TypeError:
            resume = torch.load(args.resume_checkpoint, map_location=device)
        required = ("state_dict", "optimizer_state_dict", "random_state", "numpy_random_state",
                    "torch_random_state", "epoch")
        missing = [name for name in required if name not in resume]
        if missing:
            raise RuntimeError(f"resume checkpoint lacks exact-continuation fields: {missing}")
        if int(resume["hidden"]) != args.hidden or int(resume["spatial"]) != args.spatial:
            raise RuntimeError("resume architecture does not match requested architecture")
        if int(resume["seed"]) != args.seed:
            raise RuntimeError("resume seed does not match requested seed")
        model.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        random.setstate(resume["random_state"])
        np.random.set_state(resume["numpy_random_state"])
        torch.set_rng_state(resume["torch_random_state"].cpu())
        if device.type == "cuda" and resume.get("cuda_random_state_all") is not None:
            torch.cuda.set_rng_state_all([
                state.cpu() for state in resume["cuda_random_state_all"]
            ])
        start_epoch = int(resume["epoch"])
        history_path = args.output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if len(history) != start_epoch:
            raise RuntimeError(
                f"history/checkpoint round mismatch: {len(history)} != {start_epoch}"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    wall0 = time.time()
    wall_offset = float(history[-1]["wall_seconds"]) if history else 0.0
    for local_epoch in range(args.epochs):
        epoch = start_epoch + local_epoch + 1
        model.train(); order = np.random.permutation(n)
        sums = {name: 0.0 for name in ["total", *weights]}; batches = 0
        for start in range(0, n, args.batch_size):
            index = order[start:start + args.batch_size]
            batch = tensor_batch(data, index, feature_scales, time_scale, blood_scale, device)
            output = model(batch["grid"], batch["feature"], batch["global"], batch["targets"])
            best_t = torch.from_numpy(best[index]).to(device)
            total, components = losses(output, batch, best_t, weights)
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite training loss in round {epoch}, batch {start}")
            optimizer.zero_grad(set_to_none=True); total.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"non-finite gradient in round {epoch}, batch {start}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            sums["total"] += float(total.item())
            for name, value in components.items(): sums[name] += float(value.item())
            batches += 1
        row = {"epoch": epoch, **{name: value / max(1, batches) for name, value in sums.items()},
               "wall_seconds": wall_offset + time.time() - wall0}
        history.append(row)
        checkpoint = {
            "version": "v10.6-spatial-bc-v1", "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hidden": args.hidden, "spatial": args.spatial, "seed": args.seed,
            "epoch": epoch, "feature_scales": feature_scales,
            "time_scale": time_scale, "blood_scale": blood_scale,
            "weights": weights, "states": n,
            "random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        torch.save(checkpoint, args.output_dir / f"epoch_{epoch:02d}.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(row), flush=True)
    final = args.output_dir / "final.pt"
    torch.save(checkpoint, final)
    print(json.dumps({"final": str(final), "states": n, "epoch": epoch,
                      "additional_rounds": args.epochs}), flush=True)


if __name__ == "__main__":
    main()
