#!/usr/bin/env python3
"""Verify the L1 per-joint MPJDLE export patch.

This script is intentionally static-first: it checks the source file without
importing the project, so it still works on a fresh server before MMCV/Mamba
dependencies are fully warmed up. Optionally pass an eval JSON to validate the
runtime output schema after `tools/test.py --metrics-out`.
"""
from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path


REQUIRED_MARKERS = [
    "per_joint_mpjdle_sum = np.zeros((len(self.JOINT_NAMES), 3), dtype=np.float64)",
    "mpjpe_metrics, per_joint_mpjpe, per_joint_mpjdle, matched_pred_kpts, matched_gt_kpts = matched_results",
    "per_joint_mpjdle_values = np.asarray(per_joint_mpjdle, dtype=np.float64)",
    "per_joint_mpjdle_sum += per_joint_mpjdle_values * matched_count",
    "per_joint_mpjdle_sum += miss_penalty_mm * missed_count",
    "avg_per_joint_mpjdle = per_joint_mpjdle_sum / float(total_gt_persons)",
    "export['per_joint_mpjdle'] = per_joint_mpjdle_dict",
    "per_joint_mpjdle = per_joint_error_dim.mean(dim=0).cpu().numpy() * 1000",
    "return mpjpe_metrics, per_joint_mpjpe, per_joint_mpjdle, matched_pred, matched_gt",
]


def verify_source(path: Path) -> bool:
    if not path.is_file():
        print(f"FAIL: source file not found: {path}", file=sys.stderr)
        return False

    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"FAIL: Python syntax check failed for {path}:\n{exc}", file=sys.stderr)
        return False

    src = path.read_text(encoding="utf-8", errors="replace")
    missing = [marker for marker in REQUIRED_MARKERS if marker not in src]
    if missing:
        print("FAIL: L1 source markers are missing:", file=sys.stderr)
        for marker in missing:
            print(f"  - {marker}", file=sys.stderr)
        return False

    print(f"OK: {path} contains the complete L1 per_joint_mpjdle patch.")
    return True


def verify_json(path: Path) -> bool:
    if not path.is_file():
        print(f"FAIL: eval JSON not found: {path}", file=sys.stderr)
        return False

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    ok = True
    for key in ("per_joint_mpjpe", "per_joint_mpjdle"):
        value = data.get(key)
        if not isinstance(value, dict):
            print(f"FAIL: {path} missing dict key {key!r}.", file=sys.stderr)
            ok = False
            continue
        if len(value) != 14:
            print(f"FAIL: {key} has {len(value)} entries; expected 14.", file=sys.stderr)
            ok = False

    mpjdle = data.get("per_joint_mpjdle", {})
    for joint, axes in mpjdle.items():
        if set(axes.keys()) != {"h", "v", "d"}:
            print(
                f"FAIL: per_joint_mpjdle[{joint!r}] keys are {sorted(axes.keys())}; "
                "expected h/v/d.",
                file=sys.stderr,
            )
            ok = False

    if ok:
        sample = next(iter(mpjdle.items())) if mpjdle else None
        print(f"OK: {path} has 14 per_joint_mpjpe and 14 per_joint_mpjdle entries.")
        print(f"    sample per_joint_mpjdle: {sample}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("opera/datasets/wifi_pose.py"),
        help="Path to wifi_pose.py.",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=None,
        help="Optional metrics JSON to validate after eval.",
    )
    args = parser.parse_args()

    ok = verify_source(args.source)
    if args.eval_json is not None:
        ok = verify_json(args.eval_json) and ok
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
