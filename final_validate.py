#!/usr/bin/env python3
"""Fresh-process final validation for Shapes 1-13."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "final"


def run_shape(shape: int) -> dict:
    trials, warmup, repeats = 5, 5, 15
    if shape == 6:
        trials, warmup, repeats = 2, 1, 3
    elif shape == 13:
        trials, warmup, repeats = 3, 3, 9
    command = [
        sys.executable, str(ROOT / "benchmark_harness.py"),
        "--shape", str(shape), "--impl", "dispatch",
        "--accuracy-trials", str(trials),
        "--warmup", str(warmup), "--repeats", str(repeats),
    ]
    if shape in (6, 13):
        command.extend(("--reference-batch-chunk", "64" if shape == 6 else "4"))
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True,
        timeout=600, check=False,
    )
    if not completed.stdout.strip().startswith("{"):
        return {
            "shape": shape, "process_exit_code": completed.returncode,
            "error": completed.stderr[-4000:] or completed.stdout[-4000:],
        }
    result = json.loads(completed.stdout)
    result["process_exit_code"] = completed.returncode
    return result


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for shape in range(1, 14):
        result = run_shape(shape)
        results.append(result)
        (OUTPUT / f"shape_{shape:02d}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(
            f"shape {shape}: correct={result.get('correct')} "
            f"latency_ms={result.get('latency_ms')}",
            flush=True,
        )
    summary = {
        "fresh_process": True,
        "device": "mps",
        "dtype": "float32",
        "results": results,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(result.get("correct") for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

