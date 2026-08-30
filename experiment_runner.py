#!/usr/bin/env python3
"""Run isolated synchronized benchmarks and append a meaningful experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--change", required=True)
    parser.add_argument("--lesson", default="Pending analysis")
    parser.add_argument("--decision", choices=("keep", "reject", "investigate"), default="investigate")
    parser.add_argument("--shape", type=int, action="append", required=True)
    parser.add_argument("--impl", required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--accuracy-trials", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--skip-reference-timing", action="store_true")
    parser.add_argument("--batch-chunk", type=int)
    parser.add_argument("--reference-batch-chunk", type=int)
    parser.add_argument("--compile-candidate", action="store_true")
    parser.add_argument("--trace-candidate", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    args = parser.parse_args()

    results = []
    for shape in args.shape:
        command = [
            sys.executable,
            str(ROOT / "benchmark_harness.py"),
            "--shape", str(shape),
            "--impl", args.impl,
            "--dtype", args.dtype,
            "--padding-ratio", str(args.padding_ratio),
            "--warmup", str(args.warmup),
            "--repeats", str(args.repeats),
            "--accuracy-trials", str(args.accuracy_trials),
        ]
        if args.skip_reference_timing:
            command.append("--skip-reference-timing")
        if args.batch_chunk:
            command.extend(("--batch-chunk", str(args.batch_chunk)))
        if args.reference_batch_chunk:
            command.extend(("--reference-batch-chunk", str(args.reference_batch_chunk)))
        if args.compile_candidate:
            command.extend(("--compile-candidate", "--compile-mode", args.compile_mode))
        if args.trace_candidate:
            command.append("--trace-candidate")
        started = time.time()
        try:
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True,
                timeout=args.timeout, check=False,
            )
            if completed.stdout.strip().startswith("{"):
                result = json.loads(completed.stdout)
                result["process_exit_code"] = completed.returncode
            else:
                result = {
                    "shape": shape,
                    "process_exit_code": completed.returncode,
                    "error": completed.stderr[-4000:] or completed.stdout[-4000:],
                }
        except subprocess.TimeoutExpired as error:
            result = {
                "shape": shape,
                "error": f"timeout after {args.timeout}s",
                "stdout": (error.stdout or "")[-2000:],
                "stderr": (error.stderr or "")[-2000:],
            }
        result["wall_seconds"] = time.time() - started
        results.append(result)

    record = {
        "experiment_id": args.id,
        "timestamp_unix": time.time(),
        "hypothesis": args.hypothesis,
        "targeted_shapes": args.shape,
        "implementation_change": args.change,
        "implementation": args.impl,
        "dtype": args.dtype,
        "results": results,
        "correctness": [r.get("correct") for r in results],
        "latency_ms": [r.get("latency_ms") for r in results],
        "useful_tflops": [r.get("useful_tflops") for r in results],
        "estimated_fp32_mfu": [r.get("estimated_fp32_mfu") for r in results],
        "comparison_with_reference": [r.get("reference_speedup") for r in results],
        "comparison_with_strong_obvious_baseline": None,
        "decision": args.decision,
        "lesson_learned": args.lesson,
        "aggregate_note": "INTERNAL ONLY; no official weighting inferred.",
    }
    with (ROOT / "experiments.jsonl").open("a") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
