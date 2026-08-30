#!/usr/bin/env python3
"""Same-process synchronized A/B for an earlier and current dispatcher."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import types
from pathlib import Path

import torch

from implementations import DispatchTransformer
from original.torch_transformer_benchmark import (
    BaselineTransformer,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
)
from shapes import OFFICIAL_SHAPES


def load_dispatch(revision: str):
    source = subprocess.check_output(
        ["git", "show", f"{revision}:implementations.py"], text=True
    )
    module = types.ModuleType(f"implementations_{revision.replace('-', '_')}")
    module.__file__ = f"{revision}:implementations.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.DispatchTransformer


def timed_forward(model, x, mask) -> float:
    torch.mps.synchronize()
    started = time.perf_counter_ns()
    model(x, mask)
    torch.mps.synchronize()
    return (time.perf_counter_ns() - started) / 1e6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, choices=(1, 4, 5), required=True)
    parser.add_argument("--old-revision", default="13b3d26")
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = OFFICIAL_SHAPES[args.shape - 1]
    device, dtype = torch.device("mps"), torch.float32
    torch.manual_seed(1234)
    reference = BaselineTransformer(config)
    old_model = load_dispatch(args.old_revision)(config)
    current_model = DispatchTransformer(config)
    copy_model_weights(reference, old_model)
    copy_model_weights(reference, current_model)
    reference = reference.to(device=device, dtype=dtype).eval()
    old_model = old_model.to(device=device, dtype=dtype).eval()
    current_model = current_model.to(device=device, dtype=dtype).eval()

    accuracy = []
    with torch.inference_mode():
        for trial in range(args.accuracy_trials):
            x, mask = generate_random_case(
                config, device, dtype, 1234 + trial, 0.0, 1.0
            )
            expected = reference(x, mask)
            old_output = old_model(x, mask)
            current_output = current_model(x, mask)
            torch.mps.synchronize()
            old_check = compare_outputs(expected, old_output, rtol=0.02, atol=0.002)
            current_check = compare_outputs(
                expected, current_output, rtol=0.02, atol=0.002
            )
            accuracy.append(
                {
                    "seed": 1234 + trial,
                    "old_failed_elements": old_check.failed_elements,
                    "current_failed_elements": current_check.failed_elements,
                }
            )

        x, mask = generate_random_case(config, device, dtype, 101234, 0.0, 1.0)
        for _ in range(args.warmup):
            old_model(x, mask)
            current_model(x, mask)
        torch.mps.synchronize()
        old_samples, current_samples = [], []
        for pair in range(args.pairs):
            if pair % 2 == 0:
                old_samples.append(timed_forward(old_model, x, mask))
                current_samples.append(timed_forward(current_model, x, mask))
            else:
                current_samples.append(timed_forward(current_model, x, mask))
                old_samples.append(timed_forward(old_model, x, mask))

    old_median = statistics.median(old_samples)
    current_median = statistics.median(current_samples)
    result = {
        "shape": args.shape,
        "old_revision": args.old_revision,
        "old_implementation": "DispatchTransformer",
        "current_revision": "working-tree",
        "current_implementation": "DispatchTransformer",
        "correct": all(
            item["old_failed_elements"] == 0
            and item["current_failed_elements"] == 0
            for item in accuracy
        ),
        "accuracy": accuracy,
        "old_latency_ms": old_median,
        "current_latency_ms": current_median,
        "old_over_current": old_median / current_median,
        "old_samples_ms": old_samples,
        "current_samples_ms": current_samples,
        "winner": "old" if old_median < current_median else "current",
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if result["correct"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
