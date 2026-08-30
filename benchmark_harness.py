#!/usr/bin/env python3
"""Truthful synchronized correctness/performance harness for Apple MPS."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from implementations import BatchChunkedReference, IMPLEMENTATIONS
from original.torch_transformer_benchmark import (
    BaselineTransformer,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
)
from shapes import OFFICIAL_SHAPES


# Published theoretical FP32 peak for the 8-core Apple M2 GPU used here.
# This is a literature estimate, not an official Apple specification.
ESTIMATED_M2_FP32_PEAK_TFLOPS = 2.86


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def useful_flops(config) -> int:
    """Conventional matmul FLOPs, counting only causal attention pairs."""
    b, s, d, f, layers = (
        config.batch_size,
        config.seq_len,
        config.d_model,
        config.ffn_dim,
        config.num_layers,
    )
    projections = 8 * b * s * d * d
    ffn = 4 * b * s * d * f
    attention = (2 * b * d * s * (s + 1)) if config.causal else (4 * b * d * s * s)
    return layers * (projections + ffn + attention)


def timed_samples(model, x, mask, device, warmup: int, repeats: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        synchronize(device)
        samples = []
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter_ns()
            model(x, mask)
            synchronize(device)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def exact_chunked_compare(reference, candidate, rtol: float, atol: float):
    """Apply the challenge's exact predicate in tiles to avoid MPS reduction overflow."""
    if reference.numel() <= 50_000_000:
        return compare_outputs(reference, candidate, rtol=rtol, atol=atol)
    checks = []
    weighted_mean = 0.0
    for start in range(0, reference.shape[0], 64):
        check = compare_outputs(
            reference[start : start + 64], candidate[start : start + 64],
            rtol=rtol, atol=atol,
        )
        checks.append(check)
        weighted_mean += check.mean_abs_error * check.total_elements
    total = sum(check.total_elements for check in checks)
    return SimpleNamespace(
        passed=all(check.passed for check in checks),
        failed_elements=sum(check.failed_elements for check in checks),
        total_elements=total,
        max_abs_error=max(check.max_abs_error for check in checks),
        max_relative_error=max(check.max_relative_error for check in checks),
        mean_abs_error=weighted_mean / total,
    )


def run(args: argparse.Namespace) -> dict:
    config = OFFICIAL_SHAPES[args.shape - 1]
    if args.batch_chunk:
        os.environ["TECHJAM_BATCH_CHUNK"] = str(args.batch_chunk)
    if args.reference_batch_chunk:
        os.environ["TECHJAM_REFERENCE_BATCH_CHUNK"] = str(args.reference_batch_chunk)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    reference_type = BatchChunkedReference if args.shape in (6, 13) else BaselineTransformer
    reference = reference_type(config)
    candidate = IMPLEMENTATIONS[args.impl](config)
    copy_model_weights(reference, candidate)
    reference = reference.to(device=device, dtype=dtype).eval()
    candidate = candidate.to(device=device, dtype=dtype).eval()
    if args.compile_candidate:
        candidate = torch.compile(candidate, mode=args.compile_mode)
    if args.trace_candidate:
        trace_x, trace_mask = generate_random_case(
            config, device, dtype, args.seed, args.padding_ratio, 1.0
        )
        with torch.inference_mode():
            candidate = torch.jit.trace(
                candidate, (trace_x, trace_mask), strict=False
            ).eval()
        synchronize(device)

    accuracy_trials = []
    passed = True
    with torch.inference_mode():
        for trial in range(args.accuracy_trials):
            x, mask = generate_random_case(
                config, device, dtype, args.seed + trial, args.padding_ratio, 1.0
            )
            expected = reference(x, mask)
            actual = candidate(x, mask)
            synchronize(device)
            # Large MPS candidates can leave gigabytes in the caching allocator.
            # Release only after both outputs are complete and before the
            # challenge's unchanged elementwise comparison allocates its masks.
            if device.type == "mps":
                torch.mps.empty_cache()
            check = exact_chunked_compare(
                expected, actual, rtol=args.rtol, atol=args.atol
            )
            passed &= check.passed
            accuracy_trials.append(
                {
                    "passed": check.passed,
                    "failed_elements": check.failed_elements,
                    "total_elements": check.total_elements,
                    "max_abs_error": check.max_abs_error,
                    "max_relative_error": check.max_relative_error,
                    "mean_abs_error": check.mean_abs_error,
                }
            )
            del x, mask, expected, actual

    result = {
        "shape": args.shape,
        "config": asdict(config),
        "implementation": args.impl,
        "device": str(device),
        "dtype": str(dtype),
        "correct": passed,
        "accuracy": accuracy_trials,
        "useful_flops": useful_flops(config),
        "estimated_fp32_peak_tflops": ESTIMATED_M2_FP32_PEAK_TFLOPS,
        "estimated_fp32_mfu": None,
        "estimated_fp32_mfu_note": (
            "Useful FP32 TFLOP/s divided by the published 2.86 TFLOP/s "
            "theoretical estimate; not an official Apple specification."
        ),
    }
    if not passed and not args.benchmark_on_failure:
        return result

    x, mask = generate_random_case(
        config, device, dtype, args.seed + 100_000, args.padding_ratio, 1.0
    )
    candidate_samples = timed_samples(
        candidate, x, mask, device, args.warmup, args.repeats
    )
    reference_samples = None
    if not args.skip_reference_timing:
        reference_samples = timed_samples(
            reference, x, mask, device, args.warmup, args.repeats
        )
    latency_ms = statistics.median(candidate_samples)
    measured_tflops = useful_flops(config) / (latency_ms * 1e9)
    result.update(
        {
            "latency_ms": latency_ms,
            "mean_ms": statistics.fmean(candidate_samples),
            "min_ms": min(candidate_samples),
            "samples_ms": candidate_samples,
            "useful_tflops": measured_tflops,
            "estimated_fp32_mfu": (
                measured_tflops / ESTIMATED_M2_FP32_PEAK_TFLOPS
                if dtype == torch.float32 else None
            ),
        }
    )
    if reference_samples:
        reference_ms = statistics.median(reference_samples)
        result["reference_latency_ms"] = reference_ms
        result["reference_speedup"] = reference_ms / latency_ms
        result["reference_samples_ms"] = reference_samples
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, choices=range(1, 15), required=True)
    parser.add_argument("--impl", choices=IMPLEMENTATIONS, default="reference")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--accuracy-trials", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--skip-reference-timing", action="store_true")
    parser.add_argument("--batch-chunk", type=int)
    parser.add_argument("--reference-batch-chunk", type=int)
    parser.add_argument("--benchmark-on-failure", action="store_true")
    parser.add_argument("--compile-candidate", action="store_true")
    parser.add_argument("--trace-candidate", action="store_true")
    parser.add_argument(
        "--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    outcome = run(arguments)
    rendered = json.dumps(outcome, indent=2)
    print(rendered)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    raise SystemExit(0 if outcome["correct"] else 2)
