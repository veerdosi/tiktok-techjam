#!/usr/bin/env python3
"""Isolated compute/memory probe for Shape 14 sequence-preserving shards."""

from __future__ import annotations

import argparse
import json
import time

import torch

from benchmark_harness import useful_flops
from implementations import IMPLEMENTATIONS
from original.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--impl", choices=IMPLEMENTATIONS, required=True)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--compare-reference", action="store_true")
    args = parser.parse_args()

    config = TransformerConfig(args.batch, args.seq_len, 1024, 16, 1024, 2, True)
    dtype = getattr(torch, args.dtype)
    device = torch.device("mps")
    torch.manual_seed(1234)
    reference = BaselineTransformer(config)
    candidate = IMPLEMENTATIONS[args.impl](config)
    copy_model_weights(reference, candidate)
    reference = reference.to(device=device, dtype=dtype).eval()
    candidate = candidate.to(device=device, dtype=dtype).eval()
    x, mask = generate_random_case(config, device, dtype, 1234, 0.0, 1.0)

    expected = None
    if args.compare_reference:
        with torch.inference_mode():
            expected = reference(x, mask)
        torch.mps.synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        output = candidate(x, mask)
    torch.mps.synchronize()
    latency_ms = (time.perf_counter_ns() - started) / 1e6
    result = {
        "batch": args.batch,
        "seq_len": args.seq_len,
        "implementation": args.impl,
        "dtype": args.dtype,
        "latency_ms": latency_ms,
        "useful_flops": useful_flops(config),
        "useful_tflops": useful_flops(config) / (latency_ms * 1e9),
        "output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all().item()),
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
    }
    if expected is not None:
        check = compare_outputs(expected, output, rtol=0.02, atol=0.002)
        result["correct"] = check.passed
        result["failed_elements"] = check.failed_elements
        result["max_abs_error"] = check.max_abs_error
        result["max_relative_error"] = check.max_relative_error
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("correct", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())

