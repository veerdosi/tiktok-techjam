#!/usr/bin/env python3
"""Correctness and full-length shard probe for the MLX backend."""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np
import torch

from benchmark_harness import useful_flops
from mlx_backend import MLXTransformer
from original.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    generate_random_case,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fuse-qkv", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    config = TransformerConfig(args.batch, args.seq_len, 1024, 16, 1024, 2, True)
    torch_dtype = getattr(torch, args.dtype)
    mlx_dtype = getattr(mx, args.dtype)
    torch.manual_seed(args.seed)
    reference = BaselineTransformer(config)
    candidate = MLXTransformer(reference, dtype=mlx_dtype, fuse_qkv=args.fuse_qkv)
    candidate_call = mx.compile(candidate) if args.compile else candidate
    reference = reference.to(device="mps", dtype=torch_dtype).eval()
    x, mask = generate_random_case(
        config, torch.device("mps"), torch_dtype, args.seed, 0.0, 1.0
    )
    expected = None
    if args.compare_reference:
        with torch.inference_mode():
            expected = reference(x, mask)
        torch.mps.synchronize()
    x_cpu = x.cpu().float().numpy().astype(np.float16 if args.dtype == "float16" else np.float32)
    x_mlx = mx.array(x_cpu)
    mx.eval(x_mlx)
    started = time.perf_counter_ns()
    output = candidate_call(x_mlx)
    mx.eval(output)
    latency_ms = (time.perf_counter_ns() - started) / 1e6
    result = {
        "batch": args.batch,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "seed": args.seed,
        "fuse_qkv": args.fuse_qkv,
        "compiled": args.compile,
        "latency_ms": latency_ms,
        "useful_tflops": useful_flops(config) / (latency_ms * 1e9),
        "output_shape": list(output.shape),
    }
    if expected is not None:
        output_torch = torch.from_numpy(np.asarray(output).copy()).to("mps")
        check = compare_outputs(expected, output_torch, rtol=0.02, atol=0.002)
        result.update(
            correct=check.passed,
            failed_elements=check.failed_elements,
            max_abs_error=check.max_abs_error,
            max_relative_error=check.max_relative_error,
        )
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("correct", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
