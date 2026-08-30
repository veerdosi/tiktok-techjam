#!/usr/bin/env python3
"""Measure fused causal-attention memory at real Shape-14 sequence lengths."""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    shape = (args.batch, 16, args.seq_len, 64)
    q = mx.zeros(shape, dtype=mx.float16)
    k = mx.zeros(shape, dtype=mx.float16)
    v = mx.zeros(shape, dtype=mx.float16)
    mx.eval(q, k, v)
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    started = time.perf_counter_ns()
    output = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=64**-0.5, mask="causal", force_fused=True
    )
    mx.eval(output)
    latency_ms = (time.perf_counter_ns() - started) / 1e6
    peak = mx.get_peak_memory()
    dense_fp16_bytes = args.batch * 16 * args.seq_len * args.seq_len * 2
    result = {
        "batch": args.batch,
        "seq_len": args.seq_len,
        "dtype": "float16",
        "latency_ms": latency_ms,
        "baseline_active_bytes": baseline,
        "peak_active_bytes": peak,
        "peak_over_baseline_bytes": max(0, peak - baseline),
        "dense_attention_matrix_bytes": dense_fp16_bytes,
        "peak_to_dense_matrix_ratio": peak / dense_fp16_bytes,
        "output_shape": list(output.shape),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
