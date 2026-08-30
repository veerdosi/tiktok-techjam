#!/usr/bin/env python3
"""Execute a logical Shape-14 batch using representable sequence-preserving tiles."""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from benchmark_harness import useful_flops
from mlx_backend import MLXTransformer
from original.torch_transformer_benchmark import BaselineTransformer, TransformerConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--batch-chunk", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=100_000)
    args = parser.parse_args()
    if args.batch % args.batch_chunk:
        raise SystemExit("batch must be divisible by batch-chunk for this probe")

    chunk_config = TransformerConfig(
        args.batch_chunk, args.seq_len, 1024, 16, 1024, 2, True
    )
    logical_config = TransformerConfig(args.batch, args.seq_len, 1024, 16, 1024, 2, True)
    torch_model = BaselineTransformer(chunk_config)
    model = mx.compile(MLXTransformer(torch_model, dtype=mx.float16, fuse_qkv=True))
    x = mx.zeros((args.batch_chunk, args.seq_len, 1024), dtype=mx.float16)
    mx.eval(x)
    mx.reset_peak_memory()

    calls = args.batch // args.batch_chunk
    started = time.perf_counter_ns()
    output = None
    for _ in range(calls):
        output = model(x)
        mx.eval(output)
    latency_ms = (time.perf_counter_ns() - started) / 1e6
    assert output is not None
    result = {
        "logical_batch": args.batch,
        "batch_chunk": args.batch_chunk,
        "calls": calls,
        "seq_len": args.seq_len,
        "dtype": "float16",
        "latency_ms": latency_ms,
        "useful_tflops": useful_flops(logical_config) / (latency_ms * 1e9),
        "peak_active_bytes": mx.get_peak_memory(),
        "finite": bool(mx.all(mx.isfinite(output)).item()),
        "output_shape_per_chunk": list(output.shape),
    }
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
