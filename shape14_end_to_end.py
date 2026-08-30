#!/usr/bin/env python3
"""Incremental official-FP32 Shape-14 generation, execution, and output storage.

The active path advances the challenge-style MPS RNG one logical batch tile at
a time. Historical FP16 MLX-key runs remain in the recorded evidence files and
are intentionally not another active input interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from benchmark_harness import useful_flops
from mlx_backend import MLXTransformer
from original.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
)


def milliseconds(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1e6


def generate_official_mps_tile(generator, batch_size: int, seq_len: int, dtype):
    """Draw the same FP32/FP16 tensor family as the challenge, one tile at a time."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("official-mps input generation requires an available MPS device")
    return torch.randn(
        batch_size,
        seq_len,
        1024,
        generator=generator,
        device=torch.device("mps"),
        dtype=dtype,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--batch-chunk", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--input-dtype", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument(
        "--compute-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument(
        "--state-dtype", choices=("float16", "float32"), default="float32",
        help="storage dtype for residuals, layer norms, and model outputs",
    )
    parser.add_argument(
        "--output-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument(
        "--output-mode", choices=("memmap", "digest", "none"), default="digest"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/shape14_output_fp16.mmap"),
    )
    parser.add_argument("--log", type=Path)
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--clear-cache-between-tiles", action="store_true")
    args = parser.parse_args()
    if args.batch <= 0 or args.batch_chunk <= 0 or args.seq_len <= 0:
        raise SystemExit("batch, batch-chunk, and seq-len must be positive")
    if args.batch % args.batch_chunk:
        raise SystemExit("batch must be divisible by batch-chunk")

    chunk_config = TransformerConfig(
        args.batch_chunk, args.seq_len, 1024, 16, 1024, 2, True
    )
    logical_config = TransformerConfig(
        args.batch, args.seq_len, 1024, 16, 1024, 2, True
    )
    torch.manual_seed(args.seed)
    torch_model = BaselineTransformer(chunk_config)
    input_torch_dtype = getattr(torch, args.input_dtype)
    compute_mlx_dtype = getattr(mx, args.compute_dtype)
    state_mlx_dtype = getattr(mx, args.state_dtype)
    mixed_linear_fp16 = (
        args.state_dtype == "float32" and args.compute_dtype == "float16"
    )
    mlx_model = mx.compile(
        MLXTransformer(
            torch_model,
            dtype=state_mlx_dtype,
            fuse_qkv=True,
            mixed_linear_fp16=mixed_linear_fp16,
            force_fused_attention=mixed_linear_fp16,
        )
    )
    reference = None
    if args.compare_reference:
        reference = torch_model.to(device="mps", dtype=input_torch_dtype).eval()

    official_generator = torch.Generator(device="mps")
    official_generator.manual_seed(args.seed)

    output_np_dtype = np.dtype(args.output_dtype)
    output_store = None
    if args.output_mode == "memmap":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_store = np.memmap(
            args.output,
            mode="w+",
            dtype=output_np_dtype,
            shape=(args.batch, args.seq_len, 1024),
        )

    digest = hashlib.sha256()
    failed_elements = 0
    validation_elements = 0
    max_abs_error = 0.0
    max_relative_error = 0.0
    tiles = []
    input_generation_ms = 0.0
    model_ms = 0.0
    output_store_ms = 0.0
    reference_validation_ms = 0.0
    all_finite = True
    mx.reset_peak_memory()
    end_to_end_started = time.perf_counter_ns()

    for start in range(0, args.batch, args.batch_chunk):
        end = start + args.batch_chunk
        started = time.perf_counter_ns()
        assert official_generator is not None
        source_torch = generate_official_mps_tile(
            official_generator, args.batch_chunk, args.seq_len, input_torch_dtype
        )
        torch.mps.synchronize()
        input_tile = mx.from_dlpack(source_torch)
        mx.eval(input_tile)
        generation_elapsed = milliseconds(started)

        # The official input remains FP32 when requested.  The model may use a
        # validated reduced-precision compute tile, but the cast is explicit and
        # separately accounted for in the per-tile input-generation time.
        if input_tile.dtype != compute_mlx_dtype:
            x = input_tile.astype(compute_mlx_dtype)
            mx.eval(x)
        else:
            x = input_tile
        input_dtype_bytes = (
            (end - start) * args.seq_len * 1024
            * (4 if args.input_dtype == "float32" else 2)
        )
        generation_elapsed = milliseconds(started)
        input_generation_ms += generation_elapsed

        expected = None
        validation_elapsed = 0.0
        if reference is not None:
            started = time.perf_counter_ns()
            x_torch = source_torch
            with torch.inference_mode():
                expected = reference(
                    x_torch,
                    torch.ones(
                        (args.batch_chunk, args.seq_len),
                        dtype=torch.bool,
                        device="mps",
                    ),
            )
            torch.mps.synchronize()
            validation_elapsed += milliseconds(started)
            torch.mps.empty_cache()

        started = time.perf_counter_ns()
        output = mlx_model(x)
        mx.eval(output)
        forward_elapsed = milliseconds(started)
        model_ms += forward_elapsed
        tile_finite = bool(mx.all(mx.isfinite(output)).item())
        all_finite &= tile_finite

        if reference is not None:
            started = time.perf_counter_ns()
            assert expected is not None
            actual = torch.from_numpy(np.asarray(output).copy()).to("mps")
            torch.mps.synchronize()
            check = compare_outputs(expected, actual, rtol=0.02, atol=0.002)
            validation_elapsed += milliseconds(started)
            reference_validation_ms += validation_elapsed
            failed_elements += check.failed_elements
            validation_elements += check.total_elements
            max_abs_error = max(max_abs_error, check.max_abs_error)
            max_relative_error = max(max_relative_error, check.max_relative_error)
            del expected, actual
            torch.mps.empty_cache()

        started = time.perf_counter_ns()
        host_output = np.asarray(output)
        if host_output.dtype != output_np_dtype:
            host_output = host_output.astype(output_np_dtype, copy=False)
        if output_store is not None:
            output_store[start:end] = host_output
        if args.output_mode != "none":
            digest.update(memoryview(host_output).cast("B"))
        store_elapsed = milliseconds(started)
        output_store_ms += store_elapsed

        tile = {
            "start": start,
            "end": end,
            "input_generation_ms": generation_elapsed,
            "input_dtype": args.input_dtype,
            "input_bytes": int(input_dtype_bytes),
            "model_ms": forward_elapsed,
            "output_store_ms": store_elapsed,
            "reference_validation_ms": validation_elapsed,
            "finite": tile_finite,
            "active_memory_bytes": mx.get_active_memory(),
            "cache_memory_bytes": mx.get_cache_memory(),
        }
        tiles.append(tile)
        print(json.dumps({"progress": tile}), flush=True)
        del host_output, output, x, input_tile
        del source_torch
        if args.clear_cache_between_tiles:
            mx.clear_cache()

    if output_store is not None:
        started = time.perf_counter_ns()
        output_store.flush()
        output_store_ms += milliseconds(started)
    end_to_end_ms = milliseconds(end_to_end_started)
    logical_flops = useful_flops(logical_config)
    result = {
        "logical_config": {
            "batch": args.batch,
            "seq_len": args.seq_len,
            "d_model": 1024,
            "heads": 16,
            "ffn_dim": 1024,
            "layers": 2,
            "causal": True,
        },
        "batch_chunk": args.batch_chunk,
        "sequence_execution": "MLX forced-fused causal streaming/online softmax",
        "input_dtype": args.input_dtype,
        "compute_dtype": args.compute_dtype,
        "state_dtype": args.state_dtype,
        "mixed_linear_fp16": mixed_linear_fp16,
        "output_dtype": args.output_dtype,
        "input_source": "official-mps",
        "seed": args.seed,
        "input_generation_strategy": "persistent torch.Generator on MPS, tile-by-tile",
        "output_mode": args.output_mode,
        "output_path": str(args.output) if output_store is not None else None,
        "output_bytes": args.batch * args.seq_len * 1024 * output_np_dtype.itemsize,
        "output_sha256": digest.hexdigest() if args.output_mode != "none" else None,
        "finite": all_finite,
        "correct": failed_elements == 0 if reference is not None else None,
        "failed_elements": failed_elements if reference is not None else None,
        "validation_elements": validation_elements if reference is not None else None,
        "max_abs_error": max_abs_error if reference is not None else None,
        "max_relative_error": max_relative_error if reference is not None else None,
        "end_to_end_latency_ms": end_to_end_ms,
        "model_latency_ms": model_ms,
        "input_generation_ms": input_generation_ms,
        "output_store_ms": output_store_ms,
        "reference_validation_ms": reference_validation_ms,
        "useful_tflops_end_to_end": logical_flops / (end_to_end_ms * 1e9),
        "useful_tflops_model_only": logical_flops / (model_ms * 1e9),
        "peak_active_bytes": mx.get_peak_memory(),
        "tiles": tiles,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(rendered + "\n")
    return 0 if all_finite and result.get("correct", True) is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
