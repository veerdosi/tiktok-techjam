#!/usr/bin/env python3
"""Build a conservative bandwidth/bottleneck analysis from final measurements.

PyTorch 2.13 on this installation exposes CPU-only profiler activities for
MPS, so it cannot report GPU DRAM counters.  The report therefore keeps actual
achieved bandwidth as null and separately records a reproducible one-pass
tensor-traffic proxy: one read of each model parameter plus one FP32 input and
one FP32 output pass.  The proxy is not presented as measured DRAM traffic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


M2_UNIFIED_MEMORY_BANDWIDTH_GBPS = 100.0
M2_BANDWIDTH_SOURCE = (
    "https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-"
    "performance-and-capabilities/"
)
MIXED_SHAPES = {7, 8, 9, 10, 11, 12}
BOTTLENECKS = {
    1: ("MIXED / UNCLEAR", "Small GEMMs and attention; no device counter separates launch from memory traffic."),
    2: ("DISPATCH/LAUNCH-BOUND", "Whole-graph compilation and boundary caching produced the measured win."),
    3: ("DISPATCH/LAUNCH-BOUND", "Whole-graph compilation and boundary caching produced the measured win."),
    4: ("MIXED / UNCLEAR", "Small-batch MPS kernels; packed projections help, but no GPU counter is available."),
    5: ("MIXED / UNCLEAR", "Larger small-dimension GEMMs; packed QKV wins, but launch and compute effects are coupled."),
    6: ("MIXED / UNCLEAR", "Batch-tile sweep shows a GEMM/workspace knee rather than a single bandwidth limit."),
    7: ("DISPATCH/LAUNCH-BOUND", "Head-dimension padding is required to unlock the fused attention kernel."),
    8: ("COMPUTE-BOUND", "The wide GEMM regime reaches the highest measured MFU and benefits from FP16 compute."),
    9: ("MIXED / UNCLEAR", "Small-dimension mixed kernels; MLX/PyTorch differences are measurable but not a DRAM counter."),
    10: ("MIXED / UNCLEAR", "Small-dimension mixed kernels; MLX/PyTorch differences are measurable but not a DRAM counter."),
    11: ("DISPATCH/LAUNCH-BOUND", "Head-dimension padding changes the attention dispatch regime and dominates the measured win."),
    12: ("MIXED / UNCLEAR", "Short sequence and small GEMMs leave launch and compute effects coupled."),
    13: ("COMPUTE-BOUND", "Native SDPA beats explicit query streaming and the long-sequence path is attention/GEMM heavy."),
    14: ("MIXED / UNCLEAR", "FP32 input streaming, FP16 compute, fused attention, and output digest/storage all contribute."),
}


def parameter_bytes(config: dict, mixed: bool) -> int:
    d = int(config["d_model"])
    f = int(config["ffn_dim"])
    layers = int(config["num_layers"])
    linear_parameters = layers * (4 * d * d + 2 * d * f + 5 * d + f)
    norm_parameters = layers * 4 * d + 2 * d
    linear_bytes = 2 if mixed else 4
    return linear_parameters * linear_bytes + norm_parameters * 4


def analyze_row(shape: int, config: dict, latency_ms: float, useful_tflops: float, source: str) -> dict:
    mixed = shape in MIXED_SHAPES or shape == 14
    batch = int(config["batch_size"] if "batch_size" in config else config["batch"])
    seq_len = int(config["seq_len"])
    d_model = int(config["d_model"])
    parameter_bytes_value = parameter_bytes(config, mixed)
    # Official external inputs/outputs are FP32, including the new Shape 14 path.
    boundary_bytes = 2 * batch * seq_len * d_model * 4
    proxy_bytes = parameter_bytes_value + boundary_bytes
    proxy_gbps = proxy_bytes / (latency_ms / 1000.0) / 1e9
    bottleneck, evidence = BOTTLENECKS[shape]
    return {
        "shape": shape,
        "source": source,
        "precision": "mixed (FP32 state/input, FP16 linear/attention)" if mixed else "FP32",
        "latency_ms": latency_ms,
        "useful_tflops": useful_tflops,
        "estimated_fp32_mfu": None if shape == 14 else useful_tflops / 2.86,
        "estimated_fp32_mfu_note": (
            "Not directly comparable because this stress path uses FP16 linear/attention compute."
            if shape == 14 else None
        ),
        "achieved_bandwidth_gbps": None,
        "bandwidth_utilization_pct": None,
        "bandwidth_status": "N/A: no MPS GPU DRAM counter is exposed by the installed profiler",
        "one_pass_tensor_traffic_bytes": proxy_bytes,
        "one_pass_tensor_traffic_proxy_gbps": proxy_gbps,
        "one_pass_tensor_traffic_proxy_utilization_pct": proxy_gbps,
        "bottleneck_type": bottleneck,
        "classification_evidence": evidence,
        "bandwidth_reference_gbps": M2_UNIFIED_MEMORY_BANDWIDTH_GBPS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("results/final/bandwidth_analysis.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    for shape in range(1, 14):
        result = json.loads((root / "results" / "final" / f"shape_{shape:02d}.json").read_text())
        rows.append(
            analyze_row(
                shape,
                result["config"],
                float(result["latency_ms"]),
                float(result["useful_tflops"]),
                "results/final/shape_%02d.json" % shape,
            )
        )

    shape14 = json.loads((root / "results" / "shape14_fp32_stage4_final.json").read_text())
    config14 = {
        "batch_size": shape14["logical_config"]["batch"],
        "seq_len": shape14["logical_config"]["seq_len"],
        "d_model": shape14["logical_config"]["d_model"],
        "ffn_dim": shape14["logical_config"]["ffn_dim"],
        "num_layers": shape14["logical_config"]["layers"],
    }
    rows.append(
        analyze_row(
            14,
            config14,
            float(shape14["end_to_end_latency_ms"]),
            float(shape14["useful_tflops_end_to_end"]),
            "results/shape14_fp32_stage4_final.json",
        )
    )
    output = {
        "bandwidth_reference_gbps": M2_UNIFIED_MEMORY_BANDWIDTH_GBPS,
        "bandwidth_reference_source": M2_BANDWIDTH_SOURCE,
        "measurement_limit": (
            "Actual achieved GPU DRAM bandwidth is unavailable: torch.profiler "
            "reports CPU-only activities for this PyTorch/MPS installation."
        ),
        "proxy_definition": (
            "One read of every model parameter plus one FP32 input read and one FP32 "
            "output write; this is a reproducible tensor-traffic proxy, not measured DRAM traffic."
        ),
        "rows": rows,
    }
    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
