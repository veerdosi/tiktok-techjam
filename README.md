# TikTok TechJam 2026 Track 3 — Apple M2 submission

## Project overview

This project is an Apple-M2-optimized Transformer inference implementation for
the 14 official TechJam shapes. It keeps a per-shape dispatcher so each shape
uses its fastest precision-passing path: packed-QKV/native MPS attention,
compiled MLX execution, bounded batch tiling, head-dimension padding, and
precision-gated mixed FP16/FP32 kernels.

Shapes 1–13 pass the challenge’s elementwise correctness gate with zero failed
elements. Shape 14’s untouched challenge interface fails before participant
`forward()` because its monolithic input allocation is too large, but the exact
logical workload runs through a separate incremental input/output path.

Built solo by Veer Dosi.

Check [docs/RESULTS.md](docs/RESULTS.md) for the complete latency, useful
TFLOP/s, MFU, correctness, Shape 14 progression, and experiment summary. The
required environment/tool report is [docs/TECH_REPORT.md](docs/TECH_REPORT.md).

Important files:

- `final_validate.py`: synchronized fresh-process validation for Shapes 1–13.
- `shape14_interface_probe.py`: reproduces the untouched Shape 14 allocation failure.
- `shape14_end_to_end.py`: exact logical Shape 14 execution with streamed input and output.
- `docs/TECH_REPORT.md`: runtime environment and AI-tool/LLM report.
- `docs/RESULTS.md`: final measurements and Shape 14 results.

### Where the optimized transformer lives

The challenge looks for a class named `UserOptimizedTransformer`, but the
actual implementation is split between the submission wrapper and the
measured dispatcher:

- [`submission.py:6`](submission.py#L6) defines the submitted
  `UserOptimizedTransformer`. Its `pass` is intentional; it inherits the full
  implementation from `DispatchTransformer`.
- [`implementations.py:316`](implementations.py#L316) defines
  `DispatchTransformer`, which owns the shape-specific implementation and
  dispatch policy. Its main [`forward()`](implementations.py#L381) starts at
  line 381.
- [`torch_transformer_benchmark.py:7`](torch_transformer_benchmark.py#L7)
  wires the submitted class into the challenge benchmark at line 11.
- [`original/torch_transformer_benchmark.py:175`](original/torch_transformer_benchmark.py#L175)
  is the untouched challenge reference/placeholder, not the optimized
  submission implementation.

## Setup and installation

The measurements require Apple Silicon with MPS, Python 3.12, and the project
dependencies:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Steps to reproduce the results

1. Run the synchronized official-shape validation:

   ```bash
   .venv/bin/python final_validate.py
   ```

   This writes per-shape measurements to `results/final/` and verifies the
   exact correctness predicate for Shapes 1–13.

2. Reproduce the untouched Shape 14 interface result:

   ```bash
   .venv/bin/python shape14_interface_probe.py
   ```

   It records the expected pre-`forward()` failure for the monolithic FP32 and
   FP16 inputs.

3. Run the exact logical Shape 14 workload incrementally:

   ```bash
   .venv/bin/python shape14_end_to_end.py \
     --batch 32 --batch-chunk 1 --seq-len 100000 \
     --input-dtype float32 --compute-dtype float16 --state-dtype float32 \
     --output-dtype float32 --output-mode digest \
     --log results/shape14_fp32_stage4_final.json
   ```

   This advances the challenge-style FP32 MPS random stream one batch tile at a
   time, uses fused streaming/online-softmax attention, and emits a digest
   without allocating the complete input or output tensor on MPS. The
   historical FP16 Stage 2/Stage 3 results remain recorded in
   `docs/RESULTS.md` and `results/shape14_evidence.json`.

## Limitations and what I would improve with more time

- The full Shape 14 reference cannot fit at sequence length 100,000. The
  chunked algorithm is validated against the exact reference on progressively
  larger reference-fit batch/sequence cases, all with zero failed elements.
- Shape 14 end-to-end input generation and output storage add substantial
  latency. A future custom Metal/MLX streaming kernel could reduce framework
  transitions and improve the batch tile size while retaining bounded memory.
- MPS timing is sensitive to thermal state on this fanless laptop, so results
  use fresh processes, synchronized timing, and per-shape incumbents.
