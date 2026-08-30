# Tech Report

## Runtime environment

The code and the Codex agent ran on the following machine and software stack:

| Component                   | Measured value                                      |
| --------------------------- | --------------------------------------------------- |
| Machine                     | MacBook Air, model identifier Mac14,2               |
| SoC                         | Apple M2                                            |
| CPU                         | 8 cores: 4 performance + 4 efficiency               |
| GPU                         | 8-core integrated Apple M2 GPU                      |
| Unified memory              | 8 GiB (`hw.memsize=8589934592`)                     |
| Workspace storage           | 228 GiB total                                       |
| MPS recommended working set | 5,726,633,984 bytes (5.33 GiB)                      |
| Metal `maxBufferLength`     | 4,294,967,296 bytes (4.00 GiB)                      |
| Display/Metal report        | Metal 4                                             |
| macOS                       | 26.6.2, build 25G83                                 |
| Architecture                | arm64                                               |
| Python                      | 3.12.14                                             |
| PyTorch                     | 2.13.0                                              |
| MPS                         | built=True, available=True                          |
| MLX                         | 0.32.2                                              |
| NumPy                       | 2.5.2                                               |
| psutil                      | 7.2.2                                               |
| Apple clang                 | 21.0.0 (clang-2100.1.1.101)                         |
| SDK                         | 26.5                                                |
| Xcode                       | Full Xcode not installed; Command Line Tools active |

The benchmark uses Apple Metal Performance Shaders (MPS) and Apple MLX on the
integrated M2 GPU. The original challenge benchmark was preserved unchanged;
the synchronized measurement harness explicitly selects MPS and synchronizes
before and after timed forwards.

## AI tools and LLM

The project was developed with the OpenAI Codex desktop coding
agent, using a GPT-5-Sol model. Codex was used for repository analysis,
implementation, experiment design, benchmarking, debugging, and report
generation.

Supporting development tools included the terminal and Git, PyTorch/MPS,
Apple MLX, Swift/Metal limit probes, and local Python measurement scripts.

## Significant decisions driven by results

- The first SDPA version was slower because a mask `.item()` synchronized MPS every layer. That result caused a direct-mask implementation and a head-dimension-8 specialization.
- Shape 6 batch-64 made the workload runnable; the expanded tile sweep found 384 was best while 512/1024 regressed. QKV fusion produced the retained FP32 path.
- Full-model float16 was rejected despite tiny mean error because exact evaluation found individual failures, including 259 on Shape 6. A later mixed design confined FP16 to linear/attention kernels, retained FP32 residual/norm state, passed the hard gate, and won Shapes 7–12 where dispatched.
- Shape 13 query streaming was correct but far slower than native SDPA, so it remained a memory fallback rather than the winner.
- PyTorch query streaming at the full Shape 14 sequence was memory-safe but did not finish a layer in 13 minutes. This caused a switch to the MLX family; MLX completed full-length attention in 11.72 s and the packed/compiled full shard in 23.94 s.
- Shape 14 fused-attention memory was measured at sequence 25k/50k/100k and found to scale linearly. The full logical batch 32 was then executed through batch-2 tiles in 934.62 s, and the optimized model was checked against the exact reference through sequence 6,592 on three seeds.
- The final Shape 14 interface experiment added true incremental input generation and FP16 memmap output. All 32 exact logical batch elements completed as batch-1 tiles in 1,185.80 s with finite output and 2.892 GB peak active memory; reference-fit validation remained zero-failure.
- The active Shape 14 runner now advances the challenge-style MPS FP32 random stream one batch-1 tile at a time. The exact logical batch 32 × sequence 100,000 workload completed with FP32 input/state, FP16 linear/attention compute, FP32 output digest, 1,169.986 s end-to-end latency, 1.189 useful TFLOP/s, 4.326 GB peak active/cache memory, and finite outputs for all 32 tiles.
- Apple’s published 100 GB/s M2 unified-memory figure is recorded as a device peak. Because the installed PyTorch/MPS profiler exposes CPU activities only and no GPU DRAM-byte counter, achieved bandwidth is not reported.
- MLX did not become a universal backend. Cross-shape measurements retained it only where it won; Shape 9 used a paired alternating test to remove thermal bias.
- Shape 14 remains an official challenge FAIL because its required single MPS input tensor cannot be allocated, while the separate exact-logical chunked path is reported as a PASS with its own latency and output digest. The agent did not substitute cached outputs, skip computation, weaken tolerances, or invent an official score.

## Interaction History

Check out [INTERACTION_HISTORY.md](INTERACTION_HISTORY.md)
The machine-readable final samples are in `results/final/`.
