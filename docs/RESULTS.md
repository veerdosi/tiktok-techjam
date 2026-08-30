# Results

AutoMetal passes the challenge's exact elementwise precision gate on all
ordinary official shapes, with zero failed elements on Shapes 1–13. The final
paths deliver a 4.98× geometric-mean speedup over the challenge reference and
a 4.00× geometric-mean speedup over the strong SDPA baseline. Shape 8 reaches
2.005 useful TFLOP/s and 70.12% estimated FP32 MFU; Shape 13 reaches 7.15×
reference speedup.

Shapes 1–13 pass the challenge's FP32 precision gate:
relative error below 0.02 or absolute error below 0.002 for every element.
Useful TFLOP/s is directly measured.

## Final results

| Shape | Validation status                        |      Latency | Useful TFLOP/s | Est. FP32 MFU | Reference speedup | SDPA speedup | Final path                                        |
| ----: | :--------------------------------------- | -----------: | -------------: | ------------: | ----------------: | -----------: | :------------------------------------------------ |
|     1 | **PASS**                                 |     7.206 ms |          1.044 |        36.51% |             5.37× |        2.91× | packed QKV + native MPS attention                 |
|     2 | **PASS**                                 |     0.324 ms |          0.363 |        12.70% |             8.67× |        7.99× | compiled packed-QKV MLX, cached boundary          |
|     3 | **PASS**                                 |     0.860 ms |          0.547 |        19.13% |             5.01× |       10.12× | compiled packed-QKV MLX, cached boundary          |
|     4 | **PASS**                                 |     2.240 ms |          0.840 |        29.36% |             4.42× |        2.88× | packed QKV + native MPS attention                 |
|     5 | **PASS**                                 |    12.597 ms |          1.195 |        41.77% |             5.35× |        3.31× | packed QKV + native MPS attention                 |
|     6 | **PASS**                                 |      1.144 s |          1.028 |        35.95% |             5.22× |        6.02× | batch-384 + packed QKV MPS                        |
|     7 | **PASS**                                 |     3.561 ms |          0.189 |         6.61% |             5.51× |        6.27× | FP16 linear + FP32 state + padded fused attention |
|     8 | **PASS**                                 |   209.917 ms |          2.005 |        70.12% |             2.58× |        1.61× | FP16 linear + FP32 residual/norm state            |
|     9 | **PASS**                                 |     7.048 ms |          1.068 |        37.33% |             2.77× |        2.59× | FP16 linear + FP32 residual/norm state            |
|    10 | **PASS**                                 |     7.422 ms |          1.014 |        35.45% |             4.28× |        2.82× | FP16 linear + FP32 residual/norm state            |
|    11 | **PASS**                                 |     8.358 ms |          0.900 |        31.48% |             9.27× |        9.53× | FP16 linear + FP32 state + padded fused attention |
|    12 | **PASS**                                 |     1.918 ms |          0.876 |        30.63% |             3.57× |        3.37× | FP16 linear + FP32 residual/norm state            |
|    13 | **PASS**                                 |    95.575 ms |          1.259 |        44.02% |             7.15× |        2.10× | packed QKV + native SDPA                          |
|    14 | **Official FAIL† / reference-fit PASS‡** | 1,169.986 s‡ |         1.189‡ |       41.57%‡ |               N/A |          N/A | FP32 input + FP16 kernels + FP32 state‡           |

Some paths use FP16 kernels internally while retaining FP32 state; the `Final
path` column names those choices. Every retained path on Shapes 1–13 passes the
FP32 gate with zero failed elements.

† The untouched Shape 14 generator requests one contiguous
`[32,100000,1024]` MPS allocation: 12.207 GiB in FP32 or 6.104 GiB in FP16.
It fails before participant `forward()` runs.

‡ The separate Shape 14 path keeps the exact logical dimensions, generates one
batch tile at a time, uses streamed online-softmax attention, and stores output
incrementally. It is reported separately from the challenge-script result.

## How to read the comparison columns

The MFU denominator is 2.86 TFLOP/s, the lower end of the theoretical M2 FP32
range reported in Table 1 of Hübner, Hu, Peng, and Markidis, [“Apple vs.
Oranges: Evaluating the Apple Silicon M-Series SoCs for HPC Performance and
Efficiency”](https://arxiv.org/abs/2502.05317), IEEE IPDPSW 2025. That paper
reports 2.86–3.57 theoretical FP32 TFLOP/s for the M2's 8–10 GPU-core range;
this 8-core M2 uses the 2.86 TFLOP/s endpoint. It is a published theoretical
estimate, not an Apple specification or measured hardware ceiling.

`Reference speedup` compares each path with the challenge's exact
`BaselineTransformer`. `SDPA speedup` compares it with a stronger baseline:
separate PyTorch Q/K/V projections followed by
`torch.nn.functional.scaled_dot_product_attention`, using the same weights,
inputs, masks, causal setting, dtype, and synchronized MPS timing. That baseline
does not include packed QKV, MLX graph compilation, caching, shape-specific
dispatch, tiling, padding, or the mixed-precision state policy.

## Shape 14 progression

Shape 14 keeps the official logical configuration: batch 32, sequence 100,000,
model width 1,024, 16 heads, two layers, and FFN width 1,024.

| Stage                                    | Result                                                                                                                             |
| :--------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| Official monolithic interface            | FP32 input allocation of 12.21 GiB fails before participant `forward()` runs.                                                      |
| First memory-efficient computation       | Batch-2 streaming tiles completed the full logical workload in 934.623 s at 1.489 useful TFLOP/s with 5.760 GB peak active memory. |
| Historical chunked end-to-end run        | Incremental input/output completed in 1,185.799 s at 1.173 useful TFLOP/s with 2.892 GB peak active memory.                        |
| Current official-style FP32 streamed run | All 32 batch-1 tiles completed in 1,169.986 s at 1.189 useful TFLOP/s with 4.326 GB peak active/cache memory.                      |

The current path validates against the exact reference wherever that reference
fits: zero failed elements through B32/S4096 and B1/S6592 across three seeds.
The full 100k-token reference cannot fit, so this is not presented as an
challenge Shape 14 precision PASS. A streamed replacement reference would use
the same memory-saving approach as the optimized path, so it would not be an
independent dense-reference comparison. The commands, output evidence, boundary
probes, and stage-by-stage memory details are in the appendix.

## Appendix

### Stage 1: untouched challenge interface

| Test                           | Result                                                                                 |
| ------------------------------ | -------------------------------------------------------------------------------------- |
| Untouched challenge FP32 input | FAIL in `generate_random_case`: 12.21 GiB; `participant_forward_called: false`         |
| Untouched challenge FP16 input | FAIL in `generate_random_case`: 6.10 GiB; `participant_forward_called: false`          |
| Runtime Metal limit            | `maxBufferLength = 4,294,967,296` bytes; recommended working set `5,726,633,984` bytes |

The challenge generates a contiguous `[32,100000,1024]` input before it calls
participant code. That tensor is 13,107,200,000 bytes in FP32 and
6,553,600,000 bytes in FP16. Metal returns `Invalid buffer size: 12.21 GiB` or
`6.10 GiB` before `forward()` begins.

### Stage 2: first memory-efficient computation

`shape14_chunked_batch.py` preserved the exact sequence length and removed the
quadratic attention allocation with MLX forced-fused causal streaming and
online-softmax attention. It ran the complete logical batch as sixteen batch-2
computation tiles; the historical input was reused between calls.

| Test                                        | Result                                                                                   |
| ------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Packed/compiled batch 1, sequence 100k      | 23.945 s, 1.816 useful TFLOP/s                                                           |
| Direct batch 2, sequence 100k               | 48.390 s, 1.797 useful TFLOP/s                                                           |
| Direct batch 4, sequence 100k               | Completed in 216.089 s, 0.805 useful TFLOP/s                                             |
| Logical batch 32 in batch-2 tiles           | Completed in 934.623 s, finite output, 1.489 useful TFLOP/s, 5.760 GB peak active memory |
| Largest exact-reference validation          | B1/S6592, three seeds, zero failed elements                                              |
| Fused-attention peak at S25k / S50k / S100k | 204.8 / 409.6 / 819.2 MB                                                                 |
| Corresponding dense FP16 score matrix       | 20 / 80 / 320 GB                                                                         |

### Stage 3: historical chunked end-to-end execution

The historical `shape14_end_to_end.py` generated a deterministic MLX random
input for one batch tile, evaluated the same two-layer model, stored that output
tile in an FP16 memmap, released the tile, and continued. Attention stayed
fused, tiled, and online-softmax; the full score matrix was never materialized.

| Test                | Result                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Execution status    | PASS: all 32 batch-1 tiles completed and were finite                                                                |
| Total latency       | 1,185.799 s end to end: 1,118.260 s model, 57.556 s input generation, 6.800 s output storage                        |
| Useful throughput   | 1.173 TFLOP/s end to end; 1.244 TFLOP/s model only                                                                  |
| Memory behavior     | 2.892 GB peak active memory; one 6.5536 GB FP16 output file on disk                                                 |
| Output evidence     | 6,553,600,000 bytes; SHA-256 `3e546d303820ae396bf565a8e4f49f2fca72bb8f2f2cb196cafbb95cd647d2ee`                     |
| Precision status    | PASS on B4/S1024, B8/S2048, B32/S4096, and B1/S6592 across seeds 1234/1235/1236; zero failed elements in every case |
| Change from Stage 2 | 251.176 s slower because input generation and storage became part of timing; 2.867 GB lower peak active memory      |

The explicit reference stops producing trustworthy output at S6624. At S8192,
its FP32 softmax conversion requests one 4.00 GiB buffer. Fused attention grows
linearly with sequence length: 204.8, 409.6, and 819.2 MB at S25k, S50k, and
S100k. A dense score matrix at those lengths would need 20, 80, and 320 GB in
FP16.

### Stage 4: official-style FP32 logical input

The active `shape14_end_to_end.py` advances a persistent
`torch.Generator(device="mps")` and draws challenge-style FP32 input one
batch-1 tile at a time. Each 409.6 MB FP32 tile moves through DLPack, enters the
validated FP16 linear and attention compute path, keeps FP32 residual and
normalization state, then releases before the next batch element. A streaming
FP32 digest records output without allocating a full 13.1072 GB output tensor
on MPS.

| Test                             | Result                                                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Official monolithic FP32 harness | FAIL before participant `forward()`: `Invalid buffer size: 12.21 GiB`; `participant_forward_called: false`       |
| Streamed logical FP32 input      | PASS: 32/32 batch-1 tiles completed and were finite                                                              |
| End-to-end latency               | 1,169.986 s                                                                                                      |
| Useful throughput                | 1.189 TFLOP/s end to end; 1.290 TFLOP/s model only                                                               |
| Peak active memory               | 4.326 GB, including retained allocator cache; one 409.6 MB FP32 input tile at a time                             |
| Logical input/output size        | 13.1072 GB each, represented incrementally rather than as MPS tensors                                            |
| Precision validation             | Zero failed elements for FP32 input at B1/S1024, B4/S1024, B8/S2048, B32/S4096, B1/S5000, B1/S5500, and B1/S5900 |
| Precision boundary probe         | At S6000, two standalone FP32 reference forwards on the same input disagree by 0.555 maximum absolute error      |

Stage 4 uses the official FP32 logical input stream. Its FP16 compute kernels
are a validated internal reduction. The complete 100k-token FP32 reference
cannot fit, so the streamed run is not presented as an challenge Shape 14
precision PASS.

Apple documents [`maxBufferLength`](https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength)
as the largest allocation a device can make for one buffer. Apple’s [Metal
compute guidance](https://developer.apple.com/videos/play/tech-talks/10580/?time=80)
describes querying `recommendedMaxWorkingSetSize` and splitting work across
encoders and resources. Both values are queried locally by `metal_limits.swift`.
