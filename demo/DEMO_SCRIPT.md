# AutoMetal: 3:15 demo script

## 0:00–0:20 Introduction

Open **Workloads**.

Say: “Hi, I’m Veer Dosi. This is AutoMetal, my submission for Track 3 of
TikTok TechJam 2026. I built it for the official Transformer shape set on this
8 GB 2022 M2 MacBook Air. The goal was not one generic
Transformer kernel but rather to find the fastest correct route for every shape. This right here is my dashboard that i used to monitor exepriments and results. ”

## 0:20–0:48 The workload set

Point to the 14 workload tiles and the summary strip.

Say: “Each tile is an official workload. The middle label is its input tensor
shape: batch, sequence length, and model width. Clicking a tile opens its
retained path and the measured result. I benchmarked the available routes for
each shape and kept the fastest one that passed precision test.

Across the ordinary benchmark set, AutoMetal is about five times faster than
the challenge reference and about four times faster than the strong SDPA
baseline. The FLOP-weighted MFU across all 14 logical workloads is about
42 percent.”

## 0:48–1:15 Shape 8

Select **Shape 8**.

Say: “Shape 8 is the large compute case. It finishes in about 210 milliseconds
at about 2 useful TFLOP/s, with 70 percent estimated FP32 MFU. This gives a 2.6x speedup over reference.

The winning route uses reduced-precision linear and attention compute while
keeping the residual and normalization state in FP32. That split matters here:
it reduces the expensive matrix work without losing the precision gate.”

## 1:15–1:45 The experiment trail

Open **Search log** with Shape 8 selected.

Say: “This is the archived search trail behind the final dispatcher. Each entry
keeps the hypothesis, the implementation change, the decision, and what the
experiment taught me. The log is not a performance leaderboard; it records why
a route was retained or rejected.

Click **Show discarded trials**.

I kept the discarded trials too. Some were slower, some missed the tolerance,
and some helped one shape while hurting another. That is why the final
implementation dispatches by shape instead of forcing one strategy on every
workload.”

## 1:45–2:12 The dispatcher

Open **Policy**.

Say: “This is the policy that came out of those measurements. Small shapes take
a launch-conscious compiled route. Shapes with wide matrix multiplies use the
mixed-precision path. Shape 13 takes native SDPA for its long sequence.
Shape 14 uses streamed execution.

The important point is that this is still one submission. The dispatcher checks
the shape it receives and selects among five measured routes. That lets a tiny
launch-bound workload and a 100,000-token workload both use a path that makes
sense for their regime.”

## 2:12–2:57 Shape 14

Open **Shape 14**.

Say: “Shape 14 is the stress case: batch 32, sequence length 100,000, model
width 1,024, and two layers. The official FP32 interface creates one roughly
12 GB input tensor. That request is larger than this environment can represent,
so the challenge harness fails before my forward function starts.”

Follow the three numbered stages.

Say: “First, I used streaming online-softmax attention. It removes the full
quadratic attention matrix, and the complete logical batch can then run. Next,
I streamed the input too: generate one FP32 batch tile, run its complete
100,000-token sequence, release it, and continue with the next tile.”

Point to the final stage and the metrics.

Say: “The final run completed all 32 tiles at about 1.2 TFLOP/s end-to-end with
about 4.3 GB reported peak memory. The full 100,000-token reference does not
fit on this machine, so I compared the same streamed algorithm against the exact reference at the
largest cases that fit, up to about 6,600 tokens. Every comparison passed with
zero failed elements.”

## 2:57–3:15 Close

Return to **Workloads**.

Say: “I would like to thank TIktok for organizing this amazing hackathon and challenge and giving me a chance to improve my GPU skills”
