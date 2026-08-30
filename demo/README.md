# TechJam Execution Workbench demo

This is the local React/TypeScript interface used for the TechJam demo video.
It does not run benchmarks during the presentation. Instead, it replays saved,
real project evidence so the UI is fast and deterministic.

## Run locally

```bash
cd demo
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://127.0.0.1:5173`).

`npm run dev` first runs `npm run sync:data`. The sync script reads:

- `../results/final/summary.json` (the canonical final measurement artifact)
- `../results/shape14_evidence.json`
- the current dispatcher policy in `../implementations.py`
- `public/data/experiments.jsonl`, the archived 40-entry experiment log

No network is required after `npm install`. The generated local replay snapshot
is `public/data/replay.json`.

## Important honesty boundaries

- Shapes 1–13 are shown as precision PASS from their final saved measurements.
- Shape 14 clearly distinguishes the untouched challenge interface failure
  before `forward()` from the separate streamed exact-logical execution.
- The Shape 14 replay button animates saved tile evidence. It does not claim to
  re-execute the roughly 19-minute workload during the demo.
