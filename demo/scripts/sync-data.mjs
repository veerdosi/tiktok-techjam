import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const out = resolve(root, "demo/public/data/replay.json");
const readJson = (relative) => JSON.parse(readFileSync(resolve(root, relative), "utf8"));

const historicalExperiments = readFileSync(
  resolve(root, "demo/public/data/experiments.jsonl"),
  "utf8",
)
  .trim()
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const summary = readJson("results/final/summary.json");
const finalRows = summary.results;

const policy = {
  1: "Packed QKV + native MPS attention",
  2: "Compiled MLX packed-QKV + cached boundary",
  3: "Compiled MLX packed-QKV + cached boundary",
  4: "Packed QKV + native MPS attention",
  5: "Packed QKV + native MPS attention",
  6: "Batch-384 tiled packed QKV on MPS",
  7: "FP16 linear + FP32 state + padded fused attention",
  8: "FP16 linear + FP32 residual/norm state",
  9: "FP16 linear + FP32 residual/norm state",
  10: "FP16 linear + FP32 residual/norm state",
  11: "FP16 linear + FP32 state + padded fused attention",
  12: "FP16 linear + FP32 residual/norm state",
  13: "Packed QKV + native SDPA",
  14: "FP32 input stream → FP16 fused streaming attention → FP32 state/output digest",
};

const payload = {
  generatedAt: new Date().toISOString(),
  sources: {
    finalResults: "results/final/summary.json",
    shape14: "results/shape14_evidence.json",
    experimentHistory: "demo/public/data/experiments.jsonl (archived project log)",
    dispatcher: "implementations.py: DispatchTransformer",
  },
  finalRows,
  shape14: readJson("results/shape14_evidence.json"),
  experiments: historicalExperiments,
  policy,
};

mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, `${JSON.stringify(payload, null, 2)}\n`);
console.log(`Synced ${finalRows.length} final rows and ${historicalExperiments.length} saved experiments → ${out}`);
