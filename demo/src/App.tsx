import { useEffect, useMemo, useState, type ReactNode } from "react";

type View = "overview" | "history" | "policy" | "shape14";
type Decision = "keep" | "reject" | "investigate";

type FinalRow = {
  shape: number;
  config: {
    batch_size: number;
    seq_len: number;
    d_model: number;
    num_heads: number;
    ffn_dim: number;
    num_layers: number;
  };
  correct: boolean;
  latency_ms: number;
  useful_tflops: number;
  estimated_fp32_mfu: number | null;
  reference_latency_ms?: number;
  reference_speedup?: number;
  strong_sdpa_latency_ms?: number;
  sdpa_speedup?: number;
};

type Experiment = {
  experiment_id: string;
  targeted_shapes: number[];
  hypothesis: string;
  implementation_change: string;
  decision: Decision;
  lesson_learned: string;
  results: Array<{
    shape: number;
    correct?: boolean | null;
    latency_ms?: number | string | null;
    useful_tflops?: number | null;
    error?: string;
  }>;
};

type Dataset = {
  finalRows: FinalRow[];
  shape14: Record<string, any>;
  experiments: Experiment[];
  policy: Record<number, string>;
  sources: Record<string, string>;
};

const nav: Array<{ id: View; label: string; glyph: string }> = [
  { id: "overview", label: "Workloads", glyph: "01" },
  { id: "history", label: "Search log", glyph: "02" },
  { id: "policy", label: "Policy", glyph: "03" },
  { id: "shape14", label: "Shape 14", glyph: "14" },
];

const formatLatency = (ms: number | null | undefined) => {
  if (ms == null) return "—";
  if (ms >= 1000) return `${(ms / 1000).toFixed(ms > 10000 ? 3 : 2)} s`;
  return `${ms.toFixed(3)} ms`;
};

const formatTF = (value: number | null | undefined) =>
  value == null ? "—" : `${value.toFixed(3)} TFLOP/s`;

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function Status({
  children,
  tone = "pass",
}: {
  children: ReactNode;
  tone?: "pass" | "fail" | "neutral" | "keep" | "reject";
}) {
  return <span className={`status status--${tone}`}>{children}</span>;
}

function App() {
  const [data, setData] = useState<Dataset | null>(null);
  const [view, setView] = useState<View>("overview");
  const [selectedShape, setSelectedShape] = useState(8);

  useEffect(() => {
    fetch("/data/replay.json")
      .then((result) => result.json())
      .then(setData)
      .catch(() => setData(null));
  }, []);

  const comparison = useMemo(
    () =>
      new Map(
        data?.finalRows.map((row) => [row.shape, row]) ?? [],
      ),
    [data],
  );
  const selectedExperiments = useMemo(
    () =>
      data?.experiments
        .filter((entry) => entry.targeted_shapes.includes(selectedShape))
        .slice(-7)
        .reverse() ?? [],
    [data, selectedShape],
  );

  const chooseShape = (shape: number, next: View = "overview") => {
    setSelectedShape(shape);
    setView(shape === 14 && next === "overview" ? "shape14" : next);
  };

  if (!data) {
    return (
      <main className="loading">
        <span className="signal" />
        Loading saved benchmark evidence…
      </main>
    );
  }

  return (
    <div className="app-shell">
      <aside className="rail" aria-label="Primary navigation">
        <button
          className="wordmark"
          onClick={() => setView("overview")}
          aria-label="Open AutoMetal workload overview"
        >
          <img src="/assets/automatal-mark.png" alt="" />
        </button>
        <div className="rail-line" />
        <nav>
          {nav.map((item) => (
            <button
              key={item.id}
              className={view === item.id ? "rail-item is-active" : "rail-item"}
              onClick={() => setView(item.id)}
            >
              <b>{item.glyph}</b>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">TikTok TechJam 2026 / Track 3</p>
            <h1>AutoMetal</h1>
            <p className="project-subtitle">
              Transformer execution workbench · Veer Dosi
            </p>
          </div>
          <div className="topbar-actions">
            <span className="machine">
              <i />
              M2 MacBook Air · 8 GB
            </span>
            <a
              className="github-link"
              href="https://github.com/veerdosi/tiktok-techjam"
              target="_blank"
              rel="noreferrer"
            >
              GitHub ↗
            </a>
          </div>
        </header>

        {view === "overview" && (
          <Overview
            data={data}
            comparison={comparison}
            selectedShape={selectedShape}
            onSelect={chooseShape}
            onInspect={() => setView("history")}
          />
        )}
        {view === "history" && (
          <History
            data={data}
            selectedShape={selectedShape}
            experiments={selectedExperiments}
            onSelect={chooseShape}
          />
        )}
        {view === "policy" && <Policy onSelect={chooseShape} />}
        {view === "shape14" && <Shape14 evidence={data.shape14} />}
      </main>
    </div>
  );
}

function Overview({
  data,
  comparison,
  selectedShape,
  onSelect,
  onInspect,
}: {
  data: Dataset;
  comparison: Map<number, FinalRow>;
  selectedShape: number;
  onSelect: (shape: number) => void;
  onInspect: () => void;
}) {
  const selected = data.finalRows.find((row) => row.shape === selectedShape);
  return (
    <section className="view reveal is-in">
      <div className="overview-intro">
        <div>
          <h2>Every official workload has a completion path.</h2>
        </div>
      </div>
      <div className="summary-strip">
        <Metric
          label="official workloads"
          value="14 / 14 PASS"
          detail="Shape 14 via streamed logical execution"
        />
        <Metric
          label="geometric mean"
          value="4.98×"
          detail="over challenge reference"
        />
        <Metric
          label="peak MFU"
          value="70.1%"
          detail="Shape 8 · estimated FP32"
        />
        <Metric
          label="FLOP-weighted MFU"
          value="41.56%"
          detail="all 14 logical workloads"
        />
      </div>
      <div className="overview-layout">
        <div>
          <div className="atlas-head">
            <span>OFFICIAL SHAPE SET</span>
            <span>select a workload</span>
          </div>
          <div
            className="shape-board"
            role="list"
            aria-label="All 14 official workloads"
          >
            {data.finalRows.map((row) => (
              <button
                key={row.shape}
                role="listitem"
                onClick={() => onSelect(row.shape)}
                className={`board-shape ${selectedShape === row.shape ? "is-selected" : ""}`}
              >
                <b>{row.shape}</b>
                <span>
                  [{row.config.batch_size}, {row.config.seq_len}, {row.config.d_model}]
                </span>
                <small>{formatLatency(row.latency_ms)}</small>
              </button>
            ))}
            <button
              role="listitem"
              onClick={() => onSelect(14)}
              className="board-shape"
            >
              <b>14</b>
              <span>[32, 100000, 1024]</span>
              <small>1,169.986 s</small>
            </button>
          </div>
          <div className="legend">
            <button onClick={onInspect}>Read the experiment log →</button>
          </div>
        </div>
        <Inspector
          key={selectedShape}
          selected={selected}
          comparison={comparison.get(selectedShape)}
          policy={data.policy[selectedShape]}
        />
      </div>
    </section>
  );
}

function History({
  data,
  selectedShape,
  experiments,
  onSelect,
}: {
  data: Dataset;
  selectedShape: number;
  experiments: Experiment[];
  onSelect: (shape: number, view?: View) => void;
}) {
  const [showDiscarded, setShowDiscarded] = useState(false);
  const retained = experiments.filter((entry) => entry.decision === "keep");
  const discarded = experiments.filter((entry) => entry.decision !== "keep");
  const visible = showDiscarded ? [...retained, ...discarded] : retained;
  return (
    <section className="view history-view reveal is-in">
      <div className="section-heading">
        <div>
          <h2>Hypothesis becomes policy only after the gate.</h2>
        </div>
      </div>
      <div
        className="shape-picker"
        aria-label="Select a shape for its experiment history"
      >
        {Array.from({ length: 14 }, (_, index) => index + 1).map((shape) => (
          <button
            key={shape}
            onClick={() =>
              onSelect(shape, shape === 14 ? "shape14" : "history")
            }
            className={shape === selectedShape ? "is-active" : ""}
          >
            Shape {shape}
          </button>
        ))}
      </div>
      <div className="pipeline">
        <span>profile</span>
        <i>→</i>
        <span>hypothesis</span>
        <i>→</i>
        <span>implement</span>
        <i>→</i>
        <span>precision</span>
        <i>→</i>
        <span>benchmark</span>
        <i>→</i>
        <span>keep / reject</span>
      </div>
      {experiments.length > 0 && (
        <div className="history-context">
          <p>
            <b>Retained evidence first.</b> These are the trials that informed
            the final implementation.
          </p>
          {discarded.length > 0 && (
            <button onClick={() => setShowDiscarded((current) => !current)}>
              {showDiscarded
                ? "Hide discarded trials"
                : `Show ${discarded.length} discarded trials`}
            </button>
          )}
        </div>
      )}
      <div className="experiment-list">
        {visible.length ? (
          visible.map((entry) => (
            <ExperimentRow
              key={entry.experiment_id}
              entry={entry}
            />
          ))
        ) : (
          <p className="empty">
            No retained archived experiment entry targets this shape.
          </p>
        )}
      </div>
    </section>
  );
}

function ExperimentRow({ entry }: { entry: Experiment }) {
  const tone =
    entry.decision === "keep"
      ? "keep"
      : entry.decision === "reject"
        ? "reject"
        : "neutral";
  return (
    <article className="experiment-row">
      <div className="experiment-index">
        <span>{entry.experiment_id}</span>
        <Status tone={tone}>{entry.decision.toUpperCase()}</Status>
      </div>
      <div className="experiment-copy">
        <p className="experiment-hypothesis">{entry.hypothesis}</p>
        <p>
          <b>change</b> {entry.implementation_change}
        </p>
        <p className="lesson">{entry.lesson_learned}</p>
      </div>
    </article>
  );
}

function Policy({ onSelect }: { onSelect: (shape: number) => void }) {
  const groups = [
    {
      name: "Small and launch-sensitive",
      shapes: [2, 3, 7, 11],
      next: "compiled MLX / padded fused attention",
    },
    {
      name: "Wide GEMM",
      shapes: [8],
      next: "FP16 linear kernels + FP32 state",
    },
    {
      name: "General MPS workloads",
      shapes: [1, 4, 5, 6, 9, 10, 12],
      next: "packed QKV / tiled MPS or MLX",
    },
    { name: "Long sequence", shapes: [13], next: "packed QKV + native SDPA" },
    {
      name: "Extreme sequence",
      shapes: [14],
      next: "streamed FP32 input + online attention",
    },
  ];
  return (
    <section className="view policy-view reveal is-in">
      <div className="section-heading">
        <div>
          <h2>One dispatcher. Five measured routes.</h2>
        </div>
      </div>
      <div className="policy-grid">
        {groups.map((group) => (
          <article className="policy-lane" key={group.name}>
            <header>
              <span>{group.name}</span>
              <i>→</i>
              <strong>{group.next}</strong>
            </header>
            <div className="policy-shapes">
              {group.shapes.map((shape) => (
                <button
                  key={shape}
                  onClick={() => onSelect(shape)}
                  aria-label={`Open Shape ${shape} results`}
                >
                  <b>{shape}</b>
                </button>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function Inspector({
  selected,
  comparison,
  policy,
}: {
  selected?: FinalRow;
  comparison?: FinalRow;
  policy?: string;
}) {
  if (!selected) return null;
  const isShape8 = selected.shape === 8;
  const isShape13 = selected.shape === 13;
  return (
    <aside
      className="inspector"
      aria-label="Shape inspector"
      aria-live="polite"
    >
      <div className="inspector-top">
        <p className="eyebrow">Selected workload</p>
        <span>#{selected.shape}</span>
      </div>
      <h3>
        {isShape8
          ? "Wide GEMM regime"
          : isShape13
            ? "Long sequence regime"
            : "Measured execution path"}
      </h3>
      <div className="dimensions">
        <span>B {selected.config.batch_size}</span>
        <span>S {selected.config.seq_len}</span>
        <span>D {selected.config.d_model}</span>
        <span>H {selected.config.num_heads}</span>
        <span>L {selected.config.num_layers}</span>
      </div>
      <div className="inspector-results">
        <Metric label="final" value={formatLatency(selected.latency_ms)} />
        <Metric label="throughput" value={formatTF(selected.useful_tflops)} />
        <Metric
          label="est. FP32 MFU"
          value={
            selected.estimated_fp32_mfu == null
              ? "N/A"
              : `${(selected.estimated_fp32_mfu * 100).toFixed(2)}%`
          }
        />
        <Metric
          label="reference speedup"
          value={
            selected.reference_speedup
              ? `${selected.reference_speedup.toFixed(2)}×`
              : "—"
          }
        />
      </div>
      <div className="compare-lines">
        <p>
          <span>challenge reference</span>
          <b>
            {formatLatency(
              comparison?.reference_latency_ms ?? selected.reference_latency_ms,
            )}
          </b>
        </p>
        <p>
          <span>
            strong SDPA
            {selected.sdpa_speedup
              ? ` · ${selected.sdpa_speedup.toFixed(2)}×`
              : ""}
          </span>
          <b>{formatLatency(comparison?.strong_sdpa_latency_ms)}</b>
        </p>
      </div>
      <div className="chosen">
        <p className="eyebrow">Retained strategy</p>
        <p>{policy}</p>
      </div>
    </aside>
  );
}

function Shape14({ evidence }: { evidence: Record<string, any> }) {
  const stage4 = evidence.stage4_end_to_end_official_fp32;
  const stage2 = evidence.stage2_first_memory_efficient;
  return (
    <section className="view shape14-view reveal is-in">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Shape 14 / the memory story</p>
          <h2>Streaming enables Shape 14 to run end-to-end on 8 GB.</h2>
        </div>
      </div>
      <div className="shape14-outcome">
        <Status tone="pass">FULL LOGICAL WORKLOAD: PASS</Status>
        <Metric
          label="end-to-end latency"
          value={formatLatency(stage4.latency_ms)}
        />
        <Metric
          label="useful throughput"
          value={formatTF(stage4.useful_tflops_end_to_end)}
        />
        <Metric label="reported peak" value="4.326 GB" />
        <p>
          The streamed FP32 route completes the unchanged workload; the
          untouched interface still fails before `forward()`.
        </p>
      </div>
      <div className="stage-list">
        <article className="stage stage--fail">
          <div className="stage-number">01</div>
          <div>
            <p className="eyebrow">Untouched official FP32 interface</p>
            <h3>One 12.21 GiB input allocation.</h3>
            <p>Metal rejects it before participant code is called.</p>
          </div>
          <Status tone="fail">FAIL BEFORE FORWARD</Status>
        </article>
        <div className="story-transition">
          <span>The input interface is the limit.</span>
          <b>↓</b>
          <span>Streaming removes it.</span>
        </div>
        <article className="stage">
          <div className="stage-number">02</div>
          <div>
            <p className="eyebrow">Streaming attention</p>
            <h3>The full logical compute now fits.</h3>
            <p>
              Online softmax removes the full attention matrix: Stage 2
              completed batch 32 at {formatTF(stage2.useful_tflops)}.
            </p>
          </div>
          <Status tone="pass">COMPLETE LOGICAL COMPUTATION</Status>
        </article>
        <div className="story-transition">
          <span>Attention fits.</span>
          <b>↓</b>
          <span>Stream the input too.</span>
        </div>
        <article className="stage stage--final">
          <div className="stage-number">03</div>
          <div>
            <p className="eyebrow">Official-style FP32 streamed execution</p>
            <h3>Generate one batch element. Run 100k tokens. Release it.</h3>
            <div className="execution-flow">
              <span>FP32 input tile</span>
              <b>→</b>
              <span>FP16 fused compute</span>
              <b>→</b>
              <span>FP32 state + digest</span>
            </div>
            <p>
              Repeated 32 times with no full input, output, or attention matrix
              on MPS.
            </p>
          </div>
          <Status tone="pass">32 / 32 TILES COMPLETE</Status>
        </article>
      </div>
      <div className="precision-bar">
        <div>
          <p className="eyebrow">Validation limit</p>
          <h3>The full 100k-token reference cannot fit on this 8 GB M2.</h3>
          <p>
            The streamed algorithm was compared directly with the reference at
            B=32/S=4,096 and B=1/S=6,592 across three seeds. Every comparison
            passed with zero failed elements.
          </p>
        </div>
      </div>
    </section>
  );
}

export default App;
