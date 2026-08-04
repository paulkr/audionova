"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";

import * as api from "../../../lib/api";

// The studio.
//
// THE A/B TRANSPORT IS LOUDNESS-MATCHED AND THE SWITCH IS GAPLESS. Both are
// load-bearing rather than polish: louder reliably wins a blind comparison
// whether or not it is better, and a gap between the two lets memory fill in
// the difference. The server returns the processed audio already matched, and
// both elements share a playhead so switching keeps position.
//
// THE GRAPH INSPECTOR SHOWS EVERY PARAMETER WITH ITS UNITS, plus the sentence
// the planner wrote about why each node exists. That is the differentiator: an
// engineer decides whether to trust this by reading it, not by being told it
// worked.

type Tab = "A" | "B";

export default function Studio({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);

  const [project, setProject] = useState<api.Project | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const [request, setRequest] = useState("");
  const [planner, setPlanner] = useState<"model" | "rules">("model");
  const [plan, setPlan] = useState<api.PlanResult | null>(null);
  const [graphId, setGraphId] = useState<string | null>(null);
  const [rendered, setRendered] = useState<api.RenderResult | null>(null);
  const [history, setHistory] = useState<string[]>([]);

  const [vocabulary, setVocabulary] = useState<
    { intents: { id: string; label: string }[]; quantifiers: string[] } | null
  >(null);

  // A model plan measured 128 s on a real file — past the plan's own p95 target
  // of 120 s. A label that does not move for two minutes reads as a hang, so the
  // seconds tick. This is the honest stand-in for the SSE progress stream, which
  // is Phase 1 platform work and is not built: it proves the request is alive
  // without claiming to know how far along it is.
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!busy) return setElapsed(0);
    const started = Date.now();
    const timer = setInterval(
      () => setElapsed(Math.round((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [busy]);

  const [tab, setTab] = useState<Tab>("A");
  const [playing, setPlaying] = useState(false);
  const sourceRef = useRef<HTMLAudioElement>(null);
  const processedRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    api.getProject(id).then(setProject).catch((e) => setError(String(e.message)));
    api.intents().then(setVocabulary).catch(() => {});
  }, [id]);

  // Keep both elements at the same position so switching is a comparison of
  // sound rather than of where you happen to be in the take.
  const swap = useCallback((next: Tab) => {
    const from = tab === "A" ? sourceRef.current : processedRef.current;
    const to = next === "A" ? sourceRef.current : processedRef.current;
    if (from && to) {
      to.currentTime = from.currentTime;
      if (!from.paused) {
        from.pause();
        void to.play();
      }
    }
    setTab(next);
  }, [tab]);

  const toggle = useCallback(() => {
    const el = tab === "A" ? sourceRef.current : processedRef.current;
    if (!el) return;
    if (el.paused) {
      void el.play();
      setPlaying(true);
    } else {
      el.pause();
      setPlaying(false);
    }
  }, [tab]);

  async function runPlan() {
    setBusy(planner === "model" ? "planning — this takes a minute or two" : "planning");
    setError(null);
    try {
      const result = await api.plan(id, { request, planner });
      setPlan(result);
      setGraphId(result.graph_id);
      setHistory([result.graph_id]);
      setBusy("rendering");
      setRendered(await api.render(id, result.graph_id));
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  async function runRefine(intent: string, quantifier: string) {
    if (!graphId) return;
    setBusy(`applying ${quantifier.replace("_", " ")} ${intent.replace(/_/g, " ")}`);
    setError(null);
    try {
      const result = await api.refine(id, graphId, intent, quantifier);
      if (!result.graph_id) {
        setError(result.note ?? "nothing in this chain responds to that");
        return;
      }
      setGraphId(result.graph_id);
      setHistory((h) => [...h, result.graph_id!]);
      setPlan((p) => (p && result.nodes ? { ...p, nodes: result.nodes } : p));
      setBusy("rendering");
      setRendered(await api.render(id, result.graph_id));
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  // A hand edit is a first-class turn, not a local tweak. It creates a graph on
  // the same tree with the same parent semantics as a refinement, so undo steps
  // through it and the next model turn sees the correction as what it is
  // refining. Anything else would let the planner quietly overwrite a decision
  // the user made deliberately.
  async function applyEdit(nodeId: string, param: string, value: number,
                           bandIndex?: number) {
    if (!graphId) return;
    setBusy(`setting ${param.replace(/_/g, " ")}`);
    setError(null);
    try {
      const result = await api.editGraph(id, graphId, {
        node_id: nodeId, param, value, band_index: bandIndex,
      });
      setGraphId(result.graph_id);
      setHistory((h) => [...h, result.graph_id]);
      setPlan((p) => (p ? { ...p, nodes: result.nodes } : p));
      setBusy("rendering");
      setRendered(await api.render(id, result.graph_id));
    } catch (e) {
      // The server's refusal names the actual bound for this parameter on this
      // node type. Rewording it here would only make it vaguer.
      setError(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  // An imported recipe is validated exactly like a planned one. It may predate
  // a validator, or have been edited in a text editor on the way through, and
  // arriving as a file earns it no extra trust.
  async function loadRecipe(file: File) {
    setBusy("importing the recipe");
    setError(null);
    try {
      const parsed = JSON.parse(await file.text());
      const result = await api.importRecipe(id, parsed);
      setGraphId(result.graph_id);
      setHistory((h) => [...h, result.graph_id]);
      setPlan((p) => (p ? { ...p, nodes: result.nodes } : p));
      if (result.warnings.length) setError(result.warnings.join(" "));
      setBusy("rendering");
      setRendered(await api.render(id, result.graph_id));
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  // Everything above this point runs on a 20-second excerpt, which is what
  // makes iterating feel instant. The download is the one action that has to
  // touch the whole file, so it renders it on demand rather than keeping a
  // full render in step with every refinement nobody was going to download.
  async function downloadFull() {
    if (!graphId) return;
    setBusy("rendering the full file");
    setError(null);
    try {
      const result = await api.render(id, graphId, true);
      if (!result.ok || !result.deliver_key) {
        // The gate applies to the full render on its own terms. An excerpt
        // that passed is not evidence about the parts of the file it excluded.
        setError(result.failures?.map((f) => f.message).join("; ")
                 ?? "the full render did not pass the safety gate");
        return;
      }
      // A navigation, not fetch-then-blob. The file is tens of megabytes and
      // there is no reason to hold it all in memory on the way to the disk.
      window.location.href = api.downloadUrl(id, result.deliver_key);
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  function undo() {
    if (history.length < 2) return;
    const next = history.slice(0, -1);
    setHistory(next);
    setGraphId(next[next.length - 1]);
    void api.render(id, next[next.length - 1]).then(setRendered);
  }

  if (error && !project) {
    return <Shell><p className="text-white/60">{error}</p></Shell>;
  }
  if (!project) {
    return <Shell><p className="text-white/40">loading…</p></Shell>;
  }

  const report = project.report;
  const after = rendered?.after;

  return (
    <Shell>
      <header className="mb-8 flex flex-wrap items-baseline justify-between gap-3">
        <h1 className="text-[22px] font-light tracking-tight">{project.name}</h1>
        <p className="text-[13px] text-white/40">
          {project.channels === 1 ? "mono" : `${project.channels} ch`} ·{" "}
          {Math.round(project.sample_rate / 100) / 10} kHz ·{" "}
          {Math.round(project.duration_s)} s
        </p>
      </header>

      <Section title="What it is">
        <div className="grid grid-cols-2 gap-x-8 gap-y-2 sm:grid-cols-3">
          <Metric label="loudness" value={report.loudness.integrated_lufs} unit="LUFS"
                  after={after?.loudness.integrated_lufs} />
          <Metric label="true peak" value={report.loudness.true_peak_dbtp} unit="dBTP"
                  after={after?.loudness.true_peak_dbtp} />
          <Metric label="loudness range" value={report.loudness.loudness_range_lu} unit="LU"
                  after={after?.loudness.loudness_range_lu} />
          <Metric label="crest factor" value={report.dynamics.crest_factor_db} unit="dB"
                  after={after?.dynamics.crest_factor_db} />
          <Metric label="peak to loudness" value={report.dynamics.peak_to_loudness_ratio_db}
                  unit="dB" after={after?.dynamics.peak_to_loudness_ratio_db} />
          <Metric label="HF rolloff" value={report.spectrum.hf_rolloff_db_per_octave}
                  unit="dB/oct" after={after?.spectrum.hf_rolloff_db_per_octave} />
        </div>
        {report.warnings.length > 0 && (
          <ul className="mt-4 space-y-1">
            {report.warnings.map((w) => (
              <li key={w.code} className="text-[13px] text-white/50">
                <span className="mr-2 text-white/30">{w.severity}</span>
                {w.message}
              </li>
            ))}
          </ul>
        )}
        <Spectrum bands={report.spectrum.third_octave_db}
                  after={after?.spectrum.third_octave_db} />
      </Section>

      <Section title="What you want">
        <div className="flex flex-col gap-3">
          <input
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !busy && runPlan()}
            placeholder="warmer and less boxy"
            className="w-full rounded-[10px] border border-white/12 bg-white/[0.03] px-4 py-3
                       text-[15px] outline-none placeholder:text-white/25 focus:border-white/25"
          />
          <div className="flex flex-wrap items-center gap-3">
            <button onClick={runPlan} disabled={!!busy} className="pill-filled px-5 py-2 text-[14px]">
              {busy ? `${busy} · ${elapsed}s` : "Plan"}
            </button>
            <label className="flex items-center gap-2 text-[13px] text-white/45">
              <input type="checkbox" checked={planner === "rules"}
                     onChange={(e) => setPlanner(e.target.checked ? "rules" : "model")} />
              deterministic only
              <span className="text-white/25">(no model call)</span>
            </label>
          </div>
          {error && <p className="text-[13px] text-[#ff8b8b]">{error}</p>}
        </div>
      </Section>

      {plan && (
        <>
          <Section title="What it did">
            <p className="mb-5 max-w-[70ch] text-[14px] leading-[1.6] text-white/60">
              {plan.explanation}
            </p>
            <div className="space-y-3">
              {plan.nodes.map((node) => (
                <NodeCard key={node.id} node={node} disabled={!!busy}
                          onEdit={(param, value, bandIndex) =>
                            applyEdit(node.id, param, value, bandIndex)} />
              ))}
            </div>
            {plan.declined && plan.declined.length > 0 && (
              <p className="mt-4 max-w-[70ch] text-[13px] leading-[1.55] text-white/35">
                Left alone: {plan.declined.join("; ")}.
              </p>
            )}
          </Section>

          {rendered && !rendered.ok && (
            <Section title="Not shipped">
              <p className="mb-2 text-[14px] text-[#ff8b8b]">
                This render did not pass its safety checks, so it was discarded.
              </p>
              <ul className="space-y-1">
                {rendered.failures?.map((f) => (
                  <li key={f.code} className="text-[13px] text-white/55">{f.message}</li>
                ))}
              </ul>
            </Section>
          )}

          {rendered?.ok && rendered.render_key && (
            <Section title="Listen">
              <p className="mb-4 text-[13px] text-white/40">
                Loudness-matched by {rendered.match_offset_db} dB, so you are comparing
                the processing rather than the level.
              </p>
              <div className="flex flex-wrap items-center gap-3">
                <button onClick={toggle} className="pill-filled px-6 py-2 text-[14px]">
                  {playing ? "Pause" : "Play"}
                </button>
                <div className="flex overflow-hidden rounded-full border border-white/12">
                  {(["A", "B"] as Tab[]).map((t) => (
                    <button key={t} onClick={() => swap(t)}
                            className={`px-5 py-2 text-[13px] transition-colors ${
                              tab === t ? "bg-white/12 text-white" : "text-white/45"}`}>
                      {t === "A" ? "Original" : "Processed"}
                    </button>
                  ))}
                </div>
                {history.length > 1 && (
                  <button onClick={undo} className="text-[13px] text-white/45 underline
                                                    underline-offset-4 hover:text-white/70">
                    undo
                  </button>
                )}
              </div>
              <audio ref={sourceRef} src={api.audioUrl(id, rendered.source_key!)}
                     onEnded={() => setPlaying(false)} preload="auto" />
              <audio ref={processedRef} src={api.audioUrl(id, rendered.render_key)}
                     onEnded={() => setPlaying(false)} preload="auto" />
            </Section>
          )}

          {rendered?.ok && (
            <Section title="Take it away">
              <p className="mb-4 max-w-[70ch] text-[13px] leading-[1.55] text-white/40">
                Renders the whole {Math.round(project.duration_s)} seconds through
                this chain and downloads it as 24-bit WAV at{" "}
                {Math.round(project.sample_rate / 100) / 10} kHz. Not the file you
                just auditioned: that one is pulled {rendered.match_offset_db} dB to
                match the original&rsquo;s loudness so the comparison is fair. This
                is the chain&rsquo;s own output, at its own level.
              </p>
              <div className="flex flex-wrap items-center gap-3">
                <button onClick={downloadFull} disabled={!!busy}
                        className="pill-filled px-6 py-2 text-[14px] disabled:opacity-40">
                  {busy?.startsWith("rendering the full")
                    ? `${busy} · ${elapsed}s`
                    : "Download processed WAV"}
                </button>
                {graphId && (
                  <a href={api.recipeUrl(id, graphId)} download
                     className="text-[13px] text-white/45 underline underline-offset-4
                                hover:text-white/70">
                    export recipe
                  </a>
                )}
                <label className="cursor-pointer text-[13px] text-white/45 underline
                                  underline-offset-4 hover:text-white/70">
                  import recipe
                  <input type="file" accept="application/json,.json" className="hidden"
                         onChange={(e) => {
                           const file = e.target.files?.[0];
                           e.target.value = "";
                           if (file) void loadRecipe(file);
                         }} />
                </label>
              </div>
              <p className="mt-3 text-[12px] leading-[1.5] text-white/30">
                A recipe carries the chain and the engine version that rendered
                it. Re-importing one onto the same file reproduces the render
                bit for bit; onto a different file it is a starting point, not a
                guarantee.
              </p>
            </Section>
          )}

          {vocabulary && rendered?.ok && (
            <Section title="Adjust">
              <p className="mb-4 max-w-[70ch] text-[13px] leading-[1.55] text-white/40">
                Each of these is a patch against the chain above, not a new plan —
                only what you ask for changes. The step sizes are fixed, so
                &ldquo;slightly&rdquo; means the same thing every time.
              </p>
              <div className="flex flex-wrap gap-2">
                {vocabulary.intents.map((intent) => (
                  <div key={intent.id} className="flex overflow-hidden rounded-full
                                                  border border-white/10">
                    <button disabled={!!busy}
                            onClick={() => runRefine(intent.id, "slightly")}
                            className="px-3 py-1.5 text-[12px] text-white/60
                                       hover:bg-white/8 disabled:opacity-40">
                      {intent.label}
                    </button>
                    <button disabled={!!busy}
                            onClick={() => runRefine(intent.id, "noticeably")}
                            className="border-l border-white/10 px-2 py-1.5 text-[12px]
                                       text-white/35 hover:bg-white/8 disabled:opacity-40"
                            title="noticeably — two steps instead of one">
                      ++
                    </button>
                  </div>
                ))}
              </div>
            </Section>
          )}
        </>
      )}
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <main className="mx-auto min-h-dvh w-full max-w-[880px] px-6 py-14 sm:px-10">
      {children}
    </main>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-12 border-t border-white/8 pt-6">
      <h2 className="mb-5 text-[12px] uppercase tracking-[0.14em] text-white/35">
        {title}
      </h2>
      {children}
    </section>
  );
}

function Metric({ label, value, unit, after }: {
  label: string; value: number | null | undefined;
  unit: string; after?: number | null;
}) {
  if (value == null) return null;
  const delta = after != null ? after - value : null;
  return (
    <div>
      <p className="text-[12px] text-white/35">{label}</p>
      <p className="text-[17px] font-light tabular-nums">
        {value.toFixed(2)}
        <span className="ml-1 text-[12px] text-white/30">{unit}</span>
        {delta != null && Math.abs(delta) >= 0.01 && (
          <span className={`ml-2 text-[12px] tabular-nums ${
            delta > 0 ? "text-[#7fd4a0]" : "text-[#e59a9a]"}`}>
            {delta > 0 ? "+" : ""}{delta.toFixed(2)}
          </span>
        )}
      </p>
    </div>
  );
}

// A third-octave view. Levels are drawn relative to each file's own 500 Hz to
// 4 kHz average, so the two curves compare SHAPE rather than level — the same
// reason the audio is loudness-matched.
function Spectrum({ bands, after }: {
  bands: { centre_hz: number; level_db: number | null }[];
  after?: { centre_hz: number; level_db: number | null }[];
}) {
  const usable = bands.filter((b) => b.level_db != null && b.centre_hz >= 40);
  if (usable.length < 4) return null;

  const normalise = (rows: typeof usable) => {
    const mid = rows.filter((b) => b.centre_hz >= 500 && b.centre_hz <= 4000);
    const ref = mid.reduce((a, b) => a + b.level_db!, 0) / (mid.length || 1);
    return rows.map((b) => ({ hz: b.centre_hz, rel: b.level_db! - ref }));
  };

  const source = normalise(usable);
  const processed = after
    ? normalise(after.filter((b) => b.level_db != null && b.centre_hz >= 40))
    : null;

  const span = 40;
  const height = 90;
  const y = (rel: number) =>
    height - ((Math.max(-span, Math.min(span, rel)) + span) / (2 * span)) * height;
  const x = (i: number, n: number) => (i / (n - 1)) * 100;

  const path = (rows: { rel: number }[]) =>
    rows.map((p, i) => `${i ? "L" : "M"}${x(i, rows.length).toFixed(2)},${y(p.rel).toFixed(1)}`)
        .join(" ");

  return (
    <div className="mt-6">
      <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none"
           className="h-[90px] w-full" aria-label="third-octave spectrum">
        <line x1="0" y1={y(0)} x2="100" y2={y(0)} stroke="rgba(255,255,255,0.1)"
              strokeWidth="0.4" />
        <path d={path(source)} fill="none" stroke="rgba(255,255,255,0.3)"
              strokeWidth="0.7" vectorEffect="non-scaling-stroke" />
        {processed && (
          <path d={path(processed)} fill="none" stroke="rgba(190,140,255,0.85)"
                strokeWidth="0.9" vectorEffect="non-scaling-stroke" />
        )}
      </svg>
      <div className="mt-1 flex justify-between text-[10px] tabular-nums text-white/25">
        <span>{usable[0].centre_hz < 1000 ? `${usable[0].centre_hz} Hz` : ""}</span>
        <span>1 kHz</span>
        <span>{Math.round(usable[usable.length - 1].centre_hz / 1000)} kHz</span>
      </div>
    </div>
  );
}

// A number you can retype.
//
// READ-ONLY WAS THE WRONG DEFAULT. The inspector's whole claim is that an
// engineer decides whether to trust this by reading it — and someone who can
// read "-3.2 dB at 340 Hz" and disagree wants to type "-2.4" rather than
// describe the change in words and hope. Every value the server will accept an
// edit for is editable here.
//
// It stays looking like text until focused, because a chain of forty input
// boxes reads as a form and the point is that it reads as a decision.
function Editable({ value, unit, onCommit, disabled }: {
  value: number;
  unit?: string;
  disabled?: boolean;
  onCommit: (next: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);

  if (draft === null) {
    return (
      <button
        disabled={disabled}
        onClick={() => setDraft(String(value))}
        className="rounded px-1 tabular-nums underline decoration-dotted
                   decoration-white/20 underline-offset-4 hover:bg-white/8
                   hover:decoration-white/50 disabled:opacity-40"
      >
        {value}{unit ? ` ${unit}` : ""}
      </button>
    );
  }

  const commit = () => {
    const parsed = Number(draft);
    setDraft(null);
    // NaN and no-change both mean "do nothing". Sending an unchanged value
    // would still mint a new graph, so undo would step through edits that
    // never altered anything.
    if (Number.isFinite(parsed) && parsed !== value) onCommit(parsed);
  };

  return (
    <input
      autoFocus
      value={draft}
      inputMode="decimal"
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") setDraft(null);
      }}
      className="w-20 rounded border border-white/25 bg-white/10 px-1
                 tabular-nums outline-none"
    />
  );
}

function NodeCard({ node, onEdit, disabled }: {
  node: api.GraphNode;
  disabled?: boolean;
  onEdit: (param: string, value: number, bandIndex?: number) => void;
}) {
  const bands = node.params.bands;
  const scalars = Object.entries(node.params).filter(
    ([k, v]) => k !== "bands" && typeof v !== "object",
  );

  return (
    <div className="rounded-[10px] border border-white/8 bg-white/[0.02] px-4 py-3">
      <div className="mb-1 flex items-baseline gap-2">
        <span className="text-[14px]">{node.type.replace(/_/g, " ")}</span>
        <span className="text-[11px] text-white/25">{node.id}</span>
        {node.master && (
          <span className="text-[10px] uppercase tracking-wider text-white/25">
            master
          </span>
        )}
        {node.mix !== 1 && (
          <span className="text-[11px] text-white/40">{Math.round(node.mix * 100)}% wet</span>
        )}
      </div>

      {node.rationale && (
        <p className="mb-2 max-w-[70ch] text-[12px] leading-[1.5] text-white/40">
          {node.rationale}
        </p>
      )}

      {bands && bands.length > 0 && (
        <ul className="space-y-0.5">
          {bands.map((band, i) => (
            <li key={i} className="text-[13px] tabular-nums text-white/70">
              {band.filter_type}{" "}
              <Editable value={band.center_freq_hz} unit="Hz" disabled={disabled}
                        onCommit={(v) => onEdit("center_freq_hz", v, i)} />
              {" · "}
              <Editable value={band.gain_db} unit="dB" disabled={disabled}
                        onCommit={(v) => onEdit("gain_db", v, i)} />
              {" · Q "}
              <Editable value={band.q_factor} disabled={disabled}
                        onCommit={(v) => onEdit("q_factor", v, i)} />
            </li>
          ))}
        </ul>
      )}

      {scalars.length > 0 && (
        <div className="flex flex-wrap gap-x-4 gap-y-0.5">
          {scalars.map(([key, value]) => (
            <span key={key} className="text-[12px] tabular-nums text-white/55">
              <span className="text-white/30">{key.replace(/_/g, " ")}</span>{" "}
              {typeof value === "number" ? (
                <Editable value={value} disabled={disabled}
                          onCommit={(v) => onEdit(key, v)} />
              ) : (
                String(value)
              )}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
