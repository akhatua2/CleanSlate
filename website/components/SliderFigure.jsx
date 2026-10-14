"use client";
// Figure: drag to unlearn. Every tile is one song or book the model can reproduce.
// Left wall: the works we asked it to forget. Right wall: everything else it extractable.
// The slider steps through real checkpoints of evaluation-aware SimNPO unlearning.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { memLevel } from "./lyrics";
import TileCanvas, { hash } from "./TileCanvas";
import Select from "./Select";
import "./slider.css";
import { loadText } from "./data";

// eases a number toward its target so the counters move with the tiles
function useTween(target, ms = 1100) {
  const [v, setV] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const start = performance.now(), a = from.current;
    let raf;
    const tick = (t) => {
      const k = Math.min(1, (t - start) / ms);
      const e = 1 - Math.pow(1 - k, 3);
      const x = a + (target - a) * e;
      from.current = x;
      setV(x);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

const WHEN = ["", "At 25%", "At 50%", "At 75%", "Done"];

// One plain sentence about the current checkpoint, built from the numbers. It names a
// well-extractable work nobody asked about that is (nearly) gone by now.
function sentence(m, walls, stage, last, fE, rE) {
  const total = walls.forget.length + walls.retain.length;
  if (stage === 0) {
    return `Before unlearning, ${m.name} can reproduce passages from ${total.toLocaleString()} songs and books. Drag to start.`;
  }
  // prefer a work that was lost at this step, so each stop names something new
  const gone = (x, k) => x.n[k] <= 0.15 * x.n[0];
  const cands = walls.retain.filter((x) => x.n[0] >= 15 && gone(x, stage)).sort((a, b) => b.n[0] - a.n[0]);
  const hit = cands.find((x) => !gone(x, stage - 1)) ?? cands[0];
  const ex = hit ? `, including ${hit.n[stage] === 0 ? "all" : "almost all"} of ${hit.t}` : "";
  const f = fE >= 99.95 ? "the request is fully gone" : `${fE.toFixed(1)}% of the request is gone`;
  const r = rE >= 99.5 && rE < 100 ? `almost all (${rE.toFixed(2)}%)` : `${rE.toFixed(1)}%`;
  if (stage === last) {
    return `${WHEN[stage]}: ${f}, and so is ${r} of everything else${ex}.`;
  }
  return `${WHEN[stage]}, ${f}, and ${r} of everything else${ex}.`;
}

const erasedPct = (tot, s) => (tot[0] ? 100 * (1 - tot[s] / tot[0]) : 0);

export default function SliderFigure() {
  const [d, setD] = useState(null);
  const [mk, setMk] = useState("llama-8b");
  const [stage, setStage] = useState(0);
  const [hover, setHover] = useState(null);
  const [pick, setPick] = useState(null); // selected tile {s}
  const playing = useRef(null);
  const bodyRef = useRef(null);
  const choose = (x) => setPick(x);
  const onPick = useCallback((x) => setPick(x), []);

  useEffect(() => { fetch("data/slider.json").then((r) => r.json()).then(setD); }, []);
  useEffect(() => () => clearInterval(playing.current), []);

  const m = d?.models.find((x) => x.key === mk);
  const walls = useMemo(() => {
    if (!m) return null;
    // scattered, not ranked: order by a hash of the id so the layout is stable across visits
    const by = (p) => m.songs.filter((s) => s.pool === p).sort((a, b) => hash(a.id) - hash(b.id));
    return { forget: by("forget"), retain: by("retain") };
  }, [m]);

  if (!d || !m) return <div className="sl ui">Loading…</div>;
  const last = d.stages.length - 1;
  // place each checkpoint by how much of all extractable text is gone, not by training time
  const tot = m.forget.map((f, i) => f + m.retain[i]);
  const stops = tot.map((t) => (tot[0] - t) / (tot[0] - tot[last] || 1));
  const nearest = (x) => stops.reduce((bi, p, i) => (Math.abs(p - x) < Math.abs(stops[bi] - x) ? i : bi), 0);

  const play = () => {
    clearInterval(playing.current);
    let s = stage >= last ? 0 : stage;
    setStage(s);
    playing.current = setInterval(() => {
      s += 1;
      setStage(s);
      if (s >= last) clearInterval(playing.current);
    }, 900);
  };

  const fE = erasedPct(m.forget, stage), rE = erasedPct(m.retain, stage);

  return (
    <div className="sl ui">
      <div className="sl-ask">
        We asked
        <Select
          label="Model"
          value={mk}
          options={d.models.map((x) => ({ value: x.key, label: x.name }))}
          onChange={(v) => { setMk(v); setStage(0); choose(null); }}
        />
        to forget {walls.forget.length} songs and books.
      </div>

      <div className="sl-control">
        <div className="sl-track">
          <input
            type="range" min={0} max={1000} step={1} value={Math.round(stops[stage] * 1000)}
            onChange={(e) => { clearInterval(playing.current); setStage(nearest(+e.target.value / 1000)); }}
            aria-label="Unlearning progress"
            aria-valuetext={d.stages[stage]}
            style={{ "--p": `${stops[stage] * 100}%` }}
          />
          <div className="sl-ticks">
            {d.stages.map((s, i) => {
              const crowded = i > 0 && stops[i] - stops[i - 1] < 0.09;
              const flip = crowded && i % 2 === 1;
              return (
                <button
                  key={s}
                  className={`${i === stage ? "on" : ""}${flip ? " up" : ""}`}
                  style={{ left: `${stops[i] * 100}%` }}
                  onClick={() => setStage(i)}
                >
                  <i />{s}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <p className="sl-say" aria-live="polite">{sentence(m, walls, stage, last, fE, rE)}</p>

      <div className={`sl-body${pick ? " open" : ""}`} ref={bodyRef}>
      <div className="sl-left">
      <div className="sl-walls">
        <Wall
          title={`The ${walls.forget.length} we asked it to forget`}
          kind="forget"
          songs={walls.forget}
          stage={stage}
          pct={fE}
          onHover={setHover}
          pick={pick}
          onPick={onPick}
        />
        <Wall
          title={`The ${walls.retain.length.toLocaleString()} we didn’t`}
          kind="retain"
          songs={walls.retain}
          stage={stage}
          pct={rE}
          onHover={setHover}
          pick={pick}
          onPick={onPick}
        />
      </div>

      <div className="sl-legend">
        <span><span className="sw" style={{ background: "var(--mem-7)" }} /><span className="sw" style={{ background: "var(--mem-5)" }} /><span className="sw" style={{ background: "var(--mem-2)" }} /> still extractable (darker = more of the work)</span>
        <span><span className="sw" style={{ background: "var(--gone-4)" }} /> erased, as asked</span>
        <span><span className="sw lost" /> erased, but should have stayed</span>
      </div>
      {!pick && <div className="sl-hint">Click any tile to read that work with the extractable text highlighted, then drag the slider.</div>}
      </div>

      {pick && <Lyrics key={`${mk}-${pick.id}`} s={pick} model={mk} stage={stage} onClose={() => choose(null)} />}
      </div>

      {hover && (
        <div className="tooltip" style={{ opacity: 1, left: Math.min(hover.x + 12, window.innerWidth - 270), top: hover.y + 14 }}>
          <b>{hover.s.t}</b>{hover.s.a ? <> · {hover.s.a}</> : null}
          <br /><span className="k">extractable windows:</span> {hover.s.n[0]} → <b>{hover.s.n[stage]}</b>
        </div>
      )}
    </div>
  );
}

function Wall({ title, kind, songs, stage, pct, onHover, pick, onPick }) {
  return (
    <section className={`sl-wall ${kind}`}>
      <header>
        <div className="h">{title}</div>
      </header>
      <div className="sl-big">
        <Counter pct={pct} />
        <span className="lab">gone</span>
      </div>
      <TileCanvas songs={songs} stage={stage} kind={kind} pickId={pick?.id} onHover={onHover} onPick={onPick} />
    </section>
  );
}

// the counter re-renders every frame while it eases; keep that to this one span
function Counter({ pct }) {
  const shown = useTween(pct);
  return <span className="num">{shown.toFixed(1)}%</span>;
}

// The selected work, token by token in this model's tokenizer, at the current checkpoint.
// Red: still extractable. Green: extractable before unlearning, erased now.
function Lyrics({ s, model, stage, onClose }) {
  const [tok, setTok] = useState(null);
  const bodyEl = useRef(null);
  useEffect(() => { loadText(`data/tokens_ckpt/${s.id}.json`).then(setTok); }, [s.id]);
  // open on the extractable part: scroll so the first highlighted line sits near the top
  useEffect(() => {
    const el = bodyEl.current;
    const first = el?.querySelector(".t.hot, .t.gone");
    if (el && first) el.scrollTo({ top: Math.max(0, first.offsetTop - el.offsetTop - 60), behavior: "smooth" });
  }, [tok, model]);
  const m = tok?.models[model];
  const out = [];
  if (m) {
    const pre = m.vals[0], cur = m.vals[stage];
    m.spans.forEach(([a, b], i) => {
      const before = memLevel(pre?.[i]);
      const now = memLevel(cur?.[i]);
      let cls = "t", style;
      if (now) { cls += now >= 6 ? " hot deep" : " hot"; style = { background: `var(--mem-${now})` }; }
      else if (before) cls += " gone";
      tok.text.slice(a, b).split("\n").forEach((p, k) => {
        if (k > 0) out.push(<br key={`${i}b${k}`} />);
        if (!p) return;
        const lead = p.match(/^\s*/)[0];
        if (lead) out.push(lead);
        const body = p.slice(lead.length);
        if (body) out.push(<span key={`${i}-${k}`} className={cls} style={style}>{body}</span>);
      });
    });
  }
  return (
    <div className={`sl-lyrics ${s.pool}`}>
      <div className="hd">
        <span className="t">{s.t}</span>
        {s.a && <span className="by">{s.a}</span>}
        <span className={`pool ${s.pool}`}>{s.pool === "forget" ? "asked to forget" : "never asked to forget"}</span>
        <span className="by">{s.n[0]} → <b>{s.n[stage]}</b> extractable windows</span>
        <button className="x" onClick={onClose}>close ✕</button>
      </div>
      <div className="body" ref={bodyEl}>{tok ? out : "Loading…"}</div>
      {tok && tok.total_chars > tok.text.length && (
        <div className="note">Excerpt: characters {tok.offset.toLocaleString()}–{(tok.offset + tok.text.length).toLocaleString()} of {tok.total_chars.toLocaleString()}, around the most extractable passage.</div>
      )}
    </div>
  );
}
