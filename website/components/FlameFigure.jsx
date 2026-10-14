"use client";
// Figure 1: where extraction lives. One song, six models. Each strip is a model, each mark a
// token (in that model's own tokenizer), positioned by where it sits in the text.
import { useEffect, useRef, useState } from "react";
import TokenLyrics from "./TokenLyrics";
import { STEPS, fmtP, memLevel } from "./lyrics";
import "./flame.css";
import { loadText } from "./data";

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export default function FlameFigure() {
  const [index, setIndex] = useState(null);
  const [workId, setWorkId] = useState(null);
  const [tok, setTok] = useState(null);
  const [model, setModel] = useState(null);
  const [hover, setHover] = useState(null); // {i, model, x, y}

  useEffect(() => {
    fetch("data/flame_index.json").then((r) => r.json()).then((ix) => {
      setIndex(ix);
      setWorkId(ix.featured[0]);
    });
  }, []);

  useEffect(() => {
    if (!workId || !index) return;
    loadText(`data/tokens/${workId}.json`).then((d) => {
      setTok(d);
      const w = index.works.find((x) => x.id === workId);
      let best = index.models[0].key, bestN = -1;
      for (const m of index.models) {
        const k = w.models[m.key]?.pre ?? 0;
        if (k > bestN) { bestN = k; best = m.key; }
      }
      setModel((cur) => cur ?? best);
    });
  }, [workId, index]);

  if (!index || !tok || !model || tok.id !== workId) return <div className="flame ui loading">Loading…</div>;
  const work = index.works.find((x) => x.id === workId);
  const featured = index.featured.map((id) => index.works.find((w) => w.id === id)).filter(Boolean);
  const hv = hover && tok.models[hover.model];

  return (
    <div className="flame ui">
      <nav className="tabs" aria-label="Pick a song">
        {featured.map((w) => (
          <button key={w.id} aria-pressed={w.id === workId} onClick={() => setWorkId(w.id)}>{w.title}</button>
        ))}
      </nav>

      <div className="flame-head">
        <div className="t">{work.title}</div>
        <div className="by">{[work.creators, work.year].filter(Boolean).join(" · ")}</div>
      </div>

      <div className="flame-text" onMouseLeave={() => setHover(null)}>
        <div className="who">Highlighted for <b>{index.models.find((m) => m.key === model).name}</b> · click a model below to switch</div>
        <TokenLyrics tok={tok} model={model} stage="pre" onHover={(h) => setHover({ ...h, model })} />
      </div>

      <div className="flame-rows">
        {index.models.map((m) => (
          <Row
            key={m.key}
            m={m}
            tok={tok}
            count={work.models[m.key]?.pre ?? 0}
            selected={m.key === model}
            onSelect={() => setModel(m.key)}
            onHover={(i, e) => setHover(i == null ? null : { i, model: m.key, x: e.clientX, y: e.clientY })}
          />
        ))}
      </div>

      <div className="flame-legend">
        <span className="lg"><span className="sw" style={{ background: "var(--blank)" }} /> not extractable</span>
        <span className="lg">
          10⁻³ {STEPS.map((s, i) => <span key={s} className="sw" style={{ background: `var(--mem-${i + 2})` }} />)} ≈1
          <span className="k">extraction probability p<sub>z</sub></span>
        </span>
      </div>

      {hv && (
        <div className="tooltip" style={{ opacity: 1, left: Math.min(hover.x + 12, window.innerWidth - 270), top: hover.y + 14 }}>
          <b>{index.models.find((x) => x.key === hover.model).name}</b> · token “{tok.text.slice(...hv.spans[hover.i]).trim()}”
          <br /><span className="k">p<sub>z</sub></span> {fmtP(hv.vals.pre?.[hover.i] ?? null)}
        </div>
      )}
    </div>
  );
}

function Row({ m, tok, count, selected, onSelect, onHover }) {
  const ref = useRef(null);
  const t = tok.models[m.key];
  const n = tok.text.length;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const draw = () => {
      const w = cv.clientWidth, h = 22, dpr = window.devicePixelRatio || 1;
      cv.width = Math.round(w * dpr);
      cv.height = h * dpr;
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = cssVar("--blank");
      ctx.fillRect(0, 0, w, h);
      if (!t?.vals.pre) return;
      const colors = {};
      t.spans.forEach(([a, b], i) => {
        const lv = memLevel(t.vals.pre[i]);
        if (!lv) return;
        ctx.fillStyle = colors[lv] ??= cssVar(`--mem-${lv}`);
        const x0 = (a / n) * w, x1 = Math.max((b / n) * w, x0 + 1);
        ctx.fillRect(x0, 0, x1 - x0, h);
      });
    };
    draw();
    window.addEventListener("resize", draw);
    return () => window.removeEventListener("resize", draw);
  }, [t, n]);

  const tokenAt = (e) => {
    const c = (e.nativeEvent.offsetX / e.currentTarget.clientWidth) * n;
    let lo = 0, hi = t.spans.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (t.spans[mid][1] <= c) lo = mid + 1; else hi = mid;
    }
    return lo;
  };

  return (
    <div className={`row${selected ? " sel" : ""}`}>
      <button className="name" title="Highlight the lyrics this model can reproduce" onClick={onSelect}>{m.name}</button>
      <canvas
        ref={ref}
        className="strip"
        onMouseMove={(e) => t && onHover(tokenAt(e), e)}
        onMouseLeave={() => onHover(null)}
        onClick={onSelect}
      />
      <div className="count">{count} <span className="k">extractable windows</span></div>
    </div>
  );
}
