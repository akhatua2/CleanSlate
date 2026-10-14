"use client";
// Figure: the time machine. Every song in the benchmark, by release year (x) and how many
// mid-training documents share text with it (y, log). Filled = some document copies a whole
// line (>= 8 consecutive words). Drag through the years: old songs have spread outward into
// the web; songs released after the corpus cutoff still match hundreds of documents, but only
// through stock phrases they inherited (inward diffusion). Tap a dot to read it.
import { useEffect, useMemo, useRef, useState } from "react";
import "./time.css";

const Y0 = 1970, Y1 = 2025;
const CUTOFF = 2025; // songs from here on were released after the corpus was collected

function rnd(s, k) {
  let h = 2166136261;
  for (const ch of s + k) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return ((h >>> 0) % 10000) / 10000;
}
const css = (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim();

export default function TimeFigure() {
  const [d, setD] = useState(null);
  const [year, setYear] = useState(1987);
  const [pick, setPick] = useState(null);
  const [hover, setHover] = useState(null);
  const wrap = useRef(null), cv = useRef(null), line = useRef(null);
  const [W, setW] = useState(900);

  useEffect(() => { fetch("data/time.json").then((r) => r.json()).then(setD); }, []);
  useEffect(() => {
    const ro = new ResizeObserver(() => setW(wrap.current?.clientWidth ?? 900));
    if (wrap.current) ro.observe(wrap.current);
    return () => ro.disconnect();
  }, [d]);

  const narrow = W < 640;
  const H = narrow ? 300 : 380;
  const PAD = { l: narrow ? 30 : 44, r: 12, t: 12, b: 26 };

  const geo = useMemo(() => {
    if (!d) return null;
    const maxD = Math.max(...d.songs.map((s) => s.d));
    const x = (y) => PAD.l + ((y - Y0) / (Y1 - Y0 + 1)) * (W - PAD.l - PAD.r);
    const colW = (W - PAD.l - PAD.r) / (Y1 - Y0 + 1);
    const ly = (v) => {
      const t = Math.log10(Math.max(1, v)) / Math.log10(maxD);
      return PAD.t + (1 - t) * (H - PAD.t - PAD.b);
    };
    const pts = d.songs.map((s) => ({
      ...s,
      px: x(s.y) + colW * (0.12 + 0.76 * rnd(s.id, "x")),
      py: ly(s.d) + (rnd(s.id, "y") - 0.5) * 3,
    }));
    return { pts, x, colW, ly, maxD };
  }, [d, W, H]);

  // draw the swarm: the scrubbed year in full, everything else faint
  useEffect(() => {
    const c = cv.current;
    if (!c || !geo) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const blue = css("--cor-5"), ink3 = css("--ink-3"), rule = css("--rule"), paper = css("--paper"), red = css("--mem-5");
    // after-cutoff band
    const xc = geo.x(CUTOFF);
    ctx.fillStyle = "rgba(0,0,0,0.035)";
    ctx.fillRect(xc, PAD.t, W - PAD.r - xc, H - PAD.t - PAD.b);
    // y gridlines (1, 10, 100, ...)
    ctx.fillStyle = ink3; ctx.font = `11px ${css("--sans")}`; ctx.textAlign = "right";
    for (let v = 1; v <= geo.maxD; v *= 10) {
      const y = geo.ly(v);
      ctx.fillStyle = rule; ctx.fillRect(PAD.l, Math.round(y), W - PAD.l - PAD.r, 1);
      ctx.fillStyle = ink3; ctx.fillText(String(v), PAD.l - 6, y + 4);
    }
    // x ticks
    ctx.textAlign = "center";
    for (let y = 1970; y <= 2025; y += narrow ? 10 : 5) ctx.fillText(String(y), geo.x(y) + geo.colW / 2, H - 8);
    const r = narrow ? 2.2 : 2.6;
    const drawDot = (p, on) => {
      ctx.beginPath(); ctx.arc(p.px, p.py, on ? r + 1.4 : r, 0, Math.PI * 2);
      if (p.s > 0) { ctx.fillStyle = on ? blue : blue; ctx.globalAlpha = on ? 1 : 0.16; ctx.fill(); }
      else { ctx.strokeStyle = on ? ink3 : ink3; ctx.lineWidth = 1; ctx.globalAlpha = on ? 0.95 : 0.14; ctx.fillStyle = paper; ctx.fill(); ctx.stroke(); }
    };
    for (const p of geo.pts) if (p.y !== year) drawDot(p, false);
    for (const p of geo.pts) if (p.y === year) drawDot(p, true);
    ctx.globalAlpha = 1;
    // year column outline
    ctx.strokeStyle = css("--ink-1"); ctx.lineWidth = 1;
    ctx.strokeRect(geo.x(year) + 0.5, PAD.t + 0.5, geo.colW - 1, H - PAD.t - PAD.b - 1);
    const sel = pick && geo.pts.find((p) => p.id === pick.id);
    if (sel) { ctx.strokeStyle = red; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(sel.px, sel.py, r + 5, 0, Math.PI * 2); ctx.stroke(); }
  }, [geo, year, W, H, pick, narrow]);

  // the share line, drawn up to the scrubbed year
  useEffect(() => {
    const c = line.current;
    if (!c || !geo || !d) return;
    const h = 70, dpr = window.devicePixelRatio || 1;
    c.width = Math.round(W * dpr); c.height = h * dpr;
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, h);
    const maxS = Math.max(...d.years.map((r) => r.share));
    const yy = (s) => 8 + (1 - s / maxS) * (h - 16);
    const pts = d.years.map((r) => [geo.x(r.y) + geo.colW / 2, yy(r.share), r.y]);
    ctx.strokeStyle = css("--rule"); ctx.lineWidth = 1.5; ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
    ctx.strokeStyle = css("--cor-5"); ctx.lineWidth = 2.2; ctx.beginPath();
    pts.filter((p) => p[2] <= year).forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
    const cur = pts.find((p) => p[2] === year);
    if (cur) { ctx.fillStyle = css("--cor-6"); ctx.beginPath(); ctx.arc(cur[0], cur[1], 4, 0, Math.PI * 2); ctx.fill(); }
  }, [geo, d, year, W]);

  if (!d || !geo) return <div className="tm ui">Loading…</div>;

  const yr = d.years.find((r) => r.y === year) ?? { n: 0, share: 0, med: 0 };
  const after = year >= CUTOFF;
  const pickedDetail = pick && d.detail[pick.id];
  const lateIds = Object.keys(d.detail);

  const nearest = (e) => {
    const rect = cv.current.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bd = 64;
    for (const p of geo.pts) {
      const dd = (p.px - mx) ** 2 + (p.py - my) ** 2 + (p.y === year ? 0 : 30);
      if (dd < bd) { bd = dd; best = p; }
    }
    return best;
  };

  return (
    <div className="tm ui" ref={wrap}>
      <div className="tm-head">
        <span className="tm-year">{year}</span>
        <p className="tm-say">
          {after ? (
            <>Released after the corpus was collected, yet these {yr.n} songs share text with a median of <b>{Math.round(yr.med)}</b> training documents. Only <b>{(100 * yr.share).toFixed(0)}%</b> have a whole line copied; the rest is borrowed phrasing.</>
          ) : (
            <>Of the {yr.n} songs from {year}, <b>{(100 * yr.share).toFixed(0)}%</b> have a whole line copied into the training data, and a typical one shares text with <b>{Math.round(yr.med)}</b> documents.</>
          )}
        </p>
      </div>

      <div className="tm-plot">
        <canvas
          ref={cv}
          className="tm-cv"
          style={{ height: H }}
          onPointerMove={(e) => { if (e.pointerType === "mouse") setHover(nearest(e)); }}
          onPointerLeave={() => setHover(null)}
          onClick={(e) => { const p = nearest(e); if (p) { setPick(p); setYear(p.y); } }}
          role="img"
          aria-label={`${d.songs.length} songs from ${Y0} to ${Y1}, by how many training documents share their text`}
        />
        <div className="tm-ylab">documents sharing text (log)</div>
        <div className="tm-cut" style={{ left: geo.x(CUTOFF) }}>after the corpus cutoff</div>
        {hover && (
          <div className="tm-tip" style={{ left: Math.min(Math.max(hover.px, 110), W - 110), top: hover.py + 14 }}>
            <b>{hover.t}</b> <span>· {hover.a} · {hover.y}</span>
            <div>{hover.d} documents · {hover.s > 0 ? `${hover.s} with a whole line` : "no whole-line copies"}</div>
          </div>
        )}
      </div>

      <canvas ref={line} className="tm-line" />
      <div className="tm-linelab">share of songs with a whole line copied, by release year</div>

      <input
        className="tm-scrub"
        type="range" min={Y0} max={Y1} step={1} value={year}
        onChange={(e) => { setYear(+e.target.value); setPick(null); }}
        aria-label="Release year"
        style={{ "--p": `${((year - Y0) / (Y1 - Y0)) * 100}%` }}
      />

      {/* always rendered so scrubbing to 2025 never pushes the page down */}
      {(
        <nav className={`tabs small tm-late${after ? "" : " off"}`} aria-label="2025 songs" aria-hidden={!after}>
          {lateIds.map((id) => {
            const s = d.songs.find((x) => x.id === id);
            return <button key={id} aria-pressed={pick?.id === id} onClick={() => setPick(s)}>{s.t}</button>;
          })}
        </nav>
      )}

      {pick && (
        <div className="tm-panel">
          <div className="hd">
            <span className="t">{pick.t}</span> <span className="by">{pick.a} · {pick.y}</span>
            <button className="x" onClick={() => setPick(null)} aria-label="Close">✕</button>
          </div>
          <div className="stats">{pick.d} documents share text with it · {pick.s > 0 ? `${pick.s} copy a whole line` : "none copies a whole line"}</div>
          {pickedDetail ? (
            <>
              <div className="lyr">{marked(pickedDetail.text, pickedDetail.marks)}</div>
              <div className="ex">
                {pickedDetail.examples.map((e) => (
                  <div key={e.phrase} className="doc">
                    <div className="src">“{e.phrase}” · in {e.count >= 25 ? "25+" : e.count} documents, e.g. {e.src}</div>
                    <div className="snip">{e.pre && "…"}{e.snippet.slice(0, e.at[0])}<mark>{e.snippet.slice(e.at[0], e.at[1])}</mark>{e.snippet.slice(e.at[1])}{e.post && "…"}</div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="hint">{after ? "Pick one of the songs above to read it." : "Drag to 2025 to see songs released after the corpus was collected."}</div>
          )}
        </div>
      )}

      <div className="tm-legend">
        <span><i className="sw full" /> some document copies a whole line (≥ {d.strong_words} words in a row)</span>
        <span><i className="sw" /> shares only shorter phrases</span>
      </div>
    </div>
  );
}

function marked(text, marks) {
  const out = [];
  let k = 0;
  for (const [a, b, c] of marks) {
    if (a > k) out.push(<span key={`t${k}`}>{text.slice(k, a)}</span>);
    out.push(<mark key={`m${a}`} title={`in ${c >= 25 ? "25+" : c} older documents`}>{text.slice(a, b)}</mark>);
    k = b;
  }
  out.push(<span key="end">{text.slice(k)}</span>);
  return out;
}
