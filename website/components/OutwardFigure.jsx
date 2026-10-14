"use client";
// Figure: outward diffusion as a cluster map. Every point is one of the 147 mid-training
// documents that copy at least five words of Never Gonna Give You Up, grouped by where it
// came from. Filled points copy a whole line or more; hollow ones share only a short,
// common phrase. Pages we photographed sit on the map at their point. Click to open.
import { useEffect, useMemo, useRef, useState } from "react";
import "./outward.css";

// cluster centres in a unit box (x, y), laid out like a 2-D projection
const CENTRES = {
  Code: [0.22, 0.27],
  Reddit: [0.5, 0.12],
  "Entertainment & games": [0.78, 0.29],
  "Literature & fiction": [0.87, 0.68],
  "Other web": [0.62, 0.76],
  "Synthetic Q&A": [0.36, 0.76],
  "Instruction data": [0.17, 0.66],
};
const HUB = [0.5, 0.5];
// where a cluster's single page card goes (radians, screen coords): into the nearest free space
const LONE = { Code: Math.PI, Reddit: 0, "Other web": Math.PI / 2 };
// phones: same map, re-laid out for a tall screen
const CENTRES_M = {
  Reddit: [0.3, 0.135],
  "Entertainment & games": [0.73, 0.2],
  Code: [0.24, 0.32],
  "Literature & fiction": [0.78, 0.37],
  "Synthetic Q&A": [0.27, 0.74],
  "Other web": [0.74, 0.73],
  "Instruction data": [0.5, 0.9],
};
const COLORS = {
  Code: "#2a78d6",
  Reddit: "#eb6834",
  "Entertainment & games": "#1baf7a",
  "Literature & fiction": "#4a3aa7",
  "Other web": "#8a8f98",
  "Synthetic Q&A": "#e87ba4",
  "Instruction data": "#c98500",
};

function rnd(s, k) {
  let h = 2166136261;
  for (const ch of s + k) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return ((h >>> 0) % 10000) / 10000;
}

export default function OutwardFigure() {
  const [pts, setPts] = useState(null);
  const [tip, setTip] = useState(null);
  const box = useRef(null);
  const [W, setW] = useState(900);

  useEffect(() => { fetch("data/scatter.json").then((r) => r.json()).then(setPts); }, []);
  useEffect(() => {
    const ro = new ResizeObserver(() => setW(box.current?.clientWidth ?? 900));
    if (box.current) ro.observe(box.current);
    return () => ro.disconnect();
  }, [pts]);

  const narrow = W < 640;
  const H = narrow ? Math.round(W * 1.75) : Math.round(Math.min(680, W * 0.62));
  const C = narrow ? CENTRES_M : CENTRES;

  // place each document around its cluster centre on a sunflower spiral, with a little noise
  const placed = useMemo(() => {
    if (!pts) return [];
    const byG = {};
    for (const p of pts) (byG[p.g] ??= []).push(p);
    const out = [];
    for (const [g, list] of Object.entries(byG)) {
      const [cx, cy] = C[g];
      const spread = narrow ? 0.026 : 0.02;
      list
        .sort((a, b) => (b.shot ? 1 : 0) - (a.shot ? 1 : 0) || b.w - a.w)
        .forEach((p, i) => {
          const r = spread * Math.sqrt(i + 0.6) * (1 + 0.35 * (rnd(String(p.ix), "r") - 0.5));
          const t = i * 2.39996 + rnd(String(p.ix), "t") * 0.6;
          out.push({ ...p, x: cx + r * Math.cos(t) * (narrow ? 1.25 : 1), y: cy + r * Math.sin(t) * (narrow ? 0.62 : 1.7) });
        });
    }
    return out;
  }, [pts, narrow, C]);

  const [bottoms, rights, tops] = useMemo(() => {
    const b = {}, r = {}, t = {};
    for (const p of placed) { b[p.g] = Math.max(b[p.g] ?? 0, p.y); r[p.g] = Math.max(r[p.g] ?? 0, p.x); t[p.g] = Math.min(t[p.g] ?? 1, p.y); }
    return [b, r, t];
  }, [placed]);

  // thumbnails sit outside their cloud, fanned around it, each tied to its own dot
  const thumbs = useMemo(() => {
    if (narrow) return [];
    const byG = {};
    for (const p of placed) if (p.shot) (byG[p.g] ??= []).push(p);
    const out = [];
    for (const [g, list] of Object.entries(byG)) {
      const [cx, cy] = CENTRES[g];
      // fan away from the figure's centre so cards don't collide with other clusters
      const base = Math.atan2(cy - HUB[1], cx - HUB[0]);
      list.forEach((p, k) => {
        // a lone card swings off to the side so it doesn't cover the cluster's label
        const a = list.length === 1 ? (LONE[g] ?? (cx < HUB[0] ? Math.PI : 0)) : base + (k - (list.length - 1) / 2) * (narrow ? 1.1 : 1.05);
        const dist = narrow ? 0.22 : 0.17;
        // keep the whole card inside the figure
        const hw = (narrow ? 58 : 82) / W, hh = (narrow ? 42 : 56) / H;
        const tx = Math.min(1 - hw, Math.max(hw, cx + Math.cos(a) * dist * (narrow ? 1 : 0.85)));
        const ty = Math.min(1 - hh, Math.max(hh, cy + Math.sin(a) * dist * (narrow ? 0.7 : 1.25)));
        out.push({ ...p, tx, ty });
      });
    }
    return out;
  }, [placed, narrow, W, H]);

  if (!pts) return <div className="ow ui">Loading…</div>;
  const counts = Object.fromEntries(Object.keys(CENTRES).map((g) => [g, pts.filter((p) => p.g === g).length]));
  const open = (p) => p.url && window.open(p.arch ? `https://web.archive.org/web/${p.arch.replaceAll("-", "")}/${p.url}` : p.url, "_blank", "noopener");

  return (
    <div className="ow ui">
      <div className="ow-map" ref={box} style={{ height: H }} onMouseLeave={() => setTip(null)}>
        {Object.entries(C).map(([g, [cx, cy]]) => (
          <div key={g} className={`ow-label${!narrow && LONE[g] === Math.PI / 2 ? " side" : ""}`} style={
            // labels sit under their cloud; if the cluster's page card hangs below it, beside the cloud instead
            !narrow && LONE[g] === Math.PI / 2
              ? { left: (rights[g] ?? cx) * W + 10, top: (tops[g] ?? cy) * H - 6, color: COLORS[g] }
              : { left: cx * W, top: (bottoms[g] ?? cy) * H + 10, color: COLORS[g] }
          }>
            {g} <span>{counts[g]}</span>
          </div>
        ))}

        {placed.map((p) => (
          <button
            key={p.ix}
            className={`ow-dot${p.real ? " real" : ""}${p.url ? "" : " nolink"}${narrow && p.shot ? " photo" : ""}`}
            style={{ left: p.x * W, top: p.y * H, "--c": COLORS[p.g] }}
            aria-label={`${p.host ?? p.g}: “${p.line}”`}
            onMouseEnter={(e) => setTip({ p, x: p.x * W, y: p.y * H })}
            onFocus={() => setTip({ p, x: p.x * W, y: p.y * H })}
            onClick={() => (narrow ? setTip({ p, sheet: true }) : open(p))}
          />
        ))}



        <div className="ow-hub" style={{ left: HUB[0] * W, top: HUB[1] * H }}>
          <div className="t">Never Gonna Give You Up</div>
          <div className="a">Rick Astley · 1987</div>
          <div className="l">
            We&rsquo;re no strangers to love<br />You know the rules and so do I<br />A full commitment&rsquo;s what I&rsquo;m thinking of
          </div>
        </div>



        {thumbs.map((p) => {
          const rot = (rnd(String(p.ix), "rot") - 0.5) * 6;
          return (
            <a
              key={`s${p.ix}`}
              className="ow-thumb"
              href={p.arch ? `https://web.archive.org/web/${p.arch.replaceAll("-", "")}/${p.url}` : p.url}
              target="_blank"
              rel="noreferrer"
              style={{ left: p.tx * W, top: p.ty * H, "--r": `${rot}deg`, "--c": COLORS[p.g] }}
            >
              <span className="bar"><i /><i /><i /><b>{p.host}</b></span>
              <img src={p.shot} alt={`${p.host}: “${p.line}”`} loading="lazy" />
            </a>
          );
        })}

        {tip && !tip.sheet && !narrow && (
          <div className="ow-tip" style={{ left: Math.min(Math.max(tip.x, 120), W - 120), top: tip.y + 14 }}>
            <div className="h">{tip.p.host ? <>{tip.p.host} <span>· {tip.p.g}</span></> : tip.p.g}</div>
            <div className="q">“{tip.p.line}”</div>
            <div className="k">
              {tip.p.real ? "copies a whole line" : "shares only a short phrase"}
              {tip.p.url
                ? (tip.p.shot ? " · click to open" : " · click to open the source URL (the page may have changed since it was crawled)")
                : ""}
            </div>
          </div>
        )}
      </div>

      {tip?.sheet && (
        <div className="ow-sheet" role="dialog" aria-label="Document details">
          <button className="x" onClick={() => setTip(null)} aria-label="Close">✕</button>
          <div className="h">{tip.p.host ? <>{tip.p.host} <span>· {tip.p.g}</span></> : tip.p.g}</div>
          {tip.p.shot && <img className="shot" src={tip.p.shot} alt={`${tip.p.host}: “${tip.p.line}”`} />}
          <div className="q">“{tip.p.line}”</div>
          <div className="k">{tip.p.real ? "copies a whole line" : "shares only a short phrase"}</div>
          {tip.p.url && (
            <a className="open" href={tip.p.arch ? `https://web.archive.org/web/${tip.p.arch.replaceAll("-", "")}/${tip.p.url}` : tip.p.url} target="_blank" rel="noreferrer">
              {tip.p.shot ? "Open page ↗" : "Open source URL ↗ (may have changed)"}
            </a>
          )}
        </div>
      )}

      <div className="ow-legend">
        {narrow && <span><i className="sw photo" /> tap for a screenshot of the page</span>}
        <span><i className="sw real" /> copies a whole line or more</span>
        <span><i className="sw" /> shares only a short, common phrase</span>
        <span className="k">{pts.length} documents in the mid-training data · {pts.filter((p) => p.url).length} with a source URL</span>
      </div>
    </div>
  );
}
