"use client";
// Figure: guess first, then see. (1) Draw how likely a song is to have a whole line copied
// into the training data, by release year; the real curve then draws over your guess.
// (2) A song released after the corpus was collected: tap the lines you think already
// existed, then reveal the borrowed phrases and where they come from.
import { useEffect, useMemo, useRef, useState } from "react";
import "./draw.css";

const Y0 = 1970, Y1 = 2025, YMAX = 0.5;
const GAME_LINES = 18;

export default function DrawFigure() {
  const [d, setD] = useState(null);
  useEffect(() => { fetch("data/time.json").then((r) => r.json()).then(setD); }, []);
  if (!d) return <div className="dr ui">Loading…</div>;
  return (
    <div className="dr ui">
      <DrawIt d={d} />
      <Borrowed />
    </div>
  );
}

// Coverage by release year, line emphasized: the median share of a song's words that also
// appear in other Dolmino text (>= 5-word matches), with the middle half of songs as a band.
function DrawIt({ d }) {
  const wrap = useRef(null);
  const [W, setW] = useState(900);
  useEffect(() => {
    const ro = new ResizeObserver(() => setW(wrap.current?.clientWidth ?? 900));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);
  const narrow = W < 640;
  const H = narrow ? 260 : 320;
  const P = { l: 44, r: 16, t: 16, b: 30 };
  const TOP = 0.5;
  const x = (y) => P.l + ((y - Y0) / (Y1 - Y0)) * (W - P.l - P.r);
  const yv = (v) => P.t + (1 - v / TOP) * (H - P.t - P.b);
  const cov = d.coverage;
  // 3-year rolling median keeps the line readable
  const sm = cov.map((r) => {
    const w = cov.filter((q) => Math.abs(q.y - r.y) <= 1);
    return { y: r.y, v: w.reduce((a, q) => a + q.med * q.n, 0) / w.reduce((a, q) => a + q.n, 0), q1: r.q1, q3: r.q3, raw: r.med };
  });
  const line = sm.map((r, i) => `${i ? "L" : "M"}${x(r.y).toFixed(1)},${yv(r.v).toFixed(1)}`).join("");
  const band = sm.map((r, i) => `${i ? "L" : "M"}${x(r.y).toFixed(1)},${yv(r.q3).toFixed(1)}`).join("") +
    [...sm].reverse().map((r) => `L${x(r.y).toFixed(1)},${yv(r.q1).toFixed(1)}`).join("") + "Z";
  const marks = (d.marks ?? []).map((m) => ({ ...m, r: sm.find((r) => r.y === m.y) })).filter((m) => m.r && d.covers?.[m.id]);
  const end = sm[sm.length - 1];
  const avg = Math.round(d.coverage_mean * 100);

  return (
    <section className="dr-1">
      <h3 className="dr-q">On average, about {avg}% of a hit song’s words already appear in other text in the training data, from quotes and copies to everyday phrases.</h3>
      <div className="dr-chart" ref={wrap}>
        <svg width={W} height={H} role="img" aria-label="Median share of a song's words found elsewhere in the training data, by year">
          {[0, 0.1, 0.2, 0.3, 0.4, 0.5].map((v) => (
            <g key={v}>
              <line x1={P.l} x2={W - P.r} y1={yv(v)} y2={yv(v)} className="grid" />
              <text x={P.l - 8} y={yv(v) + 4} className="ax" textAnchor="end">{Math.round(v * 100)}%</text>
            </g>
          ))}
          {[1970, 1980, 1990, 2000, 2010, 2020, 2025].filter((y) => !narrow || y % 20 === 10 || y === 1970 || y === 2025).map((y) => (
            <text key={y} x={x(y)} y={H - 8} className="ax" textAnchor="middle">{y}</text>
          ))}
          <path d={band} className="band" />
          {sm.map((r) => <circle key={r.y} cx={x(r.y)} cy={yv(r.raw)} r="3" className="sample" />)}
          <path d={line} className="real" />
          {marks.map((m) => {
            const up = m.y === 2008 || m.y === 1984 || m.y === 1995 || m.y === 2020, ms = narrow ? 30 : 40;
            const cy = yv(m.r.v);
            return (
              <g key={m.id} className="mark">
                <title>{m.t} · {m.a} · {m.y}</title>
                <line x1={x(m.y)} x2={x(m.y)} y1={cy} y2={up ? cy - 20 : cy + 20} />
                <image href={d.covers[m.id]} x={x(m.y) - ms / 2} y={up ? cy - 20 - ms : cy + 20} width={ms} height={ms} className="cover" />
              </g>
            );
          })}
          <g className="note">
            <circle cx={x(end.y)} cy={yv(end.v)} r="5" />
            <text x={x(end.y) - 10} y={yv(end.v) + 22} textAnchor="end">{end.y}: {Math.round(end.v * 100)}%</text>
          </g>
        </svg>
      </div>
      <p className="dr-sub">Share of each song’s words covered by 5+ word sequences that also occur elsewhere in Dolmino. Line: median song (3-year average); band: middle half of songs; dots: each year. Year = Billboard year-end chart.</p>
    </section>
  );
}

const ROUNDS = 99;

function srcKind(src) {
  if (!src) return "web";
  if (src.startsWith("code")) return "code";
  if (src.startsWith("synthetic")) return "qa";
  if (src.startsWith("instruction")) return "inst";
  return "web";
}
function hashStr(s) {
  let h = 2166136261;
  for (const ch of s) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return h >>> 0;
}

// one round per song: its longest borrowed phrase whose source document we have,
// plus two other songs as wrong answers
function buildRounds(d) {
  const ids = Object.keys(d.detail);
  const rounds = [];
  for (const id of ids) {
    const det = d.detail[id];
    let best = null;
    for (const l of det.lines) for (const m of l.m) {
      if (!m[3] || m[1] - m[0] < 18) continue;
      if (!best || m[1] - m[0] > best.m[1] - best.m[0]) best = { line: l.t, m };
    }
    if (!best) continue;
    const others = ids.filter((x) => x !== id).sort((p, q) => hashStr(id + p) - hashStr(id + q)).slice(0, 2);
    const options = [id, ...others].sort((p, q) => hashStr(p + "o") - hashStr(q + "o"));
    rounds.push({ id, ...best, options });
  }
  return rounds.sort((p, q) => hashStr(p.id) - hashStr(q.id)).slice(0, ROUNDS);
}

// The paper's post-cutoff examples (Figure 2): songs released after Dolmino's December 2024
// knowledge cutoff, and the phrase from each that was already in the data.
const PAPER = [
  { k: "risk_it_all", t: "Risk It All", a: "Bruno Mars", y: 2026, phrase: "I could go on and on" },
  { k: "be_her", t: "Be Her", a: "Ella Langley", y: 2026, phrase: "the love of her life" },
  { k: "fate_of_ophelia", t: "The Fate of Ophelia", a: "Taylor Swift", y: 2026, phrase: "the land, the sea, the sky" },
];

function Borrowed() {
  const [docs, setDocs] = useState(null);
  useEffect(() => { fetch("data/paper_docs.json").then((r) => r.json()).then(setDocs); }, []);
  return (
    <section className="dr-2">
      <h3 className="dr-q">These songs came out after the training data was collected. Their words were already in it.</h3>
      <p className="dr-src"><img src="dolmino.png" alt="Dolmino" height="18" /> Documents from OLMo 3’s mid-training data, collected before December 2024.</p>
      <div className="pairs">
        {PAPER.map((p) => {
          const hit = docs?.[p.phrase];
          return (
            <article key={p.k} className="pair">
              <div className="lyric">
                <img src={`covers/paper_${p.k}.jpg`} alt="" width="72" height="72" />
                <div>
                  <div className="who"><b>{p.t}</b> · {p.a} · {p.y}</div>
                  <div className="line">“<mark>{p.phrase}</mark>”</div>
                </div>
              </div>
              <div className="stack">
                {(hit?.docs ?? []).map((x, i) => (
                  <figure key={i} className={`clipdoc ${srcKind(x.src)}`} style={{ "--i": i }}>
                    <figcaption className="where"><span className="topic">{label(x.src)}</span></figcaption>
                    <div className="body">{x.pre && "…"}{x.snippet.slice(0, x.at[0])}<mark>{x.snippet.slice(x.at[0], x.at[1])}</mark>{x.snippet.slice(x.at[1])}{x.post && "…"}</div>
                  </figure>
                ))}
              </div>
            </article>
          );
        })}
      </div>
      <p className="dr-more">The paper finds more: <i>Smile</i> (Morgan Wallen) shares a line with a tweet that was scraped into FLAN, and <i>Red Terror</i> (The Weeknd) quotes a poem from 1910.</p>
    </section>
  );
}

function label(src) {
  if (src.startsWith("web: ")) return `Web page · ${src.slice(5)}`;
  return { code: "Source code", "synthetic Q&A": "Synthetic Q&A", "instruction data": "Instruction data" }[src] ?? src;
}

// the first verse and chorus, each line once
function stanza(det) {
  const seen = new Set(), out = [];
  for (const l of det.lines) {
    const k = l.t.trim().toLowerCase();
    if (!k || seen.has(k)) continue;
    seen.add(k); out.push(l);
    if (out.length >= 14) break;
  }
  return out;
}
function pieces(l) {
  const out = [];
  let k = 0;
  for (const m of [...l.m].sort((a, b) => a[0] - b[0])) {
    if (m[0] < k) continue;
    if (m[0] > k) out.push({ text: l.t.slice(k, m[0]) });
    out.push({ text: l.t.slice(m[0], m[1]), m });
    k = m[1];
  }
  if (k < l.t.length) out.push({ text: l.t.slice(k) });
  return out;
}

// a borrowed phrase, shown as a clipping of the older document it was found in
function Clipping({ m, reveal }) {
  const info = m[3];
  if (!info) return null;
  const kind = srcKind(info.src);
  const topic = info.src.startsWith("web: ") ? `Web page · ${info.src.slice(5)}` : { code: "Source code", qa: "Synthetic Q&A", inst: "Instruction data" }[kind] ?? "Web page";
  const body = info.snippet;
  return (
    <figure className={`clipdoc ${kind}`}>
      <figcaption className="where">
        <span className="topic">{topic}</span>
      </figcaption>
      <div className="body">…{body.slice(0, info.at[0])}{reveal ? <mark>{body.slice(info.at[0], info.at[1])}</mark> : body.slice(info.at[0], info.at[1])}{body.slice(info.at[1])}…</div>
    </figure>
  );
}

function marked(text, marks) {
  const out = [];
  let k = 0;
  for (const [a, b] of [...marks].sort((p, q) => p[0] - q[0])) {
    if (a < k) continue;
    if (a > k) out.push(<span key={`t${k}`}>{text.slice(k, a)}</span>);
    out.push(<mark key={`m${a}`}>{text.slice(a, b)}</mark>);
    k = b;
  }
  out.push(<span key="e">{text.slice(k)}</span>);
  return out;
}
