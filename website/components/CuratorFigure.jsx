"use client";
// Figure: you are the curator (outward diffusion). The lyrics are shaded by how many
// mid-training documents contain each stretch of text. Every mark on the right is one of
// those documents; tap it to remove it from the training data and watch what is left.
import { useEffect, useMemo, useState } from "react";
import "./curator.css";

const MIN_WORDS = 5; // the paper's 5-gram floor
const CATS = ["Web pages", "Synthetic Q&A", "Instruction tuning", "Code & software", "Wikipedia", "Math", "Other"];

function corLevel(n) {
  if (n <= 0) return 0;
  if (n === 1) return 1;
  if (n <= 3) return 2;
  if (n <= 10) return 3;
  if (n <= 30) return 4;
  if (n <= 100) return 5;
  return 6;
}

export default function CuratorFigure() {
  const [index, setIndex] = useState(null);
  const [id, setId] = useState(null);
  const [d, setD] = useState(null);
  const [removed, setRemoved] = useState(() => new Set());
  const [peek, setPeek] = useState(null); // doc index shown in the preview

  useEffect(() => {
    fetch("data/flame_index.json").then((r) => r.json()).then((ix) => {
      setIndex(ix);
      // start on the paper's running example
      const nggyu = ix.works.find((w) => w.title === "Never Gonna Give You Up");
      setId(nggyu?.id ?? ix.featured[0]);
    });
  }, []);
  useEffect(() => {
    if (!id) return;
    setD(null); setRemoved(new Set()); setPeek(null);
    fetch(`data/footprint/${id}.json`).then((r) => r.json()).then(setD);
  }, [id]);

  const docs = useMemo(() => (d ? d.docs.map((x, i) => ({ ...x, i })).filter((x) => x.words >= MIN_WORDS) : []), [d]);

  // per-character count of remaining documents containing it
  const cov = useMemo(() => {
    if (!d) return null;
    const c = new Uint16Array(d.text.length);
    for (const x of docs) if (!removed.has(x.i)) for (const [a, b] of x.spans) for (let k = a; k < b; k++) c[k]++;
    return c;
  }, [d, docs, removed]);

  if (!index || !d || !cov) return <div className="cu ui">Loading…</div>;

  const nonSpace = [...d.text].reduce((n, ch) => n + (/\s/.test(ch) ? 0 : 1), 0);
  let covered = 0;
  for (let k = 0; k < d.text.length; k++) if (cov[k] > 0 && !/\s/.test(d.text[k])) covered++;
  const pct = (100 * covered) / nonSpace;
  const left = docs.length - [...removed].filter((i) => docs.some((x) => x.i === i)).length;
  const maxCopies = cov.reduce((m, v) => Math.max(m, v), 0);

  // fullest copies first: documents covering the most characters of the song
  const fullest = [...docs].sort((a, b) => span(b) - span(a));
  const removeFullest = () => {
    const next = new Set(removed);
    let k = 0;
    for (const x of fullest) { if (!next.has(x.i)) { next.add(x.i); k++; } if (k === 10) break; }
    setRemoved(next);
  };

  const featured = index.featured.map((fid) => index.works.find((w) => w.id === fid)).filter(Boolean);
  const byCat = CATS.map((c) => [c, docs.filter((x) => x.cat === c)]).filter(([, xs]) => xs.length);

  // render the lyrics as runs of equal coverage level
  const runs = [];
  for (let k = 0; k < d.text.length;) {
    const lv = /\s/.test(d.text[k]) ? -1 : corLevel(cov[k]);
    let j = k + 1;
    while (j < d.text.length && (/\s/.test(d.text[j]) ? -1 : corLevel(cov[j])) === lv) j++;
    runs.push([k, j, lv]);
    k = j;
  }

  const pd = peek != null ? d.docs[peek] : null;

  return (
    <div className="cu ui">
      <nav className="tabs" aria-label="Pick a song">
        {featured.map((w) => (
          <button key={w.id} aria-pressed={w.id === id} onClick={() => setId(w.id)}>{w.title}</button>
        ))}
      </nav>

      <p className="cu-say">
        {removed.size === 0
          ? <>We found <b>{docs.length}</b> documents in the mid-training data that copy at least five words of <i>{d.title}</i>. Tap them to remove them.</>
          : <>You removed <b>{removed.size}</b> of {docs.length} documents. <b>{pct.toFixed(0)}%</b> of the song is still somewhere in the training data{maxCopies > 1 ? <>, and its most-copied line still appears in <b>{maxCopies}</b> documents</> : null}.</>}
      </p>

      <div className="cu-body">
        <div className="cu-lyrics">
          <div className="cu-meter">
            <span className="num">{pct.toFixed(0)}%</span>
            <span className="lab">of the song still in the corpus</span>
          </div>
          <div className="text">
            {runs.map(([a, b, lv]) =>
              lv > 0
                ? <span key={a} className="c" style={{ background: `var(--cor-${lv})`, color: lv >= 5 ? "#fff" : undefined }}>{d.text.slice(a, b)}</span>
                : <span key={a}>{d.text.slice(a, b)}</span>,
            )}
          </div>
        </div>

        <div className="cu-docs">
          <div className="cu-actions">
            <button className="cu-btn" onClick={removeFullest}>Remove the 10 fullest copies</button>
            <button className="cu-btn ghost" onClick={() => setRemoved(new Set())} disabled={!removed.size}>Put them all back</button>
          </div>
          {byCat.map(([c, xs]) => (
            <div key={c} className="cu-cat">
              <div className="h">{c} <span className="k">{xs.filter((x) => !removed.has(x.i)).length} of {xs.length}</span></div>
              <div className="marks">
                {xs.map((x) => (
                  <button
                    key={x.i}
                    className={`mark${removed.has(x.i) ? " off" : ""}${peek === x.i ? " sel" : ""}`}
                    style={{ "--w": `${Math.min(1, span(x) / 400)}` }}
                    aria-label={`${c} document, ${removed.has(x.i) ? "removed" : "in corpus"}`}
                    onClick={() => {
                      const next = new Set(removed);
                      next.has(x.i) ? next.delete(x.i) : next.add(x.i);
                      setRemoved(next);
                      setPeek(x.i);
                    }}
                    onMouseEnter={() => setPeek(x.i)}
                  />
                ))}
              </div>
            </div>
          ))}
          <div className="cu-peek" aria-live="polite">
            {pd ? (
              <>
                <div className="src">{pd.cat}{pd.cat === "Web pages" ? ` · ${pd.src}` : ""}{removed.has(peek) ? " · removed" : ""}</div>
                <div className="snip">{pd.pre && "…"}{snippet(pd)}{pd.post && "…"}</div>
              </>
            ) : <div className="src">Hover or tap a document to read where the song shows up.</div>}
          </div>
        </div>
      </div>

      <div className="cu-legend">
        <span>found in</span>
        {[1, 2, 3, 4, 5, 6].map((l) => <span key={l} className="sw" style={{ background: `var(--cor-${l})` }} />)}
        <span>1 → 100+ documents</span>
      </div>
    </div>
  );
}

function span(x) {
  return x.spans.reduce((n, [a, b]) => n + (b - a), 0);
}

function snippet(x) {
  const out = [];
  let k = 0;
  const hl = [...x.hl].sort((a, b) => a[0] - b[0]);
  for (const [a, b] of hl) {
    if (a < k) continue;
    if (a > k) out.push(<span key={`t${k}`}>{x.snippet.slice(k, a)}</span>);
    out.push(<mark key={`m${a}`}>{x.snippet.slice(a, b)}</mark>);
    k = b;
  }
  out.push(<span key="end">{x.snippet.slice(k)}</span>);
  return out;
}
