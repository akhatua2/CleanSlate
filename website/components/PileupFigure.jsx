"use client";
// Figure: outward diffusion as a pile-up. The x-axis is the song, start to end. Every row
// below is one mid-training document, drawn only where it copies the song; the skyline on
// top counts the copies at each point. Click a tower to remove every document that copies
// that line, the way a curator would, and see how much of the song is still out there.
import { useEffect, useMemo, useRef, useState } from "react";
import "./pileup.css";

const MIN_WORDS = 5; // the paper's 5-gram floor
const CATS = ["Web pages", "Synthetic Q&A", "Instruction tuning", "Code & software", "Wikipedia", "Math", "Other"];
const SKY = 110; // skyline height (px)

const span = (x) => x.spans.reduce((n, [a, b]) => n + (b - a), 0);

export default function PileupFigure() {
  const [index, setIndex] = useState(null);
  const [id, setId] = useState(null);
  const [d, setD] = useState(null);
  const [k, setK] = useState(0); // documents removed so far, fullest copies first
  const [hx, setHx] = useState(null); // hovered character position
  const [peek, setPeek] = useState(null); // clicked document
  const wrap = useRef(null);
  const cv = useRef(null);
  const anim = useRef({ alpha: new Map(), raf: 0 });
  const [w, setW] = useState(0);

  useEffect(() => {
    fetch("data/flame_index.json").then((r) => r.json()).then((ix) => {
      setIndex(ix);
      setId(ix.works.find((x) => x.title === "Never Gonna Give You Up")?.id ?? ix.featured[0]);
    });
  }, []);
  useEffect(() => {
    if (!id) return;
    setD(null); setK(0); setPeek(null); setHx(null);
    anim.current.alpha = new Map();
    fetch(`data/footprint/${id}.json`).then((r) => r.json()).then(setD);
  }, [id]);
  useEffect(() => {
    const ro = new ResizeObserver(() => setW(wrap.current?.clientWidth ?? 0));
    if (wrap.current) ro.observe(wrap.current);
    return () => ro.disconnect();
  }, [d]);

  // documents grouped by source, each group ordered by where its copy starts
  const docs = useMemo(() => {
    if (!d) return [];
    const xs = d.docs.map((x, i) => ({ ...x, i })).filter((x) => x.words >= MIN_WORDS);
    return CATS.flatMap((c) => xs.filter((x) => x.cat === c).sort((a, b) => a.spans[0][0] - b.spans[0][0]));
  }, [d]);

  const lines = useMemo(() => {
    if (!d) return [];
    const out = [];
    let a = 0;
    d.text.split("\n").forEach((ln) => { if (ln.trim()) out.push([a, a + ln.length, ln.trim()]); a += ln.length + 1; });
    return out;
  }, [d]);

  const order = useMemo(() => [...docs].sort((a, b) => span(b) - span(a)), [docs]);
  const removed = useMemo(() => new Set(order.slice(0, k).map((x) => x.i)), [order, k]);
  // what else goes with them: text that is not this song, and other benchmark songs they carry
  const collateral = useMemo(() => {
    let chars = 0;
    const songs = new Set();
    for (const x of order.slice(0, k)) {
      if (x.len != null) chars += Math.max(0, x.len - span(x));
      for (const t of x.other ?? []) songs.add(t);
    }
    return { chars, songs: [...songs].sort(), known: order.length > 0 && order[0].len != null };
  }, [order, k]);

  const cov = useMemo(() => {
    if (!d) return null;
    const c = new Uint16Array(d.text.length);
    for (const x of docs) if (!removed.has(x.i)) for (const [a, b] of x.spans) for (let k = a; k < b; k++) c[k]++;
    return c;
  }, [d, docs, removed]);

  const n = d?.text.length ?? 1;
  const rowH = docs.length ? Math.max(1.6, Math.min(4, 420 / docs.length)) : 4;
  const groupGap = 6;
  const layout = useMemo(() => {
    let y = SKY + 14, prev = null;
    return docs.map((x) => { if (prev && prev !== x.cat) y += groupGap; prev = x.cat; const r = { x, y }; y += rowH; return r; });
  }, [docs, rowH]);
  const H = layout.length ? layout[layout.length - 1].y + rowH + 4 : SKY + 20;
  const maxCov = useMemo(() => {
    if (!d) return 1;
    const c = new Uint16Array(d.text.length);
    for (const x of docs) for (const [a, b] of x.spans) for (let k = a; k < b; k++) c[k]++;
    return Math.max(1, ...c);
  }, [d, docs]);

  // draw, easing removed rows out
  useEffect(() => {
    const c = cv.current;
    if (!c || !d || !w) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr); c.height = Math.round(H * dpr); c.style.height = `${H}px`;
    const css = (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
    const blue = css("--cor-5"), light = css("--cor-2"), rule = css("--rule"), ink = css("--ink-1");
    const A = anim.current;
    const X = (k) => (k / n) * w;
    const frame = () => {
      let busy = false;
      const ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, H);
      // skyline: copies at each point (original in light blue, what remains in blue)
      const step = Math.max(1, Math.floor(n / w));
      for (let k = 0; k < n; k += step) {
        let orig = 0, now = 0;
        for (let j = k; j < Math.min(n, k + step); j++) { now = Math.max(now, cov[j]); }
        for (const x of docs) for (const [a, b] of x.spans) if (a <= k && k < b) orig++;
        const x0 = X(k), x1 = Math.max(X(k + step), x0 + 1);
        const ho = (orig / maxCov) * SKY, hn = (now / maxCov) * SKY;
        ctx.fillStyle = light; ctx.fillRect(x0, SKY - ho, x1 - x0, ho);
        ctx.fillStyle = blue; ctx.fillRect(x0, SKY - hn, x1 - x0, hn);
      }
      ctx.fillStyle = rule; ctx.fillRect(0, SKY, w, 1);
      // rows: one per document
      for (const { x, y } of layout) {
        const target = removed.has(x.i) ? 0 : 1;
        let a = A.alpha.get(x.i) ?? 1;
        if (Math.abs(a - target) > 0.01) { a += (target - a) * 0.18; busy = true; } else a = target;
        A.alpha.set(x.i, a);
        const drop = (1 - a) * 10;
        ctx.globalAlpha = 0.12 + 0.88 * a;
        ctx.fillStyle = a > 0.5 ? blue : rule;
        for (const [s, e] of x.spans) ctx.fillRect(X(s), y + drop, Math.max(1, X(e) - X(s)), rowH - (rowH > 2 ? 1 : 0));
        if (peek === x.i) { ctx.globalAlpha = 1; ctx.fillStyle = ink; ctx.fillRect(0, y, 2, rowH); }
      }
      ctx.globalAlpha = 1;
      if (hx != null) { ctx.fillStyle = ink; ctx.fillRect(X(hx), 0, 1, H); }
      A.raf = busy ? requestAnimationFrame(frame) : 0;
    };
    cancelAnimationFrame(A.raf);
    A.raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(A.raf);
  }, [d, w, H, cov, layout, removed, hx, peek, docs, maxCov, n, rowH]);

  if (!index || !d || !cov) return <div className="pu ui">Loading…</div>;

  const nonSpace = [...d.text].filter((ch) => !/\s/.test(ch)).length;
  let covered = 0;
  for (let k = 0; k < n; k++) if (cov[k] > 0 && !/\s/.test(d.text[k])) covered++;
  const pct = (100 * covered) / nonSpace;
  const top = Math.max(...cov);
  const lineAt = (k) => lines.find(([a, b]) => k >= a && k <= b);
  const hl = hx != null ? lineAt(hx) : null;

  const pos = (e) => {
    const r = cv.current.getBoundingClientRect();
    return { k: Math.max(0, Math.min(n - 1, Math.floor(((e.clientX - r.left) / r.width) * n))), y: e.clientY - r.top };
  };
  const onClick = (e) => {
    const { k, y } = pos(e);
    const row = layout.find((r) => y >= r.y && y < r.y + rowH);
    if (row) setPeek(row.x.i);
  };

  const featured = index.featured.map((fid) => index.works.find((x) => x.id === fid)).filter(Boolean);
  const pd = peek != null ? d.docs[peek] : null;

  return (
    <div className="pu ui">
      <div className="pu-top">
        <label className="pu-pick">
          <select value={id} onChange={(e) => setId(e.target.value)}>
            {featured.map((x) => <option key={x.id} value={x.id}>{x.title}</option>)}
          </select>
        </label>
        <div className="pu-meter">
          <span className="num">{pct.toFixed(0)}%</span>
          <span className="lab">of the song is still in the training data</span>
        </div>
      </div>

      <div className="pu-slide">
        <input
          type="range" min={0} max={docs.length} step={1} value={k}
          onChange={(e) => setK(+e.target.value)}
          aria-label="Documents removed, fullest copies first"
          style={{ "--p": `${(k / Math.max(1, docs.length)) * 100}%` }}
        />
        <div className="ends"><span>keep everything</span><span>remove all {docs.length}</span></div>
      </div>

      <p className="pu-say">
        {k === 0
          ? <>{docs.length} documents copy at least five words of it; one line alone is copied {top} times. Drag to delete them, fullest copies first.</>
          : <>Deleting the {k} fullest {k === 1 ? "copy" : "copies"} leaves {pct.toFixed(0)}% of the song in the training data{top ? <>, and one line still in {top} documents</> : null}.</>}
      </p>

      {collateral.known && k > 0 && (
        <div className="pu-coll">
          <div><span className="num">{(collateral.chars / 1000).toFixed(collateral.chars < 10000 ? 1 : 0)}k</span> characters deleted that are not this song</div>
          <div><span className="num">{collateral.songs.length}</span> other benchmark songs lose copies too{collateral.songs.length ? <span className="k">: {collateral.songs.slice(0, 6).join(", ")}{collateral.songs.length > 6 ? `, +${collateral.songs.length - 6} more` : ""}</span> : null}</div>
        </div>
      )}

      <div className="pu-line" aria-live="polite">
        {hl ? <><span className="q">“{hl[2]}”</span> <span className="k">in {cov[hx]} document{cov[hx] === 1 ? "" : "s"}</span></> : <span className="k">Move along the song to read each line.</span>}
      </div>

      <div className="pu-plot" ref={wrap}>
        <canvas
          ref={cv}
          className="pu-cv"
          onPointerMove={(e) => setHx(pos(e).k)}
          onPointerLeave={() => setHx(null)}
          onClick={onClick}
          role="img"
          aria-label={`${docs.length} documents copying parts of ${d.title}`}
        />
        <div className="pu-axis"><span>start of song</span><span>end</span></div>
      </div>

      <div className="pu-peek">
        {pd ? (
          <>
            <div className="src">{pd.cat}{pd.cat === "Web pages" ? ` · ${pd.src}` : ""}</div>
            <div className="snip">{pd.pre && "…"}{snippet(pd)}{pd.post && "…"}</div>
          </>
        ) : <div className="src">Click any row to read that document.</div>}
      </div>
    </div>
  );
}

function snippet(x) {
  const out = [];
  let k = 0;
  for (const [a, b] of [...x.hl].sort((p, q) => p[0] - q[0])) {
    if (a < k) continue;
    if (a > k) out.push(<span key={`t${k}`}>{x.snippet.slice(k, a)}</span>);
    out.push(<mark key={`m${a}`}>{x.snippet.slice(a, b)}</mark>);
    k = b;
  }
  out.push(<span key="end">{x.snippet.slice(k)}</span>);
  return out;
}
