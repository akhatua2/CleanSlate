"use client";
// Figure: we asked one model to forget 50 songs. Left column: songs on its forget list
// (should be erased). Right column: songs it was not asked to forget (should be untouched).
// `variant` picks how erased words are drawn in the After state.
import { useEffect, useState } from "react";
import { fmtP } from "./lyrics";
import TokenLyrics from "./TokenLyrics";
import "./flame.css";
import "./unlearn.css";

export const VARIANTS = [
  { key: "outline", label: "A · dashed outline", note: "Erased words keep an empty dashed box: something was here, nothing is now." },
  { key: "strike", label: "B · faded + struck", note: "Erased words fade and are struck through. No boxes at all." },
  { key: "tint", label: "C · faint green tint", note: "Erased words get a very light green wash, much lighter than any red." },
  { key: "plain", label: "D · plain text", note: "Nothing marks erased words. After unlearning the lyrics look like ordinary text, because nothing is extractable." },
];

const pct = (a, b) => (b ? Math.round(1000 * (1 - a / b)) / 10 : 0);

export default function UnlearnFigure({ variant = "outline" }) {
  const [meta, setMeta] = useState(null);
  const [model, setModel] = useState("llama-8b");
  const [stage, setStage] = useState("pre");

  useEffect(() => {
    fetch("data/unlearn.json").then((r) => r.json()).then(setMeta);
  }, []);
  if (!meta) return <div className="ul ui">Loading…</div>;
  const m = meta.models.find((x) => x.key === model);

  return (
    <div className={`ul ui v-${variant}`}>
      <div className="ul-bar">
        <label className="ul-model">
          <span className="lbl">We asked</span>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {meta.models.map((x) => <option key={x.key} value={x.key}>{x.name}</option>)}
          </select>
          <span className="lbl">to forget {m.forget.works} songs and books.</span>
        </label>
        <div className="seg" role="group" aria-label="Before or after unlearning">
          <button aria-pressed={stage === "pre"} onClick={() => setStage("pre")}>Before</button>
          <button aria-pressed={stage === "ea"} onClick={() => setStage("ea")}>After unlearning</button>
        </div>
      </div>

      <div className="ul-cols">
        <Column
          key={`f-${model}`}
          kind="forget"
          title="On the forget list"
          goal="should be erased"
          songs={m.forget_songs}
          model={model}
          stage={stage}
          variant={variant}
        />
        <Column
          key={`k-${model}`}
          kind="keep"
          title="Not on the list"
          goal="should stay exactly as it was"
          songs={m.keep_songs}
          model={model}
          stage={stage}
          variant={variant}
        />
      </div>

      <div className="ul-totals">
        <Total
          label={`The ${m.forget.works} requested works`}
          want="want 100% erased"
          pre={m.forget.pre}
          post={m.forget.post}
          stage={stage}
          good
        />
        <Total
          label={`The other ${m.retain.works.toLocaleString()} works`}
          want="want 0% erased"
          pre={m.retain.pre}
          post={m.retain.post}
          stage={stage}
        />
      </div>
    </div>
  );
}

function Column({ kind, title, goal, songs, model, stage, variant }) {
  const [id, setId] = useState(songs[0]?.id);
  const [data, setData] = useState(null);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    if (!id) return;
    setData(null);
    fetch(`data/tokens/${id}.json`).then((r) => r.json()).then(setData);
  }, [id]);

  const s = songs.find((x) => x.id === id);
  const t = data?.models[model];

  let verdict = null;
  if (stage === "ea" && s) {
    const erased = s.pre - s.post;
    verdict = kind === "forget"
      ? <span className={s.post === 0 ? "ok" : "bad"}>{s.post === 0 ? "✓ erased, as asked" : `✗ ${s.post} windows survive`}</span>
      : <span className={erased > 0 ? "bad" : "ok"}>{erased > 0 ? `✗ damaged: ${erased} of ${s.pre} memorized windows lost` : "✓ untouched"}</span>;
  }

  return (
    <section className={`ul-col ${kind}`}>
      <header>
        <div className="h">{title}</div>
        <div className="g">{goal}</div>
      </header>
      <nav className="tabs small" aria-label="Pick a song">
        {songs.map((x) => (
          <button key={x.id} aria-pressed={x.id === id} onClick={() => setId(x.id)}>{x.title}</button>
        ))}
      </nav>
      <div className="lyrics" onMouseLeave={() => setHover(null)}>
        {!data || data.id !== id ? <div className="more">Loading…</div> : (
          <TokenLyrics tok={data} model={model} stage={stage} onHover={setHover} />
        )}
      </div>
      <div className="ul-count">
        {s && (stage === "pre"
          ? <>{s.pre} <span className="k">extractable windows</span></>
          : <>{s.pre} → <b>{s.post}</b> <span className="k">extractable windows</span> {verdict}</>)}
      </div>
      {hover && t && (
        <div className="tooltip" style={{ opacity: 1, left: Math.min(hover.x + 12, window.innerWidth - 270), top: hover.y + 14 }}>
          token “{data.text.slice(...t.spans[hover.i]).trim()}”
          <br /><span className="k">before:</span> {fmtP(t.vals.pre?.[hover.i] ?? null)}
          <br /><span className="k">after:</span> {fmtP(t.vals.ea?.[hover.i] ?? null)}
        </div>
      )}
    </section>
  );
}

function Total({ label, want, pre, post, stage, good }) {
  const shown = stage === "pre" ? pre : post;
  const frac = pre ? shown / pre : 0;
  const erased = pct(post, pre);
  return (
    <div className={`ul-total ${good ? "forget" : "keep"}`}>
      <div className="lab">{label}<span className="k"> · {want}</span></div>
      <div className="bar" title={`${shown.toLocaleString()} of ${pre.toLocaleString()} memorized windows remain`}>
        <div className="fill" style={{ width: `${frac * 100}%` }} />
      </div>
      <div className="num">
        {stage === "pre"
          ? <>{pre.toLocaleString()} <span className="k">memorized windows</span></>
          : <><b>{erased}%</b> <span className="k">erased ({pre.toLocaleString()} → {post.toLocaleString()})</span></>}
      </div>
    </div>
  );
}
