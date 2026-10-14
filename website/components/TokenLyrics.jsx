"use client";
// Lyrics rendered token by token, using one model's own tokenizer. Each token is a plain
// highlighter mark: no padding, no extra spacing, so the text reads exactly as written.
import { memLevel } from "./lyrics";

// stage: "pre" or an after-unlearning key; erased tokens (extractable before, not after) get .gone
export default function TokenLyrics({ tok, model, stage, onHover }) {
  const m = tok.models[model];
  if (!m) return <div className="more">No run for this model.</div>;
  const pre = m.vals.pre;
  const cur = m.vals[stage];
  const text = tok.text;
  const out = [];
  m.spans.forEach(([a, b], i) => {
    const before = memLevel(pre?.[i]);
    const now = stage === "pre" ? before : memLevel(cur?.[i]);
    let cls = "t";
    let style;
    if (now) {
      cls += now >= 6 ? " hot deep" : " hot";
      style = { background: `var(--mem-${now})` };
    } else if (stage !== "pre" && before) {
      cls += " gone";
    }
    // a token can contain newlines; split so line breaks render but highlight stays on the visible part
    const parts = text.slice(a, b).split("\n");
    parts.forEach((p, k) => {
      if (k > 0) out.push(<br key={`${i}-br-${k}`} />);
      if (!p) return;
      const lead = p.match(/^\s*/)[0];
      const body = p.slice(lead.length);
      if (lead) out.push(lead);
      if (body)
        out.push(
          <span
            key={`${i}-${k}`}
            className={cls}
            style={style}
            onMouseMove={onHover ? (e) => onHover({ i, x: e.clientX, y: e.clientY }) : undefined}
          >
            {body}
          </span>,
        );
    });
  });
  return <>{out}</>;
}
