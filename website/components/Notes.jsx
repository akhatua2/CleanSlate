"use client";
// Margin notes and citations, in the spirit of a printed essay: a small numbered mark in the text,
// the note itself in the right margin on wide screens and opened in place by a tap on narrow ones.
// Citations are numbered by their order in REFS and link to the reference list at the end.
import { useState } from "react";
import "./notes.css";

export const REFS = [
  { k: "tofu", t: "TOFU: A Task of Fictitious Unlearning for LLMs", a: "Maini et al.", y: 2024, u: "https://arxiv.org/abs/2401.06121" },
  { k: "muse", t: "MUSE: Machine Unlearning Six-Way Evaluation for Language Models", a: "Shi et al.", y: 2024, u: "https://arxiv.org/abs/2407.06460" },
  { k: "openun", t: "OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics", a: "Dorna et al.", y: 2025, u: "https://arxiv.org/abs/2506.12618" },
  { k: "hayes", t: "Measuring Memorization in Language Models via Probabilistic Extraction", a: "Hayes et al.", y: 2025, u: "https://arxiv.org/abs/2410.19482" },
  { k: "cooper", t: "Extracting memorized pieces of (copyrighted) books from open-weight language models", a: "Cooper et al.", y: 2025, u: "https://arxiv.org/abs/2505.12546" },
  { k: "olmo", t: "Olmo 3", a: "Team Olmo", y: 2025, u: "https://arxiv.org/abs/2512.13961" },
  { k: "infini", t: "Infini-gram mini: Exact n-gram Search at the Internet Scale with FM-Index", a: "Xu et al.", y: 2025, u: "https://arxiv.org/abs/2506.12229" },
  { k: "liu", t: "Language Models May Verbatim Complete Text They Were Not Explicitly Trained On", a: "Liu et al.", y: 2025, u: "https://arxiv.org/abs/2503.17514" },
  { k: "simnpo", t: "Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning", a: "Fan et al.", y: 2025, u: "https://arxiv.org/abs/2410.07163" },
  { k: "undial", t: "UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models", a: "Dong et al.", y: 2025, u: "https://arxiv.org/abs/2402.10052" },
  { k: "wmdp", t: "The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning", a: "Li et al.", y: 2024, u: "https://arxiv.org/abs/2403.03218" },
];
const idx = Object.fromEntries(REFS.map((r, i) => [r.k, i + 1]));

export function Cite({ k }) {
  const keys = k.split(",").map((s) => s.trim());
  return (
    <span className="cite">
      [{keys.map((key, i) => {
        const r = REFS[idx[key] - 1];
        return (
          <span key={key}>
            {i > 0 && ", "}
            <a href={`#ref-${key}`} className="cite-a" data-tip={`${r.a}, ${r.y}. ${r.t}`}>{idx[key]}</a>
          </span>
        );
      })}]
    </span>
  );
}

export function Note({ n, children }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className={`nm${open ? " on" : ""}`} aria-expanded={open} aria-label={`Note ${n}`} onClick={() => setOpen((o) => !o)}>
        {n}
      </button>
      <span className={`sidenote${open ? " open" : ""}`} role="note">
        <span className="sn-n">{n}</span> {children}
      </span>
    </>
  );
}

export function References() {
  return (
    <ol className="refs">
      {REFS.map((r) => (
        <li key={r.k} id={`ref-${r.k}`}>
          <a href={r.u}>{r.t}</a>
          <span className="by"> · {r.a} · {r.y}</span>
        </li>
      ))}
    </ol>
  );
}
