"use client";
// A dropdown that looks like the page: an underlined serif word that opens a small paper menu.
// Keyboard: Enter/Space/↓ to open, ↑/↓ to move, Enter to pick, Esc to close.
import { useEffect, useRef, useState } from "react";
import "./select.css";

export default function Select({ value, options, onChange, label }) {
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(Math.max(0, options.findIndex((o) => o.value === value)));
  const root = useRef(null);
  const cur = options.find((o) => o.value === value) ?? options[0];

  useEffect(() => {
    if (!open) return;
    const off = (e) => { if (!root.current?.contains(e.target)) setOpen(false); };
    document.addEventListener("pointerdown", off);
    return () => document.removeEventListener("pointerdown", off);
  }, [open]);

  const pick = (i) => { onChange(options[i].value); setOpen(false); };
  const onKey = (e) => {
    if (!open && ["Enter", " ", "ArrowDown"].includes(e.key)) { e.preventDefault(); setHi(options.indexOf(cur)); setOpen(true); return; }
    if (!open) return;
    if (e.key === "Escape") { setOpen(false); }
    else if (e.key === "ArrowDown") { e.preventDefault(); setHi((h) => Math.min(options.length - 1, h + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setHi((h) => Math.max(0, h - 1)); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(hi); }
  };

  return (
    <span className="sel" ref={root}>
      <button
        type="button"
        className="sel-btn"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={label}
        onClick={() => { setHi(options.indexOf(cur)); setOpen((o) => !o); }}
        onKeyDown={onKey}
      >
        {cur.label}
        <svg width="10" height="6" viewBox="0 0 10 6" aria-hidden="true"><path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
      </button>
      {open && (
        <ul className="sel-menu" role="listbox" aria-label={label}>
          {options.map((o, i) => (
            <li
              key={o.value}
              role="option"
              aria-selected={o.value === value}
              className={`${i === hi ? "hi" : ""}${o.value === value ? " on" : ""}`}
              onPointerEnter={() => setHi(i)}
              onPointerDown={(e) => { e.preventDefault(); pick(i); }}
            >
              {o.label}
            </li>
          ))}
        </ul>
      )}
    </span>
  );
}
