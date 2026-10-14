// Server-safe drawing primitives, a JS port of the anatomy kit's react/draw.tsx.
import * as React from "react";

export function Solid({ paths, tone = "hi", crease = "faint", halo = false, lit = false, flat = false, part, inner, live = false }) {
  return (
    <g className="iso-solid" data-lit={lit ? "" : undefined} data-part={part}>
      {halo ? <path className="iso-halo" d={paths.fill} /> : null}
      <path className="iso-fill" d={paths.fill} />
      {flat ? null : paths.shades.map((d, i) => (d || live ? <path key={i} className="iso-shade" data-shade={i} d={d || ""} /> : null))}
      {flat ? null : <path className="iso-top" d={paths.top} />}
      {crease === "none" || !paths.crease ? null : <path className="iso-line iso-crease" data-tone={crease} d={paths.crease} />}
      {crease === "none" || (!paths.bevel && !live) ? null : <path className="iso-line iso-bevel" data-tone={crease} d={paths.bevel} />}
      {inner}
      <path className="iso-line iso-edge" data-tone={tone} d={paths.outline} />
    </g>
  );
}

export function Line({ d, tone = "lo", part, free }) {
  if (!d) return null;
  return <path className="iso-line" data-tone={tone} data-part={part} data-free={free ? "" : undefined} d={d} />;
}

const tenth = (v) => Math.round(v * 10) / 10;

export function Dots({ points, size = 0.5, tone = "mid" }) {
  return (
    <g className="iso-dots" data-tone={tone}>
      {points.map(([x, y], i) => (
        <circle key={i} cx={tenth(x)} cy={tenth(y)} r={size} />
      ))}
    </g>
  );
}

function Swatch({ mark }) {
  if (mark === "raised" || mark === "lit") {
    return (
      <svg className="iso-swatch" data-mark={mark} viewBox="0 0 16 10" aria-hidden="true" focusable="false">
        <path d="M2 4.5 8 1.5 14 4.5 14 6 8 9 2 6Z" />
        <path className="iso-swatch-crease" d="M2 4.5 8 7.5 14 4.5" />
      </svg>
    );
  }
  return (
    <svg className="iso-swatch" data-mark={mark} viewBox="0 0 16 10" aria-hidden="true" focusable="false">
      <path d="M1.5 5h13" />
    </svg>
  );
}

export function Plate({ fig, title, hint, readout, keys = [], controls, caption, children }) {
  return (
    <figure className="iso-figure">
      <div className="iso-plate">
        <div className="iso-plate-corners" aria-hidden="true">
          <span className="iso-plate-corner" data-corner="fig">{fig}</span>
          <span className="iso-plate-corner" data-corner="title">{title}</span>
          <span className="iso-plate-corner" data-corner="hint">{hint}</span>
          <span className="iso-plate-corner" data-corner="readout" data-readout="">{readout}</span>
        </div>
        {children}
      </div>
      <figcaption className="iso-legend">
        {controls}
        <span className="iso-keys">
          {keys.map((k) => (
            <span className="iso-key" key={k.label}>
              <Swatch mark={k.mark} />
              {k.label}
            </span>
          ))}
        </span>
        {caption ? <span className="iso-caption">{caption}</span> : null}
      </figcaption>
    </figure>
  );
}
