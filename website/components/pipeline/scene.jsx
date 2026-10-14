// CleanSlate overview: one low conveyor on one bench, read left to right.
// 1 request clipboard · training-data bags on the belt · 2 the curator, a red X-ray tunnel the belt runs
// through · 3 a Y fork: D_f box, D_r box, and a branch that ends in the not-selected bin; the two box
// branches rejoin · 4 the model (jukebox θ) and both boxes ride into the fixed unlearning booth, θ′ rides
// out · 5 the belt ends at a test console (speaker + four dials).
// World units ~1 px; x along the bench, y toward the viewer, z up. Every view uses scale 1, so a
// translateAlong() offset is the same in every SVG; only the origin differs.
import * as React from "react";
import * as k from "./iso-kit.mjs";
import { Line, Solid } from "./draw";

export const AZIMUTH = 58;
const ELEV = 30;
const PAD = 8;
const R2 = (v) => Number(v.toFixed(2));
const R1 = (v) => Number(v.toFixed(1));
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, s) => a.map((v) => v * s);
const ease = (t) => {
  const u = Math.min(1, Math.max(0, t));
  return u * u * (3 - 2 * u);
};
const lerp3 = (p, q, t) => p.map((v, j) => v + (q[j] - v) * t);

/* ---------- world layout ---------- */

const BENCH = { x: 0, w: 400, d: 76 };
const BZ = 0; // the belt is a low solid slab lying on the bench
const TOP = 12;
const BW = 14;
const AXIS = 38;
const F = [122, AXIS]; // fork
const M = [216, AXIS]; // rejoin
const A1 = [152, AXIS - 20];
const A2 = [186, AXIS - 20];
const C1 = [152, AXIS + 20];
const C2 = [172, AXIS + 20];
const A_PATH = [F, A1, A2, M]; // D_f branch (back)
const B_PATH = [F, M]; // D_r branch (straight)
const C_PATH = [F, C1, C2]; // not-selected branch (front, outward)
const IN_PATH = [[12, AXIS], F];
const OUT_PATH = [M, [322, AXIS]];
const SCAN = { x: 62, y: 27, w: 32, d: 22, h: 26 };
const TUNNEL = { z0: TOP, z1: 23 };
const BOX_F = { x: 158.5, y: A1[1] - 7, w: 21, d: 14, h: 5.6 };
const BOX_R = { x: 155, y: AXIS - 7, w: 21, d: 14, h: 5.6 };
const BIN = { x: 174, y: C2[1] - 9, w: 18, d: 18, z: 0, h: 9 };
const BOOTH = { x: 256, y: 25, w: 30, d: 26, h: 40 };
const JUKE = { w: 25.6, d: 12, body: 14.4, arch: 11.2, sc: 1.6 };
const J_IN = { x: 225, y: AXIS - 6 };
const J_OUT = { x: 292, y: AXIS - 6 };
const WALL = 1;
const PAGE = { w: 8, d: 5.6, h: 0.8 };
const CONSOLE = { x: 326, y: 24, w: 70, d: 28, h: 12 };
const PANEL = { x: 328, y: 26, w: 66, d: 3, z: 12, h: 18 };
const POST = { x: 6, y: 50, w: 3, d: 3, h: 18 };

export const GEOM = { J_IN, J_OUT, JUKE, BOOTH, BOX_F, BOX_R, TOP };

/* ---------- live geometry shared with the client ---------- */

const pathAt = (pts, u) => {
  const lens = pts.slice(1).map((p, i) => Math.hypot(...sub(p, pts[i])));
  const total = lens.reduce((a, b) => a + b, 0);
  let d = Math.min(1, Math.max(0, u)) * total;
  for (let i = 0; i < lens.length; i++) {
    if (d <= lens[i] || i === lens.length - 1) return lerp3(pts[i], pts[i + 1], lens[i] ? Math.min(1, d / lens[i]) : 1);
    d -= lens[i];
  }
  return pts[pts.length - 1];
};
const z3 = (p, z = TOP) => [p[0], p[1], z];

const inBox = (b, n) => [b.x + WALL + 3 + n * 1.4, b.y + WALL + 1 + n * 0.3, TOP + WALL + n * PAGE.h];
const inBin = (n) => [BIN.x + WALL + 2.6 + n * 4, BIN.y + WALL + 3 + n * 3, WALL + n * PAGE.h];
/** Pages sorted on the Curate step: f / r land in the boxes, x goes down the front branch into the bin. */
export const PAGES = [
  { dest: "f", dark: true, rest: inBox(BOX_F, 0) },
  { dest: "x", dark: false, rest: inBin(0) },
  { dest: "r", dark: false, rest: inBox(BOX_R, 0) },
  { dest: "f", dark: true, rest: inBox(BOX_F, 1) },
  { dest: "x", dark: false, rest: inBin(1) },
  { dest: "f", dark: false, wrong: true, rest: inBox(BOX_F, 2) },
  { dest: "r", dark: false, rest: inBox(BOX_R, 1) },
].map((p, i) => ({ ...p, i }));

/** Page plan-corner at progress u: out of the tunnel, through the fork, down its branch, into a box or the bin. */
export function pageAt(p, u) {
  const t = ease(u);
  const c = (q) => [q[0] - PAGE.w / 2, q[1] - PAGE.d / 2, TOP];
  const start = c([SCAN.x + SCAN.w + 1, AXIS]);
  if (p.dest !== "x") {
    const branch = p.dest === "f" ? [c(F), c(A1)] : [c(F)];
    const above = [p.rest[0], p.rest[1], TOP + BOX_F.h + 1.2];
    const pts = [start, ...branch, above];
    return t < 0.85 ? pathAt(pts, t / 0.85) : lerp3(above, p.rest, (t - 0.85) / 0.15);
  }
  const end = c([C2[0] + 1, C2[1]]);
  const pts = [start, c(F), c(C1), end];
  if (t < 0.75) return pathAt(pts, t / 0.75);
  const s = (t - 0.75) / 0.25;
  return [end[0] + (p.rest[0] - end[0]) * Math.sin((s * Math.PI) / 2), end[1] + (p.rest[1] - end[1]) * s, end[2] + (p.rest[2] - end[2]) * s * s];
}
/** Gate angle (deg, in plan) that sends a page down its branch. */
export const FLIP = { f: -34, r: 0, x: 34 };

/** Box offset at progress u of the Unlearn step: along its branch, through the rejoin, into the booth. */
export function boxAt(which, u) {
  const b = which === "f" ? BOX_F : BOX_R;
  const c = [b.x + b.w / 2, b.y + b.d / 2, 0];
  const pts = which === "f" ? [c, [A2[0], A2[1], 0], [M[0], AXIS, 0], [BOOTH.x + 12, AXIS, 0]] : [c, [BOOTH.x + 12, AXIS, 0]];
  return sub(pathAt(pts, ease(u)), c);
}

/** Jukebox song cards (4 x 3). After unlearning: f = blank (asked), c = blank (collateral), r = pinned and kept, o = untouched. */
export const CARDS = ["o", "f", "c", "o", "r", "f", "o", "r", "o", "c", "f", "o"];
const cardTint = (i) => `hsl(${(i * 47 + 20) % 360} 42% 82%)`;

/** Dials (illustrative only): asked songs gone, other songs kept, facts kept, general skills. Target notch at full. */
export const GAUGES = [
  { key: "gone", at: 0.84, label: "asked songs gone" },
  { key: "intact", at: 0.58, label: "other songs kept" },
  { key: "facts", at: 0.78, label: "facts kept" },
  { key: "ability", at: 0.72, label: "general skills" },
];
export const SWEEP = 120;
export const needleAngle = (v) => -SWEEP + 2 * SWEEP * v;
const DIAL = { pitch: 16.5, x0: 8.25, cy: 8.4, r: 5.4 };

/* ---------- drawing helpers ---------- */

const proj = (P, pts, closed = false) => k.pathOf(pts.map((p) => k.iso(p, P)), closed);

/** An affine map from a 2D drawing (a right, b down) onto any plane in the world. */
function planeM(P, o, u, v) {
  const O = k.iso(o, P);
  const U = sub(k.iso(add(o, u), P), O);
  const V = sub(k.iso(add(o, v), P), O);
  return `matrix(${R2(U[0])} ${R2(U[1])} ${R2(V[0])} ${R2(V[1])} ${R2(O[0])} ${R2(O[1])})`;
}

/** A convex solid from its faces (each a planar polygon), shaded like the kit. */
function Convex({ P, faces, tone = "mid", lit = false }) {
  const all = faces.flat();
  const c = mul(all.reduce((acc, p) => add(acc, p), [0, 0, 0]), 1 / all.length);
  const a = (AZIMUTH * Math.PI) / 180;
  const e = (ELEV * Math.PI) / 180;
  const view = [Math.cos(a) * Math.cos(e), Math.sin(a) * Math.cos(e), Math.sin(e)];
  const light = [-Math.sin(a), Math.cos(a)];
  const out = [];
  for (const f of faces) {
    const m = f.length;
    const d1 = sub(f[Math.floor(m / 2)], f[0]);
    const d2 = sub(f[Math.floor((3 * m) / 4)], f[Math.floor(m / 4)]);
    let n = [d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]];
    const fc = mul(f.reduce((acc, p) => add(acc, p), [0, 0, 0]), 1 / m);
    if (n[0] * (fc[0] - c[0]) + n[1] * (fc[1] - c[1]) + n[2] * (fc[2] - c[2]) < 0) n = mul(n, -1);
    const len = Math.hypot(...n) || 1;
    if ((n[0] * view[0] + n[1] * view[1] + n[2] * view[2]) / len <= 1e-6) continue;
    const up = n[2] / len;
    const h = Math.hypot(n[0], n[1]) || 1;
    const shade = Math.min(3, Math.max(0, Math.floor((((n[0] * light[0] + n[1] * light[1]) / h + 1) / 2) * 4)));
    out.push({ d: proj(P, f, true), top: up > 0.6, front: Math.abs(n[1] / len) > 0.9 && up < 0.3, shade });
  }
  return (
    <g className="iso-solid" data-lit={lit ? "" : undefined}>
      {out.map((f, i) => (f.top ? <path key={i} className="iso-top" d={f.d} /> : f.front ? <path key={i} className="iso-fill pl-front" d={f.d} /> : <path key={i} className="iso-shade" data-shade={f.shade} d={f.d} />))}
      {out.map((f, i) => (
        <path key={`e${i}`} className="iso-line iso-edge" data-tone={tone} d={f.d} />
      ))}
    </g>
  );
}
const Prism = ({ P, bottom, top, tone, lit }) => (
  <Convex P={P} tone={tone} lit={lit} faces={[bottom, top, ...[0, 1, 2, 3].map((i) => [bottom[i], bottom[(i + 1) % 4], top[(i + 1) % 4], top[i]])]} />
);
/** A prism along y from an x-z profile (convex). */
const ExtrudeY = ({ P, prof, y0, y1, tone, lit }) => (
  <Convex
    P={P}
    tone={tone}
    lit={lit}
    faces={[prof.map(([x, z]) => [x, y1, z]), prof.map(([x, z]) => [x, y0, z]), ...prof.map(([x, z], i) => {
      const [x2, z2] = prof[(i + 1) % prof.length];
      return [[x, y0, z], [x2, y0, z2], [x2, y1, z2], [x, y1, z]];
    })]}
  />
);

/** A physical cable: a solid tube (two edges, rounded caps) lying on the bench; a bead travels along it on its step. */
function Cable({ P, pts, id }) {
  const flat = pts.map((p) => k.iso(p, P));
  let len = 0;
  for (let i = 1; i < flat.length; i++) len += Math.hypot(flat[i][0] - flat[i - 1][0], flat[i][1] - flat[i - 1][1]);
  const d = k.pathOf(flat);
  return (
    <g className="pl-cable" data-cable={id} style={{ "--len": R1(len) }}>
      <path className="pl-cable-o" d={d} />
      <path className="pl-cable-i" d={d} />
      <path className="pl-cable-bead" d={d} />
    </g>
  );
}
/** A socket block on a front face (at y), bottom at z. */
const Jack = ({ P, x, y, z }) => <Solid paths={k.slabOf({ x: x - 1.3, y, w: 2.6, d: 1.2, r: 0.3 }, z, 2.6, P, 2)} tone="mid" />;

/** A flat face in space: paper fill and an outline. */
const Face = ({ P, pts, cls = "", tone = "mid" }) => (
  <g className={`pl-face ${cls}`}>
    <path className="pl-face-fill" d={proj(P, pts, true)} />
    <path className="iso-line" data-tone={tone} d={proj(P, pts, true)} />
  </g>
);

const Slab = ({ P, plan, z, h, r = 0.6, tone = "mid", lit, bevel = 0, cls }) => {
  const s = <Solid paths={k.slabOf({ r, ...plan }, z, h, P, 3, bevel)} tone={tone} lit={lit} />;
  return cls ? <g className={cls}>{s}</g> : s;
};

/** An open container: floor + back/left walls (inner, darker), then contents, then right/front walls. */
function binParts(P, b, { gap } = {}) {
  const z = b.z;
  const kp = `${b.x}-${b.y}-`;
  const back = [
    <Slab key={`${kp}floor`} P={P} plan={{ x: b.x, y: b.y, w: b.w, d: b.d, r: 0.4 }} z={z} h={WALL} cls="pl-inner" />,
    <Slab key={`${kp}wb`} P={P} plan={{ x: b.x, y: b.y, w: b.w, d: WALL, r: 0.2 }} z={z} h={b.h} cls="pl-inner" />,
    <Slab key={`${kp}wl`} P={P} plan={{ x: b.x, y: b.y, w: WALL, d: b.d, r: 0.2 }} z={z} h={b.h} cls="pl-inner" />,
  ];
  const fy = b.y + b.d - WALL;
  const front = [<Slab key={`${kp}wr`} P={P} plan={{ x: b.x + b.w - WALL, y: b.y, w: WALL, d: b.d, r: 0.2 }} z={z} h={b.h} />];
  if (gap) {
    front.push(
      <Slab key={`${kp}wf0`} P={P} plan={{ x: b.x, y: fy, w: gap[0] - b.x, d: WALL, r: 0.2 }} z={z} h={b.h} />,
      <Slab key={`${kp}wf1`} P={P} plan={{ x: gap[1], y: fy, w: b.x + b.w - gap[1], d: WALL, r: 0.2 }} z={z} h={b.h} />,
      <Slab key={`${kp}wfg`} P={P} plan={{ x: gap[0], y: fy, w: gap[1] - gap[0], d: WALL, r: 0.2 }} z={z} h={2.4} />,
    );
  } else {
    front.push(<Slab key={`${kp}wf`} P={P} plan={{ x: b.x, y: fy, w: b.w, d: WALL, r: 0.2 }} z={z} h={b.h} />);
  }
  return { back, front };
}

const PageSheet = ({ P, p }) => (
  <g data-page={p.i} className={p.dark ? "pl-dark" : p.wrong ? "pl-wrong" : undefined}>
    <Slab P={P} plan={{ x: p.rest[0], y: p.rest[1], w: PAGE.w, d: PAGE.d, r: 0.2 }} z={p.rest[2]} h={PAGE.h} />
  </g>
);

/** An open set box riding a belt, with its pages inside and one engraved line on the front. */
function SetBox({ P, b, forget }) {
  const box = { ...b, z: TOP };
  const parts = binParts(P, box);
  return (
    <g data-box={forget ? "f" : "r"}>
      {parts.back}
      {PAGES.filter((p) => p.dest === (forget ? "f" : "r")).map((p) => (
        <PageSheet key={p.i} P={P} p={p} />
      ))}
      {parts.front}
      {forget ? <path className="pl-rim-f" d={k.planOutline({ x: b.x, y: b.y, w: b.w, d: b.d, r: 0.2 }, TOP + b.h, P, 2)} /> : null}
      <g transform={k.sideMatrix([b.x, b.y + b.d], TOP + b.h, P, "left")}>
        <text className="pl-engrave" x={b.w / 2} y="2.55" textAnchor="middle">
          D<tspan className="pl-sub" dy="0.5">{forget ? "f" : "r"}</tspan>
          <tspan className="pl-engrave-s" dy="-0.5">{forget ? " forget" : " keep"}</tspan>
        </text>
      </g>
    </g>
  );
}

/** A jukebox standing at (x0, y0) on the belt: arched cabinet with grille bars, chrome trim, card window,
 *  selection buttons, coin slot, speaker grille and an engraved plaque. after: the θ′ state. */
function Jukebox({ P, x0, y0, after, part }) {
  const J = JUKE;
  const z0 = TOP;
  const zb = z0 + J.body;
  const prof = [[x0, z0], [x0 + J.w, z0]];
  for (let i = 0; i <= 14; i++) {
    const t = (i / 14) * Math.PI;
    prof.push([x0 + J.w / 2 + J.arch * Math.cos(t), zb + J.arch * Math.sin(t)]);
  }
  const yf = y0 + J.d;
  const top = zb + J.arch;
  const B0 = { w: 16, body: 9, arch: 7 };
  const ax = B0.w / 2;
  const ar = B0.arch;
  // grille bars inside the inner arch: vertical lines ending on the arc
  const bars = [];
  for (let x = 3.6; x <= 12.4; x += 1.1) {
    const dx = x - ax;
    const r = ar - 2.2;
    if (Math.abs(dx) >= r) continue;
    const y = ar - Math.sqrt(r * r - dx * dx);
    bars.push(`M${R2(x)} ${R2(y)}V${ar - 0.2}`);
  }
  const cards = CARDS.map((role, i) => {
    const x = 1.9 + (i % 4) * 3.1;
    const y = 8.5 + Math.floor(i / 4) * 2.1;
    const blank = after && (role === "f" || role === "c");
    return (
      <g key={i}>
        <rect className={blank ? "pl-card pl-card-blank" : "pl-card"} x={R1(x)} y={R1(y)} width="2.7" height="1.8" rx="0.2" fill={blank ? undefined : cardTint(i)} />
        {blank ? null : <path className="pl-card-ink" d={`M${R1(x + 0.5)} ${R1(y + 0.7)}H${R1(x + 2.2)}M${R1(x + 0.5)} ${R1(y + 1.2)}H${R1(x + 1.7)}`} />}
        {after && role === "r" ? <rect className="pl-pin" x={R1(x + 1)} y={R1(y - 0.45)} width="0.8" height="1" rx="0.15" /> : null}
      </g>
    );
  });
  return (
    <g data-part={part}>
      <ExtrudeY P={P} prof={prof} y0={y0} y1={yf} />
      <g transform={`${k.sideMatrix([x0, yf], top, P, "left")} scale(${J.sc})`}>
        <path className="pl-chrome" d={`M0.9 ${ar}A${ar - 0.9} ${ar - 0.9} 0 0 1 ${B0.w - 0.9} ${ar}`} />
        <path className="pl-arch-in" d={`M${ax - (ar - 2.2)} ${ar}A${ar - 2.2} ${ar - 2.2} 0 0 1 ${ax + (ar - 2.2)} ${ar}Z`} />
        <path className="pl-bars" d={bars.join("")} />
        <path className="pl-chrome" d={`M0 ${ar}H${B0.w}`} />
        <rect className="pl-plate" x="4.4" y={ar + 0.3} width="7.2" height="1.6" rx="0.3" />
        <text className="pl-plaque" x={ax} y={ar + 1.65} textAnchor="middle">{after ? "θ′" : "θ"}</text>
        <rect className="pl-window" x="1.4" y="8" width="13.2" height="6.8" rx="0.5" />
        {cards}
        {[0, 1, 2, 3].map((i) => (
          <circle key={i} className="pl-button" cx={R1(3.4 + i * 1.8)} cy="15.4" r="0.45" />
        ))}
        <rect className="pl-slot" x="11" y="15" width="2.4" height="0.7" rx="0.3" />
        <rect className="pl-grille" x="2.4" y="16.2" width="11.2" height="0.6" rx="0.3" />
        <path className="pl-chrome" d={`M0 ${B0.body + ar - 0.1}H${B0.w}`} />
      </g>
    </g>
  );
}

/** A low belt along a polyline (mitred bends) as one solid: fine slats, side rails, rivets on the side. */
function BeltPath({ P, pts, keyp, w = BW, rivets = true }) {
  const h = w / 2;
  const dirs = pts.slice(1).map((q, i) => {
    const dx = q[0] - pts[i][0];
    const dy = q[1] - pts[i][1];
    const L = Math.hypot(dx, dy);
    return [dx / L, dy / L, L];
  });
  const side = (off) =>
    pts.map((p, i) => {
      const a = dirs[Math.max(0, i - 1)];
      const b = dirs[Math.min(dirs.length - 1, i)];
      let nx = -(a[1] + b[1]);
      let ny = a[0] + b[0];
      const nl = Math.hypot(nx, ny) || 1;
      nx /= nl;
      ny /= nl;
      const cos = nx * -b[1] + ny * b[0];
      const m = off / Math.max(0.5, cos);
      return [p[0] + nx * m, p[1] + ny * m];
    });
  const ring = [...side(h), ...side(-h).reverse()];
  const paths = k.extrude(ring, BZ, TOP - BZ, P, { convex: false });
  const slats = [];
  const dots = [];
  dirs.forEach(([ux, uy, L], i) => {
    for (let s = 1.6; s < L - 1; s += 1.6) {
      const cx = pts[i][0] + ux * s;
      const cy = pts[i][1] + uy * s;
      slats.push(proj(P, [[cx - uy * (h - 1.4), cy + ux * (h - 1.4), TOP], [cx + uy * (h - 1.4), cy - ux * (h - 1.4), TOP]]));
    }
    if (rivets) {
      for (let s = 4; s < L - 2; s += 8) {
        const cx = pts[i][0] + ux * s - uy * h;
        const cy = pts[i][1] + uy * s + ux * h;
        if (-uy * Math.cos((AZIMUTH * Math.PI) / 180) + ux * Math.sin((AZIMUTH * Math.PI) / 180) > 0.05) dots.push(k.iso([cx, cy, TOP - 2], P));
      }
    }
  });
  const rail = (off) => proj(P, side(off).map((p) => z3(p)));
  return [
    <Solid key={`${keyp}b`} paths={paths} tone="mid" crease="lo" />,
    <path key={`${keyp}s`} className="pl-slats" d={slats.join("")} />,
    <path key={`${keyp}r`} className="iso-line" data-tone="mid" d={rail(h - 1) + rail(-(h - 1))} />,
    <g key={`${keyp}d`} className="pl-rivets">
      {dots.map(([x, y], i) => (
        <circle key={i} cx={R1(x)} cy={R1(y)} r="0.35" />
      ))}
    </g>,
  ];
}

/** A convex plate at a fork or rejoin, covering the joint (same height as the belt). */
function JointPlate({ P, at, dirs, keyp }) {
  const pts = [];
  for (const [ux, uy] of dirs) {
    for (const s of [-0.5, 4.5]) {
      pts.push([at[0] + ux * s - uy * BW / 2, at[1] + uy * s + ux * BW / 2]);
      pts.push([at[0] + ux * s + uy * BW / 2, at[1] + uy * s - ux * BW / 2]);
    }
  }
  const hull = (ps) => {
    const p = [...ps].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [];
    for (const q of p) {
      while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop();
      lo.push(q);
    }
    const up = [];
    for (const q of [...p].reverse()) {
      while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop();
      up.push(q);
    }
    return [...lo.slice(0, -1), ...up.slice(0, -1)];
  };
  const ring = hull(pts);
  return [
    <Solid key={`${keyp}p`} paths={k.extrude(ring, BZ, TOP - BZ, P, { convex: true })} tone="mid" crease="lo" />,
  ];
}
const unit = (a, b) => {
  const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
  return [(b[0] - a[0]) / d, (b[1] - a[1]) / d];
};

/** Screw heads: a ring of tiny dots at given world points. */
const Screws = ({ P, pts }) => (
  <g className="pl-screws">
    {pts.map((p, i) => {
      const [x, y] = k.iso(p, P);
      return <circle key={i} cx={R1(x)} cy={R1(y)} r="0.55" />;
    })}
  </g>
);

/* ---------- the parts list, in painter's order ---------- */
// home: the phone row that owns the part (1 = left half, 2 = right half); also: other rows that draw it too.

const SPLIT = 222; // phone rows split here (x)

const ITEMS = [
  /* the belt into the fork, with the training-data bags riding it */
  { st: 2, home: 1, box: [[{ x: 12, y: AXIS - BW / 2, w: F[0] - 12, d: BW }, 0, TOP]], draw: (P) => {
    const z = TOP;
    return [
      ...BeltPath({ P, pts: IN_PATH, keyp: "in" }),
      <Slab key="case" P={P} plan={{ x: 14, y: 33, w: 14, d: 10, r: 1.6 }} z={z} h={8.4} bevel={0.6} />,
      <Line key="zip" d={k.sideSeam({ x: 14, y: 33, w: 14, d: 10, r: 1.6 }, z + 4.2, P, 8)} tone="lo" />,
      <Slab key="caseH" P={P} plan={{ x: 18, y: 37.4, w: 6, d: 1.4, r: 0.5 }} z={z + 8.4} h={1.4} />,
      <Slab key="box" P={P} plan={{ x: 31, y: 33.5, w: 11, d: 9.5, r: 0.4 }} z={z} h={7.6} bevel={0.4} />,
      <path key="tape" className="pl-tape" d={proj(P, [[35.8, 33.5, z + 7.6], [35.8, 43, z + 7.6], [35.8, 43, z + 4.6]])} />,
      <Slab key="tote" P={P} plan={{ x: 45, y: 34, w: 10, d: 9, r: 1.2 }} z={z} h={9} bevel={0.5} />,
      <g key="strap" transform={k.sideMatrix([45, 43], z + 9, P, "left")}>
        <path className="pl-strap" d="M2.6 0C2.6 -4 7.4 -4 7.4 0" />
      </g>,
    ];
  } },

  /* 1 · request: a clipboard on a short post beside the belt's start, five song lines with record marks */
  { st: 1, home: 1, box: [[{ x: 1, y: 50, w: 13, d: 8 }, 0, POST.h + 18]], draw: (P) => {
    const o = [1.5, 51, POST.h + 16];
    const slant = Math.hypot(4, 15);
    const u = [1, 0, 0];
    const v = [0, 4 / slant, -15 / slant];
    const n = [0, 15 / slant, 4 / slant];
    const box = (a0, a1, b0, b1, c0, c1) => {
      const at = (a, b, c) => add(add(add(o, mul(u, a)), mul(v, b)), mul(n, c));
      return { bottom: [at(a0, b1, c0), at(a1, b1, c0), at(a1, b1, c1), at(a0, b1, c1)], top: [at(a0, b0, c0), at(a1, b0, c0), at(a1, b0, c1), at(a0, b0, c1)] };
    };
    const board = box(0, 11, 0, slant, -1.1, 0);
    const clip = box(3.5, 7.5, -1.1, 1.8, -1.5, 0.5);
    return [
      <Slab key="foot" P={P} plan={{ x: 3.5, y: 49.5, w: 8, d: 8, r: 1.2 }} z={0} h={1.2} bevel={0.3} />,
      <Slab key="post" P={P} plan={{ ...POST, r: 0.4 }} z={1.2} h={POST.h - 1.2} />,
      <Slab key="ledge" P={P} plan={{ x: 1.5, y: 55, w: 11, d: 2.4, r: 0.4 }} z={POST.h - 1.5} h={1.5} />,
      <Prism key="board" P={P} bottom={board.bottom} top={board.top} />,
      <g key="sheet" data-part="sheet" transform={planeM(P, add(o, mul(n, 0.12)), u, v)}>
        <rect className="pl-sheet" x="0.9" y="2" width="9.2" height={R1(slant - 3)} rx="0.3" />
        {[4.4, 6.8, 9.2, 11.6, 14].map((b, i) => (
          <g key={b}>
            <circle className="pl-rec" cx="2.5" cy={b} r="0.8" />
            <circle className="pl-rec-dot" cx="2.5" cy={b} r="0.25" />
            <path className="iso-line pl-ruled" d={`M4 ${b}H${i % 2 ? 8 : 9.2}`} />
          </g>
        ))}
      </g>,
      <Prism key="clip" P={P} bottom={clip.bottom} top={clip.top} />,
      <Jack key="jReq" P={P} x={POST.x + 1.5} y={POST.y + POST.d} z={1.2} />,
    ];
  } },

  /* 2 · the curator: a red X-ray tunnel straddling the belt, curtains at its exit, monitor, controls, vents */
  { st: 2, home: 1, box: [[SCAN, 0, SCAN.h + 13]], draw: (P) => {
    const S = SCAN;
    const fy = S.y + S.d;
    const ex = S.x + S.w;
    const cavX = fy - (AXIS + BW / 2) - 0.6;
    return [
      <Slab key="scan" P={P} plan={{ ...S, r: 1.6 }} z={0.8} h={S.h - 0.8} bevel={0.9} lit tone="hi" />,
      ...[[S.x + 1, S.y + S.d - 3], [S.x + S.w - 4, S.y + S.d - 3]].map(([x, y]) => <Slab key={`ft${x}`} P={P} plan={{ x, y, w: 3, d: 2.4, r: 0.6 }} z={0} h={0.8} />),
      <g key="scanFront" transform={k.sideMatrix([S.x, fy], S.h, P, "left")}>
        <path className="pl-mine-thin" d={`M0 3.2H${S.w}`} />
        <rect className="pl-mine-panel" x="3" y="6" width="15" height="10" rx="0.8" />
        <path className="iso-line pl-mine-mark" d="M7.6 8.2C6.6 8.2 6.4 8.7 6.4 9.5V10.4C6.4 11 6 11.2 5.4 11.2C6 11.2 6.4 11.4 6.4 12V12.9C6.4 13.7 6.6 14.2 7.6 14.2M11.4 8.2C12.4 8.2 12.6 8.7 12.6 9.5V10.4C12.6 11 13 11.2 13.6 11.2C13 11.2 12.6 11.4 12.6 12V12.9C12.6 13.7 12.4 14.2 11.4 14.2" />
        <circle className="pl-led" data-led="0" cx="15.6" cy="8.6" r="0.8" />
        <circle className="pl-led" data-led="1" cx="15.6" cy="11" r="0.8" />
        {[22.5, 27].map((cx) => (
          <g key={cx}>
            <circle className="pl-knob" cx={cx} cy="10" r="1.8" />
            <path className="pl-knob-tick" d={`M${cx} 10L${cx + 1} 8.6`} />
          </g>
        ))}
        <path className="pl-vent" d={[17.4, 18.8, 20.2, 21.6].map((y) => `M20 ${y}H${S.w - 2}`).join("")} />
        {[[1.2, 1.2], [S.w - 1.2, 1.2], [1.2, S.h - 1.6], [S.w - 1.2, S.h - 1.6]].map(([x, y]) => (
          <circle key={`${x}-${y}`} className="pl-screw" cx={x} cy={y} r="0.45" />
        ))}
      </g>,
      <g key="scanExit" transform={k.sideMatrix([ex, fy], S.h, P, "right")}>
        <rect className="pl-cavity" x={R1(cavX)} y={S.h - TUNNEL.z1 - 0.6} width={BW + 1.2} height={TUNNEL.z1 - TUNNEL.z0 + 0.6} rx="0.6" />
        <path className="pl-curtain" d={Array.from({ length: 7 }, (_, i) => `M${R1(cavX + 0.9 + i * 1.9)} ${R1(S.h - TUNNEL.z1 - 0.6)}V${R1(S.h - TUNNEL.z0 - 2.2 - (i % 2) * 0.6)}`).join("")} />
        <path className="pl-mine-thin" d={`M0 3.2H${S.d}`} />
        <path className="pl-vent" d={[6.5, 7.9, 9.3].map((y) => `M3 ${y}H${S.d - 3}`).join("")} />
      </g>,
      <Jack key="jScan" P={P} x={S.x + 12} y={fy} z={1.5} />,
      <Slab key="monNeck" P={P} plan={{ x: S.x + 17, y: S.y + 10, w: 4, d: 4, r: 0.5 }} z={S.h} h={3} lit tone="hi" />,
      <Slab key="mon" P={P} plan={{ x: S.x + 12, y: S.y + 11, w: 14, d: 3, r: 0.6 }} z={S.h + 3} h={9} bevel={0.4} lit tone="hi" />,
      <g key="screen" transform={k.sideMatrix([S.x + 12, S.y + 14], S.h + 12, P, "left")}>
        <rect className="pl-bezel" x="0.6" y="0.6" width="12.8" height="7.8" rx="0.6" />
        <rect className="pl-screen" x="1.3" y="1.3" width="11.4" height="6.4" rx="0.3" />
        <rect className="pl-xray" x="2.6" y="2.5" width="5.6" height="3.8" rx="0.6" />
        <path className="pl-xray-l" d="M3.5 3.7H6.5M3.5 5H7.4M9.8 2.6V6.4M11 2.6V6.4" />
        <path className="pl-scanline" data-part="scanline" d="M1.9 1.5V7.5" />
      </g>,
    ];
  } },

  /* cable: clipboard → scanner, down from its plug, flat along the bench with clips, up into the scanner's socket */
  { st: 1, home: 1, box: [], draw: (P) => {
    const jx = POST.x + 1.5;
    const sy = SCAN.y + SCAN.d;
    return [
      <Cable key="req" P={P} id="req" pts={[[jx, POST.y + POST.d + 0.6, 1.2], [jx, POST.y + POST.d + 0.6, 0.5], [jx, 64, 0.5], [SCAN.x + 12, 64, 0.5], [SCAN.x + 12, sy + 0.6, 0.5], [SCAN.x + 12, sy + 0.6, 1.5]]} />,
      ...[22, 44].map((x) => <Slab key={"clip" + x} P={P} plan={{ x, y: 62.9, w: 1.4, d: 2.2, r: 0.2 }} z={0} h={1.3} />),
    ];
  } },

  /* 3 · the Y fork: D_f branch (back), D_r branch (straight), not-selected branch (front, ends at the bin) */
  { st: 3, home: 1, box: [[{ x: F[0], y: 18, w: M[0] - F[0], d: 44 }, 0, TOP]], draw: (P) => [
    ...BeltPath({ P, pts: A_PATH, keyp: "a" }),
    ...BeltPath({ P, pts: B_PATH, keyp: "b" }),
    ...BeltPath({ P, pts: C_PATH, keyp: "c" }),
    <Slab key="cStop" P={P} plan={{ x: C2[0], y: C2[1] - BW / 2, w: 1, d: BW, r: 0.2 }} z={0} h={TOP + 1.2} />,
    ...JointPlate({ P, at: F, dirs: [unit(F, A1), unit(F, M), unit(F, C1)], keyp: "jf" }),
    <g key="gate" transform={k.topMatrix(F, TOP + 0.05, P)}>
      <g data-part="flip" transform={`rotate(${FLIP.f})`}>
        <rect className="pl-flip" x="-0.9" y="-0.9" width="11" height="1.8" rx="0.9" />
      </g>
      <circle className="pl-pivot-flip" r="1.5" />
      <circle className="pl-pivot-dot" r="0.45" />
    </g>,
  ] },

  /* the line out (drawn before the rejoin plate covers its start) */
  { st: 4, home: 2, box: [[{ x: M[0], y: AXIS - BW / 2, w: OUT_PATH[1][0] - M[0], d: BW }, 0, TOP]], draw: (P, view) => [
    ...BeltPath({ P, pts: [[Math.max(M[0], view.xRange[0]), AXIS], [Math.min(OUT_PATH[1][0], view.xRange[1]), AXIS]], keyp: "out" }),
  ] },
  { st: 3, home: 1, box: [], draw: (P) => [...JointPlate({ P, at: M, dirs: [unit(M, A2), unit(M, F), unit(M, [M[0] + 10, AXIS])], keyp: "jm" })] },

  /* 3 · the two set boxes on their branches, and the not-selected bin at the end of the front branch */
  { st: 3, home: 1, also: [2], box: [], draw: (P) => [<SetBox key="bf" P={P} b={BOX_F} forget />, <SetBox key="br" P={P} b={BOX_R} />] },
  { st: 3, home: 1, box: [[BIN, 0, BIN.h]], draw: (P) => {
    const b = binParts(P, BIN);
    return [...b.back, ...PAGES.filter((p) => p.dest === "x").map((p) => <PageSheet key={p.i} P={P} p={p} />), ...b.front];
  } },

  /* θ: the model before unlearning, on the belt just before the booth */
  { st: 4, home: 2, box: [[{ x: J_IN.x, y: J_IN.y, w: JUKE.w, d: JUKE.d }, TOP, JUKE.body + JUKE.arch]], draw: (P) => [<Jukebox key="jin" P={P} x0={J_IN.x} y0={J_IN.y} part="jb-in" />] },

  /* 4 · the unlearning booth (fixed): the belt runs through it; flaps at the exit, access panel, vent, lamp */
  { st: 4, home: 2, box: [[BOOTH, 0, BOOTH.h + 4]], draw: (P) => {
    const B = BOOTH;
    const fy = B.y + B.d;
    const ex = B.x + B.w;
    const cavX = fy - (AXIS + BW / 2) - 0.6;
    const openH = JUKE.body + JUKE.arch + 1.6;
    return [
      <Slab key="booth" P={P} plan={{ ...B, r: 1.2 }} z={0} h={B.h} bevel={0.9} />,
      <Line key="seam" d={k.sideSeam({ ...B, r: 1.2 }, B.h - 4, P, 3)} tone="lo" />,
      <path key="access" className="pl-panel" d={k.planOutline({ x: B.x + 5, y: B.y + 5, w: 16, d: 12, r: 0.8 }, B.h, P, 3)} />,
      <Screws key="accS" P={P} pts={[[B.x + 6.2, B.y + 6.2, B.h], [B.x + 19.8, B.y + 6.2, B.h], [B.x + 6.2, B.y + 15.8, B.h], [B.x + 19.8, B.y + 15.8, B.h]]} />,
      <Slab key="lampB" P={P} plan={{ x: B.x + 23, y: B.y + 8, w: 4, d: 4, r: 2 }} z={B.h} h={1} />,
      <Solid key="lamp" paths={k.cylinder(B.x + 25, B.y + 10, 1.4, B.h + 1, 2, P, 20)} tone="mid" />,
      <g key="bFront" transform={k.sideMatrix([B.x, fy], B.h, P, "left")}>
        <path className="pl-vent" d={[8, 9.4, 10.8, 12.2].map((y) => `M4 ${y}H16`).join("")} />
        {Array.from({ length: 7 }, (_, i) => (
          <circle key={i} className="pl-screw" cx={R1(2 + i * 4.6)} cy="2" r="0.4" />
        ))}
      </g>,
      <g key="bExit" transform={k.sideMatrix([ex, fy], B.h, P, "right")}>
        <rect className="pl-cavity" x={R1(cavX)} y={R1(B.h - TOP - openH)} width={BW + 1.2} height={openH} rx="0.6" />
        <path className="pl-curtain" d={Array.from({ length: 7 }, (_, i) => `M${R1(cavX + 0.9 + i * 1.9)} ${R1(B.h - TOP - openH)}V${R1(B.h - TOP - 4 - (i % 2) * 0.8)}`).join("")} />
      </g>,
      <Solid key="gland" paths={k.cylinder(ex - 4, B.y + 3, 1.2, B.h, 1.6, P, 16)} tone="mid" />,
    ];
  } },

  /* θ′: the updated model after the booth; asked songs and two neighbours blank, kept songs pinned */
  { st: 4, home: 2, box: [[{ x: J_OUT.x, y: J_OUT.y, w: JUKE.w, d: JUKE.d }, TOP, JUKE.body + JUKE.arch]], draw: (P) => [<Jukebox key="jout" P={P} x0={J_OUT.x} y0={J_OUT.y} after part="jb-out" />] },

  /* 5 · test console at the end of the line: speaker and four dials with full tick scales */
  { st: 5, home: 2, box: [[CONSOLE, 0, CONSOLE.h], [PANEL, PANEL.z, PANEL.h]], draw: (P) => {
    const C = CONSOLE;
    const fy = C.y + C.d;
    const dials = GAUGES.map((g, i) => {
      const cx = DIAL.x0 + i * DIAL.pitch;
      const cy = DIAL.cy;
      const r = DIAL.r;
      const ticks = Array.from({ length: 11 }, (_, j) => {
        const a = (needleAngle(j / 10) * Math.PI) / 180;
        const r0 = j % 5 === 0 ? r - 1.5 : r - 0.9;
        return `M${R2(cx + r0 * Math.sin(a))} ${R2(cy - r0 * Math.cos(a))}L${R2(cx + (r - 0.3) * Math.sin(a))} ${R2(cy - (r - 0.3) * Math.cos(a))}`;
      }).join("");
      const ta = (needleAngle(1) * Math.PI) / 180;
      return (
        <g key={g.key}>
          <circle className="pl-bezel-ring" cx={R2(cx)} cy={cy} r={r + 0.7} />
          <circle className="pl-gauge-face" cx={R2(cx)} cy={cy} r={r} />
          <path className="pl-ticks" d={ticks} />
          <path className="pl-target" d={`M${R2(cx + (r - 2.2) * Math.sin(ta))} ${R2(cy - (r - 2.2) * Math.cos(ta))}L${R2(cx + (r - 0.3) * Math.sin(ta))} ${R2(cy - (r - 0.3) * Math.cos(ta))}`} />
          <g data-needle={g.key} data-cx={R2(cx)} data-cy={cy} transform={`rotate(${needleAngle(g.at).toFixed(2)} ${R2(cx)} ${cy})`}>
            <path className="pl-needle" d={`M${R2(cx)} ${R2(cy + 1)}L${R2(cx)} ${R2(cy - r + 1)}`} />
          </g>
          <circle className="pl-pivot" cx={R2(cx)} cy={cy} r="0.7" />
          <circle className="pl-screw" cx={R2(cx - r - 0.1)} cy={R2(cy + r - 0.4)} r="0.3" />
          <circle className="pl-screw" cx={R2(cx + r + 0.1)} cy={R2(cy + r - 0.4)} r="0.3" />
        </g>
      );
    });
    return [
      <Slab key="cbody" P={P} plan={{ ...C, r: 1.4 }} z={0} h={C.h} bevel={0.8} />,
      <g key="cFront" transform={k.sideMatrix([C.x, fy], C.h, P, "left")}>
        <g data-part="cone" data-cx="7" data-cy="6">
          <circle className="pl-cone" cx="7" cy="6" r="4.6" />
          <circle className="pl-cone-in" cx="7" cy="6" r="3.3" />
          <circle className="pl-cone-in" cx="7" cy="6" r="2.1" />
          <circle className="pl-cone-cap" cx="7" cy="6" r="1" />
        </g>
        <path className="pl-vent" d={[3.6, 5, 6.4, 7.8].map((y) => `M15 ${y}H${C.w - 4}`).join("")} />
      </g>,
      <Slab key="panel" P={P} plan={{ ...PANEL, r: 0.6 }} z={PANEL.z} h={PANEL.h} bevel={0.4} />,
      <g key="dials" transform={k.sideMatrix([PANEL.x, PANEL.y + PANEL.d], PANEL.z + PANEL.h, P, "left")}>
        {dials}
      </g>,
    ];
  } },
];

/** Labels (HTML, outside the drawing's parts): station tags above each station, and a few small captions. */
const TAGS = [
  { st: 1, home: 1, text: "Request", at: [POST.x + 1.5, POST.y + 2, POST.h + 21] },
  { st: 2, home: 1, text: "Your curator", at: [SCAN.x + SCAN.w / 2, AXIS, SCAN.h + 25] },
  { st: 3, home: 1, text: "Forget / keep", at: [(F[0] + M[0]) / 2, AXIS - 8, TOP + 22] },
  { st: 4, home: 2, text: "Unlearning", at: [BOOTH.x + BOOTH.w / 2, AXIS, BOOTH.h + 12] },
  { st: 5, home: 2, text: "Evaluation", at: [CONSOLE.x + CONSOLE.w / 2, AXIS, PANEL.z + PANEL.h + 12] },
];
const NOTES = [
  { home: 1, text: "training data", at: [34, AXIS + BW / 2 + 4, 0], key: "data" },
  { home: 1, text: "not selected", at: [BIN.x + BIN.w / 2, BIN.y + BIN.d + 2, 0], key: "bin" },
  { home: 1, text: "D_f forget", at: [BOX_F.x + BOX_F.w / 2, BOX_F.y + 2, TOP + BOX_F.h + 6], key: "pf", phone: true },
  { home: 1, text: "D_r keep", at: [BOX_R.x + BOX_R.w / 2, BOX_R.y + BOX_R.d + 3, 0], key: "pr", phone: true },
  { home: 2, text: "model θ", at: [J_IN.x + JUKE.w / 2, AXIS + BW / 2 + 3, 0], key: "jin" },
  { home: 2, text: "model θ′", at: [J_OUT.x + JUKE.w / 2, AXIS + BW / 2 + 3, 0], key: "jout" },
  ...GAUGES.map((g, i) => ({ home: 2, text: g.label, at: [PANEL.x + DIAL.x0 + i * DIAL.pitch, CONSOLE.y + CONSOLE.d, 0], key: `dial${i}`, dial: true })),
];

/* ---------- views ---------- */

function buildView(id, rows, xRange) {
  const items = ITEMS.filter((it) => rows.includes(it.home) || (it.also || []).some((r) => rows.includes(r)));
  const unitP = { origin: [0, 0], scale: 1, azimuth: AZIMUTH, elevation: ELEV };
  const benchPlan = { x: xRange[0], y: 0, w: xRange[1] - xRange[0], d: BENCH.d };
  const pts = [...k.boxCorners(benchPlan, -6, 0)];
  for (const it of items) if (rows.includes(it.home)) for (const [plan, z0, h] of it.box) pts.push(...k.boxCorners(plan, z0, z0 + h));
  const tags = TAGS.filter((b) => rows.includes(b.home));
  for (const b of tags) pts.push(b.at);
  const flat = pts.map((p) => k.iso(p, unitP));
  const xs = flat.map((p) => p[0]);
  const ys = flat.map((p) => p[1]);
  const left = Math.min(...xs) - PAD;
  const top = Math.min(...ys) - PAD - 6;
  const width = Math.ceil(Math.max(...xs) + PAD - left);
  const height = Math.ceil(Math.max(...ys) + PAD - top);
  const P = { origin: [R1(-left), R1(-top)], scale: 1, azimuth: AZIMUTH, elevation: ELEV };
  const pct = (at) => {
    const [x, y] = k.iso(at, P);
    return { left: `${((x / width) * 100).toFixed(2)}%`, top: `${((y / height) * 100).toFixed(2)}%` };
  };
  const notes = NOTES.filter((n) => rows.includes(n.home));
  const dialTop = Math.max(...notes.filter((n) => n.dial).map((n) => k.iso(n.at, P)[1]), 0) + 3;
  const labels = [
    ...tags.map((t) => ({ kind: "tag", st: t.st, text: t.text, pos: pct(t.at), key: `t${t.st}` })),
    ...notes.map((n) => {
      const pos = pct(n.at);
      if (n.dial) pos.top = `${((dialTop / height) * 100).toFixed(2)}%`;
      return { kind: n.dial ? "dial" : n.phone ? "phone" : "note", text: n.text, pos, key: n.key };
    }),
  ];
  return { id, rows, P, width, height, items, benchPlan, xRange, labels };
}

export const VIEWS = {
  wide: buildView("wide", [1, 2], [0, BENCH.w]),
  row1: buildView("row1", [1], [0, SPLIT]),
  row2: buildView("row2", [2], [SPLIT, BENCH.w]),
};

export function ViewArt({ view }) {
  const P = view.P;
  const B = view.benchPlan;
  return (
    <>
      <Solid paths={k.slabOf({ ...B, r: 2 }, -6, 6, P, 3, 1.4)} tone="lo" halo />
      <Line d={k.planOutline(k.insetPlan({ ...B, r: 2 }, 2.4), 0, P, 3)} tone="faint" />
      <Screws P={P} pts={[[B.x + 4.5, 4.5, 0], [B.x + B.w - 4.5, 4.5, 0], [B.x + 4.5, B.d - 4.5, 0], [B.x + B.w - 4.5, B.d - 4.5, 0]]} />
      {view.items.map((it, i) => (
        <g key={i} data-st={it.st}>
          {it.draw(P, view)}
        </g>
      ))}
    </>
  );
}

export function ViewLabels({ view }) {
  return view.labels.map((l) =>
    l.kind === "tag" ? (
      <span key={l.key} className="pl-tagl" data-tag={l.st} style={l.pos}>
        <b>{l.st}</b>
        {l.text}
      </span>
    ) : (
      <span key={l.key} className={`pl-note pl-note-${l.kind}`} style={l.pos}>
        {l.kind === "phone" ? (
          <>
            D<sub>{l.text[2]}</sub>
            {l.text.slice(3)}
          </>
        ) : (
          l.text
        )}
      </span>
    ),
  );
}

export { k as kit };
