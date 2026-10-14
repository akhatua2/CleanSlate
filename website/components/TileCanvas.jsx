"use client";
// One wall of tiles drawn on a single canvas. Every tile is a song or book; its red layer
// crossfades with what is left underneath (green where erasing was the goal, hatched
// damage elsewhere). One animation loop handles the random fades and, when the layout
// width changes, glides every tile from where it was on the page to its new spot.
import { useEffect, useRef } from "react";

const GAP = 3;
const FADE_MS = 600, FADE_SPREAD = 900;
const MOVE_MS = 650, MOVE_SPREAD = 220;

export function hash(id) {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619);
  return h >>> 0;
}

// red level by number of extractable windows still left in the work
export function level(n) {
  if (n <= 0) return 0;
  if (n <= 2) return 2;
  if (n <= 5) return 3;
  if (n <= 10) return 4;
  if (n <= 20) return 5;
  if (n <= 40) return 6;
  return 7;
}

const easeOut = (k) => 1 - Math.pow(1 - k, 3);
const easeInOut = (k) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);

function rgb(hex) {
  const v = parseInt(hex.replace("#", ""), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export default function TileCanvas({ songs, stage, kind, pickId, onPick, onHover }) {
  const wrap = useRef(null);
  const cv = useRef(null);
  const st = useRef(null); // all mutable animation state lives here, outside React
  const props = useRef({});
  props.current = { songs, onPick, onHover, pickId };

  // (re)initialise when the set of works changes (model switch)
  useEffect(() => {
    const n = songs.length;
    const css = (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
    const mem = [null, null, ...[2, 3, 4, 5, 6, 7].map((i) => rgb(css(`--mem-${i}`)))];
    const s = {
      n, mem,
      green: css("--gone-4"), ink: css("--ink-1"),
      // per-tile: position (from/to/start), alpha and colour (from/to/start)
      x: new Float32Array(n), y: new Float32Array(n),
      fx: new Float32Array(n), fy: new Float32Array(n), tx: new Float32Array(n), ty: new Float32Array(n), ps: new Float64Array(n),
      a: new Float32Array(n), fa: new Float32Array(n), ta: new Float32Array(n),
      c: Array.from({ length: n }, () => [0, 0, 0]), fc: Array.from({ length: n }, () => [0, 0, 0]),
      tc: Array.from({ length: n }, () => [0, 0, 0]), as: new Float64Array(n),
      size: 14, cols: 1, w: 0, h: 0, origin: null, raf: 0, laidOut: false, hover: -1,
    };
    for (let i = 0; i < n; i++) {
      const lv = level(songs[i].n[stage]);
      s.a[i] = s.fa[i] = s.ta[i] = lv ? 1 : 0;
      const col = mem[lv || 2];
      s.c[i] = [...col]; s.fc[i] = [...col]; s.tc[i] = [...col];
    }
    st.current = s;
    layout(false);
    return () => cancelAnimationFrame(s.raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [songs]);

  // a stage change starts a random-delayed fade per tile
  useEffect(() => {
    const s = st.current;
    if (!s) return;
    const now = performance.now();
    songs.forEach((w, i) => {
      const lv = level(w.n[stage]);
      sample(s, i, now);
      s.fa[i] = s.a[i]; s.ta[i] = lv ? 1 : 0;
      s.fc[i] = [...s.c[i]]; s.tc[i] = [...(s.mem[lv || 2])];
      s.as[i] = now + (hash(w.id + ":" + stage) % FADE_SPREAD);
    });
    kick();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage]);

  useEffect(() => { kick(); }, [pickId]);

  useEffect(() => {
    const ro = new ResizeObserver(() => layout(true));
    ro.observe(wrap.current);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function layout(animate) {
    const s = st.current, el = wrap.current, c = cv.current;
    if (!s || !el || !c) return;
    const w = el.clientWidth;
    if (!w) return;
    const size = w < 420 ? 12 : 14;
    const cols = Math.max(1, Math.floor((w + GAP) / (size + GAP)));
    const rows = Math.ceil(s.n / cols);
    const h = Math.max(size, rows * (size + GAP) - GAP);
    const rect = el.getBoundingClientRect();
    const origin = { x: rect.left + window.scrollX, y: rect.top + window.scrollY };
    const now = performance.now();
    // FLIP across the page: where each tile was (page coords) becomes its starting point
    const dx = s.origin ? s.origin.x - origin.x : 0, dy = s.origin ? s.origin.y - origin.y : 0;
    for (let i = 0; i < s.n; i++) {
      const tx = (i % cols) * (size + GAP), ty = Math.floor(i / cols) * (size + GAP);
      if (animate && s.laidOut) {
        sample(s, i, now);
        s.fx[i] = s.x[i] + dx; s.fy[i] = s.y[i] + dy;
        s.ps[i] = now + (hash(props.current.songs[i].id) % MOVE_SPREAD);
      } else {
        s.fx[i] = s.x[i] = tx; s.fy[i] = s.y[i] = ty; s.ps[i] = 0;
      }
      s.tx[i] = tx; s.ty[i] = ty;
    }
    Object.assign(s, { size, cols, w, h, origin, laidOut: true });
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.height = `${h}px`;
    kick();
  }

  function sample(s, i, now) {
    const pk = s.ps[i] ? Math.min(1, Math.max(0, (now - s.ps[i]) / MOVE_MS)) : 1;
    const pe = easeOut(pk);
    s.x[i] = s.fx[i] + (s.tx[i] - s.fx[i]) * pe;
    s.y[i] = s.fy[i] + (s.ty[i] - s.fy[i]) * pe;
    const ak = Math.min(1, Math.max(0, (now - s.as[i]) / FADE_MS));
    const ae = easeInOut(ak);
    s.a[i] = s.fa[i] + (s.ta[i] - s.fa[i]) * ae;
    for (let j = 0; j < 3; j++) s.c[i][j] = s.fc[i][j] + (s.tc[i][j] - s.fc[i][j]) * ae;
    return pk < 1 || ak < 1;
  }

  function kick() {
    const s = st.current;
    if (!s || s.raf) return;
    const frame = () => {
      const now = performance.now();
      let busy = false;
      for (let i = 0; i < s.n; i++) busy = sample(s, i, now) || busy;
      draw(s);
      s.raf = busy ? requestAnimationFrame(frame) : 0;
    };
    s.raf = requestAnimationFrame(frame);
  }

  function hatch(ctx) {
    const p = document.createElement("canvas");
    p.width = p.height = 4;
    const g = p.getContext("2d");
    g.fillStyle = "#f1efe9"; g.fillRect(0, 0, 4, 4);
    g.strokeStyle = "#d9d6cd"; g.lineWidth = 1.6;
    g.beginPath(); g.moveTo(0, 4); g.lineTo(4, 0); g.moveTo(-1, 1); g.lineTo(1, -1); g.moveTo(3, 5); g.lineTo(5, 3); g.stroke();
    return ctx.createPattern(p, "repeat");
  }

  function draw(s) {
    const c = cv.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, s.w, s.h);
    s.pattern ??= hatch(ctx);
    const r = 3, z = s.size;
    const { pickId: pid, songs: ws } = props.current;
    for (let i = 0; i < s.n; i++) {
      const x = s.x[i], y = s.y[i], a = s.a[i];
      if (a < 0.999) {
        ctx.globalAlpha = 1 - a;
        ctx.beginPath(); ctx.roundRect(x, y, z, z, r);
        if (kind === "forget") { ctx.fillStyle = s.green; ctx.fill(); }
        else {
          ctx.fillStyle = s.pattern; ctx.fill();
          ctx.strokeStyle = "#c9c5ba"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.roundRect(x + 0.5, y + 0.5, z - 1, z - 1, r - 0.5); ctx.stroke();
        }
      }
      if (a > 0.001) {
        ctx.globalAlpha = a;
        const [cr, cg, cb] = s.c[i];
        ctx.fillStyle = `rgb(${cr | 0},${cg | 0},${cb | 0})`;
        ctx.beginPath(); ctx.roundRect(x, y, z, z, r); ctx.fill();
      }
      if (ws[i]?.id === pid || i === s.hover) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = s.ink; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.roundRect(x - 2, y - 2, z + 4, z + 4, r + 2); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  const at = (e) => {
    const s = st.current;
    if (!s) return -1;
    const rect = cv.current.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const col = Math.floor(px / (s.size + GAP)), row = Math.floor(py / (s.size + GAP));
    if (col < 0 || col >= s.cols || px % (s.size + GAP) > s.size || py % (s.size + GAP) > s.size) return -1;
    const i = row * s.cols + col;
    return i < s.n ? i : -1;
  };

  return (
    <div className="tiles-wrap" ref={wrap}>
      <canvas
        ref={cv}
        className="tiles-cv"
        role="img"
        aria-label={`${songs.length} works; tap one to read it`}
        onPointerMove={(e) => {
          if (e.pointerType !== "mouse") return;
          const i = at(e), s = st.current;
          if (s && i !== s.hover) { s.hover = i; kick(); draw(s); }
          props.current.onHover(i >= 0 ? { s: props.current.songs[i], x: e.clientX, y: e.clientY } : null);
        }}
        onPointerLeave={() => { const s = st.current; if (s) { s.hover = -1; draw(s); } props.current.onHover(null); }}
        onClick={(e) => { const i = at(e); if (i >= 0) props.current.onPick(props.current.songs[i]); }}
      />
    </div>
  );
}
