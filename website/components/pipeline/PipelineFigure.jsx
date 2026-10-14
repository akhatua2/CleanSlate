"use client";

import * as React from "react";
import "./iso.css";
import "./pipeline.css";
import { FLIP, GAUGES, GEOM, PAGES, VIEWS, ViewArt, ViewLabels, boxAt, kit as k, needleAngle, pageAt } from "./scene";

const Df = (
  <>
    D<sub>f</sub>
  </>
);
const Dr = (
  <>
    D<sub>r</sub>
  </>
);
const STEPS = [
  { n: 1, name: "Request", text: <>A request arrives: stop the model from reproducing these songs.</> },
  { n: 2, name: "Curate", text: <>Your curator is the scanner: it checks data from any source against the request and decides what goes where. This is the part you submit.</> },
  { n: 3, name: "Two sets", text: <>It returns what to forget ({Df}) and, optionally, what to keep ({Dr}). Everything else passes through.</> },
  { n: 4, name: "Unlearn", text: <>A fixed unlearning method takes the model and both sets, and produces an updated model: songs in {Df} fade, {Dr} holds, and some others fade too.</> },
  { n: 5, name: "Evaluate", text: <>We score the updated model: requested songs gone, other songs intact, facts still known, general ability unchanged.</> },
];
const DEFAULT_STEP = 2;

const LABEL =
  "One low conveyor on a bench, left to right. 1, Request: a clipboard on a post listing five songs, cabled to 2, Your curator: a red X-ray tunnel the belt runs through, carrying bags of training data. 3, Forget / keep: past the tunnel a gate forks the belt into three branches: a box marked D sub f forget with a red rim, a box marked D sub r keep, and a branch that ends in a not-selected bin. The two box branches rejoin and carry the boxes, with a jukebox labelled model theta whose song cards are all filled, into 4, Unlearning: a fixed booth. A jukebox labelled model theta prime rides out with the asked songs and two neighbours blank and the kept songs pinned. 5, Evaluation: the belt ends at a console with a speaker and four dials: asked songs gone, other songs kept, facts kept, general skills.";

const ease = (t) => {
  const u = Math.min(1, Math.max(0, t));
  return u * u * (3 - 2 * u);
};
const clamp01 = (t) => Math.min(1, Math.max(0, t));
const still = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const J_IN_DX = GEOM.BOOTH.x + 10 - GEOM.J_IN.x;
const J_OUT_FROM = GEOM.BOOTH.x + GEOM.BOOTH.w + 0.5 - GEOM.J_OUT.x;
const P3_AT = (i) => 0.2 + i * 0.6;
const P3_DUR = 1.6;

/** Rest pose: everything in its finished state, as rendered on the server. */
const restPose = () => ({
  sheet: 0,
  scan: 0,
  pages: PAGES.map(() => ({ u: 1, op: 1 })),
  flip: FLIP.f,
  boxes: { u: 0, op: 1 },
  jIn: { dx: 0, op: 1 },
  jOut: { dx: 0, op: 1 },
  needles: GAUGES.map((g) => needleAngle(g.at)),
  cone: 1,
});

/** Motion for each step: tracks {at, dur, run(t, pose)} in seconds. */
const TRACKS = {
  1: [{ at: 0, dur: 1.3, run: (t, p) => { p.sheet = 1.4 * Math.sin(Math.PI * ease(t)); } }],
  2: [{ at: 0, dur: 2.4, run: (t, p) => { p.scan = t < 1 ? (t * 2) % 1 : 0; } }],
  3: [
    ...PAGES.map((pg, i) => ({ at: P3_AT(i), dur: P3_DUR, run: (t, p) => { p.pages[i] = { u: t, op: t <= 0 ? 0 : Math.min(1, t * 8) }; } })),
    {
      at: 0,
      dur: P3_AT(PAGES.length - 1) + P3_DUR,
      run: (t, p) => {
        const s = t * (P3_AT(PAGES.length - 1) + P3_DUR);
        let cur = PAGES[0];
        for (const pg of PAGES) if (s >= P3_AT(pg.i) - 0.25) cur = pg;
        p.flip = FLIP[cur.dest];
      },
    },
  ],
  4: [
    { at: 0, dur: 1.4, run: (t, p) => { p.jIn = { dx: J_IN_DX * ease(t), op: t >= 1 ? 0 : 1 }; } },
    { at: 1.4, dur: 2.2, run: (t, p) => { if (t <= 0) return; p.boxes = { u: t, op: t >= 1 ? 0 : 1 }; } },
    { at: 3.8, dur: 2.0, run: (t, p) => { p.jOut = { dx: J_OUT_FROM * (1 - ease(t)), op: t <= 0 ? 0 : Math.min(1, t * 3) }; } },
    { at: 6.0, dur: 0.8, run: (t, p) => { if (t <= 0) return; p.boxes = { u: 0, op: t }; p.jIn = { dx: 0, op: t }; } },
  ],
  5: [
    ...GAUGES.map((g, i) => ({ at: i * 0.18, dur: 1.5, run: (t, p) => { p.needles[i] = needleAngle(g.at * ease(t)); } })),
    { at: 0, dur: 1.8, run: (t, p) => { p.cone = 1 + 0.07 * Math.sin(Math.PI * 6 * t) * (1 - t); } },
  ],
};

function startPose(step) {
  const p = restPose();
  for (const tr of TRACKS[step]) tr.run(0, p);
  return p;
}

function Scene({ view, className }) {
  return (
    <div className={`pl-stage ${className}`}>
      <svg className="iso-svg" data-view={view.id} viewBox={`0 0 ${view.width} ${view.height}`} role="img" aria-label={LABEL} focusable="false">
        <ViewArt view={view} />
      </svg>
      <div className="pl-labels" aria-hidden="true">
        <ViewLabels view={view} />
      </div>
    </div>
  );
}

const shown = (view, x) => (x >= view.xRange[0] && x <= view.xRange[1] ? 1 : 0);

export default function PipelineFigure() {
  const root = React.useRef(null);
  const [step, setStep] = React.useState(DEFAULT_STEP);
  const anim = React.useRef({ raf: 0 });

  const apply = React.useCallback((pose) => {
    const node = root.current;
    if (!node) return;
    node.querySelectorAll("svg[data-view]").forEach((svg) => {
      const view = VIEWS[svg.dataset.view];
      const P = view.P;
      const sheet = svg.querySelector('[data-part="sheet"]');
      if (sheet) {
        if (!sheet.dataset.base) sheet.dataset.base = sheet.getAttribute("transform") || "";
        sheet.setAttribute("transform", `${sheet.dataset.base} translate(0 ${(-pose.sheet).toFixed(2)})`);
      }
      svg.querySelector('[data-part="scanline"]')?.setAttribute("transform", `translate(${(pose.scan * 12).toFixed(2)} 0)`);
      PAGES.forEach((pg, i) => {
        const g = svg.querySelector(`[data-page="${i}"]`);
        if (!g) return;
        const at = pageAt(pg, pose.pages[i].u);
        g.setAttribute("transform", k.translateAlong(P, [at[0] - pg.rest[0], at[1] - pg.rest[1], at[2] - pg.rest[2]]));
        g.setAttribute("opacity", pose.pages[i].op.toFixed(3));
      });
      svg.querySelector('[data-part="flip"]')?.setAttribute("transform", `rotate(${pose.flip.toFixed(1)})`);
      for (const which of ["f", "r"]) {
        const g = svg.querySelector(`[data-box="${which}"]`);
        if (!g) continue;
        const off = boxAt(which, pose.boxes.u);
        const b = which === "f" ? GEOM.BOX_F : GEOM.BOX_R;
        g.setAttribute("transform", k.translateAlong(P, off));
        g.setAttribute("opacity", (pose.boxes.op * shown(view, b.x + b.w / 2 + off[0])).toFixed(3));
      }
      const jin = svg.querySelector('[data-part="jb-in"]');
      if (jin) {
        jin.setAttribute("transform", k.translateAlong(P, [pose.jIn.dx, 0, 0]));
        jin.setAttribute("opacity", pose.jIn.op.toFixed(3));
      }
      const jout = svg.querySelector('[data-part="jb-out"]');
      if (jout) {
        jout.setAttribute("transform", k.translateAlong(P, [pose.jOut.dx, 0, 0]));
        jout.setAttribute("opacity", pose.jOut.op.toFixed(3));
      }
      GAUGES.forEach((g, i) => {
        const n = svg.querySelector(`[data-needle="${g.key}"]`);
        if (n) n.setAttribute("transform", `rotate(${pose.needles[i].toFixed(2)} ${n.dataset.cx} ${n.dataset.cy})`);
      });
      const cone = svg.querySelector('[data-part="cone"]');
      if (cone) {
        const { cx, cy } = cone.dataset;
        cone.setAttribute("transform", `translate(${cx} ${cy}) scale(${pose.cone.toFixed(3)}) translate(${-cx} ${-cy})`);
      }
    });
  }, []);

  const play = React.useCallback(
    (n) => {
      const a = anim.current;
      cancelAnimationFrame(a.raf);
      if (still()) {
        apply(restPose());
        return;
      }
      apply(startPose(n));
      const tracks = TRACKS[n];
      const end = Math.max(...tracks.map((t) => t.at + t.dur));
      let t0 = 0;
      const frame = (now) => {
        if (!t0) t0 = now;
        const s = (now - t0) / 1000;
        const pose = restPose();
        for (const tr of tracks) tr.run(clamp01((s - tr.at) / tr.dur), pose);
        apply(pose);
        if (s < end) a.raf = requestAnimationFrame(frame);
      };
      a.raf = requestAnimationFrame(frame);
    },
    [apply],
  );

  React.useEffect(() => {
    const node = root.current;
    apply(restPose());
    if (!node || still()) return undefined;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          play(DEFAULT_STEP);
          io.disconnect();
        }
      },
      { threshold: 0.35 },
    );
    io.observe(node);
    const a = anim.current;
    return () => {
      io.disconnect();
      cancelAnimationFrame(a.raf);
    };
  }, [apply, play]);

  const choose = (n) => {
    setStep(n);
    play(n);
    root.current?.querySelector(`[data-step-button="${n}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: still() ? "auto" : "smooth" });
  };

  const onKey = (e) => {
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const n = ((step - 1 + d + 5) % 5) + 1;
    choose(n);
    root.current?.querySelector(`[data-step-button="${n}"]`)?.focus();
  };

  const art = React.useMemo(
    () => (
      <>
        <Scene view={VIEWS.wide} className="pl-wide" />
        <div className="pl-narrow">
          <Scene view={VIEWS.row1} className="pl-rowv" />
          <Scene view={VIEWS.row2} className="pl-rowv" />
        </div>
      </>
    ),
    [],
  );

  return (
    <div className="pl iso" data-theme="light" data-step={step} ref={root}>
      <p className="pl-lede">
        You submit the <span className="pl-lede-mine">curator</span>, and optionally your own data. It scans data against the request, with access to the model, and returns what to forget and what to keep. Everything else is fixed.
      </p>
      <figure className="iso-figure">
        <div className="iso-plate">
          <div className="iso-plate-corners" aria-hidden="true">
            <span className="iso-plate-corner" data-corner="fig">Fig 3</span>
            <span className="iso-plate-corner" data-corner="title">CleanSlate overview</span>
          </div>
          {art}
        </div>
        <figcaption className="pl-below">
          <div className="pl-steps" role="group" aria-label="Pipeline steps" onKeyDown={onKey}>
            {STEPS.map((s) => (
              <button key={s.n} type="button" data-step-button={s.n} aria-current={s.n === step ? "step" : undefined} tabIndex={s.n === step ? 0 : -1} onClick={() => choose(s.n)}>
                <span className="pl-step-n">{s.n}</span>
                <span className="pl-step-name">{s.name}</span>
              </button>
            ))}
          </div>
          <p className="pl-sentence" aria-live="polite">
            {STEPS[step - 1].text}
          </p>
          <div className="pl-keys">
            <span className="pl-key"><i className="pl-sw" data-sw="mine" />your curator (the part you submit)</span>
            <span className="pl-key"><i className="pl-sw" data-sw="fixed" />fixed by the benchmark</span>
            <span className="pl-key"><i className="pl-sw" data-sw="dark" />matches the request</span>
          </div>
        </figcaption>
      </figure>
    </div>
  );
}
