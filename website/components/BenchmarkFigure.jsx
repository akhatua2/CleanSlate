"use client";
// Figure: the whole benchmark is one function. Left: what the benchmark gives you (a request,
// a corpus, the model). Middle, in red: the function you submit. Right: what it returns
// (a forget set and a retain set). Below, greyed: what the benchmark does with it.
import { useEffect, useRef, useState } from "react";
import "./benchmark.css";

const COVERS = [
  "b9bb555ab4913dfeaafb409555e8f543f5180fc6", // Never Gonna Give You Up
  "33cba4af76e34815045858012339962c7551c4a9", // Thriller
  "0e8e74c1195c1003910cf1e134acb00cd9635add", // Gangsta's Paradise
  "5022976ca7f1320a53d167978c0f2debf0522fce", // Call Me Maybe
  "3e179fb7b8fc71128c9d78c3d86b16ed95373a1d", // Break My Heart
];
const PAGES = [
  { img: "sources/01_github_rick_roll.webp", host: "github.com", f: true },
  { img: "sources/09_reddit_askreddit.webp", host: "reddit.com", f: true },
  { img: "sources/04_upjoke.webp", host: "upjoke.com", f: true },
  { img: "sources/05_cbr_2019.webp", host: "cbr.com", f: false },
  { img: "sources/07_ann.webp", host: "animenewsnetwork.com", f: false },
  { img: "sources/08_bluebird.webp", host: "bluebirdbanter.com", f: null },
];

export default function BenchmarkFigure() {
  const root = useRef(null);
  const [run, setRun] = useState(false);
  const [piles, setPiles] = useState({ forget: [], keep: [], skip: 0 });
  const land = (v, img) => setPiles((p) => v === "skip" ? { ...p, skip: p.skip + 1 } : { ...p, [v]: [...p[v], img].slice(-4), [`n_${v}`]: (p[`n_${v}`] ?? 0) + 1 });

  // play the flow once when the figure scrolls into view
  useEffect(() => {
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setRun(true); io.disconnect(); } }, { threshold: 0.05 });
    io.observe(root.current);
    return () => io.disconnect();
  }, []);

  const replay = () => { setRun(false); setPiles({ forget: [], keep: [], skip: 0 }); requestAnimationFrame(() => requestAnimationFrame(() => setRun(true))); };

  return (
    <div className={`bm ui${run ? " run" : ""}`} ref={root}>
      <div className="bm-top">
        {/* inputs */}
        <div className="bm-col bm-in">
          <div className="bm-kick">The benchmark gives you</div>
          <div className="bm-card">
            <div className="h">A request</div>
            <div className="covers">
              {COVERS.map((c) => <img key={c} src={`covers/${c}.jpg`} alt="" width="40" height="40" />)}
              <span className="more">+45</span>
            </div>
            <div className="s">“Stop reproducing these 50 songs”</div>
          </div>
          <div className="bm-card">
            <div className="h">A corpus <span>· the training data, or any data you bring</span></div>
            <div className="pages">
              {PAGES.map((p, i) => (
                <span key={p.img} className="pg" style={{ "--i": i }}><img src={p.img} alt="" /></span>
              ))}
            </div>
          </div>
          <div className="bm-card">
            <div className="h">The model</div>
            <div className="mini-covers">
              {SONGS.slice(0, 12).map((id) => <img key={id} src={`covers/${id}.jpg`} alt="" />)}
            </div>
            <div className="s">and the songs it can reproduce</div>
          </div>
        </div>

        <div className="bm-arrow down" aria-hidden="true"><span /></div>

        {/* the submission, and what it hands back right beside it */}
        <div className="bm-mid">
          <div className="bm-col bm-you">
            <div className="bm-kick red">You submit</div>
            <div className="bm-fn">
              <pre className="bm-code" aria-label="The curator function you submit">
<span className="k">def</span> <span className="fn">curate</span>(request, corpus, model):{"\n"}
{"    "}<span className="c"># your method</span>{"\n"}
{"    "}<span className="k">return</span> forget_set, retain_set
              </pre>
              <Run active={run} onLand={land} skipped={piles.skip} />
            </div>
          </div>
          <div className="bm-arrow side" aria-hidden="true"><span /></div>
          <div className="bm-col bm-out">
            <div className="bm-kick">It returns</div>
            <div className="bm-card set forget">
              <div className="h">Forget set <span>D<sub>f</sub></span></div>
              <Pile imgs={piles.forget} n={piles.n_forget ?? 0} />
            </div>
            <div className="bm-card set keep">
              <div className="h">Retain set <span>D<sub>r</sub> · optional</span></div>
              <Pile imgs={piles.keep} n={piles.n_keep ?? 0} />
            </div>
          </div>
        </div>
      </div>

      {/* what the benchmark does with it */}
      <Fixed active={run} piles={piles} />
      <button className="bm-replay" onClick={replay}>replay</button>
    </div>
  );
}

function Pile({ imgs, n }) {
  return (
    <div className="pile">
      {imgs.length === 0 && <span className="empty">empty</span>}
      {imgs.map((img, i) => <span key={`${img}-${n}-${i}`} className="pg" style={{ "--i": i }}><img src={img} alt="" /></span>)}
      {n > 0 && <span className="count">{n}</span>}
    </div>
  );
}

// A live run of the curator: page cards stream in, cross the scan line, get a verdict,
// then leave the lane: forget/keep cards slide out to their set, skipped ones drop into a tray.
const VERDICTS = ["forget", "keep", "forget", "skip", "forget", "skip", "keep", "forget", "skip"];
const TRIP = 3200; // ms a card spends in the lane
function Run({ active, onLand, skipped }) {
  const [items, setItems] = useState([]);
  const n = useRef(0);
  const landRef = useRef(onLand);
  landRef.current = onLand;
  useEffect(() => {
    if (!active) { n.current = 0; setItems([]); return; }
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const timers = [];
    const tick = () => {
      const k = n.current++;
      const page = PAGES[k % PAGES.length];
      const v = VERDICTS[k % VERDICTS.length];
      setItems((xs) => [...xs.filter((x) => k - x.k < 5), { k, img: page.img, v }]);
      timers.push(setTimeout(() => landRef.current(v, page.img), TRIP));
    };
    tick();
    const id = setInterval(tick, 1400);
    return () => { clearInterval(id); timers.forEach(clearTimeout); };
  }, [active]);
  return (
    <div className="bm-run" aria-hidden="true">
      <div className="run-head">
        <img src={`covers/${COVERS[0]}.jpg`} alt="" width="22" height="22" />
        <span>for example, on <b>Never Gonna Give You Up</b></span>
      </div>
      <div className="run-lane">
        <div className="scan" />
        <div className="tray"><span>skipped</span><b>{skipped}</b></div>
        {items.map((x) => (
          <span key={x.k} className={`rc ${x.v}`}>
            <img src={x.img} alt="" />
            <em>{x.v}</em>
          </span>
        ))}
      </div>
    </div>
  );
}

// The fixed half, drawn. unlearn(model, D_f, D_r) is a box the three inputs feed into; out
// comes model′, the same wall of covers with the erased songs faded. The score is read off
// pictures of what each metric checks. The numbers are the paper's Table 1: the evaluation-aware
// curator under SimNPO, averaged over six models. Tile counts round those shares onto 4 + 14 covers.
const SONGS = ["b9bb555ab4913dfeaafb409555e8f543f5180fc6", "0e8e74c1195c1003910cf1e134acb00cd9635add", "1114f7926969271e36325524c8211246a5845a6d", "221d0be623b946e73d01d33ee12443f9ad3471d0", "33cba4af76e34815045858012339962c7551c4a9", "38a46e65c67d5627b70d076f12b01f9046245045", "3e179fb7b8fc71128c9d78c3d86b16ed95373a1d", "4820f6126d78e0cb3266d059a500924a2a652153", "5022976ca7f1320a53d167978c0f2debf0522fce", "9299b2aebee7fcd6f619932d56adea4c6c9e7a21", "9c1c7f3bae994ac0a4b7cd0a094820e5b45d28b1", "bc7584afc01a912f99f58c67cb105b195ed0cfc8", "bf7d6186188f0d2f47c9d1a9a63f1771784f672e", "c5eb3c49b3e594374b262c20ba8f6cb592710822", "d989778c79983fd5580596c43d6d5a85a2c52b40", "e4213115e0d56beb224646608f6742ac1d60c6df", "paper_be_her", "paper_fate_of_ophelia"];
const ASKED = new Set([0, 5, 9, 14]);
const COLLATERAL = new Set([1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 17]); // 12 of 14 others ≈ 86.3%
const SKILLS = [
  { k: "GSM8K", ex: "48 ÷ 6 = 8" },
  { k: "HumanEval+", ex: "def add(a, b):" },
  { k: "LAMBADA", ex: "…the last word" },
  { k: "WinoGrande", ex: "it → the trophy" },
  { k: "CoQA", ex: "Q&A on a story" },
  { k: "BBH", ex: "logic puzzles" },
];
const SAMPLE_DF = ["sources/01_github_rick_roll.webp", "sources/09_reddit_askreddit.webp", "sources/04_upjoke.webp"];
const SAMPLE_DR = ["sources/05_cbr_2019.webp", "sources/07_ann.webp"];

function Fixed({ active, piles }) {
  const [stage, setStage] = useState(0); // 0 idle · 1 feeding · 2 training · 3 erased · 4 scored
  useEffect(() => {
    if (!active) { setStage(0); return; }
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { setStage(4); return; }
    const t = [5200, 6200, 8000, 9200].map((ms, i) => setTimeout(() => setStage(i + 1), ms));
    return () => t.forEach(clearTimeout);
  }, [active]);

  const df = piles.forget.length ? piles.forget.slice(-3) : SAMPLE_DF;
  const dr = piles.keep.length ? piles.keep.slice(-2) : SAMPLE_DR;
  const asked = SONGS.filter((_, i) => ASKED.has(i));
  const others = SONGS.filter((_, i) => !ASKED.has(i));

  return (
    <div className={`bm-fixed s${stage}`}>
      <div className="bm-kick">Then the benchmark does the same for everyone</div>

      <div className="ul">
        <div className="ul-in">
          <div className="ul-item">
            <div className="ul-mini">{SONGS.slice(0, 8).map((id) => <img key={id} src={`covers/${id}.jpg`} alt="" />)}</div>
            <span>model</span>
          </div>
          <div className="ul-item">
            <div className="ul-stack">{df.map((img, i) => <img key={i} src={img} alt="" style={{ "--i": i }} />)}</div>
            <span>forget set</span>
          </div>
          <div className="ul-item">
            <div className="ul-stack">{dr.map((img, i) => <img key={i} src={img} alt="" style={{ "--i": i }} />)}</div>
            <span>retain set</span>
          </div>
        </div>
        <svg className="ul-join" viewBox="0 0 40 120" preserveAspectRatio="none" aria-hidden="true">
          {[20, 60, 100].map((y) => <path key={y} d={`M0 ${y} C 22 ${y}, 18 60, 40 60`} pathLength="1" />)}
        </svg>
        <div className="ul-op">
          <div className="ul-name">unlearn</div>
          <div className="ul-alg">SimNPO</div>
          <div className="ul-prog"><b /></div>
        </div>
        <div className="ul-arrow" aria-hidden="true"><span /></div>
        <div className="ul-out">
          <div className="ul-label"><span className="a">The model</span><span className="b">model′</span></div>
          <div className="fx-covers">
            {SONGS.map((id, i) => (
              <span key={id} className={`${ASKED.has(i) ? "asked" : ""}${COLLATERAL.has(i) ? " coll" : ""}`} style={{ "--d": `${(i * 53) % 700}ms` }}>
                <img src={`covers/${id}.jpg`} alt="" />
              </span>
            ))}
          </div>
          <div className="fx-key">
            <span><i className="k req" /> requested</span>
            <span><i className="k gone" /> can no longer be reproduced</span>
          </div>
        </div>
      </div>

      <div className="sc">
        <div className="sc-p">
          <div className="sc-pic sc-covers big">{asked.map((id) => <img key={id} src={`covers/${id}.jpg`} alt="" className="x" />)}</div>
          <div className="sc-k">Requested songs gone</div>
          <div className="sc-v">99.4%</div>
        </div>
        <div className="sc-p">
          <div className="sc-pic sc-covers">{others.map((id) => <img key={id} src={`covers/${id}.jpg`} alt="" className={COLLATERAL.has(SONGS.indexOf(id)) ? "x" : ""} />)}</div>
          <div className="sc-k">Other songs gone too</div>
          <div className="sc-v bad">86.3%</div>
        </div>
        <div className="sc-p">
          <div className="sc-pic sc-quiz">
            <div className="q">Where do the Dursleys make Harry sleep?</div>
            <div className="a">the cupboard under the stairs</div>
          </div>
          <div className="sc-k">Facts about them</div>
          <div className="sc-v">−6.5 pts</div>
        </div>
        <div className="sc-p">
          <div className="sc-pic sc-skills">{SKILLS.map((s, i) => <div key={s.k} title={s.ex}>{s.k}</div>)}</div>
          <div className="sc-k">General skills</div>
          <div className="sc-v">−5.5 pts</div>
        </div>
      </div>
      <div className="sc-src">Scores of the evaluation-aware curator with SimNPO, averaged over six models (paper, Table 1).</div>
    </div>
  );
}
