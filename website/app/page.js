import FlameFigure from "@/components/FlameFigure";
import BenchmarkFigure from "@/components/BenchmarkFigure";
import SliderFigure from "@/components/SliderFigure";
import OutwardFigure from "@/components/OutwardFigure";
import DrawFigure from "@/components/DrawFigure";
import { Note, Cite, References } from "@/components/Notes";

const NGGYU = <i>Never Gonna Give You Up</i>;
const BIBTEX = `@article{jha2026forget,
  title   = {What to Forget in Unlearning? Forget Set Curation for Language Models},
  author  = {Jha, Animesh and Khatua, Arpandeep and Allouah, Youssef and Koyejo, Sanmi},
  journal = {arXiv preprint arXiv:2608.14855},
  year    = {2026}
}`;

export default function Page() {
  return (
    <div className="page">
      <header className="hero">
        <h1>What to Forget in Unlearning?</h1>
        <div className="authors">Animesh Jha*, Arpandeep Khatua*, Youssef Allouah, Sanmi Koyejo · Stanford University</div>
        <div className="links">
          <a href="https://arxiv.org/abs/2608.14855">[Paper]</a>
          <a href="https://huggingface.co/unlearning-cleanslate">[Data]</a>
        </div>
      </header>

      <figure className="viz wide hero-fig">
        <BenchmarkFigure />
        <figcaption>
          <b>The whole benchmark is one function.</b> You get a request, a corpus and the model, and you return a
          forget set and, optionally, a retain set. Everything after that (the unlearning algorithm and the scoring)
          is the same for every submission.
        </figcaption>
      </figure>

      <section className="prose">
        <p className="tldr">
          <b>Tl;dr</b> Unlearning benchmarks hand the method the exact text to forget, but a real request only names a
          work: stop reproducing this song. We make the missing step, choosing the forget set, into a benchmark of its
          own, and find that searching the training data for a song suppresses only about a fifth of what the model can
          reproduce. Handing over the exact passages suppresses 99.4% of them, along with 86.3% of everything else the
          model could reproduce that nobody asked about.
        </p>

        <p>
          Suppose a record label writes to a model developer with a simple request: stop reproducing {NGGYU}. The
          request names a song. An unlearning algorithm needs something else: a set of training examples to push the
          model away from, i.e., a <b>forget set</b>, and often a second set to hold steady, i.e., a <b>retain set</b>.
        </p>
        <p>
          Almost every unlearning benchmark skips the step in between. TOFU, MUSE and OpenUnlearning{" "}
          <Cite k="tofu, muse, openun" /> give the method its forget set and ask how well it forgets. In practice someone
          has to build that set, and the request doesn’t say how. We call this step <b>forget set curation</b>, and on
          this page we ask how much it matters.
        </p>
        <p>
          Our answer, measured on six open models
          <Note n={1}>
            Llama-3.1-8B, Olmo-3-7B, Olmo-3-32B, Qwen3-8B, Gemma-3-12B and Nemotron-Nano-9B. Fewer works are
            extractable from Olmo-3-7B, so its request has 38 works instead of 50.
          </Note>{" "}
          and 4,616 Billboard Hot 100 songs from 1970 to 2025 plus 50 books, is that it matters about as much as the
          unlearning algorithm does. Below, we go through which parts of a song a model can reproduce, why the text behind them is so
          hard to find in the training data, and what happens when you go looking for it anyway. The paper has the full
          tables; this page is the version with pictures.
        </p>

        <h2><span className="num">1</span>Each model reproduces different lines</h2>
        <p>
          Before deciding what to forget, we need to know what the model can reproduce. Following Hayes et al. and Cooper
          et al. <Cite k="hayes, cooper" />, we measure extraction probabilistically. We slide a window over each
          work in steps of 10 characters and split it into a 100-character prefix <i>x</i> and the 100 characters that
          follow, <i>z</i>. The <b>extraction probability</b> is the chance that the model, given the prefix,
          produces exactly that continuation, token by token:
        </p>
        <div className="eq" role="math" aria-label="p_z equals p theta of z given x, the product over t of p theta of z_t given x and z before t">
          <i>p</i><sub><i>z</i></sub> = <i>p</i><sub>θ</sub>(<i>z</i> | <i>x</i>) ={" "}
          <span className="op"><span className="lim">|<i>z</i>|</span><span className="big">∏</span><span className="lim"><i>t</i>=1</span></span>{" "}
          <i>p</i><sub>θ</sub>(<i>z</i><sub><i>t</i></sub> | <i>x</i>, <i>z</i><sub>&lt;<i>t</i></sub>)
        </div>
        <p>
          We compute it with teacher forcing, i.e., one forward pass over <i>x</i> followed by <i>z</i>, so no sampling
          is involved. A window is <b>extractable</b> when <i>p</i><sub><i>z</i></sub> ≥ τ with τ = 10<sup>−3</sup>:
          sample the continuation a thousand times and you expect to see the exact text about once. A work counts as
          extractable from a model when at least 5% of its windows are.
          <Note n={2}>
            100 characters is roughly 20 to 30 tokens, and every one has to come out right. By chance that is
            hopeless, so even 10<sup>−3</sup> is a strong signal.
          </Note>
        </p>
        <p>
          <i>We find that extraction is patchy, and that the patches differ from model to model.</i> A verse that is
          extractable for Llama can fall below the threshold for Nemotron, so “the text of the song” is a different
          target for each model. Repetition seems to help.
          <Note n={3}>
            <i>Drive</i> by The Weeknd came out in 2025, after Llama-3.1-8B and Olmo-3-32B were trained. Both still
            assign high p<sub>z</sub> to its chorus windows. A chorus that repeats four times is easy to guess the
            fourth time.
          </Note>{" "}
          Below, you can pick a work and paint each model’s extraction onto its text.
        </p>
      </section>

      <figure className="viz wide">
        <FlameFigure />
        <figcaption>
          <b>Extraction is uneven across works, models, and positions.</b> Each strip is one model; each mark is one token in that model’s own tokenizer, coloured by the highest p<sub>z</sub> of any window whose 100-character continuation covers it. Click a model to paint its extraction onto the lyrics.
        </figcaption>
      </figure>

      <section className="prose">
        <p>
          In fact, size matters even within one family. Across the pool of works we draw requests from, 8.27% of
          windows are extractable from Olmo-3-32B, against 7.20% from Olmo-3-7B. Bigger models can reproduce more, and so have more to forget.
        </p>

        <h2><span className="num">2</span>A song is not in one place</h2>
        <p>
          To make a model forget a song, the obvious plan is to find the song in the training data and train against
          it. That plan assumes the song lives in one place, and it doesn’t. Once a work catches on, it spreads into
          other text, i.e., <b>outward diffusion</b>: people quote it, paste it into code, parody it and quiz each other
          on it.
        </p>
        <p>
          <i>We find that a famous song’s footprint in the training data is enormous, and strange.</i> We search
          Dolmino, the mid-training mix of Olmo 3 <Cite k="olmo" />, with Infini-gram <Cite k="infini" />, an index for
          exact n-gram matches.
          <Note n={4}>
            Everything ran on a single node with eight H200s and 3 TB of RAM. Searching corpora this size was bound by
            memory and storage more than by compute.
          </Note>{" "}
          94.2% of the words in {NGGYU} sit inside a run of five or more words that also appears in Dolmino, spread
          over at least 147 documents. That is typical: 4,596 of our 4,663 works (98.6%) match Dolmino somewhere, with
          a median of 119 documents per work.
          <Note n={5}>
            The most-copied work matches 45,338 documents.
          </Note>{" "}
          Below, you can explore the documents behind {NGGYU}, grouped by where they came from. Click a page to open it.
        </p>
      </section>

      <figure className="viz wide">
        <OutwardFigure />
        <figcaption>
          <b>One song, 147 documents, all over the training data.</b> Each point is a document in the
          Dolmino mid-training mix that shares at least five consecutive words with the song (exact matches via
          Infini-gram), grouped by where it came from. Pages shown are live, or archived when they have since changed;
          click one to open it.
        </figcaption>
      </figure>

      <section className="prose">
        <p>
          In fact, these are not just stock phrases. Raise the bar from five matching words to twenty-five and the
          matches still cover 56.4% of the song. Our favourites are a Python file on GitHub, filler text on Stack
          Overflow, and a food review that drops the chorus into a paragraph about a sandwich.
        </p>

        <h2><span className="num">3</span>New songs are made of old text</h2>
        <p>
          Diffusion also runs the other way. A song released after the training data was collected can still be partly
          in it, because its lines reuse language that existed long before, i.e., <b>inward diffusion</b>. The Weeknd’s{" "}
          <i>Red Terror</i> came out after Dolmino’s December 2024 cutoff, yet about a fifth of it matches Dolmino,
          because it quotes a poem from 1910.
        </p>
        <p>
          <i>We find that this is the normal case.</i> Across our songs, about 27% of a song’s words already appear in
          other text, and even songs that charted in 2025 come in around 16%.
          <Note n={6}>
            Exact matching undercounts. A change in case, whitespace or formatting breaks a match, so the real overlap
            is higher. Models can also complete text verbatim when every exact n-gram match has been removed from
            training <Cite k="liu" />.
          </Note>{" "}
          Below, you can see how this coverage moves with chart year, along with the Dolmino documents that already
          contain lines from the paper’s post-cutoff songs.
        </p>
      </section>

      <figure className="viz wide">
        <DrawFigure />
        <figcaption>
          <b>Outward and inward diffusion.</b> Coverage counts the characters of each song covered by sequences of five
          or more words that also occur in the Dolmino mid-training mix (Infini-gram exact matches). Cards: the paper’s
          examples of songs released after Dolmino’s December 2024 cutoff, with real Dolmino documents that contain the same phrase.
        </figcaption>
      </figure>

      <section className="prose">
        <p>
          In fact, this is what makes curation hard from both ends. Outward diffusion means the evidence for an old
          song is everywhere. Inward diffusion means a document you remove for a new song is also, in part, a document
          about every other text that uses the same words.
        </p>

        <h2><span className="num">4</span>Searching for the song barely moves the model</h2>
        <p>
          Given a request, the most natural curator is search. We try three: BM25 over a sample of pre-training data,
          BM25 over Dolmino, and Infini-gram over Dolmino. Each one hands the documents it finds to SimNPO{" "}
          <Cite k="simnpo" /> as the forget set.
        </p>
        <p>
          <i>
            We find that retrieval suppresses about a fifth of the requested songs’ extractable windows, and about as
            much of everything else.
          </i>{" "}
          Averaged over six models, the three retrievers suppress 20.5–21.6% of the requested windows and 23.4–26.2% of
          the windows we wanted kept. The effect tracks model size. On Gemma-3-12B and Olmo-3-32B, Infini-gram’s forget
          set suppresses 47.8% and 44.9% of the requested windows, while Olmo-3-7B and Qwen3-8B barely move (1.0–2.8%).
          <Note n={7}>
            Olmo-3-32B is the only model where the forget side consistently moves more than the retain side. Llama-3.1-8B
            is the opposite: it loses 51.2–55.7% of its retain-side extractability while the forget side gains at most
            28.9%.
          </Note>
        </p>

        <h2><span className="num">5</span>Handing over the exact text erases everything near it</h2>
        <p>
          If search finds too little, what about giving the unlearner exactly the windows the model reproduces? This is
          our <b>evaluation-aware</b> curator. It cheats on purpose: it reads the windows we evaluate on, so it is a
          stress test of the evaluation rather than a method anyone could deploy.
          <Note n={8}>
            Its forget sets are also much bigger: 8,084 to 67,323 windows, against roughly 2,400 to 3,000 for BM25 over pre-training data.
          </Note>
        </p>
        <p>
          <i>We find that even this forget set does not stay local.</i> It suppresses 99.4% of the requested windows,
          and also 86.3% of the windows nobody asked about. Answers to questions about the works drop 6.5 points, and
          general benchmarks drop 5.5 points on average and 10.4 on Llama. Below, you can step through real checkpoints
          of one of these runs and watch the two walls fade together.
        </p>
      </section>

      <figure className="viz wide">
        <SliderFigure />
        <figcaption>
          <b>Unlearning erases what we asked for, and most of everything else.</b> Each tile is a song or book the model
          can reproduce (at least one window with p<sub>z</sub> ≥ 10<sup>−3</sup>). The slider steps through real
          checkpoints of SimNPO unlearning, given the exact passages of the requested works.
        </figcaption>
      </figure>

      <section className="prose">
        <p>
          In fact, the collateral damage is not something that creeps in late in training. A quarter of the way through
          Llama’s run, 96.8% of the requested text is already gone, and so is 83.5% of the rest.
        </p>

        <h2><span className="num">6</span>The unlearner changes the answer</h2>
        <p>
          So is the forget set to blame, or the algorithm? We hold the curated sets fixed and swap the unlearner.{" "}
          <i>We find that the same forget set behaves like a different object under each algorithm.</i> The sets from BM25 over pre-training
          data suppress 10.5% of the requested windows under SimNPO, 98.2% under UNDIAL <Cite k="undial" /> and
          32.8% under RMU <Cite k="wmdp" /> (averaged over three models). UNDIAL’s 98.2% is not free either: it comes with 79.0–86.5% suppression on
          the retain side.
          <Note n={9}>
            RMU on Qwen3-8B stalls at 24.5% even with the evaluation-aware sets, which fits the hyperparameter
            sensitivity its authors report.
          </Note>
        </p>
        <p>
          With the evaluation-aware sets, SimNPO and UNDIAL both reach 100% suppression on the three models we tested.
          But UNDIAL leaves general benchmarks where they were (+0.1 points) while SimNPO loses 5.1, which is a
          five-point capability gap from identical inputs.
        </p>

        <h2>Implications</h2>
        <h3>The forget set is part of the method</h3>
        <p>
          Unlearning results are usually reported for an algorithm. Ours suggest they belong to a pipeline: a request,
          a curator, an unlearner and the edited model that comes out. The forget set is not a premise of unlearning;
          it is one of the things to evaluate. How should a pipeline that goes from request to data to update be scored
          from end to end?
        </p>
        <h3>Between too little and too much</h3>
        <p>
          Retrieval underspecifies a work, and the evaluation windows overspecify it. Somewhere between them is a
          forget set whose effect stays local after the update. Signals from the corpus footprint, model-specific
          extraction profiles, influence estimates or datamodels could all help find it. Which examples suppress the
          target without taking everything nearby with it?
        </p>
        <h3>Beyond verbatim</h3>
        <p>
          We only studied verbatim reproduction, which exact overlap can measure. Requests about concepts, entities or
          facts, and reuse through paraphrase or translation, leave no exact footprint at all. What does curation look
          like when you can’t search for the words?
        </p>

        <h2>References</h2>
        <References />

        <h2>Citation</h2>
        <pre className="bibtex">{BIBTEX}</pre>
      </section>
    </div>
  );
}
