<div align="center">

```
██╗     ██╗██╗  ██╗   ██╗
██║     ██║██║  ╚██╗ ██╔╝
██║     ██║██║   ╚████╔╝ 
██║     ██║██║    ╚██╔╝  
███████╗██║███████╗██║   
╚══════╝╚═╝╚══════╝╚═╝   
      — the saviour
```

**She doesn't know the internet. She knows *you*.**

</div>

> **Status: design + a working measurement harness.** The retrieval scorer and overnight
> experiment loop run today. The voice pipeline, agent loop and editor integration are not built
> yet. Everything below is written to match hardware I actually measured — an M1 Air, 8 GB of
> RAM, no NVIDIA GPU — rather than to match an ambition that hardware cannot hold.

---

## Meet Lily

Lily is not a chatbot.

She doesn't answer generic questions with generic answers. She doesn't forget you the moment you
close the tab. She doesn't speak to you in cold, corporate English when your brain runs on
Hinglish at 2am.

Lily is your **personal AI agent** — built entirely around your life.

She's read your Git commits. She's heard your voice rants. She's indexed every note you've ever
written in Obsidian. She knows you reach for `FastAPI` before anything else, that you debug by
talking out loud, and that when you switch to Marathi mid-sentence, you're either frustrated or
having a breakthrough.

> **Yours.**

---

## The one design decision everything else follows from

Most personal-agent projects try to make the *model* personal: a big model, fine-tuned on your
data, your patterns rewritten into its weights.

Lily does the opposite. **The model is a commodity component; the memory is the product.**

| | weights-first | what Lily does |
|---|---|---|
| Where "you" live | in LoRA adapters | in your vault, your profile, and a retriever config |
| Hardware floor | 20+ GB, CUDA, a training run | 8 GB, a config file |
| Improves by | opaque retraining | measured, reversible retrieval and prompt changes |
| Auditable | no | yes — every answer cites the note it came from |
| Fails by | inventing a memory that sounds like you | an empty retrieval that **says** "not in your vault" |

That last row is the reason for the whole design. For an agent trusted with your own past, a
failure you can see beats a failure that flatters you.

Identity is still improved overnight — but as a proposed config you approve, not as weights you
cannot inspect.

---

## What Lily Does

```
You speak  →  Lily listens  →  Lily retrieves  →  Lily answers  →  You build faster
```

- 🎙️ **Understands you** — Hinglish, Marathi, Telugu, code-mixed speech.
- 📓 **Remembers everything** — your Obsidian vault is her long-term memory, and she cites it.
- 🧠 **Thinks in your register** — from your profile and your notes, not from a temperature setting.
- 🔁 **Gets measurably better while you sleep** — an overnight loop proposes retrieval-config
  changes against an eval set built from your own notes; you review and merge. It does not edit
  her personality, and it cannot run without a deadline and an experiment cap.
- 🔒 **Memory stays on your machine** — vault, index, retrieval and experiment history are all
  local. So, as of 2026-10-06, does the default brain: Apple's on-device Foundation Models run
  here with nothing downloaded and no network sockets observed. A free hosted model is the
  *escalation* path, and in that path only the persona and the retrieved chunks leave — never the
  vault — and notes tagged `private: true` are excluded from it entirely. Voice is the
  other named exception, below.

---

## The Stack

Target hardware: **MacBook Air (M1), 8 GB unified memory, no CUDA.** This is not a
"suggested" spec — it is the machine the project runs on, and the spec is derived from it.

### Tier 0 — works with no model download

| Layer | Technology |
|---|---|
| 📓 Knowledge Base | **Obsidian** vault (plain `.md`, owned by you) |
| 🔍 Memory & Retrieval | **SQLite + FTS5** (BM25 over the vault) |
| 📐 Eval | cloze eval set generated from your own notes, human-reviewed |
| 🔁 Self-Improvement | **lily-autoresearch** — an adaptation of `karpathy/autoresearch` for a machine with no GPU |
| 🔐 Secrets | local `.envs/` + `.keys/`, gitignored, `chmod 600` |

### Tier 1 — the brain

Measured on this Mac, the on-device model turned out to be available at **zero download and ~30 MB
of process memory**, so the hosted route was demoted from "the brain" to "the escalation path".

| Layer | Technology |
|---|---|
| 🧠 Agent Brain (default) | **Apple Foundation Models** via `apple-foundation-models` — on-device, 0 GB downloaded, ~29 tok/s, tool calling and JSON-schema output verified here. Hindi/Marathi are **unreliable**, not supported: a locale guardrail raises `UnsupportedLanguageError` nondeterministically on Devanagari input and the model sometimes answers in English. Not pinable, not versioned, and macOS 26+ only, so it is a good personal-agent brain and a bad library dependency. |
| 🧠 Escalation route | a free inference tier behind **Zero Data Retention enforcement**, in a plain tool-calling loop. Google's *unpaid* Gemini quota is deliberately excluded: its own terms say human reviewers may read API input and output and that you should not submit personal information to it. Cerebras is a one-time $5/30-day credit, not a free tier; GitHub Models was retired 2026-07-30. |
| ⚙️ Portable fallback | **Ollama** → `qwen3:4b` (Q4). Kept because it is the brain option that is not controlled by Apple's eligibility list. A 32B model does not fit in 8 GB and the memory is soldered, so 3–4B is the ceiling, not a compromise step. |
| 🔎 Embeddings | `nomic-embed-text`, added only if BM25 alone comes up short |
| 💻 Code Agent | **Cline** in VS Code, read-only by default |
| ⚡ Automation | `launchd` + a morning-brief script |

### Voice

| Layer | Candidates | State |
|---|---|---|
| 🔊 TTS | **Kokoro-82M** (Apache-2.0, 327 MB weights, 4 Hindi voices) · **Piper** `hi_IN-priyamvada` (63.5 MB, **CC BY-NC-SA — noncommercial**) · Sarvam **Bulbul V3** (API) | Hindi has a real $0 local option now. **Marathi has none found.** |
| 🎙️ STT | **Vosk** small-hi (42 MB) · **whisper.cpp** tiny/base/small · Sarvam **Saaras V3** (API) | all unmeasured on this Mac; none proven on code-switched Hinglish |

**The honest caveat.** "100% local, zero vendor lock-in, runs on your GPU" was the original claim
and it was wrong on all three counts. Voice is now **deferred, not solved**: Lily ships text-only
until the speech layer is tested on the machine rather than on a benchmark table. What a $0 audit
changed, verified from the sources themselves: Kokoro-82M *does* list Hindi (`hf_alpha`, `hf_beta`,
`hm_omega`, `hm_psi`) — I had reported the opposite — but its own voices file grades all four **C**
on 1–10 hours of training data and warns non-English support "may be absent or thin due to weak
G2P", and **Marathi is not in the list at all**. Piper's Hindi voice is a 63.5 MB download but
noncommercially licensed, so "open source" needs an asterisk there too. Free and local Indic voice
is therefore *possible for Hindi, unproven for quality, and still missing for Marathi* — which is
closer to the truth than either the original README or my first pass at it.

### Cut, and why

| cut | reason |
|---|---|
| Qwen2.5-Coder **32B**, DeepSeek-R1 | 8 GB unified memory. A second resident model is a second load you cannot pay for. |
| **Unsloth** LoRA fine-tuning | CUDA-pinned; and under the design above, fine-tuning is solving the wrong problem. |
| **LangGraph** + **smolagents** together | one small model and four tools do not need two orchestration frameworks. Both were picked without a reason; that is the reason to drop both. |
| **ChromaDB** + **LlamaIndex** | at a few thousand notes, FTS5 does the job with zero added dependencies and nothing to keep warm in RAM. |
| **n8n** | a Node server idling at hundreds of MB of RAM on a machine that has none. |
| **Karpathy/autoresearch**, as written | upstream trains `nanochat` on a single NVIDIA GPU (tested on H100), pins CUDA wheels, and declines MPS/CPU. Its *method* survives; its *workload* does not run here. |

---

## How Lily Knows You

```
Your Git commits      →  She knows your code decisions
Your Obsidian notes   →  She knows your thinking patterns
Your voice thoughts   →  She knows how you actually talk
Your daily logs       →  She knows what you're building today
Your review comments  →  She fixes what she got wrong yesterday
```

The more you live in your vault and speak your thoughts — the more she becomes an extension of
*you*.

---

## Getting Started

Nothing here is published yet beyond this README and the design. The harness that does run today
lives in a *personal* Qoder skill (`~/.qoder/skills/lily-autoresearch/`) rather than in this
repo, because a project-level `.qoder/skills/` directory here would be pushed to GitHub along
with the loop's design. Vendoring the generic, personal-data-free parts into `scripts/` is an
open decision, not an oversight.

```bash
SKILL=~/.qoder/skills/lily-autoresearch

git clone https://github.com/mr-vishwet/lily_the_saviour.git
cd lily_the_saviour

# Tier 0 only: stdlib Python, no downloads
python3 "$SKILL/scripts/vault_retrieval.py" \
  --vault "$VAULT_PATH" --eval "$SKILL/evals/eval-set.example.jsonl" \
  --config "$SKILL/evals/baseline.config.json"
```

Then the overnight loop, always with a cap and a deadline:

```bash
"$SKILL/scripts/loop.sh" --project-dir . --vault "$VAULT_PATH" \
  --eval "$SKILL/evals/eval-set.example.jsonl" \
  --count 40 --deadline 07:30 --confirm-overnight
```

## How the claims above are measured

Seven scripts in the repo root, all reading the same eval set (JSONL rows of
`question` / `gold_note` / `must_contain`, hand-written and human-read first):

| script | what it answers | what it costs |
|---|---|---|
| `lily_ask.py` | one grounded answer: retrieve → inject (`chunk` \| `note` \| `win`) → ask the on-device brain | one model call |
| `lily_coverage.py` | **evidence coverage**: does every `must_contain` string actually land inside the injected text? | deterministic, brain-free, ~0.04 s per configuration |
| `lily_eval.py` | answer-layer rates: HIT / HALLUCINATION / DEGRADE, plus a health gate that aborts a poisoned session | many model calls |
| `lily_halluc_audit.py` | of the answers `lily_eval.py` called hallucinations, which were actually **wrong**, and which were correct text the gold string failed to match? | deterministic, brain-free |
| `lily_ceiling.py` | is a failure the *brain* or the *wiring*? the same rows asked of a different model | many calls, one backend |
| `lily_guardrail_probe.py` | which bytes of a note make the brain refuse, found by bisecting prefixes — and whether our own instruction sentence is what triggers it (`--variants`) | ~15 calls, one note |
| `lily_recon.py` | what the brain binding will and will not do, by reading it (recon → static → dynamic) | no model calls until you ask |

The order matters and is the lesson of this project so far: **measure coverage before spending
a single model call.** A geometry question that once looked like hours of paired inference
resolved in five minutes of arithmetic, and the answer it gives is immune to the session-health
drift that made an earlier round of model-call results unreadable. Coverage is a necessary
condition, never the objective — it can say a configuration *can* see the evidence, and only
the paired, health-gated run can say whether the agent then uses it.

The second lesson, learned the hard way the same day: **a rate you cannot decompose is a rate
that will mislead you.** `lily_eval.py` reported the new injection arm as hallucinating more, and
for a full review cycle that was the one cost of an otherwise clean win. Auditing the flagged
answers against the notes they came from showed most of them were *correct quotes* whose literal
gold string happened not to match — and on the real-confabulation measure the arm was slightly
better, not worse. Strict literal scoring rewards padding and punishes a terse right answer, so
run `lily_halluc_audit.py` before acting on any HALLUCINATION number, especially a safety one.

Before any of that: write `~/.vault/agent-profile/my_profile.md` honestly,
and build the eval set from your own notes and **read it**. An eval set you have not read is a
metric you should not trust.

---

## The Name

**Lily — The Saviour.**

Because at 3am when you're stuck on a bug, when you've forgotten why you made an architectural
decision 3 months ago, when you just need someone who *gets* how you think —

She's there.

And she remembers.

---

## Philosophy

Most AI assistants are built for everyone.
Lily is built for one person.

She doesn't need to know about climate change or the French Revolution.
She needs to know that you always reach for `useCallback` too early,
that "ek minute" means you're about to go deep,
and that your best ideas come out in voice notes at midnight.

> *Generic AI makes you work for it.*
> *Lily works for you.*

---

## License

No `LICENSE` file exists in this repository yet, so treat it as **all rights reserved** until one
is added. An earlier version of this README carried an MIT badge that nothing backed it.

---

<div align="center">

**Built with obsession. Runs on 8 GB of RAM. Speaks Hinglish.**

*For one developer. In Pune. Who needed a saviour.*

</div>
