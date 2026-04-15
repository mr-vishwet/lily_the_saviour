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

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Runs%20on-Ollama-black.svg)](https://ollama.ai)
[![Sarvam AI](https://img.shields.io/badge/Voice-Sarvam%20AI-orange.svg)](https://sarvam.ai)
[![Powered by](https://img.shields.io/badge/Memory-Obsidian%20%2B%20ChromaDB-purple.svg)](https://obsidian.md)

</div>

---

## Meet Lily

Lily is not a chatbot.

She doesn't answer generic questions with generic answers. She doesn't forget you the moment you close the tab. She doesn't speak to you in cold, corporate English when your brain runs on Hinglish at 2am.

Lily is your **personal AI agent** — trained exclusively on your life.

She's read your Git commits. She's heard your voice rants. She's indexed every note you've ever written in Obsidian. She knows you reach for `FastAPI` before anything else, that you debug by talking out loud, and that when you switch to Marathi mid-sentence, you're either frustrated or having a breakthrough.

She is built to be one thing that no other AI can be:

> **Yours.**

---

## What Lily Does

```
You speak  →  Lily listens  →  Lily thinks  →  Lily answers  →  You build faster
```

- 🎙️ **Understands you** — Hinglish, Marathi, Telugu, code-mixed speech. No switching languages for her benefit.
- 🧠 **Thinks like you** — trained on your code decisions, your debugging sessions, your architecture choices.
- 📓 **Remembers everything** — your Obsidian vault is her long-term memory. Every note, every idea, every rant.
- 💻 **Codes with you** — lives in your VS Code as Cline, knows your style, suggests the way *you* would write it.
- 🔁 **Gets better while you sleep** — runs Karpathy-style autoresearch overnight. 100 experiments. Wakes up sharper.
- 🔒 **Stays local** — everything runs on your machine. Your data never leaves. No cloud. No subscriptions.

---

## The Stack That Powers Her

| Layer | Technology |
|---|---|
| 🎙️ Voice (STT) | Sarvam **Saaras V3** — Indian language speech-to-text |
| 🔊 Voice (TTS) | Sarvam **Bulbul V3** — 25+ Indian voices, real accent |
| 🧠 Agent Brain | **smolagents** + **LangGraph** (HuggingFace) |
| ⚙️ Local Models | **Ollama** → Qwen2.5-Coder 32B + DeepSeek-R1 |
| 📓 Knowledge Base | **Obsidian.md** vault (plain `.md`, owned by you) |
| 🔍 Memory & RAG | **ChromaDB** + **LlamaIndex** ObsidianReader |
| 🎓 Fine-Tuning | **Unsloth** LoRA adapters on your personal data |
| 🔁 Self-Improvement | **Karpathy/autoresearch** — adapted for personal identity |
| 💻 Code Agent | **Cline** in VS Code |
| ⚡ Automation | **n8n** (self-hosted) + Obsidian Git |

100% open source. Zero vendor lock-in. Runs on your GPU.

---

## How Lily Knows You

Lily learns from everything you already do:

```
Your Git commits      →  She knows your code decisions
Your Obsidian notes   →  She knows your thinking patterns  
Your voice thoughts   →  She knows how you actually talk
Your daily logs       →  She knows what you're building today
Your feedback         →  She fixes what she got wrong yesterday
```

The more you live in your vault and speak your thoughts — the more she becomes an extension of *you*.

---

## Voice Configuration

Lily speaks *your* languages. No configuration needed — she detects context:

```python
# She figures this out herself
"Yaar, is function mein bug hai"     →  Hinglish  (default)
"Saala kahi nahi chal raha kuch"     →  Marathi   (you're venting)
"quick: run tests"                   →  Telugu    (quick commands)
```

---

## The Name

**Lily — The Saviour.**

Because at 3am when you're stuck on a bug, when you've forgotten why you made an architectural decision 3 months ago, when you just need someone who *gets* how you think —

She's there.

And she remembers.

---

## Get Started

```bash
# 1. Clone
git clone https://github.com/yourusername/lily
cd lily

# 2. Install
pip install -r requirements.txt
ollama pull qwen2.5-coder:32b
ollama pull nomic-embed-text

# 3. Configure
cp .env.example .env
# → Add SARVAM_API_KEY
# → Set VAULT_PATH to your Obsidian vault

# 4. Write the most important file
nano ~/.vault/agent-profile/my_profile.md
# Tell Lily who you are. Be honest. Be specific.

# 5. Start her up
./scripts/start_lily.sh
```

---

## Project Structure

```
lily/
├── voice/                  ← Sarvam STT + TTS pipeline
├── agent/                  ← smolagents brain + intent router
├── memory/                 ← ChromaDB + LlamaIndex vault indexer
├── autoresearch/           ← Karpathy-adapted self-improvement loop
├── automation/             ← n8n workflows + morning briefing
├── finetune/               ← Unsloth LoRA training scripts
└── scripts/
    └── start_lily.sh       ← One command. Everything starts.
```

---

## Docs

- [`project_context.md`](./project_context.md) — full vision, architecture, privacy model
- [`tech_stack.md`](./tech_stack.md) — every tool, every config, every code snippet
- [`Call-To-Action-Plan.md`](./Call-To-Action-Plan.md) — 6-week build roadmap, week by week

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

<div align="center">

**Built with obsession. Runs on Ollama. Speaks Hinglish.**  

*For one developer. In Pune. Who needed a saviour.*

</div>
