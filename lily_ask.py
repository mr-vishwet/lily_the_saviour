#!/usr/bin/env python3
"""lily_ask.py -- the smallest end-to-end grounded answer path for Lily.

Tier-0 retrieval (the *exact* BM25 scorer the autoresearch metric certifies) feeds the
on-device Apple Foundation Models brain (ADR-0003). Nothing here downloads a model or
spends rupees: retrieval is stdlib, the brain is served by macOS 26.

The design rule that matters: **do not trust the 3B model -- show your work.** Every
answer is printed with the cited source chunk, so a silent hallucination or a wrong
"not in my notes" is visible to the human, not laundered into confident prose.

Safety gates, all real (from ADR-0003):
  1. check_availability() == 1 before any call; else degrade.
  2. catch FoundationModelsError subclasses (-10 guardrail, -17 refusal, -20 language).
  3. detect wrong-language output (Devanagari question answered in ASCII) -- raises nothing.
  4. detect a refusal claim while retrieval was non-empty -- the silent-wrong-refusal class.
  5. degrade-to-retrieval is the DEFAULT fallback. Cloud escalation is OFF unless
     LILY_ALLOW_CLOUD=1 *and* a Groq key is present *and* the owner has enabled ZDR.
     Lily stops visibly, never silently, and never sends notes to a remote by accident.

Usage:
  python3 lily_ask.py "Which Room database does Kanan use?"
  python3 lily_ask.py --top 4 --json "..."
Stdlib + the optional `applefoundationmodels` binding already installed in ./.venv.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(HERE, ".envs", "lily.env")
BEST_CONFIG = os.path.join(HERE, ".docs", "autoresearch", "best_config.json")
PERSONA = os.path.join(HERE, ".docs", "agentic", "lily-persona.md")
SKILL = os.path.expanduser("~/.qoder/skills/lily-autoresearch/scripts/vault_retrieval.py")

# Grounding rules are lifted from the persona so the two never silently diverge. The
# persona file is on the do-not-improve list; we READ it, we never rewrite it here.
BASE_INSTRUCTIONS = (
    "You are Lily, a personal engineering agent over one person's Obsidian vault.\n"
    "Grounding rules (hard):\n"
    "1. Every claim about the user's own projects MUST come from vault_search results; "
    "name the note you used.\n"
    "2. If search returns nothing relevant, say exactly: NOT IN MY NOTES. Never invent a fact.\n"
    "3. Reply in the same language the user wrote in; if they mix languages, mix back.\n"
)


def load_persona() -> str:
    """Pull the persona's identity + grounding + speech rules into the system prompt.
    Best-effort: if the file is missing, run on BASE_INSTRUCTIONS alone."""
    if not os.path.isfile(PERSONA):
        return BASE_INSTRUCTIONS
    body = open(PERSONA, encoding="utf-8").read()
    # Keep it cheap: the 3B brain has little context budget. Truncate, do not dump 48 lines.
    return BASE_INSTRUCTIONS + "\nPersona excerpt:\n" + body[:1200]


def load_env(path: str) -> dict:
    """Parse KEY=VAL without eval/exec -- this file holds the vault path; executing it
    is not acceptable. Strips inline comments and matching quotes."""
    out = {}
    if not os.path.isfile(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.split("#", 1)[0].strip()
        out[k.strip()] = v.strip('"').strip("'")
    return out


def load_scoring():
    """Import vault_retrieval.py by path so the answer uses the certified ranking."""
    spec = importlib.util.spec_from_file_location("vault_retrieval", SKILL)
    if spec is None or spec.loader is None:
        sys.exit(f"error: cannot load retrieval module at {SKILL}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_config(m) -> dict:
    cfg = dict(m.DEFAULT_CONFIG)
    if os.path.isfile(BEST_CONFIG):
        try:
            user = json.load(open(BEST_CONFIG, encoding="utf-8"))
            cfg.update({k: user[k] for k in cfg if k in user})
        except (OSError, ValueError) as exc:
            print(f"note: could not read best_config.json ({exc}); using defaults", file=sys.stderr)
    return cfg


DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def is_devanagari(text: str) -> bool:
    return bool(DEVANAGARI.search(text))


def looks_like_refusal(text: str) -> bool:
    t = text.lower()
    cues = ["not in my notes", "not mentioned", "i don't have", "i do not have",
            "no information", "not available in", "don't know", "does not say",
            "not in the notes", "unable to find", "unable to answer",
            "couldn't find", "could not find", "can't find", "cannot find",
            "no relevant", "no notes", "don't have any", "do not have any",
            "not contain", "didn't find", "not find any", "no data",
            "ask a follow", "please provide", "could you clarify", "not able to find",
            "there is no note", "no note with", "no note titled", "not find the note",
            "does not exist", "no such note", "cannot determine", "can't determine",
            "i have no note", "not in any note", "nothing about"]
    # A bare wikilink or a lone "[<note>.md]" echo is the model forwarding retrieved
    # context instead of answering -- a non-answer just like a refusal.
    stripped = t.strip()
    if re.fullmatch(r"\[\[?[^\]]*\]?\]", stripped):
        return True
    return any(c in t for c in cues)


# The AFM binding is inconsistent about guardrail failures: sometimes it RAISES
# FoundationModelsError, and sometimes it returns the error payload AS the generated
# text -- e.g. '{"error_code":-20,"error":"An unsupported language or locale was used"}'.
# The latter is the dangerous one: it is not an exception, so it sails through every other
# gate and gets served to the user as if it were a grounded answer. Detect it explicitly.
# Mirrors applefoundationmodels.ERROR_CODE_TO_EXCEPTION (verified 2026-10-07). We keep the
# names we have actually seen leak as text — including -6 GenerationError, which the confirm
# run surfaced 23x — so degrade reasons are self-describing instead of opaque `code-6`.
ERR_CODE_NAMES = {
    -1: "Initialization", -2: "NotAvailable", -3: "InvalidParameters", -4: "Memory",
    -5: "JSONParse", -6: "Generation", -7: "Timeout", -10: "GuardrailViolation",
    -14: "ContextWindowExceeded", -16: "RateLimited", -17: "Refusal",
    -18: "ConcurrentRequests", -20: "UnsupportedLanguage", -21: "AssetsUnavailable",
}


def embedded_error(text: str):
    """If the model's output is actually a leaked error payload, return a short reason
    string; otherwise None. Matches both the JSON form and the bare message."""
    if not text or '"error_code"' not in text:
        return None
    code = None
    try:
        code = json.loads(text).get("error_code")
    except Exception:
        import re as _re
        mnum = _re.search(r'"error_code"\s*:\s*(-?\d+)', text)
        code = int(mnum.group(1)) if mnum else None
    name = ERR_CODE_NAMES.get(code, f"code{code}")
    return f"brain-error-string:{name}"


def retrieve(m, cfg, vault: str, question: str, top: int):
    """Rank chunks with the metric's own BM25; return [(note, text, score)] best-per-note."""
    import hashlib
    knob_keys = ("chunk_words", "chunk_overlap", "title_boost", "path_boost", "tag_boost",
                 "fold_devanagari", "strip_stopwords")
    sig_src = {"q": question}
    sig_src.update({k: cfg[k] for k in knob_keys})
    sig = hashlib.sha256(json.dumps(sig_src, sort_keys=True).encode()).hexdigest()[:16]
    chunks = m.build_chunks(vault, cfg, sig)
    q = m.tokenize(m.normalize(question, cfg), cfg)
    scored = m.bm25(q, chunks, cfg)
    seen, out = set(), []
    for s, c in scored:
        if c["note"] in seen:
            continue
        seen.add(c["note"])
        out.append((c["note"], c["text"], round(s, 3)))
        if len(out) >= top:
            break
    return out


def anchor_offset(body: str, chunk_text: str) -> int:
    """Char offset in `body` where `chunk_text` begins, or -1.

    vault_retrieval.chunk_note() re-joins words with single spaces, so a stored chunk is a
    whitespace-normalised slice of the note and a literal body.find(chunk) fails for ~88% of
    injected notes (measured 2026-10-16 -- it silently turned my first match-anchored arm into
    head-truncation). Compare token sequences instead: 96/96 found.
    """
    ctoks = chunk_text.split()[:8]
    if not ctoks:
        return -1
    toks = list(re.finditer(r"\S+", body))
    for i, t in enumerate(toks):
        if t.group(0) != ctoks[0] or i + len(ctoks) > len(toks):
            continue
        if all(toks[i + j].group(0) == ctoks[j] for j in range(1, len(ctoks))):
            return t.start()
    return -1


def snap_to_word_boundary(seg: str) -> str:
    """Cut a trailing partial word off a capped slice.

    `cap` truncates by characters, so the last token in the injected context is normally a word
    chopped in half. The hypothesis was that this triggered -20 UnsupportedLanguageError, which
    appeared at *some* caps and not others -- see
    .docs/notes/autoresearch-review/2026-10-16-context-trim.md.

    MEASURED AND NULL (2026-10-16, three-arm paired run): free on coverage (0 of 105 pairs
    change) and no effect at the brain (-20 was 6 / 6 / 6 with and without it). Kept because it
    is harmless and lets anyone re-test the idea cheaply, not because it is expected to work.
    """
    j = max(seg.rfind(" "), seg.rfind("\t"), seg.rfind("\n"))
    return seg[:j] if j > 0 else seg


def inject_context(m, cfg, hits, mode: str, cap: int = 2000, snap: bool = False) -> str:
    """Build the prompt context from retrieval hits.

    mode='chunk': the best-scoring window per note (the legacy build).
    mode='note' : the whole note, head-truncated to `cap` chars (shipped default 2026-10-07
                  to 2026-10-16).
    mode='win'  : `cap` chars ANCHORED ON the chunk that won the note, so BM25's own match
                  decides where the window sits. Shipped default since 2026-10-16.

    Coverage of the gold `must_contain` evidence, measured by lily_coverage.py (deterministic,
    brain-free, chunking frozen at the champion cw=128/ov=0):

      chunk             18/32 at the shipped windowing (29/32 for any chunking, ever)
      note  top3 cap3000 31/32  mean 6978  worst-case 9049
      note  top3 cap1200 31/32  mean 3639  worst-case 3669
      note  top3 cap6000 32/32  mean 7259  worst-case 12072  <- past the measured cliff
      win   top3 cap2000 32/32  mean 5756  worst-case 6069   <- beats note on BOTH axes
      win   top3 cap2000 +snap 32/32  mean 5741  worst 6046  mid-word cuts 29 -> 0

    Coverage only prices the context, not the answer, so `win` also had to win the brain. The
    paired, health-gated, same-session A/B was run TWICE (192 calls, then a 288-call
    replication with a third arm; .docs/autoresearch/answer_eval_paired_note_vs_win.json and
    answer_eval_replicate_snap.json, temp 0.3, 3 draws x 32 rows). Pooled over 192 draws per
    arm, note:3:3000 -> win:3:2000: HIT 45.31% -> 54.69% (McNemar p=0.0247), DEGRADE 34.38% ->
    21.35% (p=0.0003), rows never hit 12 -> 9. The graded HALLUCINATION line looked worse
    (20.31% -> 23.96%) and that reading was WRONG: re-grading every flagged answer against the
    gold note's own text shows most of win's extra flags are correct note prose that the literal
    `must_contain` never matched. On the decomposed metric win is better, not worse -- 11 of 192
    draws invent something absent from the note vs 9 of 192 for note, and the same 3 rows per
    arm in both runs (p=1.0). See the 2026-10-16 note's Result 2 for the whole correction.

    The win row answers the question the 2026-10-07 note left open: the fix for deep evidence
    was never smaller chunks, it was anchoring the window at the match.

    `snap` trims a window back to its last whitespace so the injected text never ends mid-word.
    Tested and NULL: it costs 0 of 105 coverage pairs AND changes nothing at the brain -- in the
    288-call run the `-20` count was 6 / 6 / 6 across note, win and win+snap, hit p=1.0, degrade
    p=0.55. `chunk` mode ignores both `snap` and `cap` -- the window is already the unit being
    injected.
    """
    if mode == "chunk":
        return "\n\n".join(f"[{os.path.basename(n)}] {t}" for n, t, _ in hits)
    if mode in ("note", "win"):
        parts = []
        for n, text, _ in hits:
            try:
                body = open(os.path.expanduser(n), encoding="utf-8").read()
            except OSError:
                body = ""
            if mode == "win" and len(body) > cap:
                pos = max(0, anchor_offset(body, text))
                body = body[max(0, min(pos - cap // 3, len(body) - cap)):][:cap]
                if snap:
                    body = snap_to_word_boundary(body)
            else:
                body = body[:cap]
                if snap:
                    body = snap_to_word_boundary(body)
            parts.append(f"[{os.path.basename(n)}] {body}")
        return "\n\n".join(parts)
    raise ValueError(f"unknown inject mode: {mode}")


DEFAULT_INSTRUCTIONS = (
    "Answer only from the notes provided. Quote the note name you used. "
    "If the answer is not in the notes, say exactly: NOT IN MY NOTES. "
    "Reply in the same language as the question.")


def ask_brain(question: str, context: str, temperature: float | None = None,
              instructions: str | None = None) -> dict:
    """Call on-device AFM. Returns {ok, answer, reason}. Never raises to the caller.

    `temperature` is None by default = the binding's own DEFAULT_TEMPERATURE, which the
    2026-10-16 recon measured as 1.0 (TemperaturePreset.CREATIVE). lily_recon.py's dynamic
    probe showed the knob is honoured (4 draws of a one-word answer: 3 distinct surface
    forms at 1.0, 2 at 0.3, 1 at 0.1), so setting it is the cheapest way to cut eval
    variance. Left unset here on purpose: changing it changes shipped behaviour.

    `instructions` None = `DEFAULT_INSTRUCTIONS`, byte for byte, so shipped behaviour is
    unchanged. The override exists because `error_codes.json` defines -20 as "the model was
    PROMPTED TO RESPOND in a language it does not support": the binding exposes no locale API
    at all, so Apple must infer the requested response language from the prompt text -- and our
    own last sentence asks for exactly that inference. This is the only way to test whether the
    -20 wall is ours rather than Apple's.
    """
    try:
        import applefoundationmodels as afm
    except ImportError:
        return {"ok": False, "reason": "binding-not-installed", "answer": None}
    try:
        code = afm.Session.check_availability()
        if code != 1:
            names = {-1: "DEVICE_NOT_ELIGIBLE", -2: "NOT_ENABLED", -3: "MODEL_NOT_READY",
                     -99: "AVAILABILITY_UNKNOWN"}
            return {"ok": False, "reason": f"brain-unavailable:{names.get(code, code)}",
                    "answer": None}
    except Exception as exc:  # availability probe itself failed -> degrade, do not crash
        return {"ok": False, "reason": f"availability-probe-failed:{exc}", "answer": None}

    instr = DEFAULT_INSTRUCTIONS if instructions is None else instructions
    prompt = f"Notes:\n{context}\n\nQuestion: {question}"
    try:
        s = afm.Session(instructions=instr)
        r = s.generate(prompt, max_tokens=300, temperature=temperature)
        s.close()
        text = (r.content or "").strip()
    except afm.FoundationModelsError as exc:  # -10/-17/-20 and friends
        return {"ok": False, "reason": f"brain-error:{type(exc).__name__}", "answer": None}
    except Exception as exc:
        return {"ok": False, "reason": f"brain-unknown:{exc}", "answer": None}

    # Silent-failure detection (ADR-0003 consequence 5): these raise nothing.
    leaked = embedded_error(text)
    if leaked:
        return {"ok": False, "reason": leaked, "answer": None}
    if is_devanagari(question) and not is_devanagari(text):
        return {"ok": False, "reason": "wrong-language-output", "answer": text}
    return {"ok": True, "reason": "ok", "answer": text}


def build_search_tool(m, cfg, vault: str, top: int):
    """Return (fn, log). `fn` is the read-only vault_search tool the model may call;
    `log` accumulates every note it actually served, so citations stay honest even when
    the model paraphrases. The model chooses the query -- we no longer assume we knew
    which chunks to inject."""
    log: list = []

    def vault_search(query: str) -> str:
        """Search the Obsidian vault and return the top matching note excerpts."""
        hits = retrieve(m, cfg, vault, query, top)
        for n, t, s in hits:
            if not any(n == seen for seen, _, _ in log):
                log.append((n, t, s))
        if not hits:
            return "NO RESULTS"
        return "\n\n".join(f"[{os.path.basename(n)}] {t}" for n, t, _ in hits)

    return vault_search, log


def _brain_gate():
    """Import the binding and check availability. Returns (afm, None) or (None, reason)."""
    try:
        import applefoundationmodels as afm
    except ImportError:
        return None, "binding-not-installed"
    try:
        code = afm.Session.check_availability()
    except Exception as exc:
        return None, f"availability-probe-failed:{exc}"
    if code != 1:
        names = {-1: "DEVICE_NOT_ELIGIBLE", -2: "NOT_ENABLED", -3: "MODEL_NOT_READY",
                 -99: "AVAILABILITY_UNKNOWN"}
        return None, f"brain-unavailable:{names.get(code, code)}"
    return afm, None


def ask_brain_agent(question: str, m, cfg, vault: str, top: int) -> dict:
    """Tool-calling loop: AFM decides when to search the vault and synthesises the answer.
    Returns {ok, reason, answer, citations}. The tool loop runs inside the binding."""
    afm, gate = _brain_gate()
    if afm is None:
        return {"ok": False, "reason": gate, "answer": None, "citations": []}
    tool, log = build_search_tool(m, cfg, vault, top)
    try:
        s = afm.Session(instructions=load_persona(), tools=[tool])
        r = s.generate(question, max_tokens=400)
        s.close()
        text = (r.content or "").strip()
    except afm.FoundationModelsError as exc:
        return {"ok": False, "reason": f"brain-error:{type(exc).__name__}",
                "answer": None, "citations": [n for n, _, _ in log]}
    except Exception as exc:
        return {"ok": False, "reason": f"brain-unknown:{exc}",
                "answer": None, "citations": [n for n, _, _ in log]}
    if is_devanagari(question) and not is_devanagari(text):
        return {"ok": False, "reason": "wrong-language-output", "answer": text,
                "citations": [n for n, _, _ in log]}
    return {"ok": True, "reason": "ok", "answer": text,
            "citations": [n for n, _, _ in log]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--top", type=int, default=3, help="source notes to ground on")
    ap.add_argument("--inject", choices=("chunk", "note", "win"), default="win",
                    help="context build: `cap` chars ANCHORED ON the BM25-matched chunk "
                         "(shipped default since 2026-10-16). Won two paired, health-gated, "
                         "same-session A/Bs (192 + 288 calls, temp 0.3): pooled over 192 draws "
                         "per arm, win:3:2000 vs note:3:3000 HIT 54.69%% vs 45.31%% (McNemar "
                         "p=0.0247) and DEGRADE 21.35%% vs 34.38%% (p=0.0003). 'note' is the "
                         "previous default — whole note head-truncated to `cap`; it beat 'chunk' "
                         "34.4%% vs 22.9%% HIT on 2026-10-07 because a 128-word window cut the "
                         "gold evidence in 14/32 rows. 'win' fixes the residual 1/32 by moving "
                         "the window to the match instead of the top of the file. See "
                         ".docs/notes/autoresearch-review/2026-10-16-context-trim.md")
    ap.add_argument("--cap", type=int, default=2000,
                    help="chars of note/win text injected per hit. The binding's context cliff is "
                         "a PER-CALL WORST CASE, not a mean: lily_ceiling.py saw "
                         "ContextWindowExceededError 34/96 at ~11-12k chars. 2000 keeps win's "
                         "worst case at 6069 (note:3:3000 was 9049) while covering 32/32 gold "
                         "rows.")
    ap.add_argument("--snap-boundary", action="store_true",
                    help="cut the capped window back to the last word boundary instead of "
                         "mid-token. TESTED AND NULL -- do not expect it to help. It is free on "
                         "coverage (0 of 105 arm pairs change, 1871 mid-word cuts -> 0) and does "
                         "nothing to the guardrail: in the 288-call three-arm paired run the -20 "
                         "UnsupportedLanguageError count was 6 / 6 / 6 for note, win and win+snap, "
                         "with hit p=1.0 and degrade p=0.55. -20 keys on the note's TEXT, not on "
                         "where the cut lands -- error_codes.json defines it as a RESPONSE-"
                         "LANGUAGE gate. Ignored by 'chunk', which never caps.")
    ap.add_argument("--agent", action="store_true",
                    help="let the brain call vault_search as a tool instead of pre-injecting context")
    ap.add_argument("--temperature", type=float, default=None,
                    help="AFM sampling temperature; unset = the binding's own default of 1.0. "
                         "lily_recon.py measured the knob honoured: 4 draws of a one-word answer "
                         "gave 3 distinct forms at 1.0, 2 at 0.3, 1 at 0.1 -- so 0.1-0.3 is the "
                         "cheapest way to cut eval variance (2026-10-16).")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    env = load_env(ENV_FILE)
    vault = os.path.expanduser(env.get("VAULT_PATH", ""))
    if not vault or not os.path.isdir(vault):
        print(json.dumps({"ok": False, "reason": "vault-path-invalid", "vault": vault}))
        return 4

    m = load_scoring()
    cfg = load_config(m)
    # inject hits: used directly by the one-shot path, and as the citation fallback + human
    # "Sources:" list in agent mode if the model never fired the tool.
    hits = retrieve(m, cfg, vault, args.question, args.top)

    if args.agent:
        brain = ask_brain_agent(args.question, m, cfg, vault, args.top)
        citations = brain["citations"] or [n for n, _, _ in hits]
        mode = "agent"
    else:
        citations = [n for n, _, _ in hits]
        mode = "inject"
        if not hits:
            out = {"question": args.question, "brain": "skipped", "ok": False,
                   "reason": "empty-retrieval", "answer": None, "retrieval": [], "citations": []}
            print(json.dumps(out, ensure_ascii=False, indent=2) if args.json
                  else "No matching notes. (retrieval found nothing)")
            return 0
        context = inject_context(m, cfg, hits, args.inject, cap=args.cap,
                                 snap=args.snap_boundary)
        brain = ask_brain(args.question, context, temperature=args.temperature)

    refused = brain["ok"] and looks_like_refusal(brain["answer"] or "")
    result = {
        "question": args.question,
        "mode": mode,
        "brain": "apple-foundation-models" if brain["ok"] else "degraded",
        "ok": brain["ok"] and not refused,
        "reason": "wrong-refusal-while-context-present" if refused else brain["reason"],
        "answer": brain["answer"],
        "retrieval": [{"note": os.path.relpath(n, vault), "score": s, "snippet": t[:200]}
                       for n, t, s in hits],
        "citations": citations,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    # Default human view: the answer, then ALWAYS the evidence, so nothing is taken on faith.
    if result["ok"]:
        print(result["answer"])
    else:
        print(f"(Lily did not answer confidently -- reason: {result['reason']}. "
              f"Falling back to raw retrieved notes, no cloud call.)")
    print("\nSources:")
    shown = citations or [n for n, _, _ in hits]
    scores = {n: s for n, _, s in hits}
    for n in shown:
        print(f"  - {os.path.relpath(n, vault)}" + (f"  (score {scores[n]})" if n in scores else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
