#!/usr/bin/env python3
"""lily_coverage.py -- deterministic, brain-free search over the INJECTION config space.

WHY THIS EXISTS (2026-10-07). The paired A/B showed the answer layer's real constraint was
evidence coverage, not model quality: the chunk build sent the gold evidence in 18/32 rows,
whole-note in 32/32. Coverage is computable WITHOUT calling AFM, so unlike the answer layer
(it costs ~70 min for one paired run and can be poisoned by a sick session) this metric is
instant, deterministic, and reproducible. It answers the open question the merge left:

    can a SURGICAL chunk config reach full coverage at a smaller context than whole-note,
    staying under the measured AFM context cliff?

OBJECTIVE (constrained, not "maximize"): maximise coverage@top subject to mean injected
context staying under --max-ctx (default 6000 chars, the ~6k measured safe zone; top-5
whole-notes blew the window with ContextWindowExceededError x34/96).

Hard rules inherited from lily-autoresearch: reads the eval set and vault, writes only its
own results file; never touches persona, never edits vault_retrieval.py's scorer.
"""
from __future__ import annotations
import argparse, importlib.util, itertools, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "evals", "lily.handwritten.jsonl")


def load_scoring():
    skill = os.path.expanduser("~/.qoder/skills/lily-autoresearch/scripts/vault_retrieval.py")
    spec = importlib.util.spec_from_file_location("vault_retrieval", skill)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_ask():
    spec = importlib.util.spec_from_file_location("lily_ask", os.path.join(HERE, "lily_ask.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rank_per_note(m, cfg, chunks, q_tokens):
    """BM25 then best-chunk-per-note, exactly as lily_ask.retrieve() does."""
    scored = m.bm25(q_tokens, chunks, cfg)
    seen, out = set(), []
    for s, c in scored:
        if c["note"] in seen:
            continue
        seen.add(c["note"])
        out.append((c["note"], c["text"]))
    return out


NOTE_BODY: dict[str, str] = {}


def note_body(path: str) -> str:
    if path not in NOTE_BODY:
        try:
            NOTE_BODY[path] = open(os.path.expanduser(path), encoding="utf-8").read()
        except OSError:
            NOTE_BODY[path] = ""
    return NOTE_BODY[path]


def anchor_offset(body: str, chunk_text: str) -> int:
    """Char offset in `body` where this chunk starts.

    chunk_note() re-joins words with single spaces, so the stored chunk text is
    whitespace-normalised and a literal body.find(chunk) misses ~88% of the time (measured
    2026-10-16, and the reason my first `win` run was a silent no-op). Match on the token
    sequence instead.
    """
    ctoks = chunk_text.split()[:8]
    if not ctoks:
        return -1
    toks = [m for m in re.finditer(r"\S+", body)]
    for i, t in enumerate(toks):
        if t.group(0) != ctoks[0] or i + len(ctoks) > len(toks):
            continue
        if all(toks[i + j].group(0) == ctoks[j] for j in range(1, len(ctoks))):
            return t.start()
    return -1


def midword_cut(body: str, start: int, stop: int) -> bool:
    """True when the slice [start:stop] is truncated AND the cut lands inside a word.

    Motivated by the 2026-10-16 guardrail hunt: `Data Layer.md` cut at 77 chars ("… main
    Retrofit interf") raised UnsupportedLanguageError on 2/2 draws while the same note cut at
    62 answered 2/2 -- so the arbitrary byte boundary, not the content, was the trigger.
    UnsupportedLanguage is what the binding surfaces when AFM's on-device language ID rejects
    the prompt, and a dangling word fragment is exactly what makes English look not-English.
    Every cap-based truncation here cuts at an arbitrary char, so this is a property of the
    ARM, computable without calling the brain.
    """
    if stop >= len(body):
        return False
    a, b = body[stop - 1], body[stop]
    return a.isalnum() and b.isalnum()


def build_ctx(ask, m, cfg, hits, mode, cap, snap=False):
    """Returns (ctx, mid_word_cuts).

    note/win delegate to `lily_ask.inject_context` rather than reimplementing it: the metric
    must be computed over the text the SHIPPED path would inject, or the grid optimises a
    build nobody runs. (chunk keeps a local build because ask's chunk mode ignores `cap`, and
    the cap sweep is the point of this harness.)

    snap=True asks for the word-boundary variant of the same arm, so the coverage PRICE of
    snapping is measured in the same units as the coverage it buys -- snapping is a guardrail
    hypothesis, and a hypothesis that costs rows has to show that cost before the AFM calls
    are spent on it.
    """
    if mode == "chunk":
        segs = []
        for _, t in hits:
            seg = t[:cap]
            if snap:
                seg = ask.snap_to_word_boundary(seg)
            segs.append(seg)
        return "\n\n".join(f"[{os.path.basename(n)}] {s}"
                           for (n, _), s in zip(hits, segs)), (0 if snap else
                          sum(1 for _, t in hits if midword_cut(t, 0, cap)))
    ctx = ask.inject_context(m, cfg, [(n, t, 0.0) for n, t in hits], mode, cap=cap, snap=snap)
    cuts = 0
    for n, t in hits:
        body = note_body(n)
        if mode == "win" and len(body) > cap:
            pos = max(0, anchor_offset(body, t))
            start = max(0, min(pos - cap // 3, len(body) - cap))
            seg = body[start:][:cap]
        else:
            start = 0
            seg = body[:cap]
        if snap:
            # count on the text actually injected, not on the pre-snap slice
            seg = ask.snap_to_word_boundary(seg)
        cuts += midword_cut(body, start, start + len(seg))
    return ctx, cuts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-words", default="32,64,128,256,512,1024")
    ap.add_argument("--overlap", default="0,8,32,64,128")
    ap.add_argument("--tops", default="1,2,3,4,5")
    ap.add_argument("--caps", default="500,1000,2000,3000,4500,8000")
    ap.add_argument("--modes", default="chunk,note,win")
    ap.add_argument("--eval", dest="evalset", default=EVAL,
                    help="JSONL rows with must_contain; must be the same set the loop scores on, "
                         "otherwise the sweep optimises coverage of questions nobody asks.")
    ap.add_argument("--snap", default="off", choices=["off", "on", "both"],
                    help="score the word-boundary-snapped cut too (lily_ask.snap_to_word_boundary). "
                         "Default off keeps the historical grid; `both` costs ~2x the arms and buys "
                         "the one number the -20 hypothesis is missing: how many rows of coverage "
                         "snapping actually gives back.")
    ap.add_argument("--max-ctx", type=int, default=8500,
                    help="feasibility budget (chars). Default 8500 = the largest arm known to "
                         "complete on AFM (top-3 whole-note, cap 3000, 0 ContextWindowExceeded "
                         "in the 96-draw paired run). top-5 whole-note (~11-12k) blew 34/96.")
    ap.add_argument("--budgets", default="6000,8500",
                    help="budgets to report frontiers against. 6000 = the conservative measured "
                         "safe zone; 8500 = top-3 whole-note (cap 3000), which completed the "
                         "paired A/B with no ContextWindowExceeded. top-5 (~11-12k) blew 34/96.")
    ap.add_argument("--base-config", default=os.path.join(HERE, ".docs/autoresearch/best_config.json"))
    ap.add_argument("--out", default=os.path.join(HERE, ".tmp/coverage_search.tsv"),
                    help="default is scratch (.tmp/); pass an explicit .docs/ path to bank it")
    ap.add_argument("--report", default=os.path.join(HERE, ".tmp/coverage_search.json"))
    args = ap.parse_args()

    ask = load_ask()
    env = ask.load_env(ask.ENV_FILE)
    vault = os.path.expanduser(env.get("VAULT_PATH", ""))
    if not vault or not os.path.isdir(vault):
        print(f"error: vault path invalid: {vault}", file=sys.stderr)
        return 4

    m = load_scoring()
    rows = [json.loads(l) for l in open(args.evalset, encoding="utf-8") if l.strip()]
    base = json.load(open(args.base_config))
    # ranker boosts stay at the measured champion; only the *injection geometry* is searched.
    frozen = {k: base[k] for k in ("title_boost", "path_boost", "tag_boost", "k1", "b",
                                   "freshness_weight", "half_life_days", "fold_devanagari",
                                   "strip_stopwords") if k in base}
    qtoks = []
    for r in rows:
        cfg_probe = {**frozen, "chunk_words": 128, "chunk_overlap": 0}
        qtoks.append(m.tokenize(m.normalize(r["question"], cfg_probe), cfg_probe))

    cw = [int(x) for x in args.chunk_words.split(",")]
    ov = [int(x) for x in args.overlap.split(",")]
    tops = [int(x) for x in args.tops.split(",")]
    caps = [int(x) for x in args.caps.split(",")]
    modes = args.modes.split(",")
    snaps = {"off": [False], "on": [True], "both": [False, True]}[args.snap]
    # Rows with gold evidence. `expect_absent` rows are excluded from the coverage denominator:
    # all() over an empty must_contain is True, so including them would give every arm a free
    # covered row and inflate the one metric that is supposed to be honest.
    n_answer = sum(1 for r in rows if r.get("must_contain"))

    fh = open(args.out, "a", encoding="utf-8")
    if fh.tell() == 0:
        fh.write("mode\tchunk_words\tchunk_overlap\ttop\tcap\tcoverage\tmean_ctx\tmax_ctx\tfeasible\tmidword_cuts\tsnap\n")
    results = []
    t0 = time.time()
    n_rankings = 0
    for chunk_words, chunk_overlap in itertools.product(cw, ov):
        cfg = {**frozen, "chunk_words": chunk_words, "chunk_overlap": chunk_overlap}
        sig = f"cov-{chunk_words}-{chunk_overlap}"
        chunks = m.build_chunks(vault, cfg, sig)
        per_row = []
        for q, r in zip(qtoks, rows):
            per_row.append((rank_per_note(m, cfg, chunks, q), r))
        n_rankings += len(rows)
        for mode, top, cap, snap in itertools.product(modes, tops, caps, snaps):
            cov, tot_len, max_len, cuts = 0, 0, 0, 0
            for hits, r in per_row:
                sel = hits[:top]
                ctx, ncuts = build_ctx(ask, m, cfg, sel, mode, cap, snap=snap)
                cuts += ncuts
                tot_len += len(ctx)
                max_len = max(max_len, len(ctx))
                # An `expect_absent` row has no gold string, and `all()` over an empty list is
                # True -- so scoring it would hand every arm a free covered row and inflate
                # coverage by exactly the number of honesty rows added. Coverage is a measure of
                # *evidence reaching the prompt*; a question with no evidence has nothing to cover.
                if r.get("must_contain") and all(s.lower() in ctx.lower()
                                                 for s in r["must_contain"]):
                    cov += 1
            mean_ctx = round(tot_len / len(rows))
            feasible = mean_ctx <= args.max_ctx
            row = {"mode": mode, "chunk_words": chunk_words, "chunk_overlap": chunk_overlap,
                   "top": top, "cap": cap, "coverage": cov, "rows": n_answer,
                   "mean_ctx": mean_ctx, "max_ctx": max_len, "feasible": feasible,
                   "midword_cuts": cuts, "snap": snap}
            results.append(row)
            fh.write(f"{mode}\t{chunk_words}\t{chunk_overlap}\t{top}\t{cap}\t{cov}\t"
                     f"{mean_ctx}\t{max_len}\t{1 if feasible else 0}\t{cuts}\t{1 if snap else 0}\n")
        print(f"  [progress] chunk_words={chunk_words} overlap={chunk_overlap} "
              f"({time.time()-t0:.0f}s, {len(results)} arms scored)", file=sys.stderr, flush=True)
    fh.close()

    budgets = [int(x) for x in args.budgets.split(",")]
    per_budget = {}
    for b in budgets:
        ok = [r for r in results if r["mean_ctx"] <= b]
        bc = max((r["coverage"] for r in ok), default=-1)
        per_budget[str(b)] = {
            "arms": len(ok), "best_coverage": bc,
            "cheapest_arms_at_best_coverage": sorted(
                [r for r in ok if r["coverage"] == bc], key=lambda r: r["mean_ctx"])[:10],
        }
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "eval_rows": len(rows),
           "max_ctx_budget": args.max_ctx, "arms_scored": len(results),
           "seconds": round(time.time() - t0, 1), "rankings_done": n_rankings,
           "per_context_budget": per_budget,
           "best_feasible_coverage": max((r["coverage"] for r in results if r["feasible"]), default=-1),
           "full_coverage_arms_any_budget": sorted([r for r in results if r["coverage"] == n_answer],
                                                   key=lambda r: r["mean_ctx"])[:12]}
    json.dump(out, open(args.report, "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
