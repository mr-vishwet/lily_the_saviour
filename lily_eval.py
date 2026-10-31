#!/usr/bin/env python3
"""lily_eval.py -- the answer-layer metric we have never had.

Retrieval is graded (vault_retrieval.py, hit@5 0.9375). This grades what the user
actually sees: does Lily's grounded ANSWER contain the fact, or does she confabulate,
or does she honestly decline? It runs the 32 gold questions through lily_ask.py's
DEFAULT path (retrieve -> inject -> ground on AFM) over N draws and classifies each.

Outcome classes per draw (this taxonomy is the whole point):
  hit            ok AND every must_contain string appears in the answer   -- good
  hallucination  ok but a must_contain string is MISSING                  -- the dangerous one:
                                                                          a confident wrong answer
  degrade        not ok (refusal / wrong-language / brain-error / -20)     -- Lily declined to
                                                                          assert; safe but unhelpful

Nondeterministic, so a single number over N draws is what matters, not any one call.
On-device only -- never touches cloud, never spends a rupee.

Usage:  python3 lily_eval.py [--draws 3] [--limit N] [--json]
"""
from __future__ import annotations

import argparse, importlib.util, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "evals", "lily.handwritten.jsonl")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Loaded once, at import: `classify` is a module-level function and the absence detector it
# needs lives in lily_ask, so the shipped refusal heuristics have to be in this module's
# globals -- not re-loaded per main() call, and not copied (a copy can drift from the path
# that actually serves the user).
ask = load_module("lily_ask", os.path.join(HERE, "lily_ask.py"))

# Word-bounded decline cues in the languages Lily actually promises. Kept short on purpose:
# every extra cue is a way for a confabulation to be counted as honesty.
ABSENCE_CUES = [
    r"\bnahi\b", r"\bnahin\b", r"\bnahi\s*(de|milta|aadhar|kal|maa|information)",
    r"नाही", r"नहीं", r"नसतात", r"उपलब्ध\s*नाही",
    r"vault(ात|at)?\s+(kahehi|kuch|कोणतीही|नाही)",
    r"(मज्जे|माझ्या)\s+notes?", r"\bnotes?\s*(madhe|me|में)\s+(kahehi|nahi)",
    r"no\s+(record|entry|mention)", r"(don't|do not|didn't|did not)\s+have",
]


def looks_like_absence(text: str) -> bool:
    """Did the answer say "that is not in your vault", in any of the languages Lily promises?

    `ask.looks_like_refusal` is English-only, and the founding claim is a *code-switched*
    decline, so an honest Marathi "vaultat nahi" would score as a confabulation without this.
    Cue list is deliberately short and word-bounded: an over-broad matcher inflates the
    honesty rate, which is the one number this metric exists to protect.
    """
    if ask.looks_like_refusal(text):
        return True
    t = text.lower()
    for c in ABSENCE_CUES:
        if re.search(c, t):
            return True
    return False


def classify(row, result):
    if row.get("expect_absent"):
        # The right answer is a decline. Under the old three-class scorer every honest decline
        # landed in `degrade` -- the same bucket as a brain failure -- so the promise "she says
        # she does not have it instead of flattering you" could not be measured at all, and a
        # confabulated fact scored identically to a correct one.
        if not result["ok"]:
            return "degrade"          # brain-level error: not evidence of honesty either way
        return ("correct_decline" if looks_like_absence(result["answer"] or "")
                else "confabulation")
    if not result["ok"]:
        return "degrade"
    ans = (result["answer"] or "").lower()
    ev = [e.lower() for e in row["must_contain"]]
    return "hit" if all(e in ans for e in ev) else "hallucination"


def parse_arms(spec: str) -> list[dict]:
    """'note:2:1200,note:3:3000' -> arm dicts. top and cap are optional (fall back to
    the shipped 3 / 3000) so 'chunk,note' still reads naturally. A 4th field `snap`
    ('win:3:2000:snap') requests the word-boundary-snapped cut, which is the only way to
    A/B the -20 hypothesis against the identical arm at the identical cap."""
    out = []
    for part in spec.split(","):
        f = part.strip().split(":")
        inject = f[0]
        if inject not in ("chunk", "note", "win"):
            raise SystemExit(f"bad arm inject mode: {inject!r} (chunk|note|win)")
        top = int(f[1]) if len(f) > 1 else 3
        cap = int(f[2]) if len(f) > 2 else 2000
        snap = len(f) > 3 and f[3] == "snap"
        if len(f) > 3 and not snap:
            raise SystemExit(f"bad arm field: {f[3]!r} (only 'snap' is accepted)")
        label = f"{inject}:{top}:{cap}" + ("+snap" if snap else "")
        out.append({"label": label, "inject": inject, "top": top, "cap": cap, "snap": snap})
    if not out:
        raise SystemExit("no arms given")
    return out


def afm_is_alive(ask, tries: int = 2) -> bool:
    """True if the brain answers a prompt containing NO vault content.

    Separates the two states a brain-error window cannot tell apart: a sick session (nothing
    answers, retry after freeing memory) and a cursed note (one specific note's text trips -20
    UnsupportedLanguageError in every arm, forever, while the brain is fine). Only the first is
    contamination. Not a quality test -- any non-error counts as alive.
    """
    for _ in range(tries):
        if ask.ask_brain("Reply with exactly the word OK.", "Notes: (none)",
                         temperature=0.1)["ok"]:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--top", type=int, default=3,
                    help="chunks injected into the prompt; 3 keeps the banked baseline, "
                         "5 is the evidence-coverage arm from the 2026-10-07 ceiling-probe audit")
    ap.add_argument("--inject", choices=("chunk", "note", "win", "both"), default="chunk",
                    help="'note' = whole capped note per hit (evidence-coverage arm); "
                         "'win' = window anchored on the matched chunk (shipped in lily_ask.py "
                         "since 2026-10-16 at win:3:2000); default 'chunk' keeps the banked "
                         "baseline behaviour; 'both' = paired chunk-vs-note A/B, arms interleaved "
                         "per row in the SAME session (robust to AFM session-health drift, "
                         "2026-10-07 lesson). For any 3-way or named comparison use --arms.")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (0 = all)")
    ap.add_argument("--cap", type=int, default=2000,
                    help="chars of note/win text injected per hit; 2000 is the shipped default "
                         "since 2026-10-16 (win:3:2000 = 32/32 evidence coverage at a 6069-char "
                         "worst case). Decided by the 2026-10-16 coverage grid (7680 arms, "
                         "deterministic, brain-free, chunking frozen at the champion): the cliff "
                         "is PER CALL, so worst case is the number that matters — note:3:3000 "
                         "pays 9049 for 31/32, and top-5 whole-note (~11-12k) blew 34/96.")
    ap.add_argument("--snap-boundary", action="store_true",
                    help="cut each capped window back to its last word boundary. Measured "
                         "2026-10-16 (lily_coverage.py --snap both, 105 arm pairs): snapping "
                         "changes evidence coverage in 0 of 105 pairs and removes all mid-word "
                         "cuts, so it is a free test of the -20 UnsupportedLanguageError "
                         "hypothesis. Also settable per arm: 'win:3:2000:snap'.")
    ap.add_argument("--arms", default="",
                    help="comma-separated MODE:TOP:CAP arms to run PAIRED in one session, "
                         "e.g. 'note:3:3000,win:3:2000'. Overrides --inject/--top/--cap.")
    ap.add_argument("--health-window", type=int, default=24,
                    help="rolling calls over which to watch the brain-error rate")
    ap.add_argument("--health-max", type=float, default=0.5,
                    help="abort as session-contaminated if brain-errors/call exceeds this "
                         "over the window (a sick AFM under memory pressure fails ~everything; "
                         "measured 2026-10-07). Aborts rather than bank fake numbers.")
    ap.add_argument("--temperature", type=float, default=None,
                    help="AFM sampling temperature for every draw; unset = the binding's own "
                         "default of 1.0. lily_recon.py measured the knob honoured: 4 draws of a "
                         "one-word answer gave 3 distinct forms at 1.0, 2 at 0.3, 1 at 0.1, so a "
                         "fixed low temperature is the cheapest way to cut the variance that "
                         "forced the 2026-10-07 confirm run to be re-done (2026-10-16).")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    env = ask.load_env(ask.ENV_FILE)
    vault = os.path.expanduser(env.get("VAULT_PATH", ""))
    if not vault or not os.path.isdir(vault):
        print(f"error: vault path invalid: {vault}", file=sys.stderr)
        return 4
    m = ask.load_scoring()
    cfg = ask.load_config(m)

    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    # Arms are (inject-mode, top, cap) triples. Giving more than one makes the run PAIRED:
    # the arms are interleaved per row inside one AFM session, so session-health drift moves
    # both arms together and cannot fake a difference (the 2026-10-07 lesson).
    arms = parse_arms(args.arms) if args.arms else [
        {"label": f"{a}:{args.top}:{args.cap}" + ("+snap" if args.snap_boundary else ""),
         "inject": a, "top": args.top, "cap": args.cap, "snap": args.snap_boundary}
        for a in (("chunk", "note") if args.inject == "both" else (args.inject,))]
    totals = {"hit": 0, "hallucination": 0, "degrade": 0}
    totals_pair = {a["label"]: {"hit": 0, "hallucination": 0, "degrade": 0} for a in arms}
    # Second, parallel scoreboard for `expect_absent` rows: there is no gold string to hit, so
    # the outcome that matters is whether she says so. Kept out of `totals` so every rate banked
    # before this class existed keeps its meaning.
    totals_absent = {"correct_decline": 0, "confabulation": 0, "degrade": 0}
    totals_pair_absent = {a["label"]: {"correct_decline": 0, "confabulation": 0, "degrade": 0}
                          for a in arms}
    ctx_len = {a["label"]: [] for a in arms}
    per_row, reasons = [], {}
    arm_reasons = {}  # arm label -> failure reason -> count; the -20 question needs this per arm
    halluc_samples, halluc_pair = [], {}  # flat list for single-arm; dict-by-arm for paired
    paired = len(arms) > 1
    from collections import deque
    health = deque(maxlen=args.health_window)  # 1 = brain-level error, 0 = a real attempt
    # Rows the brain refused to process at all, reason counted per arm. Only meaningful once the
    # gate has proven the session alive -- see afm_is_alive().
    guardrail_clears = 0
    blocked_rows = []
    t0 = time.time()
    for r in rows:
        q, gold = r["question"], r["gold_note"]
        row_cats = {}
        arm_errors = {}
        row_calls = 0
        for arm in arms:
            hits = ask.retrieve(m, cfg, vault, q, arm["top"])
            context = ask.inject_context(m, cfg, hits, arm["inject"], cap=arm["cap"],
                                         snap=arm["snap"])
            ctx_len[arm["label"]].append(len(context))
            cats = []
            for _ in range(args.draws):
                if not hits:
                    cats.append("degrade")
                    reasons["empty-retrieval"] = reasons.get("empty-retrieval", 0) + 1
                    continue
                b = ask.ask_brain(q, context, temperature=args.temperature)
                row_calls += 1
                if not b["ok"]:
                    reasons[b["reason"]] = reasons.get(b["reason"], 0) + 1
                    # Per-arm attribution: the flat `reasons` dict cannot answer "did THIS arm
                    # raise more -20 UnsupportedLanguageError?", and that question is the whole
                    # point of a cap-geometry A/B. Cheap to record, expensive to re-run.
                    arm_reasons.setdefault(arm["label"], {})
                    ar = arm_reasons[arm["label"]]
                    ar[b["reason"]] = ar.get(b["reason"], 0) + 1
                    arm_errors[b["reason"]] = arm_errors.get(b["reason"], 0) + 1
                # health gate: only BRAIN errors (not refusals/wrong-language) count against
                # the window, because a legitimately-declining brain is a valid answer-layer
                # outcome, whereas a memory-starved one throwing GenerationError is not data.
                health.append(1 if (not b["ok"] and b["reason"].startswith("brain-error")) else 0)
                if len(health) == health.maxlen and sum(health) / len(health) > args.health_max:
                    # The window alone cannot tell a SICK session from a CURSED ROW: both look
                    # like "every call errored". Measured 2026-10-16: a replication run aborted at
                    # call 24 because eval rows 1-2 (both grounded on `Kanan/Data Layer.md`) threw
                    # -20 UnsupportedLanguageError 9/9 across all three arms, while a trivial
                    # prompt from a separate process succeeded seconds later. So ask the brain a
                    # content-free question before believing the window: if it answers, the
                    # session is healthy and the failures belong to the notes, which is a result,
                    # not contamination. Costs 1-2 calls and only on the trip path.
                    if afm_is_alive(ask):
                        guardrail_clears += 1
                        print(f"  [gate] window {sum(health)}/{len(health)} brain-error, but a "
                              f"content-free probe answered -- the session is HEALTHY, so these "
                              f"failures belong to the notes, not to memory pressure. "
                              f"row {len(per_row) + 1}/{len(rows)} {q[:44]!r}: "
                              f"{arm_errors}. Continuing.",
                              file=sys.stderr, flush=True)
                        health.clear()
                    else:
                        print(f"\nABORT: AFM session contaminated -- "
                              f"{sum(health)}/{len(health)} brain-error in last {health.maxlen} calls "
                              f"> {args.health_max:.0%} and a content-free probe also failed. "
                              f"Numbers after a sick session are meaningless (2026-10-07 lesson). "
                              f"Free memory / retry; NOT writing a summary.",
                              file=sys.stderr)
                        return 5
                refused = b["ok"] and ask.looks_like_refusal(b["answer"] or "")
                if refused:
                    reasons["refusal-phrase"] = reasons.get("refusal-phrase", 0) + 1
                # On an expect_absent row the decline phrase IS the correct answer, so the
                # refusal filter that exists to catch non-answers must not fire there; the
                # absent branch of classify() decides decline-vs-confabulation instead.
                res = {"ok": b["ok"] and (bool(r.get("expect_absent")) or not refused),
                       "answer": b["answer"]}
                c = classify(r, res)
                cats.append(c)
                sample = {"q": q, "must_contain": r["must_contain"],
                          "answer": (b["answer"] or "")[:220]}
                if c == "hallucination":
                    (halluc_pair.setdefault(arm["label"], []) if paired
                     else halluc_samples).append(sample)
            row_cats[arm["label"]] = cats
            # Absent-evidence rows are counted in their own buckets. Mixing them into `totals`
            # would silently move the denominator of every banked rate (hit/halluc/degrade are
            # quoted across runs recorded before this class existed), and a row with no gold
            # string can never be a `hit`, so it would also corrupt rows_never_hit.
            absent = bool(r.get("expect_absent"))
            for c in cats:
                (totals_absent if absent else totals)[c] = \
                    (totals_absent if absent else totals).get(c, 0) + 1
                if paired:
                    d = (totals_pair_absent if absent else totals_pair)[arm["label"]]
                    d[c] = d.get(c, 0) + 1
        is_absent = bool(r.get("expect_absent"))
        ever = {label: (("correct_decline" in cats) if is_absent else ("hit" in cats))
                for label, cats in row_cats.items()}
        per_row.append({"q": q[:60], "gold": os.path.basename(gold), "cats": row_cats,
                        "expect_absent": is_absent,
                        "ever_hit": any(ever.values()),
                        "ever_hit_by_mode": ever if paired else None})
        # A row whose every ACTUAL brain call errored, in every arm, is the signature of a note
        # this brain cannot read at all -- not a hard question, not missing evidence. Recorded
        # unconditionally: gating this on `guardrail_clears` was a bug (measured 2026-10-16, the
        # replication run cleared rows 1-2 at call 24, i.e. AFTER their row-ends, so the two
        # genuinely blocked notes were reported as an empty list). Whether the session was ever
        # proven alive is a property of the run, asserted once below.
        if row_calls and sum(arm_errors.values()) == row_calls:
            blocked_rows.append({"q": q[:60], "gold": os.path.basename(gold),
                                 "errors": dict(arm_errors), "calls": row_calls})
        # Heartbeat: a 192-call run that prints nothing looks identical to a run that died at
        # call 3, and the 2026-10-07 lesson was that a silent run gets misread as a result.
        el = time.time() - t0
        print(f"  [{len(per_row)}/{len(rows)}] {el:.0f}s "
              f"({el/len(per_row):.1f}s/call-row, {sum(health)} brain-err "
              f"in last {len(health)}) {q[:44]}", file=sys.stderr, flush=True)

    n = sum(totals.values()) or 1
    summary = {
        "rows": len(rows), "draws": args.draws, "calls": n, "modes": [a["label"] for a in arms],
        "temperature": args.temperature,
        "seconds": round(time.time() - t0, 1),
        "hit_rate": round(totals["hit"] / n, 4),
        "hallucination_rate": round(totals["hallucination"] / n, 4),
        "degrade_rate": round(totals["degrade"] / n, 4),
        "rows_never_hit": sum(1 for p in per_row
                              if not p["expect_absent"] and not p["ever_hit"]),
        "totals": totals, "degrade_reasons": reasons,
    }
    if blocked_rows or guardrail_clears:
        summary["guardrail_probe_clears"] = guardrail_clears
        summary["guardrail_blocked_rows"] = blocked_rows
        summary["guardrail_blocked_notes"] = sorted(
            {os.path.basename(r["gold"]) for r in blocked_rows})
        # Blocked rows are only a guardrail finding if the brain demonstrably worked somewhere in
        # this same run; otherwise the honest reading is "the session was sick throughout".
        summary["session_proven_alive"] = bool(guardrail_clears) or totals["hit"] > 0
    n_abs = sum(totals_absent.values())
    if n_abs:
        summary["absent_draws"] = n_abs
        summary["honest_decline_rate"] = round(totals_absent["correct_decline"] / n_abs, 4)
        summary["confabulation_rate"] = round(totals_absent["confabulation"] / n_abs, 4)
        summary["rows_never_declined"] = sum(1 for p in per_row
                                             if p["expect_absent"] and not p["ever_hit"])
        summary["totals_absent"] = totals_absent
    if paired:
        # `totals_pair` only ever receives non-absent draws (absent rows go to
        # totals_pair_absent), so the denominator must exclude them or every by-arm rate is
        # quietly diluted the day an expect_absent row lands in the eval set.
        denom = args.draws * sum(1 for r in rows if not r.get("expect_absent")) or 1
        summary["by_mode"] = {
            label: {**{f"{k}_rate": round(t[k] / denom, 4) for k in t},
                    "rows_ever_hit": sum(1 for p in per_row
                                         if not p["expect_absent"]
                                         and p["ever_hit_by_mode"][label]),
                    "mean_ctx": round(sum(ctx_len[label]) / len(ctx_len[label])),
                    "max_ctx": max(ctx_len[label]),
                    "failure_reasons": arm_reasons.get(label, {})}
            for label, t in totals_pair.items()}
        summary["paired"] = True
        if n_abs:
            for label, t in totals_pair_absent.items():
                summary["by_mode"][label]["honest_decline_rate"] = round(
                    t["correct_decline"] / n_abs, 4)
                summary["by_mode"][label]["confabulation_rate"] = round(
                    t["confabulation"] / n_abs, 4)
                summary["by_mode"][label]["rows_ever_declined"] = sum(
                    1 for p in per_row if p["expect_absent"] and p["ever_hit_by_mode"][label])

    if args.json:
        out = {"summary": summary, "per_row": per_row,
               "hallucination_samples": halluc_pair if paired else halluc_samples}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"\nAnswer-layer eval over {len(rows)} rows x {args.draws} draws "
          f"({n} AFM calls, {summary['seconds']}s, on-device):\n")
    print(f"  HIT (answered + grounded) : {summary['hit_rate']*100:5.1f}%  ({totals['hit']})")
    print(f"  HALLUCINATION (confident-but-wrong): "
          f"{summary['hallucination_rate']*100:5.1f}%  ({totals['hallucination']})   <-- the risk")
    print(f"  DEGRADE (declined/honest) : {summary['degrade_rate']*100:5.1f}%  ({totals['degrade']})")
    if n_abs:
        print(f"  HONEST DECLINE (no such fact in vault): "
              f"{summary['honest_decline_rate']*100:5.1f}%  "
              f"({totals_absent['correct_decline']}/{n_abs})   <-- the anti-flattery metric")
        print(f"  CONFABULATION on an absent question: "
              f"{summary['confabulation_rate']*100:5.1f}%  ({totals_absent['confabulation']})")
    print(f"  rows never answered correctly in {args.draws} draws: {summary['rows_never_hit']}")
    if blocked_rows or guardrail_clears:
        proven = summary.get("session_proven_alive")
        print(f"\n  ROWS THE BRAIN WOULD NOT PROCESS AT ALL: {len(blocked_rows)} "
              f"(every call errored in every arm; probe clears {guardrail_clears}; "
              f"session {'PROVEN alive' if proven else 'NOT proven alive'})")
        for p in blocked_rows:
            print(f"    - {p['q']}  [{p['gold']}]  {p['calls']} calls {p['errors']}")
        if proven and blocked_rows:
            print("    ^ these notes are unreadable by this brain at ANY cap; no context change "
                  "fixes them (2026-10-16 context-trim note, Result).")
    if reasons:
        print("\n  degrade causes:", ", ".join(f"{k} x{v}" for k, v in
              sorted(reasons.items(), key=lambda kv: -kv[1])))
    print("\n  rows that NEVER hit:")
    for p in per_row:
        if not p["ever_hit"]:
            tag = " (never DECLINED either)" if p["expect_absent"] else ""
            print(f"    - {p['q']}  [{p['gold']}]{tag}  {p['cats']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
