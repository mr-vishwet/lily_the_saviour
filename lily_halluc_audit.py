"""Decompose a graded HALLUCINATION into "wrong" vs "right but the gold string missed it".

The scorer calls an answer a hallucination when a `must_contain` literal is absent. That is the
right shape for HIT, but it makes the HALLUCINATION headline unsafe to act on: a model that
quotes the correct sentence verbatim, without reproducing the markdown around the gold string,
is scored as confident-but-wrong. Measured 2026-10-16 -- over two paired runs, 21 of 39 flagged
`note` draws and 33 of 46 flagged `win` draws were correct text from the very note the question
asked about, and the real confabulation rates were 5.73% vs 4.69%. The arm the headline metric
called careless was the careful one.

So: never read `hallucination_rate` as a safety number without running this over it.

    .venv/bin/python lily_halluc_audit.py .docs/autoresearch/answer_eval_replicate_snap.json

Deterministic and brain-free: it re-reads the vault notes the answers were supposed to come from.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "evals", "lily.handwritten.jsonl")

STOP = {"which", "what", "that", "this", "does", "used", "uses", "with", "from", "have", "has",
        "note", "notes", "answer", "question", "the", "and", "for", "are", "api", "apis",
        "instead", "before", "through"}

# share/dump/quote chars that carry no meaning but dominate literal comparison
_PUNCT = re.compile(r"[*`_>\[\]()#|~]")


def norm(s: str) -> str:
    """Lowercase, links unwrapped, markdown decoration dropped."""
    s = re.sub(r"\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]", r"\1", s)
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def tokens(s: str) -> set:
    """Meaningful tokens only: 4+ chars, stop words out. Answers are sentences; gold notes are
    not, so anything that survives this is content the model either got from the note or did not."""
    return {t for t in re.findall(r"[a-z0-9][a-z0-9./_+:-]{3,}", norm(s)) if t not in STOP}


def gold_paths(evalset: str) -> dict:
    out = {}
    for line in open(evalset, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[norm(r["question"])[:40]] = os.path.expanduser(r["gold_note"])
    return out


def classify(answer: str, q: str, paths: dict, cache: dict, thresh: float) -> tuple:
    """(label, coverage) for one flagged answer against its own gold note."""
    key = norm(q)[:40]
    if key not in cache:
        p = paths.get(key)
        try:
            cache[key] = tokens(open(p, encoding="utf-8").read()) if p else None
        except OSError:
            cache[key] = None
    gt = cache[key]
    if gt is None:
        return "NO-GOLD-FILE", 0.0
    at = tokens(answer)
    if not at:
        return "NO-SIGNAL", 0.0
    cov = len(at & gt) / len(at)
    if cov >= thresh:
        return "IN-GOLD", cov          # correct content, unmatched literal
    if cov >= thresh / 2:
        return "PARTIAL", cov
    return "INVENTED", cov             # genuinely confident-but-wrong


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="lily_eval.py --json artefacts")
    ap.add_argument("--eval", dest="evalset", default=EVAL)
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="fraction of the answer's tokens that must appear in the gold note to "
                         "call it correct-but-unmatched (default 0.8, i.e. near-verbatim)")
    ap.add_argument("--show-invented", action="store_true")
    args = ap.parse_args()

    paths = gold_paths(args.evalset)
    cache = {}
    for run in args.runs:
        d = json.load(open(run, encoding="utf-8"))
        s = d["summary"]
        samples = d["hallucination_samples"]
        by_arm = samples.items() if isinstance(samples, dict) else [("pooled", samples)]
        # Draws actually scored per arm: rows with a gold string to hit. expect_absent rows are
        # graded on a different axis, so counting them would dilute this rate.
        n_draws = s["draws"] * sum(1 for p in d["per_row"] if not p.get("expect_absent"))
        print(f"\n{os.path.basename(run)}  ({s['calls']} calls, {s['draws']} draws x "
              f"{s['rows']} rows, temp {s['temperature']}, {n_draws} scored draws per arm)")
        for arm, ss in by_arm:
            tally = {}
            for x in ss:
                label, cov = classify(x["answer"], x["q"], paths, cache, args.threshold)
                tally.setdefault(label, []).append((cov, x))
            inv = len(tally.get("INVENTED", []))
            print(f"  {arm:22} graded {len(ss):3}  ->  IN-GOLD {len(tally.get('IN-GOLD', [])):3} "
                  f"PARTIAL {len(tally.get('PARTIAL', [])):2} INVENTED {inv:2}"
                  f"   TRUE confabulation rate {100 * inv / n_draws:5.2f}%")
            if args.show_invented:
                for cov, x in sorted(tally.get("INVENTED", [])):
                    print(f"      cov={cov:.2f} {x['answer'][:72]!r}  [{x['q'][:40]}]")
    print("\nRead: compare the TRUE rate across arms. The graded count is not the risk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
