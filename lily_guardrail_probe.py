"""Localise the -20 UnsupportedLanguageError on the notes that trip it.

HISTORY OF THIS TOOL. It used to BISECT for a "smallest tripping prefix", on the assumption that
once a prefix fails every longer prefix fails. That assumption is false and the bisect's output was
meaningless: measured 2026-10-16 on Kanan/Data Layer.md, 29- and 70-char prefixes answer, the
79-char prefix fails, a 1932-char suffix of the same note answers, and every prefix from 219 chars
onward fails. Bisecting a non-monotone predicate returns whatever the search path landed on, so it
is replaced by an exhaustive boundary sweep.

What the sweep plus a draw-repeat established, and what is NOT worth re-testing:
  - The failure is DETERMINISTIC per body (5/5 vs 0/5 across five bodies), so it is a property of
    the text, not session drift. Use draws>=5 only to confirm that, not to hunt variance.
  - No content span explains it. Deleting the credential vocabulary, the em-dash, or the mid-word
    fragment from the tripping line all leave it failing.
  - Our instructions are irrelevant: `--variants` returns identical verdicts for the shipped text,
    the shipped text minus "Reply in the same language as the question.", a forced-English line,
    and NO instructions at all. Keep the flag if you doubt that; do not expect a new answer.
  - The rescue those findings implied (stop sending the note head) is coverage-free (32/32 -> 32/32)
    and pre-registered against a paired 8-draw run -- and it FAILED (0/16 vs 3/16, p=0.234).
    See .docs/notes/autoresearch-review/2026-10-16-context-trim.md, Result 3.

Runs from the project dir; takes one arg (a note path).
"""
import argparse, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ask = load("lily_ask", os.path.join(HERE, "lily_ask.py"))
Q = "Which Room database file name does Kanan persist to on device?"

# Kept because they are the measured null, not because they are promising. `bare` proves the wall
# exists with no instructions argument at all.
VARIANTS = {
    "shipped": None,
    "no-lang-line": ("Answer only from the notes provided. Quote the note name you used. "
                     "If the answer is not in the notes, say exactly: NOT IN MY NOTES."),
    "force-english": ("Answer only from the notes provided. Quote the note name you used. "
                      "If the answer is not in the notes, say exactly: NOT IN MY NOTES. "
                      "Reply in English."),
    "bare": "",
}


def verdict(body: str, instructions: str | None = None) -> str:
    r = ask.ask_brain(Q, body, temperature=0.1, instructions=instructions)
    return "ok" if r["ok"] else r["reason"].replace("brain-error:", "")


def repeat(body: str, draws: int) -> list[str]:
    return [verdict(body) for _ in range(draws)]


def variant_test(body: str, label: str) -> None:
    print(f"\n  instruction variants on {label}:")
    for name, instr in VARIANTS.items():
        outs = [verdict(body, instr) for _ in range(2)]
        tag = "  <== UNBLOCKED" if all(o == "ok" for o in outs) else ""
        print(f"    {name:14} {outs}{tag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("note")
    ap.add_argument("--variants", action="store_true",
                    help="compare instruction variants on the shortest failing body (measured null)")
    ap.add_argument("--draws", type=int, default=1,
                    help="draws per body; 5 is what established determinism, 1 is enough to map")
    ap.add_argument("--json", dest="jsonout")
    args = ap.parse_args()

    notes = os.path.expanduser(args.note)
    body = open(notes, encoding="utf-8").read()
    print(f"note: {os.path.basename(notes)}  chars={len(body)}  lines={body.count(chr(10))}")

    lines = body.split("\n")
    marks, rows = [], []
    for i in range(1, len(lines) + 1):
        prefix = "\n".join(lines[:i])
        outs = repeat(prefix, args.draws)
        ok = sum(o == "ok" for o in outs)
        marks.append("." if ok == len(outs) else ("x" if ok == 0 else "?"))
        rows.append({"lines": i, "chars": len(prefix), "ok": ok, "draws": len(outs)})
        print(f"  lines 1-{i:2} chars={len(prefix):5} ok {ok}/{len(outs)}  {''.join(marks)}")

    failing = [r for r in rows if r["ok"] < r["draws"]]
    passing = [r for r in rows if r["ok"] == r["draws"]]
    if not failing:
        print("  no line-boundary prefix trips the guardrail -- this note is askable")
    elif not passing:
        print("  every prefix trips, including line 1 -- the note is askable only with the head gone")
    else:
        onset = min(r["chars"] for r in failing)
        # Non-monotone by construction, so report the set rather than a single threshold.
        print(f"  onset: smallest FAILING line-boundary prefix = {onset} chars; "
              f"{len(passing)} shorter prefixes answer")
        print(f"  NOT a threshold: {[r['chars'] for r in passing]} answer while {onset} fails. "
              f"Recover from the failing boundary and subtract it out, then ask again without it.")
    if args.jsonout:
        json.dump({"note": notes, "q": Q, "rows": rows}, open(args.jsonout, "w", encoding="utf-8"),
                  indent=1)
    if args.variants:
        target = min((r for r in failing), key=lambda r: r["chars"], default=None)
        if target:
            variant_test("\n".join(lines[:target["lines"]]), f"the {target['chars']}-char failing prefix")
        variant_test(body, "the whole note")
    return 0


if __name__ == "__main__":
    sys.exit(main())
