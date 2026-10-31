#!/usr/bin/env python3
"""lily_ceiling.py -- the ceiling probe: is AFM's ~30% answer-hit a 3B capability wall?

This runs the SAME 32 gold questions through the SAME retrieve -> inject -> classify
pipeline as lily_eval.py, with ONE variable changed: the brain. Instead of on-device AFM
it calls a local Ollama model. If a slightly different model grounds dramatically better on
identical context, the bottleneck is partly the AFM brain; if it fails the same way, more
prompt work on AFM is pointless and the limit is the model size / hardware, not our wiring.

To give the probe model the best possible shot (this is a CEILING probe, not an apples
-n- apples benchmark), we decode greedily (temperature 0) -- MORE favourable than AFM got
in the baseline. We do NOT change retrieval, context, the instruction, or the scorer.

Usage:  python3 lily_ceiling.py [--model qwen3:4b] [--draws 3] [--server http://localhost:11434]
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "evals", "lily.handwritten.jsonl")

# The exact instruction string from lily_ask.ask_brain -- kept byte-identical so the
# only thing the A/B varies is the generator, never the prompt.
INSTR = ("Answer only from the notes provided. Quote the note name you used. "
         "If the answer is not in the notes, say exactly: NOT IN MY NOTES. "
         "Reply in the same language as the question.")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chat_generate(base_url, model, api_key, question, context, timeout=180):
    """Call any OpenAI-compatible /chat/completions endpoint: Ollama (/v1), Groq,
    Cloudflare Workers AI, OpenRouter -- one code path for the whole ceiling sweep.
    Greedy (temperature 0) so the probe model is deterministic and gets its best shot."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model, "temperature": 0, "messages":
            [{"role": "system", "content": INSTR},
             {"role": "user", "content": f"Notes:\n{context}\n\nQuestion: {question}"}]}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return {"ok": True, "answer": (data["choices"][0]["message"]["content"] or "").strip()}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()[:180]
        except Exception:
            pass
        return {"ok": False, "reason": f"http-{exc.code}:{detail}", "answer": None}
    except Exception as exc:
        return {"ok": False, "reason": f"gen-error:{type(exc).__name__}:{exc}", "answer": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--draws", type=int, default=1, help="greedy => deterministic, 1 suffices")
    ap.add_argument("--server", default="http://localhost:11434",
                    help="Ollama host; used to derive --base-url when the latter is empty")
    ap.add_argument("--base-url", default="",
                    help="OpenAI-compatible base, e.g. https://api.groq.com/openai/v1")
    ap.add_argument("--api-key-env", default="CEILING_API_KEY",
                    help="NAME of the env var holding the bearer key (never pass the key itself)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=3,
                    help="chunks injected; 3 matches the banked AFM baseline and the first probe")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    base_url = args.base_url or (args.server.rstrip("/") + "/v1")
    api_key = os.environ.get(args.api_key_env, "")

    ask = load_module("lily_ask", os.path.join(HERE, "lily_ask.py"))
    ev = load_module("lily_eval", os.path.join(HERE, "lily_eval.py"))
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

    # warmup: one tiny call so a local model's cold load isn't charged to draw 1;
    # hosted endpoints are already warm, so a failure here just surfaces per-draw.
    chat_generate(base_url, args.model, api_key, "ping", "warmup")

    totals = {"hit": 0, "hallucination": 0, "degrade": 0}
    per_row, reasons, halluc = [], {}, []
    t0 = time.time()
    for r in rows:
        q, gold = r["question"], r["gold_note"]
        hits = ask.retrieve(m, cfg, vault, q, args.top)
        context = "\n\n".join(f"[{os.path.basename(n)}] {t}" for n, t, _ in hits)
        cats = []
        for _ in range(args.draws):
            if not hits:
                cats.append("degrade")
                reasons["empty-retrieval"] = reasons.get("empty-retrieval", 0) + 1
                continue
            g = chat_generate(base_url, args.model, api_key, q, context)
            if not g["ok"]:
                reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1
                cats.append("degrade")
                continue
            refused = ask.looks_like_refusal(g["answer"] or "")
            if refused:
                reasons["refusal-phrase"] = reasons.get("refusal-phrase", 0) + 1
            res = {"ok": not refused, "answer": g["answer"]}
            c = ev.classify(r, res)
            cats.append(c)
            if c == "hallucination":
                halluc.append({"q": q, "must_contain": r["must_contain"],
                               "answer": (g["answer"] or "")[:220]})
        for c in cats:
            totals[c] += 1
        per_row.append({"q": q[:60], "gold": os.path.basename(gold), "cats": cats,
                        "ever_hit": "hit" in cats})

    n = sum(totals.values()) or 1
    summary = {"model": args.model, "rows": len(rows), "draws": args.draws, "calls": n,
               "seconds": round(time.time() - t0, 1),
               "hit_rate": round(totals["hit"] / n, 4),
               "hallucination_rate": round(totals["hallucination"] / n, 4),
               "degrade_rate": round(totals["degrade"] / n, 4),
               "rows_never_hit": sum(1 for p in per_row if not p["ever_hit"]),
               "totals": totals, "degrade_reasons": reasons}

    if args.json:
        print(json.dumps({"summary": summary, "per_row": per_row,
                          "hallucination_samples": halluc}, ensure_ascii=False, indent=2))
        return 0

    print(f"\nCEILING PROBE  model={args.model}  {len(rows)} rows x {args.draws} draws "
          f"({n} calls, {summary['seconds']}s, greedy temp=0):\n")
    print(f"  HIT              : {summary['hit_rate']*100:5.1f}%  ({totals['hit']})")
    print(f"  HALLUCINATION    : {summary['hallucination_rate']*100:5.1f}%  ({totals['hallucination']})")
    print(f"  DEGRADE          : {summary['degrade_rate']*100:5.1f}%  ({totals['degrade']})")
    print(f"  rows never hit   : {summary['rows_never_hit']}")
    print("\n  AFM baseline for comparison: HIT 30.2% / HALLUC 35.4% / DEGRADE 34.4%")
    if reasons:
        print("\n  degrade causes:", ", ".join(f"{k} x{v}" for k, v in
              sorted(reasons.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
