#!/usr/bin/env python3
"""lily_recon.py -- an internal reverse-engineering harness, in the shape of morluto/rea.

REA's method is: recon (what is this thing?) -> static (what does it expose without
running?) -> dynamic (what does it actually do?) -> report (claims tied to evidence).
We adopt the method, not the toolchain: this machine has 8 GB RAM and ~5 GB free disk,
so Ghidra/Hopper are not installable. Everything REA does through a GUI, this does with
`otool`, `strings`, `inspect`, and bounded live probes.

AUTHORIZATION. REA's own constraint is "software you own or are authorized to examine".
The only built-in target is the Apple FoundationModels *binding installed in our own
venv* plus the dylib it ships -- files on this disk, read by our own code, examined for
interoperability with a personal agent. Anything else needs an explicit allowlist entry
with a stated basis; `--target` refuses otherwise. This tool never copies, dumps or
transmits the system's machine learning assets.

WHY THIS EXISTS (the payoff is measured, not assumed). Every real gain this project has
had came from treating AFM as a black box and probing it, not from its docs:
  - errors returned as *text* with a success-shaped response (ADR-0003 consequence 5),
  - the context cliff (top-3 whole-notes ~6-8k ok; top-5 ~11-12k -> ContextWindowExceeded x34/96),
  - session-health poisoning: check_availability() returns AVAILABLE while every generate()
    raises GenerationError.
This harness makes those findings reproducible and adds a standing question set.
"""
from __future__ import annotations
import argparse, hashlib, importlib, inspect, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

# The allowlist IS the authorization gate. basis = why we may examine it.
TARGETS = {
    "afm": {
        "path": ".venv/lib/python3.12/site-packages/applefoundationmodels",
        "module": "applefoundationmodels",
        "basis": "Python package vendored into this project's own venv; binaries on this "
                 "disk; examined for interoperability with software we are writing.",
    },
}

CAPABILITY_RE = re.compile(
    r"(?i)(context|token|buffer|language|locale|temperature|available|guardrail|refus|schema|max)")


def sh(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- recon
def phase_recon(t: dict) -> dict:
    root = os.path.join(HERE, t["path"])
    files, digest = [], {}
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if n.endswith((".pyc",)) or "__pycache__" in dirpath:
                continue
            p = os.path.join(dirpath, n)
            rel = os.path.relpath(p, root)
            files.append(rel)
            if n.endswith((".so", ".dylib", ".json")):
                digest[rel] = {"bytes": os.path.getsize(p), "sha256_16": sha256(p)}
    out = {"root": root, "files": files, "binaries": digest}
    for rel, info in digest.items():
        if rel.endswith((".so", ".dylib")):
            rc, txt = sh(["otool", "-L", os.path.join(root, rel)])
            info["links"] = [ln.strip() for ln in txt.splitlines()[1:] if ln.strip()]
            info["otool_rc"] = rc
    sys_fw = "/System/Library/Frameworks/FoundationModels.framework/Versions/A/FoundationModels"
    out["system_framework_on_disk"] = os.path.exists(sys_fw)
    out["note"] = ("FoundationModels.framework ships only Resources/ + _CodeSignature: the "
                   "code lives in the dyld shared cache, so no static string analysis of the "
                   "Apple-side binary is possible here. The bundled libfoundation_models.dylib "
                   "is the whole observable static surface." if not out["system_framework_on_disk"]
                   else "")
    return out


# ---------------------------------------------------------------- static
def phase_static(t: dict) -> dict:
    mod = importlib.import_module(t["module"])
    pub = [n for n in dir(mod) if not n.startswith("_")]
    root = os.path.join(HERE, t["path"])
    codes = json.load(open(os.path.join(root, "error_codes.json"), encoding="utf-8"))
    ec = {c["code"]: {"name": c["name"], "swift_case": c["swift_case"], "desc": c["description"]}
          for c in codes}
    rc, txt = sh(["strings", "-a", os.path.join(root, "libfoundation_models.dylib")])
    cap = sorted({s.strip() for s in txt.splitlines()
                  if 6 <= len(s.strip()) <= 120 and CAPABILITY_RE.search(s)})
    sess = mod.Session
    api = {
        "session_methods": [n for n in dir(sess) if not n.startswith("_")],
        "generate_sig": str(inspect.signature(sess.generate)),
        "init_sig": str(inspect.signature(sess.__init__)),
        "defaults": {"temperature": getattr(mod, "DEFAULT_TEMPERATURE", None),
                     "max_tokens": getattr(mod, "DEFAULT_MAX_TOKENS", None),
                     "temperature_bounds": [getattr(mod, "MIN_TEMPERATURE", None),
                                            getattr(mod, "MAX_TEMPERATURE", None)]},
        "presets": {k: v for k, v in vars(mod.TemperaturePreset).items() if not k.startswith("_")},
        "availability_codes": {int(v): k for k, v in
                                getattr(mod.Availability, "__members__", {}).items()},
        "version": getattr(mod, "__version__", None),
    }
    # the contract-vs-call audit: which documented knobs does OUR code actually pass?
    ours = open(os.path.join(HERE, "lily_ask.py"), encoding="utf-8").read()
    used = {k: bool(re.search(rf"{k}\s*=", ours)) for k in ("temperature", "max_tokens", "schema", "stream")}
    return {"public_api": pub, "error_codes": ec, "capability_strings": cap,
            "api": api, "our_call_audit": used}


# ---------------------------------------------------------------- dynamic
class Health:
    """Abort on a contaminated session, exactly like lily_eval's gate: a sick AFM produces
    numbers that look like findings and are not."""

    def __init__(self, window: int = 12, max_bad: float = 0.5):
        self.h: list[int] = []
        self.window, self.max_bad = window, max_bad

    def feed(self, bad: bool) -> bool:
        self.h.append(1 if bad else 0)
        tail = self.h[-self.window:]
        return len(tail) == self.window and sum(tail) / len(tail) > self.max_bad


def phase_dynamic(t: dict, probes: list[str], max_calls: int) -> dict:
    import applefoundationmodels as afm
    res, calls, health = {}, 0, Health()

    def one(prompt: str, **kw):
        nonlocal calls
        if calls >= max_calls:
            return None, "budget-exhausted", 0
        calls += 1
        s = None
        t0 = time.time()
        try:
            s = afm.Session(instructions="You are a terse test harness. Answer in at most 12 words.")
            r = s.generate(prompt, max_tokens=kw.get("max_tokens", 60),
                           temperature=kw.get("temperature"))
            text = (r.content or "") if r is not None else ""
            leaked = re.search(r'\{"error_code"\s*:\s*(-?\d+)', text)
            if leaked:
                return text, f"leaked-as-text:{leaked.group(1)}", time.time() - t0
            return text, "ok", time.time() - t0
        except Exception as exc:
            return None, f"raised:{type(exc).__name__}", time.time() - t0
        finally:
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass

    if "health" in probes:
        code = afm.Session.check_availability()
        reason = afm.Session.get_availability_reason()
        ready = afm.Session.is_ready()
        txt, status, dt = one("Say the word READY.")
        res["health"] = {"availability_code": code, "reason": reason, "is_ready": ready,
                         "simple_call": status, "seconds": round(dt, 2),
                         "finding": ("availability says OK but generation failed -- session-health "
                                     "poisoning confirmed" if code == 1 and status.startswith(("raised", "leaked"))
                                     else "availability and a real call agree")}

    if "cliff" in probes:
        # Bisection on injected-context length: the budget our whole design is constrained by.
        lo, hi, blow = 1, 0, None
        probe_pts = [3000, 6000, 9000, 12000, 16000]
        curve = []
        for n in probe_pts:
            txt, status, dt = one("Summarise in one line: " + ("filler word " * (n // 12))[:n])
            blew = status == "raised:ContextWindowExceededError"
            curve.append({"chars": n, "status": status, "blew": blew})
            if blew and blow is None:
                blow = n
            if health.feed(status.startswith("brain") or status.startswith("leaked")):
                res["cliff"] = {"aborted": "session contaminated mid-probe", "curve": curve}
                return res
        res["cliff"] = {"first_failure_at": blow, "curve": curve,
                        "note": "points are coarse on purpose; each is one call"}

    if "language" in probes:
        langs = {"ascii_en": "Reply in English: what is a vault?",
                 "devanagari_hi": "केवल दो शब्दों में उत्तर दें: क्या है?",
                 "devanagari_mr": "फक्त दो शब्दांत उत्तर द्या: काय?",
                 "hinglish": "Vault ke baare me do words me batao"}
        out = {}
        for k, q in langs.items():
            txt, status, dt = one(q)
            out[k] = {"status": status, "devanagari_out": bool(txt and re.search(r"[\u0900-\u097F]", txt)),
                      "sample": (txt or "")[:60]}
            health.feed(status.startswith("raised"))
        res["language"] = {"matrix": out,
                           "finding": "UnsupportedLanguageError is nondeterministic: score over "
                                      "many draws, never on one"}

    if "temperature" in probes:
        # The question this harness exists to ask: our code never passes temperature, so every
        # eval draw samples at DEFAULT_TEMPERATURE=1.0 (CREATIVE). Does FACTUAL=0.3 give
        # repeatable answers? If yes, A/B runs need fewer draws.
        q = "One word. Which is larger: 9 or 11?"
        out = {}
        for temp in (1.0, 0.3, 0.1):
            ans = []
            for _ in range(4):
                txt, status, _dt = one(q, temperature=temp, max_tokens=8)
                ans.append((txt or status).strip().lower()[:20])
                health.feed(status.startswith("raised"))
            uniq = len({a for a in ans if a})
            out[str(temp)] = {"answers": ans, "unique": uniq}
        res["temperature"] = {
            "matrix": out,
            "finding": "reproducibility knob: unique-answer count at each temperature. "
                       "lily_ask.py passes no temperature at all (see our_call_audit)."}

    res["_calls_used"] = calls
    return res


# ---------------------------------------------------------------- report
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="afm", choices=sorted(TARGETS))
    ap.add_argument("--phases", default="recon,static",
                    help="add 'dynamic' to spend live AFM calls (budgeted, health-gated)")
    ap.add_argument("--probes", default="health,temperature",
                    help="subset for dynamic: health,cliff,language,temperature")
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--out", default="", help="default: .tmp/recon-<target>-<date>.md")
    args = ap.parse_args()

    t = TARGETS[args.target]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    findings: dict = {"target": args.target, "authorization_basis": t["basis"],
                      "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "phases": phases}
    if "recon" in phases:
        findings["recon"] = phase_recon(t)
    if "static" in phases:
        findings["static"] = phase_static(t)
    if "dynamic" in phases:
        findings["dynamic"] = phase_dynamic(t, [p.strip() for p in args.probes.split(",")],
                                           args.max_calls)

    out = args.out or os.path.join(HERE, ".tmp", f"recon-{args.target}-{time.strftime('%Y%m%d')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(findings, open(out, "w", encoding="utf-8"), indent=2)

    s = findings.get("static", {})
    print(f"target: {args.target}  basis: {t['basis'][:60]}...")
    if "recon" in findings:
        r = findings["recon"]
        print(f"recon: {len(r['files'])} files, {len(r['binaries'])} binaries, "
              f"system framework on disk: {r['system_framework_on_disk']}")
    if s:
        print(f"static: {len(s['public_api'])} public symbols, {len(s['error_codes'])} error codes, "
              f"{len(s['capability_strings'])} capability strings")
        print(f"  defaults: temperature={s['api']['defaults']['temperature']} "
              f"max_tokens={s['api']['defaults']['max_tokens']} presets={s['api']['presets']}")
        print(f"  our code passes: {s['our_call_audit']}")
    if "dynamic" in findings:
        print(json.dumps({k: v for k, v in findings["dynamic"].items()}, indent=2)[:3000])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
