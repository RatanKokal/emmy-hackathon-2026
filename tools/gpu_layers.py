#!/usr/bin/env python3
"""GPU validation layers: fixtures, corpus, sweeps, differential, sanitizer.

Run on the box from the synced repo root. Gates come from tests/expectations.json.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECT = json.load(open(ROOT / "tests" / "expectations.json"))
FAILURES = []


def gate(name, ok, detail=""):
    ok = bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def layer_fixtures():
    print("== layer: fixtures")
    tmp = ROOT / "tests" / ".gpu"
    tmp.mkdir(exist_ok=True)
    ok_all = True
    for d in sorted((ROOT / "tests" / "fixtures").iterdir()):
        ir = d / "kernel.ir.json"
        if not ir.exists():
            continue
        work = tmp / d.name          # isolated root so one crash can't kill others
        kdir = work / d.name
        kdir.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.name != "kernel.ir.json":
                (kdir / f.name).write_bytes(f.read_bytes())
        r = run(f"{sys.executable} compile.py {ir}")
        (kdir / "k.ptx").write_text(r.stdout)
        r = run(f"{sys.executable} tools/harness/gpu_run.py {work}")
        got = kdir / "got.bin"
        if not got.exists():
            import re as _re
            cu = _re.search(r"CUresult \d+", r.stderr or "")
            gate(f"fixture:{d.name}", False, f"no output ({cu.group(0) if cu else 'crash'})")
            ok_all = False
            continue
        r = run(f"{sys.executable} tools/harness/gpu_check.py {work}")
        residue = EXPECT["fixtures"].get(d.name, "")
        m = re.search(r"bad (\d+)/", r.stdout)
        bad_n = int(m.group(1)) if m else 0
        if "PASS" in r.stdout:
            gate(f"fixture:{d.name}", True)
        elif "residual_bad=" in residue:
            want = int(residue.split("residual_bad=")[1].split()[0])
            gate(f"fixture:{d.name}", bad_n == want, f"residual {bad_n}/{want}")
        else:
            gate(f"fixture:{d.name}", False, r.stdout.strip().split("\n")[0][:60])
    return ok_all


def layer_corpus():
    print("== layer: corpus (official runner)")
    ok = True
    for d in sorted(p for p in (ROOT / "problems").iterdir() if p.is_dir()):
        r = run(f"{sys.executable} submit.py {d} --compiler "
                f"{ROOT}/compile.py")
        passes = r.stdout.count("PASS")
        gate(f"official:{d.name}", passes >= 1 and "FAIL" not in r.stdout)
    r = run(f"{sys.executable} tools/harness/regression.py", cwd=str(ROOT), timeout=600)
    m = [ln for ln in r.stdout.split("\n") if "== " in ln and "PASS" in ln]
    exp = EXPECT["regression"]
    ok &= gate("regression", bool(m) and f"{exp['pass']} PASS" in m[0] and
               all(n in r.stdout for n in exp["fail_named"]), m[0][:70] if m else r.stderr[:70])
    return ok


def layer_sweeps():
    print("== layer: sweeps")
    ok = True
    r = run(f"{sys.executable} tools/harness/semantic_sweep.py", cwd=str(ROOT), timeout=600)
    exp = EXPECT["semantic"]
    ok &= gate("semantic", f"{exp['ok']} OK, 0 BAD" in r.stdout, r.stdout.strip().split("\n")[-1][:60])
    r = run(f"{sys.executable} tools/harness/op_matrix.py compile.py", cwd=str(ROOT), timeout=600)
    exp = EXPECT["op_matrix"]
    emitted = f"emitted {exp['emitted']} / {exp['total']}" in r.stdout
    bad = 0
    for f in (ROOT / "sweep_out").glob("*.ptx"):
        rr = run(f"ptxas -arch=sm_89 -o /dev/null {f}")
        if rr.returncode != 0:
            bad += 1
    ok &= gate("op_matrix", emitted and bad == exp["ptxas_bad"], f"emitted={emitted}, ptxas_bad={bad}")
    r = run(f"{sys.executable} tools/harness/statement_sweep.py", cwd=str(ROOT), timeout=900)
    n = len(list((ROOT / "tests" / ".stmt_sweep").glob("*/*.ir.json")))
    ok &= gate("statement_sweep", n == EXPECT["statement_sweep"]["built"], f"{n} problems")
    for s in (1, 2, 3):
        r = run(f"{sys.executable} tools/harness/seed_regression.py {s}", cwd=str(ROOT), timeout=900)
        exp = EXPECT["seeds"]
        ok &= gate(f"seed{s}", f"{exp['pass']} PASS, {exp['fail']} FAIL, {exp['skip']} SKIP / {exp['total']}" in r.stdout,
                   r.stdout.strip().split("\n")[-1][:60])
    return ok


def layer_differential():
    print("== layer: differential")
    r = run(f"{sys.executable} tools/harness/differential.py tests/hsx tests/synth tests/cases",
            cwd=str(ROOT), timeout=1200)
    exp = EXPECT["differential"]
    counts = {v: r.stdout.count(f" {k}  ") for k, v in
              (("OK", "ok"), ("DIFF", "diff"), ("ERROR", "error"))}
    ok = (counts["ok"] == exp["ok"] and counts["diff"] == exp["diff"]
          and counts["error"] == exp["error"])
    return gate("differential", ok, f"OK={counts['ok']} DIFF={counts['diff']} ERR={counts['error']} (expect {exp['ok']}/{exp['diff']}/{exp['error']})")


def layer_sanitizer():
    print("== layer: sanitizer")
    import os
    san = os.environ.get(
        "EMMY_SANITIZER",
        str(ROOT / ".sanitizer" / "root" / "usr" / "local" / "cuda-13.4" / "bin" / "compute-sanitizer"))
    if not Path(san).exists() and run(f"command -v compute-sanitizer").returncode != 0:
        return gate("sanitizer", True, "SKIPPED — no compute-sanitizer (run tools/setup_box.py)")
    bad = 0
    n = 0
    tmp = ROOT / "tests" / ".gpu"
    for d in sorted((ROOT / "tests" / "fixtures").iterdir()):
        ir = d / "kernel.ir.json"
        if not ir.exists():
            continue
        kdir = tmp / d.name / d.name
        if not (kdir / "k.ptx").exists():
            r = run(f"{sys.executable} compile.py {ir}")
            kdir.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.name != "kernel.ir.json":
                    (kdir / f.name).write_bytes(f.read_bytes())
            (kdir / "k.ptx").write_text(r.stdout)
        n += 1
        r = run(f'"{san}" --tool=memcheck --error-exitcode=9 '
                f"{sys.executable} tools/harness/gpu_run.py {tmp / d.name}")
        if "ERROR SUMMARY: 0 errors" not in r.stdout:
            bad += 1
            lines = [ln for ln in (r.stdout + r.stderr).split("\n") if ln.strip()]
            gate(f"sanitizer:{d.name}", False, lines[-1][:80] if lines else "no output")
    return gate("sanitizer", bad == 0, f"{n} fixtures memchecked, {bad} flagged")


LAYERS = {"fixtures": layer_fixtures, "corpus": layer_corpus, "sweeps": layer_sweeps,
          "differential": layer_differential, "sanitizer": layer_sanitizer}


def main():
    layers = sys.argv[1:] or list(LAYERS)
    for layer in layers:
        if layer in LAYERS:
            LAYERS[layer]()
        else:
            gate(layer, False, "unknown")
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} gate(s): {FAILURES}")
        return 1
    print("all GPU gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
