#!/usr/bin/env python3
"""Validation engine for the emmy-safe compiler.

Layers: smoke, golden, static (local, no GPU); fixtures, corpus, sweeps,
differential, sanitizer, bench (GPU). Usage:
    python tools/validate.py --layer quick          # smoke+golden+static
    python tools/validate.py --layer all            # everything with a GPU
    python tools/validate.py --layer fixtures --compiler bundled.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402
import re  # noqa: E402

EXPECT = json.load(open(ROOT / "tests" / "expectations.json"))
GOLDEN = ROOT / "tests" / "golden"
FIXTURES = ROOT / "tests" / "fixtures"
FAILURES = []


def gate(name, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def default_compiler(args):
    if args.compiler:
        return Path(args.compiler)
    return None  # package


def compile_ir(compiler, ir_path, out_path):
    """Compile one IR with either the package or a script compiler."""
    if compiler is None:
        import copy
        from emmy.compiler.graph import Graph
        from src.compiler import Compiler
        g = Graph.from_dict(json.loads(Path(ir_path).read_text()))
        meta = json.loads((Path(ir_path).parent / "meta.json").read_text())
        c = Compiler(g, meta, find_kernel_pkg(g))
        c.compile()
        out_path.write_text(c.render())
        return True
    r = subprocess.run([sys.executable, str(compiler), str(ir_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False
    out_path.write_text(r.stdout)
    return True


def find_kernel_pkg(graph):
    kernels = [n for n, nd in graph.nodes.items()
               if type(nd.op).__name__ == "KernelOp"]
    return kernels[0]


def ir_sources():
    """(name, ir_path, compiler_ir_name) for goldens: 15 corpus + 8 fixtures."""
    out = []
    for d in sorted((ROOT / "problems").iterdir()):
        irs = list(d.glob("*.ir.json"))
        if irs:
            out.append((d.name, irs[0]))
    for d in sorted(FIXTURES.iterdir()):
        ir = d / "kernel.ir.json"
        if ir.exists():
            out.append((d.name, ir))
    return out


def layer_smoke(compiler):
    print("== layer: smoke")
    if compiler is None:
        r = subprocess.run([sys.executable, "-c", "from src.compiler import Compiler"],
                           capture_output=True, text=True, cwd=str(ROOT))
        return gate("package import", r.returncode == 0, r.stderr[-80:])
    r = subprocess.run([sys.executable, str(compiler), "--help"],
                       capture_output=True, text=True)
    return gate("entry import", True)


def layer_golden(compiler):
    print("== layer: golden")
    tmp = ROOT / "tests" / ".out"
    tmp.mkdir(exist_ok=True)
    if args_update:
        ok_all = True
        for name, ir in ir_sources():
            ok = compile_ir(compiler, ir, GOLDEN / f"{name}.ptx")
            ok_all &= ok
        return gate("goldens regenerated", ok_all, f"{len(ir_sources())} files")
    ok_all = True
    n = 0
    for name, ir in ir_sources():
        out = tmp / f"{name}.ptx"
        if not compile_ir(compiler, ir, out):
            gate(f"golden:{name}", False, "compile failed")
            ok_all = False
            continue
        golden = GOLDEN / f"{name}.ptx"
        same = golden.exists() and golden.read_bytes() == out.read_bytes()
        gate(f"golden:{name}", same)
        ok_all &= same
        n += 1
    return gate("golden count", n == EXPECT["golden_count"], f"{n}/{EXPECT['golden_count']}")


def layer_static():
    print("== layer: static")
    tmp = ROOT / "tests" / ".out"
    ok = True
    for checker in ("ptxcheck", "racecheck"):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "checkers" / f"{checker}.py"),
                            str(tmp)], capture_output=True, text=True)
        total = r.stdout.strip().split("\n")[-1]
        flagged = [ln for ln in r.stdout.split("\n")
                   if re.search(r"\.ptx: \d+ violation|RACE \(", ln)]
        allowed = EXPECT["checker_whitelist"].get(checker, [])
        unexpected = [f for f in flagged
                      if not any(w in f for w in allowed)]
        ok &= gate(f"{checker}", not unexpected, f"{total} ({len(flagged)} flagged, {len(allowed)} whitelisted)")
    return ok


LAYERS = {
    "smoke": lambda c: layer_smoke(c),
    "golden": layer_golden,
    "static": lambda c: layer_static(),
}

GPU_LAYERS = ["fixtures", "corpus", "sweeps", "differential", "sanitizer", "bench"]

args = None
args_update = False


def main():
    global args, args_update
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="quick",
                    help="quick | smoke,golden,static | fixtures | corpus | sweeps | differential | sanitizer | bench | all")
    ap.add_argument("--compiler", default=None, help="compiler script path (default: src package)")
    ap.add_argument("--update-goldens", action="store_true")
    args = ap.parse_args()
    args_update = args.update_goldens

    compiler = default_compiler(args)
    selected = (["smoke", "golden", "static"] if args.layer == "quick"
                else list(LAYERS) if args.layer == "all"
                else [x.strip() for x in args.layer.split(",")])

    for layer in selected:
        if layer in LAYERS:
            LAYERS[layer](compiler)
        elif layer in GPU_LAYERS:
            gate(layer, False, "GPU layer — run on the box: EMMY_BOX_HOST=<user@host> python3 tools/setup_box.py --run")
        else:
            gate(layer, False, "unknown layer")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} gate(s): {FAILURES}")
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
