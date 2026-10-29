#!/usr/bin/env python3.12
"""Parallel regression: persistent-process compile (emmy imported once) +
threaded runner phase."""
import concurrent.futures as cf
import importlib.util
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from refs import REFS  # noqa: E402

RUNNER = "http://runner.riftstack.ai:48701"
HOME = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[2]
NP_DTYPE = {"f32": np.float32, "f64": np.float64, "f16": np.float16,
            "i32": np.int32, "i64": np.int64}
SKIP = {"mm64", "mm_rect"}

spec = importlib.util.spec_from_file_location("cc", str(HOME / "compile.py"))
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

print_lock = threading.Lock()
RESULTS = {}


def compile_all(problems):
    """Phase 1: persistent process — emmy imported once, all kernels compiled."""
    t0 = time.perf_counter()
    out = {}
    for name, d, ir_path, meta in problems:
        try:
            graph = cc.Graph.from_dict(json.load(open(ir_path)))
            kernel = cc.find_kernel(graph)
            comp = cc.Compiler(graph, meta, kernel)
            comp.compile()
            out[name] = (comp.render(), None)
        except Exception as e:  # noqa: BLE001
            out[name] = (None, f"{type(e).__name__}: {str(e)[:150]}")
    print(f"[compile phase] {len(out)} kernels in {time.perf_counter() - t0:.1f}s "
          f"({(time.perf_counter() - t0) / max(len(out), 1) * 1000:.0f} ms/kernel)",
          flush=True)
    return out


def run_one(name, d, meta, ptx, cerr):
    k = meta["kernels"][0]
    if ptx is None:
        return name, False, f"compile failed: {cerr}"
    c = requests.post(f"{RUNNER}/compile", json={"ptx": ptx}, timeout=(10, 300)).json()
    if not c.get("ok"):
        return name, False, f"ptxas: {c.get('stderr', '')[:120]}"
    if k["entry"] not in c.get("entry_points", []):
        return name, False, f"entry {c.get('entry_points')} != {k['entry']}"

    vals = {}
    run_args = []
    for a in k["args"]:
        if a.get("scalar"):
            run_args.append({"name": a["name"], "dtype": a["dtype"], "scalar": True,
                             "value": a["value"]})
            continue
        a2 = {"name": a["name"], "dtype": a["dtype"], "shape": a["shape"]}
        if a.get("output"):
            a2["output"] = True
        elif "data" in a:
            a2["data"] = a["data"]
            vals[a["name"]] = np.array(a["data"], dtype=NP_DTYPE.get(a["dtype"], np.float32))
        elif "seed" in a:
            a2["seed"] = a["seed"]
            vals[a["name"]] = np.random.default_rng(a["seed"]).standard_normal(
                a["shape"]).astype(NP_DTYPE.get(a["dtype"], np.float32))
        run_args.append(a2)

    need = max((math.prod(a["shape"]) for a in k["args"] if a.get("output")), default=1)
    r = {}
    for _ in range(40):
        r = requests.post(f"{RUNNER}/run", json={
            "cubin_id": c["cubin_id"], "entry": k["entry"], "grid": k["grid"],
            "block": k["block"], "smem_bytes": k["smem_bytes"],
            "zero_outputs": k["zero_outputs"], "args": run_args,
            "warmup": 0, "iters": 1, "max_return": max(4096, need)},
            timeout=(10, 300)).json()
        if r.get("stage") != "busy":
            break
        time.sleep(2)
    if not r.get("ok"):
        return name, False, f"launch: {str(r)[:120]}"

    ref = REFS[name](vals).astype(np.float64)
    out_name = next(a["name"] for a in k["args"] if a.get("output"))
    got = np.array(r["outputs"][out_name]["values"], dtype=np.float64).reshape(ref.shape)
    ok = bool(np.allclose(got, ref, rtol=meta["rtol"], atol=meta["atol"]))
    err = float(np.max(np.abs(got - ref))) if got.size else 0.0
    return name, ok, f"max abs err {err:.3e} ({r.get('median_us', '?')} us)"


def main():
    problems = []
    for root, kind in ((ROOT / "tests" / "hsx", "hsx"), (ROOT / "tests" / "synth", "synth"), (ROOT / "tests" / "cases", "test")):
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name in SKIP:
                continue
            irs = sorted(d.glob("*.ir.json"))
            m = d / "meta.json"
            if irs and m.exists():
                problems.append((d.name, d, irs[0], json.load(open(m))))
    print(f"[collect] {len(problems)} problems", flush=True)

    ptxs = compile_all(problems)

    t0 = time.perf_counter()
    npass = nfail = 0
    fails = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_one, name, d, meta, ptxs[name][0], ptxs[name][1]):
                (name, d, meta) for name, d, ir, meta in problems}
        for fut in cf.as_completed(futs):
            name, ok, msg = fut.result()
            tag = "PASS" if ok else "FAIL"
            with print_lock:
                print(f"[{name}] {tag} {msg}", flush=True)
            if ok:
                npass += 1
            else:
                nfail += 1
                fails.append(name)
    dt = time.perf_counter() - t0
    print(f"== {npass} PASS, {nfail} FAIL in {dt:.0f}s "
          f"({dt / len(problems) * 1000:.0f} ms/problem); failing: {fails}", flush=True)


if __name__ == "__main__":
    main()
