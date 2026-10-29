#!/usr/bin/env python3.12
"""Run every corpus + synthetic problem through the OFFICIAL runner."""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from refs import REFS

RUNNER = "http://runner.riftstack.ai:48701"
REPO = str(Path(__file__).resolve().parents[2])
HSX = Path(__file__).resolve().parents[2] / "tests" / "hsx"
SYN = Path(__file__).resolve().parents[2] / "tests" / "synth"
TST = Path(__file__).resolve().parents[2] / "tests" / "cases"
MARK = Path(__file__).resolve().parents[2] / "tests" / "markers"
MARK.mkdir(exist_ok=True)

SKIP = {"mm64", "mm_rect"}  # TMA — out of scope
NP_DTYPE = {"f32": np.float32, "f64": np.float64, "f16": np.float16,
            "i32": np.int32, "i64": np.int64}


def collect():
    probs = []
    for d in sorted(HSX.iterdir()):
        if d.is_dir() and d.name not in SKIP:
            probs.append((d.name, d))
    for d in sorted(SYN.iterdir()):
        if d.is_dir():
            probs.append((d.name, d))
    for d in sorted(TST.iterdir()):
        if d.is_dir():
            probs.append((d.name, d))
    return probs


def local_values(meta):
    """name -> numpy array for reference computation (seeds mirrored locally)."""
    vals = {}
    for a in meta["kernels"][0]["args"]:
        if a.get("scalar"):
            continue
        if "data" in a:
            vals[a["name"]] = np.array(a["data"], dtype=NP_DTYPE.get(a["dtype"], np.float32))
        elif "seed" in a:
            vals[a["name"]] = np.random.default_rng(a["seed"]).standard_normal(a["shape"]).astype(NP_DTYPE.get(a["dtype"], np.float32))
    return vals


def run_problem(name, d):
    ir = next(d.glob("*.ir.json"))
    meta = json.load(open(d / "meta.json"))
    k = meta["kernels"][0]
    r = subprocess.run([sys.executable, f"{REPO}/compile.py", str(ir)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"compile.py failed: {r.stderr.strip()[-300:]}"
    ptx = r.stdout
    c = requests.post(f"{RUNNER}/compile", json={"ptx": ptx}, timeout=(10, 300)).json()
    if not c.get("ok"):
        return False, f"ptxas rejected: {c.get('stderr', '')[-300:]}"
    if k["entry"] not in c.get("entry_points", []):
        return False, f"entry points {c.get('entry_points')} lack {k['entry']!r}"

    run_args = []
    for a in k["args"]:
        if a.get("scalar"):
            run_args.append({"name": a["name"], "dtype": a["dtype"],
                             "scalar": True, "value": a["value"]})
            continue
        a2 = {"name": a["name"], "dtype": a["dtype"], "shape": a["shape"]}
        if a.get("output"):
            a2["output"] = True
        elif "seed" in a:
            a2["seed"] = a["seed"]
        elif "data" in a:
            a2["data"] = a["data"]
        run_args.append(a2)

    need = max((math.prod(a["shape"]) for a in k["args"] if a.get("output")), default=1)
    r = {}
    for _ in range(40):
        r = requests.post(f"{RUNNER}/run", json={
            "cubin_id": c["cubin_id"], "entry": k["entry"],
            "grid": k["grid"], "block": k["block"], "smem_bytes": k["smem_bytes"],
            "zero_outputs": k["zero_outputs"], "args": run_args,
            "warmup": 0, "iters": 1, "max_return": max(4096, need),
        }, timeout=(10, 300)).json()
        if r.get("stage") != "busy":
            break
        time.sleep(2)
    if not r.get("ok"):
        return False, f"launch failed: {str(r)[:300]}"

    vals = local_values(meta)
    ref = REFS[name](vals)
    out_name = next(a["name"] for a in k["args"] if a.get("output"))
    got = np.array(r["outputs"][out_name]["values"], dtype=np.float64).reshape(ref.shape)
    ref = ref.astype(np.float64)
    ok = np.allclose(got, ref, rtol=meta["rtol"], atol=meta["atol"])
    err = float(np.max(np.abs(got - ref))) if got.size else 0.0
    return ok, f"max abs err {err:.3e} (rtol={meta['rtol']} atol={meta['atol']}, {r.get('median_us', '?')} us)"


def main():
    only = sys.argv[1:] or None
    probs = collect()
    if only:
        probs = [(n, d) for n, d in probs if n in only]
    npass = nfail = 0
    fails = []
    for name, d in probs:
        mark = MARK / f"{name}.done"
        if mark.exists() and not only:
            verdict = mark.read_text().splitlines()[0]
            print(f"[{name}] CACHED {verdict}")
            if verdict.startswith("PASS"):
                npass += 1
            else:
                nfail += 1
                fails.append(name)
            continue
        try:
            ok, msg = run_problem(name, d)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"exception: {type(e).__name__} {e}"
        print(f"[{name}] {'PASS' if ok else 'FAIL'} {msg}", flush=True)
        mark.write_text(("PASS " if ok else "FAIL ") + msg)
        if ok:
            npass += 1
        else:
            nfail += 1
            fails.append(name)
        time.sleep(0.3)
    print(f"== {npass} PASS, {nfail} FAIL; failing: {fails}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
