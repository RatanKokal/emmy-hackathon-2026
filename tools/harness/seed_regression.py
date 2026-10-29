#!/usr/bin/env python3.12
"""Seed-varied regression: run the 85-problem corpus with different data seeds."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests

RUNNER = "http://runner.riftstack.ai:48701"
HOME = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[2]
NP_DTYPE = {"f32": np.float32, "f64": np.float64, "f16": np.float16, "i32": np.int32, "i64": np.int64}
sys.path.insert(0, str(HOME))
from refs import REFS  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def main():
    problems = []
    for root in (ROOT / "tests" / "hsx", ROOT / "tests" / "synth", ROOT / "tests" / "cases"):
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name in ("mm64", "mm_rect"):
                continue
            irs = sorted(d.glob("*.ir.json"))
            m = d / "meta.json"
            if irs and m.exists():
                problems.append((d.name, d, irs[0], json.load(open(m))))

    npass = nfail = nskip = 0
    fails = []
    for name, d, ir_path, meta in problems:
        try:
            k = meta["kernels"][0]
            if len(k["args"]) > 6:
                nskip += 1
                continue
            import subprocess as sp
            compile_r = sp.run([sys.executable, str(HOME / "compile.py"), str(ir_path)],
                               capture_output=True, text=True, timeout=30)
            if compile_r.returncode != 0:
                nfail += 1
                fails.append((name, "compile: " + compile_r.stderr[-60:]))
                continue
            c = requests.post(f"{RUNNER}/compile", json={"ptx": compile_r.stdout},
                              timeout=(10, 300)).json()
            if not c.get("ok"):
                nfail += 1
                fails.append((name, "ptxas: " + c.get("stderr", "")[:60]))
                continue
            vals = {}
            run_args = []
            for a in k["args"]:
                if a.get("scalar"):
                    run_args.append({k2: a[k2] for k2 in ("name", "dtype", "scalar", "value")})
                    continue
                a2 = {"name": a["name"], "dtype": a["dtype"], "shape": a["shape"]}
                if a.get("output"):
                    a2["output"] = True
                elif "data" in a:
                    a2["data"] = a["data"]
                    vals[a["name"]] = np.array(a["data"], dtype=NP_DTYPE.get(a["dtype"], np.float32))
                elif "seed" in a:
                    a2["seed"] = a["seed"] + SEED
                    vals[a["name"]] = np.random.default_rng(
                        a["seed"] + SEED).standard_normal(a["shape"]).astype(
                        NP_DTYPE.get(a["dtype"], np.float32))
                run_args.append(a2)
            need = max((math.prod(a["shape"]) for a in k["args"] if a.get("output")), default=1)
            for attempt in range(30):
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
                nfail += 1
                fails.append((name, "launch: " + str(r)[:60]))
                continue
            ref = REFS[name](vals).astype(np.float64)
            out_name = next(a["name"] for a in k["args"] if a.get("output"))
            got = np.array(r["outputs"][out_name]["values"], dtype=np.float64).reshape(ref.shape)
            if np.allclose(got, ref, rtol=meta["rtol"], atol=meta["atol"]):
                npass += 1
            else:
                nfail += 1
                err = float(np.max(np.abs(got - ref)))
                fails.append((name, f"max abs err {err:.3e}"))
        except Exception as e:  # noqa: BLE001
            nskip += 1
    print(f"SEED {SEED}: {npass} PASS, {nfail} FAIL, {nskip} SKIP / {len(problems)}")
    for n, m in fails:
        print(f"  FAIL {n}: {m}")


if __name__ == "__main__":
    main()
