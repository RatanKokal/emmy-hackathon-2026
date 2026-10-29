#!/usr/bin/env python3.12
"""Semantic sweep v3: harness bugs fixed (refs, tolerance, op mapping)."""
import concurrent.futures as cf
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests

HOME = Path(__file__).resolve().parents[2]
RUNNER = "http://runner.riftstack.ai:48701"
NP_DTYPE = {"f32": np.float32, "f64": np.float64, "f16": np.float16,
            "i32": np.int32, "i64": np.int64}
N = 64

spec = importlib.util.spec_from_file_location("cc", str(HOME / "compile.py"))
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from emmy.compiler.graph import Graph  # noqa: E402
from emmy.compiler.tensor import Tensor  # noqa: E402
from emmy.compiler.ir.kernel.ir import KernelOp, Tile, Axis, Assign, Load, Write  # noqa: E402
from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: E402
from emmy.compiler.ir.expr import Var  # noqa: E402
from emmy.compiler.ir.atom import DataType  # noqa: E402
from emmy.compiler.ir.axis import Dim  # noqa: E402
from emmy.compiler.ir.base import InputOp  # noqa: E402

_erf = np.vectorize(math.erf)


def _sig(x):
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


F = ("f32", "f16", "f64")
I = ("i32", "i64")
OPS = {
    "add":        (2, lambda a: a[0] + a[1], 0, 0, "f", F + I),
    "subtract":   (2, lambda a: a[0] - a[1], 0, 0, "f", F + I),
    "multiply":   (2, lambda a: a[0] * a[1], 0, 0, "f", F + I),
    "divide":     (2, lambda a: a[0] / a[1], 0, 0, "fnz", F),
    "maximum":    (2, lambda a: np.maximum(a[0], a[1]), 0, 0, "f", F + I),
    "minimum":    (2, lambda a: np.minimum(a[0], a[1]), 0, 0, "f", F + I),
    "neg":        (1, lambda a: -a[0], 0, 0, "f", F + I),
    "negative":   (1, lambda a: -a[0], 0, 0, "f", F + I),
    "abs":        (1, lambda a: np.abs(a[0]), 0, 0, "f", F + I),
    "copy":       (1, lambda a: a[0], 0, 0, "f", F + I),
    "square":     (1, lambda a: a[0] * a[0], 0, 0, "f", F + I),
    "sqrt":       (1, lambda a: np.sqrt(np.abs(a[0])), 0, 0, "abs", F),
    "rsqrt":      (1, lambda a: 1.0 / np.sqrt(np.abs(a[0])), 2e-5, 2e-5, "abs", F),
    "reciprocal": (1, lambda a: 1.0 / a[0], 0, 0, "fnz", F),
    "exp":        (1, lambda a: np.exp(np.clip(a[0], -80, 80)), 2e-5, 2e-5, "f", F),
    "exp2":       (1, lambda a: np.exp2(np.clip(a[0], -120, 120)), 2e-5, 2e-5, "f", F),
    "log":        (1, lambda a: np.log(np.abs(a[0])), 2e-5, 2e-5, "abs", F),
    "log2":       (1, lambda a: np.log2(np.abs(a[0])), 2e-5, 2e-5, "abs", F),
    "log1p":      (1, lambda a: np.log1p(np.abs(a[0])), 2e-5, 2e-5, "abs", F),
    "sin":        (1, lambda a: np.sin(a[0]), 1e-4, 1e-4, "f", F),
    "cos":        (1, lambda a: np.cos(a[0]), 1e-4, 1e-4, "f", F),
    "tanh":       (1, lambda a: np.tanh(a[0]), 1e-5, 1e-5, "f", F),
    "sigmoid":    (1, lambda a: _sig(a[0]), 2e-5, 2e-5, "f", F),
    "silu":       (1, lambda a: a[0] * _sig(a[0]), 2e-5, 2e-5, "f", F),
    "softplus":   (1, lambda a: np.log1p(np.exp(-np.abs(a[0].astype(np.float64)))) + np.maximum(a[0], 0), 2e-5, 2e-5, "f", F),
    "erf":        (1, lambda a: _erf(a[0]), 1e-5, 1e-5, "f", F),
    "pow":        (2, lambda a: np.power(a[0], 3.0), 2e-4, 2e-4, "pow", F),
    "floor":      (1, lambda a: np.floor(a[0]), 0, 0, "f", F),
    "ceil":       (1, lambda a: np.ceil(a[0]), 0, 0, "f", F),
    "round":      (1, lambda a: np.rint(a[0]), 0, 0, "f", F),
    "trunc":      (1, lambda a: np.trunc(a[0]), 0, 0, "f", F),
    "equal":         (2, lambda a: (a[0] == a[1]).astype(a[0].dtype), 0, 0, "f", F + I),
    "not_equal":     (2, lambda a: (a[0] != a[1]).astype(a[0].dtype), 0, 0, "f", F + I),
    "greater":       (2, lambda a: (a[0] > a[1]).astype(a[0].dtype), 0, 0, "f", F + I),
    "less":          (2, lambda a: (a[0] < a[1]).astype(a[0].dtype), 0, 0, "f", F + I),
    "greater_equal": (2, lambda a: (a[0] >= a[1]).astype(a[0].dtype), 0, 0, "f", F + I),
    "less_equal":    (2, lambda a: (a[0] <= a[1]).astype(a[0].dtype), 0, 0, "f", F + I),
    "where":      (3, lambda a: np.where(a[0] != 0, a[1], a[2]), 0, 0, "f", F + I),
    "sign":       (1, lambda a: np.sign(a[0]), 0, 0, "f", F),
    "fma":        (3, lambda a: a[0] * a[1] + a[2], 0, 0, "f", F),
    "mul_add":    (3, lambda a: a[0] * a[1] + a[2], 0, 0, "f", F),
    "left_shift":   (2, lambda a: a[0] << np.clip(a[1], 0, 30).astype("int32"), 0, 0, "shift", I),
    "right_shift":  (2, lambda a: a[0] >> np.clip(a[1], 0, 30).astype("int32"), 0, 0, "shift", I),
    "bitwise_and":  (2, lambda a: a[0] & a[1], 0, 0, "i", I),
    "bitwise_or":   (2, lambda a: a[0] | a[1], 0, 0, "i", I),
    "bitwise_xor":  (2, lambda a: a[0] ^ a[1], 0, 0, "i", I),
    "bitwise_not":  (1, lambda a: ~a[0], 0, 0, "i", I),
    "logical_and":  (2, lambda a: ((a[0] != 0) & (a[1] != 0)).astype(a[0].dtype), 0, 0, "i", I),
    "logical_or":   (2, lambda a: ((a[0] != 0) | (a[1] != 0)).astype(a[0].dtype), 0, 0, "i", I),
    "logical_not":  (1, lambda a: (a[0] == 0).astype(a[0].dtype), 0, 0, "i", I),
    "bitwise_count": (1, lambda a: np.array([bin(int(v) & (2 ** 64 - 1 if "64" in a[0].dtype.name else 2 ** 32 - 1)).count("1")
                                             for v in a[0].ravel()], dtype=a[0].dtype).reshape(a[0].shape), 0, 0, "i", I),
    "remainder":  (2, lambda a: np.fmod(a[0], a[1]), 0, 0, "fnz", F),
    "mod":        (2, lambda a: np.fmod(a[0], a[1]), 0, 0, "fnz", F),
    "fmod":       (2, lambda a: np.fmod(a[0], a[1]), 0, 0, "fnz", F),
    "floor_divide": (2, lambda a: np.floor_divide(a[0], a[1]), 0, 0, "fnz", F),
    # int remainder/mod/fmod: C % semantics (sign of dividend)
    "remainder_int": (2, lambda a: (a[0] - a[1] * np.trunc(a[0].astype(np.float64) / a[1].astype(np.float64))).astype(a[0].dtype), 0, 0, "inz", I),
    "mod_int":       (2, lambda a: (a[0] - a[1] * np.trunc(a[0].astype(np.float64) / a[1].astype(np.float64))).astype(a[0].dtype), 0, 0, "inz", I),
}
INT_OPS = {"left_shift", "right_shift", "bitwise_and", "bitwise_or", "bitwise_xor",
           "bitwise_not", "logical_and", "logical_or", "logical_not", "bitwise_count"}
I32_REMAP = {"remainder": "remainder", "mod": "fmod", "fmod": "fmod"}


def gen_data(kind, dt, rng):
    d = NP_DTYPE[dt]
    if kind == "abs":
        return np.abs(rng.standard_normal(N).astype(d)) + 0.25
    if kind == "fnz":
        v = rng.standard_normal(N).astype(d)
        v[v == 0] = 1.0
        return v
    if kind == "shift":
        return rng.integers(0, 30, N).astype(d)
    if kind == "pow":
        return np.full(N, 3.0).astype(d)
    if kind == "i":
        v = rng.integers(-100, 100, N).astype(d)
        v[v == 0] = 3
        return v
    return rng.standard_normal(N).astype(d)


def build_graph(op, dt, arity):
    D = DataType(name=dt, np=NP_DTYPE[dt], nbytes=np.dtype(NP_DTYPE[dt]).itemsize)
    g = Graph()
    ax = Axis(name="i", extent=Dim(N))
    bufs = [("x", D)] + ([("y", D)] if arity >= 2 else []) + ([("z", D)] if arity == 3 else [])
    for bn, bd in bufs:
        g.add_node(InputOp(), inputs=[], outputs=[Tensor(name=bn, shape=(Dim(N),), dtype=bd)],
                   node_id=bn)
    out_t = Tensor(name="out", shape=(Dim(N),), dtype=D)
    loads = [Load(names=(f"a{i}",), input=bn, index=(Var(name="i"),), dtype=bd)
             for i, (bn, bd) in enumerate(bufs)]
    args = tuple(f"a{i}" for i in range(arity))
    stmts = loads
    if op in ("fma", "mul_add"):
        stmts.append(Assign(name="m", op=ElementwiseImpl("multiply"),
                            args=(args[0], args[1]), dtype=D))
        stmts.append(Assign(name="v", op=ElementwiseImpl("add"),
                            args=("m", args[2]), dtype=D))
    elif op == "frac":
        stmts.append(Assign(name="t", op=ElementwiseImpl("trunc"), args=(args[0],), dtype=D))
        stmts.append(Assign(name="v", op=ElementwiseImpl("sub"), args=(args[0], "t"), dtype=D))
    elif dt in ("i32", "i64") and op in ("remainder", "mod", "fmod"):
        # emmy renders int % through these names; F32_ONLY currently blocks them
        stmts.append(Assign(name="v", op=ElementwiseImpl(op), args=args, dtype=D))
    else:
        emmy_op = "negative" if op == "neg" else op
        stmts.append(Assign(name="v", op=ElementwiseImpl(emmy_op), args=args, dtype=D))
    stmts.append(Write(output="out", index=(Var(name="i"),), values=("v",), value_dtype=D,
                       atomic=False, swizzle="NONE"))
    tile = Tile(axes=(ax,), body=tuple(stmts), block_threads=None)
    kop = KernelOp(name=f"k_{op}_{dt}", body=type(tile.body)([tile]))
    g.add_node(kop, inputs=[bn for bn, _ in bufs], outputs=[out_t], node_id="k")
    return g


def compile_graph(g, meta):
    kernel = cc.find_kernel(g)
    comp = cc.Compiler(g, meta, kernel)
    comp.compile()
    return comp.render()


def run_cell(name, meta, ptx, feeds, ref, rtol, atol):
    r = requests.post(f"{RUNNER}/compile", json={"ptx": ptx}, timeout=(10, 300)).json()
    if not r.get("ok"):
        return "PTXAS", r.get("stderr", "")[:80]
    k = meta["kernels"][0]
    run_args = []
    for a in k["args"]:
        if a.get("output"):
            run_args.append({"name": a["name"], "dtype": a["dtype"], "shape": a["shape"],
                             "output": True})
        else:
            run_args.append({"name": a["name"], "dtype": a["dtype"], "shape": a["shape"],
                             "data": np.asarray(feeds[a["name"]]).ravel().tolist()})
    need = max((math.prod(a["shape"]) for a in k["args"] if a.get("output")), default=1)
    for _ in range(30):
        r = requests.post(f"{RUNNER}/run", json={
            "cubin_id": r["cubin_id"], "entry": k["entry"], "grid": k["grid"],
            "block": k["block"], "smem_bytes": k["smem_bytes"], "zero_outputs": [],
            "args": run_args, "warmup": 0, "iters": 1, "max_return": max(64, need)},
            timeout=(10, 300)).json()
        if r.get("stage") != "busy":
            break
        time.sleep(2)
    if not r.get("ok"):
        return "RUNFAIL", str(r.get("stderr", ""))[:80]
    out_name = next(a["name"] for a in k["args"] if a.get("output"))
    got = np.array(r["outputs"][out_name]["values"], dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if got.shape != ref.shape:
        return "SHAPE", f"{got.shape} vs {ref.shape}"
    bad = ~np.isclose(got, ref, rtol=max(rtol, 1e-7), atol=max(atol, 1e-7), equal_nan=True)
    if not bad.any():
        return "OK", ""
    i = int(np.flatnonzero(bad)[0])
    return "WRONG", f"got={got[i]:.6g} ref={ref[i]:.6g} at [{i}] ({bad.sum()} bad)"


def quantize(ref, dt):
    """f16 outputs are f16-quantized by the kernel; compare at f16 precision."""
    if dt == "f16":
        return ref.astype(np.float16).astype(np.float64)
    return np.asarray(ref, dtype=np.float64)


def cell_tol(dt, rtol, atol):
    if dt == "f16":
        return max(rtol, 5e-4), max(atol, 4e-3)
    return rtol, atol


def work(cell):
    op, dt, arity, ref, rtol, atol, kind = cell
    name = f"{op}_{dt}"
    feeds = {}
    for i in range(arity):
        bn = ("x", "y", "z")[i]
        feeds[bn] = gen_data(kind, dt, RNG)
    try:
        g = build_graph(op, dt, arity)
        meta = {"id": name, "rtol": rtol, "atol": atol,
                "kernels": [{"node": "k", "entry": f"k_{op}_{dt}",
                             "arg_order": list(feeds) + ["out"],
                             "grid": [2, 1, 1], "block": [32, 1, 1], "smem_bytes": 0,
                             "zero_outputs": [], "final": True,
                             "args": [{"name": bn, "dtype": dt, "shape": [N],
                                       "data": np.asarray(v).ravel().tolist()}
                                      for bn, v in feeds.items()]
                                     + [{"name": "out", "dtype": dt, "shape": [N],
                                         "output": True}]}]}
        ptx = compile_graph(g, meta)
        rref = ref([feeds["x"], feeds.get("y"), feeds.get("z")][:arity])
        rref = quantize(rref, dt)
        rt2, at2 = cell_tol(dt, rtol, atol)
        return run_cell(name, meta, ptx, feeds, rref, rt2, at2)
    except Exception as e:  # noqa: BLE001
        return "ERR", f"{type(e).__name__}: {str(e)[:90]}"


RNG = np.random.default_rng(101)


def main():
    cells = []
    for op, (arity, ref, rtol, atol, kind, dts) in OPS.items():
        for dt in dts:
            if op.endswith("_int"):
                continue
            if dt == "f64" and op in ("exp", "exp_fast", "exp2", "log", "log2", "log1p",
                                       "sigmoid", "silu", "softplus", "tanh", "erf",
                                       "pow", "sin", "cos"):
                continue  # designed F32_ONLY raises
            if op in ("remainder", "mod", "fmod") and dt in I:
                key = op + "_int"
                a2, r2, rt2, at2, k2, d2 = OPS[key]
                cells.append((op, dt, a2, r2, rt2, at2, k2))
                continue
            cells.append((op, dt, arity, ref, rtol, atol, kind))
    print(f"[semantic sweep v3] {len(cells)} cells", flush=True)

    npass = nbad = 0
    bad_rows = []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(work, c): c for c in cells}
        for fut in cf.as_completed(futs):
            op, dt = futs[fut][0], futs[fut][1]
            status, msg = fut.result()
            if status == "OK":
                npass += 1
                print(f"[{op}_{dt}] OK", flush=True)
            else:
                nbad += 1
                bad_rows.append((f"{op}_{dt}", status, msg))
                print(f"[{op}_{dt}] {status} {msg}", flush=True)
    print(f"== SEMANTIC SWEEP v3: {npass} OK, {nbad} BAD "
          f"in {time.perf_counter() - t0:.0f}s", flush=True)
    for n, s, m in sorted(bad_rows):
        print(f"  {s:8s} {n}: {m}", flush=True)


if __name__ == "__main__":
    main()
