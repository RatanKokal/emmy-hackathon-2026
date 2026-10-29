#!/usr/bin/env python3
"""Tier 2b: emmy-compiler differential + speed baseline on the H200."""
import copy
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.render import render_kernelop
from emmy.compiler.tensor import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from np_dtypes import NP_DTYPE  # noqa: E402

import ctypes  # noqa: E402

cu = ctypes.CDLL("libcuda.so.1")


def ck(rv, what=""):
    if rv != 0:
        raise RuntimeError(f"{what}: CUresult {rv}")


def render_ref(ir_path, out_cu):
    g = Graph.from_dict(json.load(open(ir_path)))
    kn = [n for n, nd in g.nodes.items() if type(nd.op).__name__ == "KernelOp"][0]
    op = g.nodes[kn].op
    tensors = {}
    for node in g.nodes.values():
        for t in node.outputs:
            tensors[t.name] = t
    cu_src = render_kernelop(op, tensors=tensors)
    Path(out_cu).write_text(cu_src)
    # ref param order: emmy derives from the op
    ins = op.inputs.values() if hasattr(op.inputs, "values") else op.inputs
    outs = op.outputs.values() if hasattr(op.outputs, "values") else op.outputs
    return [t.name for t in ins] + [t.name for t in outs], op.name or kn


def build_inputs(meta):
    k = meta["kernels"][0]
    arrays = {}
    for a in k["args"]:
        if a.get("scalar"):
            continue
        n = 1
        for e in a["shape"]:
            n *= int(e)
        if "data" in a:
            arr = np.array(a["data"], dtype=NP_DTYPE.get(a["dtype"], np.float32))
            arr = (arr * np.ones(n, dtype=arr.dtype))[:n].reshape(a["shape"])
        else:
            arr = np.random.default_rng(a.get("seed", 0)).standard_normal(a["shape"])
            arr = arr.astype(NP_DTYPE.get(a["dtype"], np.float32))
        arrays[a["name"]] = arr
    return arrays


class Kernel:
    def __init__(self, image, entry, is_ptx):
        ck(cu.cuInit(0), "cuInit")
        self.image = image
        mod = ctypes.c_void_p()
        ck(cu.cuModuleLoadData(ctypes.byref(mod), image), "load")
        self.mod = mod
        fn = ctypes.c_void_p()
        ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, entry.encode()), f"fn {entry}")
        self.fn = fn

    def run(self, ptrs_order, grid, block, smem, out_name, out_bytes, zero_first):
        if zero_first:
            pass
        args = (ctypes.c_uint64 * len(ptrs_order))(*ptrs_order)
        params = (ctypes.c_void_p * len(ptrs_order))(
            *[ctypes.cast(ctypes.byref(args, 8 * i), ctypes.c_void_p) for i in range(len(ptrs_order))]
        )
        gx, gy, gz = grid
        bx, by, bz = block
        if smem > 49152:
            ck(cu.cuFuncSetAttribute(self.fn, 8, smem), "smem attr")
        ck(cu.cuLaunchKernel(self.fn, gx, gy, gz, bx, by, bz, smem, None, params, None), "launch")
        ck(cu.cuCtxSynchronize(), "sync")


def main(roots):
    ck(cu.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "dev")
    ctx = ctypes.c_void_p()
    ck(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "ctx")

    rows = []
    for root in roots:
        for d in sorted(Path(root).iterdir()):
            if not d.is_dir():
                continue
            irs = sorted(d.glob("*.ir.json"))
            metas = d / "meta.json"
            if not irs or not metas.exists():
                continue
            name = d.name
            try:
                rows.append(process(name, d, irs[0], json.load(open(metas))))
            except Exception as e:  # noqa: BLE001
                rows.append((name, "ERR", str(e)[:120], "", "", ""))
    print(f"{'problem':16s} {'match':6s} {'maxdiff':>10s} {'ours-us':>9s} {'emmy-us':>9s} {'ratio':>6s}")
    for r in rows:
        if r[1] == "ERR":
            print(f"{r[0]:16s} ERROR  {r[2]}")
        else:
            print(f"{r[0]:16s} {r[1]:6s} {r[2]:>10.2e} {r[3]:>9.1f} {r[4]:>9.1f} {r[5]:>6.2f}")


def process(name, d, ir_path, meta):
    k = meta["kernels"][0]
    if len(k["args"]) > 0 and any(a.get("intermediate") for a in k["args"]):
        return (name, "SKIP", "multi-kernel", 0, 0, 0)

    # --- ours
    r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir_path)],
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        return (name, "ERR", "compile.py: " + r.stderr.strip()[-80:], 0, 0, 0)
    our_ptx = r.stdout.encode()

    # --- ref: emmy render + nvcc
    cu_dir = d / "_ref"
    cu_dir.mkdir(exist_ok=True)
    ref_order, entry = render_ref(ir_path, cu_dir / "ref.cu")
    nb = subprocess.run(["nvcc", "-arch=sm_90", "-cubin", "-o", str(cu_dir / "ref.cubin"),
                         str(cu_dir / "ref.cu")], capture_output=True, text=True)
    if nb.returncode != 0:
        return (name, "ERR", "nvcc: " + nb.stderr.strip()[-80:], 0, 0, 0)
    ref_cubin = (cu_dir / "ref.cubin").read_bytes()

    # --- buffers
    arrays = build_inputs(meta)
    ptrs_meta, ptrs_ref = {}, {}
    outs = {}
    for a in k["args"]:
        n = 1
        for e in a["shape"]:
            n *= int(e)
        nb_bytes = n * np.dtype(NP_DTYPE.get(a["dtype"], np.float32)).itemsize
        p = ctypes.c_ulonglong()
        ck(cu.cuMemAlloc_v2(ctypes.byref(p), max(nb_bytes, 4)), "alloc")
        ptrs_meta[a["name"]] = p.value
        ptrs_ref[a["name"]] = p.value  # same buffers for both kernels
        if a.get("output"):
            outs[a["name"]] = (n, nb_bytes, a["dtype"], p.value)
        else:
            host = arrays[a["name"]]
            ck(cu.cuMemcpyHtoD_v2(ctypes.c_void_p(p.value), host.ctypes.data_as(ctypes.c_void_p),
                                  host.nbytes), "HtoD")

    our_kernel = Kernel(our_ptx, k["entry"], True)
    ref_kernel = Kernel(ref_cubin, entry, False)

    def run_one(kernel, order, zero):
        if zero:
            for on, (n, nb, dt, p) in outs.items():
                ck(cu.cuMemsetD32_v2(ctypes.c_void_p(p), 0, max(nb // 4, 1)), "memset")
        kernel.run([ptrs_meta[n] for n in order], k["grid"], k["block"], k["smem_bytes"],
                   None, None, False)

    # correctness: ours vs emmy's own compiled kernel
    run_one(ref_kernel, ref_order, True)
    ref_out = {}
    for on, (n, nb, dt, p) in outs.items():
        host = np.zeros(n, dtype=NP_DTYPE.get(dt, np.float32))
        ck(cu.cuMemcpyDtoH_v2(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(p), nb), "DtoH")
        ref_out[on] = host
    run_one(our_kernel, k["arg_order"], True)
    our_out = {}
    for on, (n, nb, dt, p) in outs.items():
        host = np.zeros(n, dtype=NP_DTYPE.get(dt, np.float32))
        ck(cu.cuMemcpyDtoH_v2(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(p), nb), "DtoH")
        our_out[on] = host

    match, maxdiff = True, 0.0
    for on in outs:
        a, b = our_out[on].astype(np.float64), ref_out[on].astype(np.float64)
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        maxdiff = max(maxdiff, d)
        tol = meta.get("rtol", 1e-5) * max(1.0, float(np.max(np.abs(b)))) + meta.get("atol", 1e-5)
        if not (d <= tol):
            match = False

    # timing: median of 30 single launches, outputs re-zeroed between
    def timed(kernel, order):
        ts = []
        for _ in range(3):
            run_one(kernel, order, True)
        for _ in range(30):
            t0 = time.perf_counter()
            run_one(kernel, order, True)
            ts.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(ts)

    ours_us = timed(our_kernel, k["arg_order"])
    emmy_us = timed(ref_kernel, ref_order)
    ratio = ours_us / emmy_us if emmy_us else 0
    return (name, "OK" if match else "DIFF", maxdiff, ours_us, emmy_us, ratio)


if __name__ == "__main__":
    main(sys.argv[1:])
