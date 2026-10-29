#!/usr/bin/env python3
"""gpurun2: dtype-correct, multi-output bundle runner (replaces gpurun)."""
import ctypes
import json
import sys
from pathlib import Path

import numpy as np

cu = ctypes.CDLL("libcuda.so.1")

NP = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64,
      "u32": np.uint32, "u64": np.uint64, "i16": np.int16, "u16": np.uint16,
      "u8": np.uint8, "bool": np.bool_, "f16": np.float16}
# bf16 has no numpy equivalent; treat as raw u16 (size-correct)


def ck(rv, what=""):
    if rv != 0:
        raise RuntimeError(f"{what}: CUresult {rv}")


def itemsize(dt):
    return np.dtype(NP.get(dt, np.uint16)).itemsize


def make_input(a, d):
    n = 1
    for e in a["shape"]:
        n *= int(e)
    f = d / f"in_{a['name']}.bin"
    if f.exists():
        return np.fromfile(f, dtype=np.uint8)[:n * itemsize(a["dtype"])].tobytes()
    if "data" in a:
        arr = np.array(a["data"]).astype(NP.get(a["dtype"], np.float32))
        return (arr * np.ones(n, dtype=arr.dtype))[:n].reshape(a["shape"]).tobytes()
    dt = np.dtype(NP.get(a["dtype"], np.float32))
    rng = np.random.default_rng(a.get("seed", 0))
    if dt.kind in "iu":
        arr = rng.integers(1, 100, size=a["shape"]).astype(dt)
    else:
        arr = rng.standard_normal(a["shape"]).astype(dt)
    return arr.tobytes()


def main():
    root = Path(sys.argv[1])
    ck(cu.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    ck(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (d / "k.ptx").exists() or (d / "k.ptx").stat().st_size < 100:
            print(f"{d.name}: SKIP (no ptx)")
            continue
        meta = json.load(open(d / "meta.json"))
        k = meta["kernels"][0]
        ptx = (d / "k.ptx").read_bytes()
        mod = ctypes.c_void_p()
        ck(cu.cuModuleLoadData(ctypes.byref(mod), ptx), f"load {d.name}")
        fn = ctypes.c_void_p()
        ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, k["entry"].encode()), "fn")

        ptrs, bufs = {}, {}
        for a in k["args"]:
            n = 1
            for e in a["shape"]:
                n *= int(e)
            nb = max(n * itemsize(a["dtype"]), 4)
            p = ctypes.c_ulonglong()
            ck(cu.cuMemAlloc_v2(ctypes.byref(p), nb), "alloc")
            if a.get("output"):
                host = ctypes.create_string_buffer(nb)   # zero-filled
            else:
                raw = make_input(a, d)
                host = ctypes.create_string_buffer(raw, nb)
            ck(cu.cuMemcpyHtoD_v2(p, ctypes.cast(host, ctypes.c_void_p), nb), "HtoD")
            ptrs[a["name"]] = p.value
            bufs[a["name"]] = (host, nb, a.get("output", False))

        args = (ctypes.c_uint64 * len(k["arg_order"]))(*[ptrs[n] for n in k["arg_order"]])
        params = (ctypes.c_void_p * len(k["arg_order"]))(
            *[ctypes.cast(ctypes.byref(args, 8 * i), ctypes.c_void_p) for i in range(len(k["arg_order"]))])
        gx, gy, gz = k["grid"]
        bx, by, bz = k["block"]
        if k["smem_bytes"] > 49152:
            ck(cu.cuFuncSetAttribute(fn, 8, k["smem_bytes"]), "smem attr")
        ck(cu.cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, k["smem_bytes"], None, params, None), "launch")
        ck(cu.cuCtxSynchronize(), "sync")

        outs = [n for n in k["arg_order"] if bufs[n][2]]
        for name in outs:
            host, nb, _ = bufs[name]
            ck(cu.cuMemcpyDtoH_v2(ctypes.cast(host, ctypes.c_void_p), ctypes.c_ulonglong(ptrs[name]), nb), "DtoH")
            (d / f"got_{name}.bin").write_bytes(bytes(host.raw))
            if name == outs[-1]:
                (d / "got.bin").write_bytes(bytes(host.raw))
        for p in ptrs.values():
            cu.cuMemFree_v2(ctypes.c_ulonglong(p))
        cu.cuModuleUnload(mod)
        print(f"{d.name}: ran grid={k['grid']} block={k['block']}")
    cu.cuCtxDestroy_v2(ctx)


if __name__ == "__main__":
    main()
