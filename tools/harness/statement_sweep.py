#!/usr/bin/env python3.12
"""Statement-level sweep via host mutation (add_big) — one emitter path per problem."""
import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
from emmy.compiler.graph import Graph
from emmy.compiler.ir.atom import DataType
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import (KernelOp, Assign, Load, Write, Accum, Smem, Sync,
                                        WarpShuffle, TreeHalve, CpAsyncCopy, CpAsyncCommit,
                                        CpAsyncWait)
from emmy.compiler.ir.stmt.leaves import Init
from emmy.compiler.ir.elementwise import ElementwiseImpl

REPO = Path(__file__).resolve().parents[2]
HSX = REPO / "tests" / "hsx"
OUT = REPO / "tests" / ".stmt_sweep"

DT = {}
for n, npn, nb in (("f32", "float32", 4), ("f64", "float64", 8), ("f16", "float16", 2),
                   ("i32", "int32", 4), ("i64", "int64", 8), ("u32", "uint32", 4),
                   ("u64", "uint64", 8), ("i16", "int16", 2), ("u8", "uint8", 1),
                   ("bf16", "uint16", 2), ("bool", "bool", 1)):
    DT[n] = DataType(name=n, np=np.dtype(npn), nbytes=nb)

I32 = DT["i32"]
N = 8192


def load_host():
    raw = json.load(open(next((HSX / "add_big").glob("*.ir.json"))))
    g = Graph.from_dict(copy.deepcopy(raw))
    meta = json.load(open(HSX / "add_big" / "meta.json"))
    return raw, g, meta


def kernel_parts(g):
    kn = [n for n, nd in g.nodes.items() if type(nd.op).__name__ == "KernelOp"][0]
    return kn, g.nodes[kn].op, g.nodes[kn].op.body[0]


def rebase(g, kn, op, tile, body):
    new_op = dataclasses.replace(op, body=type(op.body)([dataclasses.replace(tile, body=tuple(body))]))
    g.nodes[kn] = dataclasses.replace(g.nodes[kn], op=new_op)


def write_problem(name, raw, g, meta, smem_bytes=0, arg_dt=None):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    kn, op, _ = kernel_parts(g)
    entry = op.name or meta["kernels"][0].get("entry") or name
    raw = copy.deepcopy(raw)
    raw["nodes"][kn]["op_fields"]["body"] = [repr(s) for s in op.body]
    (d / f"{entry}.ir.json").write_text(json.dumps(raw, indent=1))
    meta = copy.deepcopy(meta)
    k = meta["kernels"][0]
    k["smem_bytes"] = smem_bytes
    if arg_dt:
        for a in k["args"]:
            a["dtype"] = arg_dt
    (d / "meta.json").write_text(json.dumps({**meta, "rtol": 1e-5, "atol": 1e-5}, indent=1))
    return name


def host_pieces(g):
    kn, op, tile = kernel_parts(g)
    idx = {}
    for s in tile.body:
        if type(s).__name__ == "Load" and s.input == "x":
            u = s.names[0].split("__u")[1]
            idx[u] = s.index
    return kn, op, tile, idx


built = []
FLOATS = ("f32", "f64")
INTS = ("i32", "i64", "u32", "u64")

# ---- 1. scalar load->write, global, every dtype
for dt in DT:
    raw, g, meta = load_host()
    kn, op, tile, idx = host_pieces(g)
    D = DT[dt]
    ns = []
    for s in tile.body:
        k2 = type(s).__name__
        if k2 == "Load" and s.input == "x":
            u = s.names[0].split("__u")[1]
            s = dataclasses.replace(s, dtype=D, names=(f"v__u{u}",))
        elif k2 == "Load" and s.input == "add_c1":
            u = s.names[0].split("__u")[1]
            s = dataclasses.replace(s, names=(f"c__u{u}",))
        elif k2 == "Assign":
            u = s.name.split("__u")[1]
            s = dataclasses.replace(s, op=ElementwiseImpl("copy"), args=(f"v__u{u}",), name=f"w__u{u}", dtype=D)
        elif k2 == "Write":
            if dt == "bool":
                for j, u in enumerate(("0", "1", "2", "3")):
                    ns.append(Write(output="add", index=(Literal(value=j, dtype=I32),),
                                    values=(f"w__u{u}",), value_dtype=D, atomic=False, swizzle="NONE"))
                continue
            s = dataclasses.replace(s, values=tuple(f"w__u{u}" for u in ("0", "1", "2", "3")), value_dtype=D)
        ns.append(s)
    rebase(g, kn, op, tile, ns)
    built.append(write_problem(f"loadg_{dt}", raw, g, meta, arg_dt=dt))

# ---- 2. vector-4 load
for dt in ("f32", "f64", "i32", "i64", "u32", "u64"):
    raw, g, meta = load_host()
    kn, op, tile, idx = host_pieces(g)
    D = DT[dt]
    ns = [Load(names=("v0", "v1", "v2", "v3"), input="x", index=idx["0"], dtype=D)]
    for j, v in enumerate(("v0", "v1", "v2", "v3")):
        ns.append(Write(output="add", index=(Literal(value=j, dtype=I32),), values=(v,),
                        value_dtype=D, atomic=False, swizzle="NONE"))
    rebase(g, kn, op, tile, ns)
    built.append(write_problem(f"loadv4_{dt}", raw, g, meta, arg_dt=dt))

# ---- 3. atomic write (scalar, per emmy)
for dt in FLOATS + INTS:
    raw, g, meta = load_host()
    kn, op, tile, idx = host_pieces(g)
    D = DT[dt]
    ns = []
    for u in ("0", "1", "2", "3"):
        ns.append(Load(names=(f"v__u{u}",), input="x", index=idx[u], dtype=D))
        ns.append(Write(output="add", index=idx[u], values=(f"v__u{u}",), value_dtype=D,
                        atomic=True, swizzle="NONE"))
    rebase(g, kn, op, tile, ns)
    built.append(write_problem(f"writeatomic_{dt}", raw, g, meta, arg_dt=dt))

# ---- 4. vector-4 write
for dt in ("f32", "f64", "i32", "i64", "u32", "u64"):
    raw, g, meta = load_host()
    kn, op, tile, idx = host_pieces(g)
    D = DT[dt]
    ns = []
    for s in tile.body:
        k2 = type(s).__name__
        if k2 == "Load" and s.input == "x":
            u = s.names[0].split("__u")[1]
            s = dataclasses.replace(s, dtype=D, names=(f"v__u{u}",))
        elif k2 == "Load" and s.input == "add_c1":
            u = s.names[0].split("__u")[1]
            s = dataclasses.replace(s, names=(f"c__u{u}",))
        elif k2 == "Assign":
            u = s.name.split("__u")[1]
            s = dataclasses.replace(s, op=ElementwiseImpl("copy"), args=(f"v__u{u}",), name=f"w__u{u}", dtype=D)
        ns.append(s)
    ns = [s for s in ns if type(s).__name__ != "Write"]
    ns.append(Write(output="add", index=idx["0"], values=("w__u0", "w__u1", "w__u2", "w__u3"),
                    value_dtype=D, atomic=False, swizzle="NONE"))
    rebase(g, kn, op, tile, ns)
    built.append(write_problem(f"writev4_{dt}", raw, g, meta, arg_dt=dt))

# ---- 5. Accum: op x dtype
for dt in FLOATS + INTS:
    for opn in ("add", "multiply", "maximum", "minimum"):
        raw, g, meta = load_host()
        kn, op, tile, idx = host_pieces(g)
        D = DT[dt]
        if dt.startswith("f"):
            ident = {"add": 0.0, "multiply": 1.0, "maximum": -1e30, "minimum": 1e30}[opn]
        else:
            w = 64 if "64" in dt else 32
            sgn = 1 if dt.startswith("i") else 0
            ident = {"add": 0, "multiply": 1, "maximum": -(2 ** (w - 1)) if sgn else 0,
                     "minimum": 2 ** (w - 1) - 1 if sgn else 2 ** w - 1}[opn]
        ns = [
            Init(name="acc", identity=ident, dtype=D),
            Load(names=("v",), input="x", index=idx["0"], dtype=D),
            Accum(name="acc", value="v", op=ElementwiseImpl(opn), dtype=D, axes=("i",), base=None),
            Write(output="add", index=idx["0"], values=("acc",), value_dtype=D, atomic=False, swizzle="NONE"),
        ]
        rebase(g, kn, op, tile, ns)
        built.append(write_problem(f"accum_{opn}_{dt}", raw, g, meta, arg_dt=dt))

# ---- 6. Init every dtype
for dt in FLOATS + INTS:
    raw, g, meta = load_host()
    kn, op, tile, idx = host_pieces(g)
    D = DT[dt]
    ns = [
        Init(name="acc", identity=0, dtype=D),
        Write(output="add", index=idx["0"], values=("acc",), value_dtype=D, atomic=False, swizzle="NONE"),
    ]
    rebase(g, kn, op, tile, ns)
    built.append(write_problem(f"init_{dt}", raw, g, meta, arg_dt=dt))

# ---- 7. Smem round-trip every dtype
SMEM_HOST = {"f32": "float", "f64": "double", "f16": "half", "bf16": "__nv_bfloat16",
             "i32": "int", "i64": "long long", "u32": "unsigned int", "u64": "unsigned long long",
             "i16": "short", "u8": "char"}
for dt in DT:
    if dt == "bool":
        continue
    raw, g, meta = load_host()
    kn, op, tile, idx = host_pieces(g)
    D = DT[dt]
    ns = [
        Smem(name="buf", extents=(N,), dtype=SMEM_HOST[dt], align=0),
        Load(names=("v",), input="x", index=idx["0"], dtype=D),
        Write(output="buf", index=idx["0"], values=("v",), value_dtype=D, atomic=False, swizzle="NONE"),
        Sync(barrier_id=0, count=None, warp=False),
        Load(names=("w",), input="buf", index=idx["0"], dtype=D),
        Write(output="add", index=idx["0"], values=("w",), value_dtype=D, atomic=False, swizzle="NONE"),
    ]
    rebase(g, kn, op, tile, ns)
    built.append(write_problem(f"smem_{dt}", raw, g, meta, smem_bytes=N * DT[dt].nbytes, arg_dt=dt))

# ---- 8. Sync variants
for tag, sync in (("sync_full", Sync(barrier_id=0, count=None, warp=False)),
                  ("sync_count", Sync(barrier_id=0, count=32, warp=False)),
                  ("sync_named", Sync(barrier_id=1, count=32, warp=False)),
                  ("sync_warp", Sync(barrier_id=0, count=None, warp=True))):
    raw, g, meta = load_host()
    kn, op, tile, idx = host_pieces(g)
    ns = [sync] + [s for s in tile.body if type(s).__name__ != "Write"] + \
         [s for s in tile.body if type(s).__name__ == "Write"]
    rebase(g, kn, op, tile, ns)
    built.append(write_problem(tag, raw, g, meta))

# ---- 9. WarpShuffle every carrier dtype
for dt in FLOATS + INTS:
    raw, g, meta = load_host()
    kn, op, tile, idx = host_pieces(g)
    D = DT[dt]
    ns = [
        Load(names=("v",), input="x", index=idx["0"], dtype=D),
        Accum(name="acc", value="v", op=ElementwiseImpl("add"), dtype=D, axes=("i",), base=None),
        WarpShuffle(state=("acc",), state_b=("acc_o",), length=32, dtype=D,
                    combine_states=(Assign(name="acc", op=ElementwiseImpl("add"),
                                           args=("acc", "acc_o"), dtype=D),)),
        Write(output="add", index=idx["0"], values=("acc",), value_dtype=D, atomic=False, swizzle="NONE"),
    ]
    rebase(g, kn, op, tile, ns)
    built.append(write_problem(f"shuffle_{dt}", raw, g, meta))

# ---- 10. TreeHalve every carrier dtype
SMALL = {"f32": 8, "f64": 16, "i32": 8, "i64": 16, "u32": 8, "u64": 16}
for dt in FLOATS + INTS:
    raw, g, meta = load_host()
    kn, op, tile, idx = host_pieces(g)
    D = DT[dt]
    smem_host = {"f32": "float", "f64": "double", "i32": "int", "i64": "long long",
                 "u32": "unsigned int", "u64": "unsigned long long"}[dt]
    ns = [
        Smem(name="tbuf", extents=(2,), dtype=smem_host, align=0),
        Load(names=("v",), input="x", index=idx["0"], dtype=D),
        Accum(name="acc", value="v", op=ElementwiseImpl("add"), dtype=D, axes=("i",), base=None),
        Write(output="tbuf", index=(Var(name="warp"),), values=("acc",), value_dtype=D,
              atomic=False, swizzle="NONE"),
        Sync(barrier_id=0, count=None, warp=False),
        TreeHalve(bufs=("tbuf",), state=("acc",), state_b=("acc_o",), length=2, tid_var="warp",
                  dtype=D, barrier_id=0, barrier_count=None, inner=None,
                  combine_states=(Assign(name="acc", op=ElementwiseImpl("add"),
                                         args=("acc", "acc_o"), dtype=D),)),
        Write(output="add", index=idx["0"], values=("acc",), value_dtype=D, atomic=False, swizzle="NONE"),
    ]
    rebase(g, kn, op, tile, ns)
    built.append(write_problem(f"tree_{dt}", raw, g, meta, smem_bytes=SMALL[dt]))

# ---- 11. CpAsync 4/8/16
for nbytes in (4, 8, 16):
    raw, g, meta = load_host()
    kn, op, tile, idx = host_pieces(g)
    ns = [
        Smem(name="cbuf", extents=(N,), dtype="float", align=16),
        CpAsyncCopy(smem="cbuf", smem_index=idx["0"], src="x", src_index=idx["0"],
                    nbytes=nbytes, swizzle="NONE", lane_index=None, lane_rows=None),
        CpAsyncCommit(),
        CpAsyncWait(group=0),
        Sync(barrier_id=0, count=None, warp=False),
        Load(names=("v",), input="cbuf", index=idx["0"], dtype=DT["f32"]),
        Write(output="add", index=idx["0"], values=("v",), value_dtype=DT["f32"],
              atomic=False, swizzle="NONE"),
    ]
    rebase(g, kn, op, tile, ns)
    built.append(write_problem(f"cpasync_{nbytes}", raw, g, meta, smem_bytes=N * 4))

print(f"built {len(built)} statement-sweep problems")
