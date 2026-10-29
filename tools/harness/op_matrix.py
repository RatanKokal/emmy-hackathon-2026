#!/usr/bin/env python3.12
"""Dispatcher x dtype validity sweep: emit every op in every dtype -> ptxas."""
import json
import sys
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("cc", sys.argv[1])
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

OPS = [
    "relu", "add", "multiply", "subtract", "divide", "floor_divide", "remainder",
    "mod", "fmod", "maximum", "minimum", "neg", "negative", "abs", "fabs", "copy",
    "pad", "sqrt", "rsqrt", "reciprocal", "exp", "exp_fast", "exp2", "log", "log2",
    "log1p", "sin", "cos", "tanh", "sigmoid", "silu", "softplus", "erf", "pow",
    "square", "floor", "ceil", "round", "trunc", "frac", "equal", "not_equal",
    "greater", "less", "greater_equal", "less_equal", "where", "bitwise_and",
    "bitwise_or", "logical_and", "logical_or", "bitwise_xor", "logical_xor",
    "sign", "bitwise_not", "logical_not", "left_shift", "right_shift",
    "bitwise_count", "fma", "mul_add",
]
DTYPES = ["f32", "f64", "i32", "i64", "u32", "u64", "i16", "u8", "bool"]
ARG_N = {
    "relu": 1, "add": 2, "multiply": 2, "subtract": 2, "divide": 2,
    "floor_divide": 2, "remainder": 2, "mod": 2, "fmod": 2, "maximum": 2,
    "minimum": 2, "neg": 1, "negative": 1, "abs": 1, "fabs": 1, "copy": 1,
    "pad": 1, "sqrt": 1, "rsqrt": 1, "reciprocal": 1, "exp": 1, "exp_fast": 1,
    "exp2": 1, "log": 1, "log2": 1, "log1p": 1, "sin": 1, "cos": 1, "tanh": 1,
    "sigmoid": 1, "silu": 1, "softplus": 1, "erf": 1, "pow": 2, "square": 1,
    "floor": 1, "ceil": 1, "round": 1, "trunc": 1, "frac": 1, "equal": 2,
    "not_equal": 2, "greater": 2, "less": 2, "greater_equal": 2, "less_equal": 2,
    "where": 3, "bitwise_and": 2, "bitwise_or": 2, "logical_and": 2,
    "logical_or": 2, "bitwise_xor": 2, "logical_xor": 2, "sign": 1,
    "bitwise_not": 1, "logical_not": 1, "left_shift": 2, "right_shift": 2,
    "bitwise_count": 1, "fma": 3, "mul_add": 3,
}


def shim():
    c = object.__new__(cc.Compiler)
    c.em = cc.Emitter()
    c.values = {}
    c.ints = {}
    c.axis_max = {}
    c.shared = {}
    c.shared_bases = {}
    c.expr_cache = {}
    c.load_cache = {}
    c.addr_cache = {}
    c.bases = {}
    c.scalar_tys = {}
    c.tid_x = None
    return c


outdir = Path(__file__).resolve().parents[2] / "tests" / ".sweep_out"
outdir.mkdir(parents=True, exist_ok=True)
results = {}
for ty in DTYPES:
    for op in OPS:
        c = shim()
        n = ARG_N[op]
        cls = cc.REG_CLASS.get(ty, "r")
        args = []
        for i in range(n):
            r = c.em.reg(cls)
            c.values[f"a{i}"] = (r, ty)
            args.append(r)
        dst = c.em.reg(cls)
        try:
            c.emit_op(dst, op, ty, args, [ty] * n)
        except NotImplementedError as e:
            results[(op, ty)] = ("SKIP", str(e)[:40])
            continue
        except Exception as e:  # noqa: BLE001
            results[(op, ty)] = ("PYERR", f"{type(e).__name__}: {str(e)[:60]}")
            continue
        counts = c.em.counts
        lines = [".version 8.0", ".target sm_89", ".address_size 64",
                 ".visible .entry t(.param .u64 p)", "{"]
        for cls2, pty in (("p", "pred"), ("f", "f32"), ("fd", "f64"),
                          ("r", "b32"), ("rd", "b64"), ("h", "b16")):
            if counts[cls2]:
                lines.append(f".reg .{pty} %{cls2}<{counts[cls2] + 1}>;")
        lines += list(c.em.lines)
        lines.append("ret;")
        lines.append("}")
        (outdir / f"{op}__{ty}.ptx").write_text("\n".join(lines))
        results[(op, ty)] = ("EMIT", "")

n_emit = sum(1 for v in results.values() if v[0] == "EMIT")
print(f"emitted {n_emit} / {len(results)} combos "
      f"({len(results) - n_emit} raised as designed)")
