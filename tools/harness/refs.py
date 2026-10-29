"""Numpy references for hsx corpus + synthetic problems (torch-free)."""
import math
import numpy as np

def _castexpr_ref(v):
    x = np.asarray(v["x"], dtype=np.float32)          # gather source
    idx = np.asarray(v["add_c1"], dtype=np.float32)   # gather indices (0..8)
    return idx + x[np.trunc(idx).astype(np.int64)]    # the IR keeps add(in0, gathered)


_erf = np.vectorize(math.erf)


def _ctrunc(a, b):
    """C '%' semantics computed the same way the PTX does (f32 div + trunc)."""
    a = np.asarray(a, dtype=np.float32)
    q = (a / b).astype(np.float32)
    t = np.trunc(q).astype(np.float32)
    return (a - (t * b).astype(np.float32)).astype(np.float32)


def _attn(q, kt, v, scale):
    s = q @ kt.T if kt.shape[0] != q.shape[1] else q @ kt
    s = s / scale
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    p = e / e.sum(-1, keepdims=True)
    return p @ v


def _ln(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * w + b


def _rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * w


REFS = {
    "add2d":      lambda v: v["x"] + v["y"],
    "add_big":    lambda v: v["x"] + 3.5,
    "attn128b":   lambda v: _attn(v["q"], v["kt"], v["v"], 8.0),
    "attn32":     lambda v: _attn(v["q"], v["kt"], v["v"], 4.0),
    "attn64b":    lambda v: _attn(v["q"], v["kt"], v["v"], 8.0),
    "attn96":     lambda v: _attn(v["q"], v["kt"], v["v"], 8.0),
    "ceil":       lambda v: np.ceil(v["x"] * 3),
    "clamp":      lambda v: np.clip(v["x"], -1.0, 1.0),
    "cos":        lambda v: np.cos(v["x"]),
    "exp":        lambda v: np.exp(v["x"]),
    "exp2":       lambda v: np.exp2(v["x"]),
    "floor":      lambda v: np.floor(v["x"] * 3),
    "gelu":       lambda v: 0.5 * v["x"] * (1 + _erf(v["x"] / np.sqrt(2.0))),
    "layernorm":  lambda v: _ln(v["x"], v["w"], v["b"]),
    "log":        lambda v: np.log(np.abs(v["x"]) + 0.5),
    "log2":       lambda v: np.log2(np.abs(v["x"]) + 0.5),
    "mean_rows_big": lambda v: v["x"].mean(-1),
    "minimum":    lambda v: np.minimum(v["x"], 0.0),
    "mm128":      lambda v: v["x"] @ v["y"],
    "mm256":      lambda v: v["x"] @ v["y"],
    "mm512":      lambda v: v["x"] @ v["y"],
    "mm64a":      lambda v: v["x"] @ v["y"],
    "mm_rect2":   lambda v: v["x"] @ v["y"],
    "mul2d_bc":   lambda v: v["x"] * v["y"],
    "neg_abs":    lambda v: np.abs(v["x"]) * -2.0,
    "pow":        lambda v: (np.abs(v["x"]) + 0.5) ** 2.5,
    "relu2d":     lambda v: np.maximum(v["x"], 0) * 3.0 - 1.0,
    "relu_max":   lambda v: np.maximum(v["x"], 0) + np.maximum(v["x"], 0.0),
    "rmsnorm2":   lambda v: _rms(v["x"], v["w"]),
    "round":      lambda v: np.round(v["x"] * 4) / 4,
    "rsqrt":      lambda v: 1.0 / np.sqrt(np.abs(v["x"]) + 0.5),
    "sigmoid":    lambda v: 1 / (1 + np.exp(-v["x"])),
    "silu":       lambda v: v["x"] / (1 + np.exp(-v["x"])),
    "sin":        lambda v: np.sin(v["x"]),
    "softmax0":   lambda v: np.exp(v["x"] - v["x"].max(0, keepdims=True)) / np.exp(v["x"] - v["x"].max(0, keepdims=True)).sum(0, keepdims=True),
    "softmax_big": lambda v: np.exp(v["x"] - v["x"].max(-1, keepdims=True)) / np.exp(v["x"] - v["x"].max(-1, keepdims=True)).sum(-1, keepdims=True),
    "softplus":   lambda v: np.log1p(np.exp(-np.abs(v["x"]))) + np.maximum(v["x"], 0),
    "sqrt":       lambda v: np.sqrt(np.abs(v["x"])),
    "square_chain": lambda v: v["x"] ** 2 + v["x"] * 2.0 + 1.0,
    "sum_big":    lambda v: np.array([v["x"].sum()]),
    "sum_cols":   lambda v: v["x"].sum(0),
    "sum_rows":   lambda v: v["x"].sum(-1),
    "tanh":       lambda v: np.tanh(v["x"]),
    "where":      lambda v: np.where(v["x"] > 0, v["x"], -v["x"]),
    # synth
    "synth_scalar_param": lambda v: v["x"] + v["y"],
    "synth_aux_threads": lambda v: v["x"].sum(-1),
    "synth_raster_even": lambda v: v["x"] + v["y"],
    "synth_i32_square": lambda v: np.asarray(v["x"], dtype=np.int32) ** 2,
    "synth_i64_square": lambda v: np.asarray(v["x"], dtype=np.int64) ** 2,
    "synth_f64_add": lambda v: np.asarray(v["x"], dtype=np.float64) + 3.5,
    "synth_f64_relu": lambda v: np.maximum(np.asarray(v["x"], dtype=np.float64), 0.0),
    "synth_f16_scale": lambda v: np.asarray(v["x"], dtype=np.float16) * np.float16(2.0),
    "synth_remainder": lambda v: _ctrunc(3.0, np.asarray(v["x"], dtype=np.float32)),
    "synth_mod": lambda v: _ctrunc(3.0, np.asarray(v["x"], dtype=np.float32)),
    "synth_fmod": lambda v: _ctrunc(3.0, np.asarray(v["x"], dtype=np.float32)),
    "synth_cond_else": lambda v: np.abs(np.asarray(v["x"], dtype=np.float32)),
    "synth_accum_under_cond": lambda v: np.where(v["x"] > 0, v["x"], 0.0).sum(-1),
    "synth_subwarp_shuffle": lambda v: v["x"].sum(-1),
    "synth_named_barrier": lambda v: v["x"].sum(-1),
    "synth_warp_sync": lambda v: v["x"].sum(-1),
    "synth_static_smem": lambda v: v["x"].sum(-1),
    "synth_grid_y": lambda v: v["x"] + v["y"],
    "synth_grid_z": lambda v: v["x"] + 3.5,
    "synth_zero_trip": lambda v: np.zeros(v["x"].shape[0], dtype=np.float32),
    # fail-first tests
    "test_sync_partial": lambda v: v["x"].sum(-1),
    "test_i64_shift": lambda v: (np.asarray(v["x"], dtype=np.int64) >> 1),
    "test_i64_cmp": lambda v: (np.asarray(v["x"], dtype=np.int64) > 0).astype(np.int64),
    "test_cross_store": lambda v: np.trunc(v["x"] + 3.5).astype(np.int32),
    "test_f16_accum": lambda v: v["x"].sum(-1).astype(np.float16),
    "test_f64_wsum": lambda v: v["x"].sum(-1),
    "test_nan_neq": lambda v: (v["x"] == 0).astype(np.float32),
    "test_pow_neg": lambda v: np.power(np.asarray(v["x"], dtype=np.float32), 3.0),
    # fail-first batch 2
    "test_builtin_index": lambda v: 3.5 + v["x"][(np.arange(v["x"].size) // 4) % 256],
    "test_castexpr": _castexpr_ref,
    "test_flatindex_let": lambda v: v["x"].copy(),
    "test_sin_large": lambda v: np.sin(np.asarray(v["x"], dtype=np.float32)),
    "test_cos_large": lambda v: np.cos(np.asarray(v["x"], dtype=np.float32)),
    "test_log_subnormal": lambda v: np.log(np.asarray(v["x"], dtype=np.float32)),
    "test_pow_zero": lambda v: np.ones_like(np.asarray(v["x"], dtype=np.float32)),
}

