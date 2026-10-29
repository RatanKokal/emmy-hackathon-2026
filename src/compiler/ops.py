# Extracted verbatim from compile_speed_safe.py (validated checkpoint f289d05).
# op emission dispatcher (emit_op) and transcendental helpers
import struct
import sys
from math import prod
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var

from .tables import (INT_TYPES, REG_CLASS, PTX_TY, WIDEN_AT_LOAD, SMEM_NBYTES,
                     F32_ONLY, UNROLL_CAP, PURE_OPS, COMMUTATIVE, _MEM_KINDS,
                     CF_KINDS)


class OpsMixin:
    def emit_op(self, dst, op, ty, args, arg_tys=None):
        if op in self.F32_ONLY and ty != "f32":
            raise NotImplementedError(f"{op} requires f32, got {ty}")
        if ty in ("i16", "u8", "u16"):
            ty = "i32" if ty == "i16" else "u32"
        if ty != "f32" and op in ("exp", "exp_fast", "exp2", "log", "log2", "log1p",
                                  "sigmoid", "silu", "softplus", "tanh", "erf", "pow",
                                  "sin", "cos"):
            # no f64 approximation instructions exist; libm-port out of scope
            raise NotImplementedError(f"{op} requires f32, got {ty}")
        if ty not in ("f32", "f64") and op in ("sqrt", "rsqrt", "reciprocal"):
            raise NotImplementedError(f"{op} requires a float dtype, got {ty}")
        p = self.ptx(ty)
        rcls = REG_CLASS[ty]
        rp = ("rn." + p) if ty.startswith("f") else p
        tmpl = self._SIMPLE_EMIT.get(op)
        if tmpl is not None:
            self.em.emit(tmpl.format(dst=dst, a0=args[0],
                                     a1=args[1] if len(args) > 1 else None,
                                     p=p, rp=rp))
            return
        hname = self._OP_HANDLER.get(op)
        if hname is None:
            raise NotImplementedError(f"elementwise op {op!r}")
        getattr(self, hname)(op, dst, ty, p, rp, rcls, args, arg_tys)

    def _mul_ins(self, ty, p, rp):
        return f"mul.{rp}" if ty.startswith("f") else f"mul.lo.{p}"

    def _sq_ins(self, ty, p):
        return f"mul.{p}" if ty.startswith("f") else f"mul.lo.{p}"

    def _div_ins(self, ty, p):
        return f"div.rn.{p}" if ty.startswith("f") else f"div.{p}"


    _SIMPLE_EMIT = {
        "add": "add.{rp} {dst}, {a0}, {a1};",
        "copy": "mov.{p} {dst}, {a0};",
        "exp2": "ex2.approx.f32 {dst}, {a0};",
        "maximum": "max.{p} {dst}, {a0}, {a1};",
        "minimum": "min.{p} {dst}, {a0}, {a1};",
        "pad": "mov.{p} {dst}, {a0};",
        "reciprocal": "rcp.rn.{p} {dst}, {a0};",
        "sqrt": "sqrt.rn.{p} {dst}, {a0};",
        "subtract": "sub.{rp} {dst}, {a0}, {a1};",
    }
    _OP_HANDLER = {
        "abs": "_op_abs",
        "bitwise_and": "_op_bitwise_and",
        "bitwise_count": "_op_bitwise_count",
        "bitwise_not": "_op_bitwise_not",
        "bitwise_or": "_op_bitwise_and",
        "bitwise_xor": "_op_bitwise_xor",
        "ceil": "_op_floor",
        "cos": "_op_sin",
        "divide": "_op_divide",
        "equal": "_op_equal",
        "erf": "_op_erf",
        "exp": "_op_exp",
        "exp_fast": "_op_exp",
        "fabs": "_op_abs",
        "floor": "_op_floor",
        "floor_divide": "_op_floor_divide",
        "fma": "_op_fma",
        "fmod": "_op_floor_divide",
        "frac": "_op_frac",
        "greater": "_op_equal",
        "greater_equal": "_op_equal",
        "left_shift": "_op_left_shift",
        "less": "_op_equal",
        "less_equal": "_op_equal",
        "log": "_op_log",
        "log1p": "_op_log1p",
        "log2": "_op_log",
        "logical_and": "_op_bitwise_and",
        "logical_not": "_op_bitwise_not",
        "logical_or": "_op_bitwise_and",
        "logical_xor": "_op_bitwise_xor",
        "mod": "_op_floor_divide",
        "mul_add": "_op_fma",
        "multiply": "_op_multiply",
        "neg": "_op_neg",
        "negative": "_op_neg",
        "not_equal": "_op_equal",
        "pow": "_op_pow",
        "relu": "_op_relu",
        "remainder": "_op_floor_divide",
        "right_shift": "_op_left_shift",
        "round": "_op_floor",
        "rsqrt": "_op_rsqrt",
        "sigmoid": "_op_sigmoid",
        "sign": "_op_sign",
        "silu": "_op_silu",
        "sin": "_op_sin",
        "softplus": "_op_softplus",
        "square": "_op_square",
        "tanh": "_op_tanh",
        "trunc": "_op_floor",
        "where": "_op_where",
    }

    def _op_relu(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty == "f64":
            zero = "0d0000000000000000"
        elif ty.startswith("f"):
            zero = "0f00000000"
        else:
            zero = "0"
        self.em.emit(f"max.{p} {dst}, {args[0]}, {zero};")

    def _op_multiply(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        ins = f"mul.{rp}" if ty.startswith("f") else f"mul.lo.{p}"
        self.em.emit(f"{ins} {dst}, {args[0]}, {args[1]};")

    def _op_divide(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty.startswith("f"):
            self.em.emit(f"div.rn.{p} {dst}, {args[0]}, {args[1]};")
        else:
            self.em.emit(f"div.{p} {dst}, {args[0]}, {args[1]};")

    def _op_floor_divide(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty.startswith("f"):
            w = self.ptx(ty)
            rcls = REG_CLASS[ty]
            q = self.em.reg(rcls)
            self.emit_op(q, "divide", ty, args)
            if op == "floor_divide":
                self.em.emit(f"cvt.rmi.{w}.{w} {dst}, {q};")
            else:
                # C '%' — what emmy's renderer emits for remainder/mod/fmod
                # (truncated quotient, remainder keeps the dividend's sign).
                t = self.em.reg(rcls)
                self.em.emit(f"cvt.rzi.{w}.{w} {t}, {q};")
                self.em.emit(f"mul.rn.{w} {t}, {t}, {args[1]};")
                self.em.emit(f"sub.rn.{w} {dst}, {args[0]}, {t};")
        elif op == "floor_divide":
            self.em.emit(f"div.{p} {dst}, {args[0]}, {args[1]};")
        else:
            self.em.emit(f"rem.{p} {dst}, {args[0]}, {args[1]};")

    def _op_neg(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty in ("u32", "u64", "bool", "u8", "u16"):
            self.em.emit(f"sub.{p} {dst}, 0, {args[0]};")
        else:
            self.em.emit(f"neg.{p} {dst}, {args[0]};")

    def _op_abs(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty in ("u32", "u64", "u8", "u16", "bool"):
            self.em.emit(f"mov.{p} {dst}, {args[0]};")  # unsigned abs is identity
        else:
            self.em.emit(f"abs.{p} {dst}, {args[0]};")

    def _op_rsqrt(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        t = self.em.reg(rcls)
        self.em.emit(f"sqrt.rn.{p} {t}, {args[0]};")
        self.em.emit(f"rcp.rn.{p} {dst}, {t};")

    def _op_exp(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty != "f32":
            raise NotImplementedError("exp only implemented for f32")
        self.fexp(dst, args[0])

    def _op_log(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        self.emit_log(dst, args[0], op == "log2")

    def _op_log1p(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        t = self.em.reg("f")
        self.em.emit(f"add.rn.f32 {t}, {args[0]}, {self.fconst(1.0)};")
        self.emit_log(dst, t, False)

    def _op_sin(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        # Range-reduce with f64 arithmetic: r = x - round(x/2pi)*2pi is
        # exact to ~|x|*1.6e-16 for every f32 input, keeping sin/cos.approx
        # inside their quadrant-accuracy envelope.
        TWO_PI = struct.unpack("<Q", struct.pack("<d", 6.283185307179586))[0]
        INV = struct.unpack("<Q", struct.pack("<d", 0.15915494309189535))[0]
        xd = self.em.reg("fd")
        self.em.emit(f"cvt.f64.f32 {xd}, {args[0]};")
        t = self.em.reg("fd")
        self.em.emit(f"mul.f64 {t}, {xd}, 0d{INV:016X};")
        kd = self.em.reg("fd")
        self.em.emit(f"cvt.rni.f64.f64 {kd}, {t};")
        rd = self.em.reg("fd")
        self.em.emit(f"fma.rn.f64 {rd}, {kd}, 0d{TWO_PI ^ (1 << 63):016X}, {xd};")
        # Cody-Waite 2-term correction: subtract the 2π_hi representation error
        r2 = self.em.reg("fd")
        self.em.emit(f"fma.rn.f64 {r2}, {kd}, 0dBCC12EDBFE997F88, {rd};")
        rf = self.em.reg("f")
        self.em.emit(f"cvt.rn.f32.f64 {rf}, {r2};")
        self.em.emit(f"{op}.approx.f32 {dst}, {rf};")

    def _op_tanh(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        # tanh(x) = (e^{2x}-1)/(e^{2x}+1); saturate past |20| where e^{2x}
        # overflows, since ex2.approx alone would yield inf/inf = NaN.
        w = self.ptx(ty)
        rcls = REG_CLASS[ty]
        t = self.em.reg(rcls)
        self.em.emit(f"mul.rn.{w} {t}, {args[0]}, {self.float_literal(2.0, ty)};")
        e = self.em.reg(rcls)
        self.fexp(e, t, ty)
        num = self.em.reg(rcls)
        den = self.em.reg(rcls)
        self.em.emit(f"add.rn.{w} {num}, {e}, {self.float_literal(-1.0, ty)};")
        self.em.emit(f"add.rn.{w} {den}, {e}, {self.float_literal(1.0, ty)};")
        self.em.emit(f"rcp.rn.{w} {den}, {den};")
        self.em.emit(f"mul.rn.{w} {t}, {num}, {den};")
        hi = self.em.reg("p")
        lo = self.em.reg("p")
        self.em.emit(f"setp.gt.{w} {hi}, {args[0]}, {self.float_literal(20.0, ty)};")
        self.em.emit(f"setp.lt.{w} {lo}, {args[0]}, {self.float_literal(-20.0, ty)};")
        mid = self.em.reg(rcls)
        self.em.emit(f"selp.{w} {mid}, {self.float_literal(-1.0, ty)}, {t}, {lo};")
        self.em.emit(f"selp.{w} {dst}, {self.float_literal(1.0, ty)}, {mid}, {hi};")

    def _op_sigmoid(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        self.emit_sigmoid(dst, args[0], False, ty)

    def _op_silu(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        self.emit_sigmoid(dst, args[0], True, ty)

    def _op_softplus(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        w = self.ptx(ty)
        rcls = REG_CLASS[ty]
        e = self.em.reg(rcls)
        self.fexp(e, args[0], ty)
        t = self.em.reg(rcls)
        self.em.emit(f"add.rn.{w} {t}, {e}, {self.float_literal(1.0, ty)};")
        l = self.em.reg(rcls)
        self.emit_log(l, t, False, ty)
        bp = self.em.reg("p")
        self.em.emit(f"setp.gt.{w} {bp}, {args[0]}, {self.float_literal(20.0, ty)};")
        self.em.emit(f"selp.{w} {dst}, {args[0]}, {l}, {bp};")

    def _op_erf(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        self.emit_erf(dst, args[0], ty)

    def _op_square(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        ins = f"mul.{p}" if ty.startswith("f") else f"mul.lo.{p}"
        self.em.emit(f"{ins} {dst}, {args[0]}, {args[0]};")

    def _op_floor(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty not in ("f32", "f64"):
            self.em.emit(f"mov.{p} {dst}, {args[0]};")  # integers: exact identity
        else:
            mode = {"floor": "rmi", "ceil": "rpi", "round": "rni", "trunc": "rzi"}[op]
            self.em.emit(f"cvt.{mode}.{p}.{p} {dst}, {args[0]};")

    def _op_frac(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty not in ("f32", "f64"):
            self.em.emit(f"mov.{p} {dst}, 0;")  # integers have no fractional part
        else:
            t = self.em.reg(rcls)
            self.em.emit(f"cvt.rzi.{p}.{p} {t}, {args[0]};")
            self.em.emit(f"sub.rn.{p} {dst}, {args[0]}, {t};")

    def _op_pow(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        # powf semantics: |x|^y magnitude; negative x -> sign(y even?+:-1)
        # for integer y, NaN otherwise (libm/torch agree).
        w = self.ptx(ty)
        rcls = REG_CLASS[ty]
        F0 = self.float_literal(0.0, ty)
        F1 = self.float_literal(1.0, ty)
        ax = self.em.reg(rcls)
        self.em.emit(f"abs.{w} {ax}, {args[0]};")
        l = self.em.reg(rcls)
        self.emit_log(l, ax, True, ty)
        m = self.em.reg(rcls)
        self.em.emit(f"mul.rn.{w} {m}, {l}, {args[1]};")
        mag = self.em.reg(rcls)
        self.em.emit(f"ex2.approx.{w} {mag}, {m};")
        neg = self.em.reg("p")
        self.em.emit(f"setp.lt.{w} {neg}, {args[0]}, {F0};")
        yi = self.em.reg(rcls)
        self.em.emit(f"cvt.rzi.{w}.{w} {yi}, {args[1]};")
        isint = self.em.reg("p")
        self.em.emit(f"setp.eq.{w} {isint}, {args[1]}, {yi};")
        half = self.em.reg(rcls)
        self.em.emit(f"mul.rn.{w} {half}, {args[1]}, {self.float_literal(0.5, ty)};")
        hi = self.em.reg(rcls)
        self.em.emit(f"cvt.rzi.{w}.{w} {hi}, {half};")
        iseven = self.em.reg("p")
        self.em.emit(f"setp.eq.{w} {iseven}, {half}, {hi};")
        oddv = self.em.reg(rcls)
        self.em.emit(f"selp.{w} {oddv}, {F1}, {self.float_literal(-1.0, ty)}, {iseven};")
        s1 = self.em.reg(rcls)
        NANV = "0d7FF8000000000000" if ty == "f64" else "0f7FC00000"
        self.em.emit(f"selp.{w} {s1}, {oddv}, {NANV}, {isint};")
        sgn = self.em.reg(rcls)
        self.em.emit(f"selp.{w} {sgn}, {s1}, {F1}, {neg};")
        self.em.emit(f"mul.rn.{w} {dst}, {sgn}, {mag};")
        # CUDA powf special values:
        #   pow(x, +/-0) == 1 for every x
        #   pow(+1, y) == 1 for every y
        #   pow(-1, +/-inf) == 1
        #   pow(-0, odd positive integer) == -0
        #   pow(-0, odd negative integer) == -inf
        yzero = self.em.reg("p")
        self.em.emit(f"setp.eq.{w} {yzero}, {args[1]}, {F0};")
        pone = self.em.reg("p")
        self.em.emit(f"setp.eq.{w} {pone}, {args[0]}, {F1};")
        neg1 = self.em.reg("p")
        self.em.emit(
            f"setp.eq.{w} {neg1}, {args[0]}, "
            f"{self.float_literal(-1.0, ty)};"
        )
        INFV = "0d7FF0000000000000" if ty == "f64" else "0f7F800000"
        ay = self.em.reg(rcls)
        self.em.emit(f"abs.{w} {ay}, {args[1]};")
        ninf = self.em.reg("p")
        self.em.emit(f"setp.eq.{w} {ninf}, {ay}, {INFV};")
        self.em.emit(f"and.pred {ninf}, {ninf}, {neg1};")
        # Floating equality cannot distinguish +0 from -0. Inspect the
        # sign bit instead.
        NEGZ = "0d8000000000000000" if ty == "f64" else "0f80000000"
        xbits = self.em.reg("r" if ty == "f32" else "rd")
        bits = 32 if ty == "f32" else 64
        sign_bit = "0x80000000" if ty == "f32" else "0x8000000000000000"
        self.em.emit(f"mov.b{bits} {xbits}, {args[0]};")
        xnegz = self.em.reg("p")
        self.em.emit(f"setp.eq.u{bits} {xnegz}, {xbits}, {sign_bit};")
        ypos = self.em.reg("p")
        self.em.emit(f"setp.gt.{w} {ypos}, {args[1]}, {F0};")
        odd = self.em.reg("p")
        self.em.emit(f"not.pred {odd}, {iseven};")
        self.em.emit(f"and.pred {odd}, {odd}, {isint};")
        zero_odd = self.em.reg("p")
        self.em.emit(f"and.pred {zero_odd}, {xnegz}, {odd};")
        special = self.em.reg("p")
        self.em.emit(f"or.pred {special}, {yzero}, {pone};")
        self.em.emit(f"or.pred {special}, {special}, {ninf};")
        self.em.emit(f"or.pred {special}, {special}, {zero_odd};")
        NEGINF = "0dFFF0000000000000" if ty == "f64" else "0fFF800000"
        zero_value = self.em.reg(rcls)
        self.em.emit(f"selp.{w} {zero_value}, {NEGZ}, {NEGINF}, {ypos};")
        spv = self.em.reg(rcls)
        self.em.emit(f"selp.{w} {spv}, {zero_value}, {F1}, {zero_odd};")
        self.em.emit(f"selp.{w} {dst}, {spv}, {dst}, {special};")

    def _op_equal(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        self.emit_mask_compare(dst, op, ty, arg_tys[0] if arg_tys else ty, args[0], args[1])

    def _op_where(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        p0 = self.mask_pred(args[0], arg_tys[0] if arg_tys else "f32")
        if p in ("u32", "s32") and arg_tys and len(arg_tys) > 2 and arg_tys[1].startswith("f"):
            b1 = self.bool_reg(args[1], arg_tys[1])
            b2 = self.bool_reg(args[2], arg_tys[2])
            self.em.emit(f"selp.{p} {dst}, {b1}, {b2}, {p0};")
        else:
            self.em.emit(f"selp.{p} {dst}, {args[1]}, {args[2]}, {p0};")

    def _op_bitwise_and(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        is_and = op in ("bitwise_and", "logical_and")
        ins = "and" if is_and else "or"
        logical = op.startswith("logical")
        if ty.startswith("f"):
            w = "f64" if ty == "f64" else "f32"
            z = self.fconst(0.0) if ty == "f32" else "0d0000000000000000"
            one = self.fconst(1.0) if ty == "f32" else "0d3FF0000000000000"
            p0 = self.em.reg("p")
            p1 = self.em.reg("p")
            self.em.emit(f"setp.ne.{w} {p0}, {args[0]}, {z};")
            self.em.emit(f"setp.ne.{w} {p1}, {args[1]}, {z};")
            self.em.emit(f"{ins}.pred {p0}, {p0}, {p1};")
            self.em.emit(f"selp.{w} {dst}, {one}, {z}, {p0};")
        elif logical:
            p0 = self.em.reg("p")
            p1 = self.em.reg("p")
            w = self.ptx(ty)
            self.em.emit(f"setp.ne.{w} {p0}, {args[0]}, 0;")
            self.em.emit(f"setp.ne.{w} {p1}, {args[1]}, 0;")
            self.em.emit(f"{ins}.pred {p0}, {p0}, {p1};")
            if ty in ("i64", "u64"):
                t32 = self.em.reg("r")
                self.em.emit(f"selp.u32 {t32}, 1, 0, {p0};")
                self.em.emit(f"cvt.{self.ptx(ty)}.u32 {dst}, {t32};")
            else:
                self.em.emit(f"selp.u32 {dst}, 1, 0, {p0};")
        else:
            w = "b64" if ty in ("i64", "u64") else "b32"
            self.em.emit(f"{ins}.{w} {dst}, {args[0]}, {args[1]};")

    def _op_bitwise_xor(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty.startswith("f"):
            w = "f64" if ty == "f64" else "f32"
            z = self.fconst(0.0) if ty == "f32" else "0d0000000000000000"
            one = self.fconst(1.0) if ty == "f32" else "0d3FF0000000000000"
            p0 = self.em.reg("p")
            p1 = self.em.reg("p")
            self.em.emit(f"setp.ne.{w} {p0}, {args[0]}, {z};")
            self.em.emit(f"setp.ne.{w} {p1}, {args[1]}, {z};")
            self.em.emit(f"xor.pred {p0}, {p0}, {p1};")
            self.em.emit(f"selp.{w} {dst}, {one}, {z}, {p0};")
        elif op == "logical_xor":
            p0 = self.em.reg("p")
            p1 = self.em.reg("p")
            w = self.ptx(ty)
            self.em.emit(f"setp.ne.{w} {p0}, {args[0]}, 0;")
            self.em.emit(f"setp.ne.{w} {p1}, {args[1]}, 0;")
            self.em.emit(f"xor.pred {p0}, {p0}, {p1};")
            if ty in ("i64", "u64"):
                t32 = self.em.reg("r")
                self.em.emit(f"selp.u32 {t32}, 1, 0, {p0};")
                self.em.emit(f"cvt.{self.ptx(ty)}.u32 {dst}, {t32};")
            else:
                self.em.emit(f"selp.u32 {dst}, 1, 0, {p0};")
        else:
            w = "b64" if ty in ("i64", "u64") else "b32"
            self.em.emit(f"xor.{w} {dst}, {args[0]}, {args[1]};")

    def _op_sign(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty.startswith("f"):
            w = self.ptx(ty)
            rcls = REG_CLASS[ty]
            pn = self.em.reg("p")
            pp = self.em.reg("p")
            t = self.em.reg(rcls)
            F0 = self.float_literal(0.0, ty)
            self.em.emit(f"setp.lt.{w} {pn}, {args[0]}, {F0};")
            self.em.emit(f"setp.gt.{w} {pp}, {args[0]}, {F0};")
            self.em.emit(f"selp.{w} {t}, {self.float_literal(-1.0, ty)}, {F0}, {pn};")
            self.em.emit(f"selp.{w} {dst}, {self.float_literal(1.0, ty)}, {t}, {pp};")
        else:
            # int sign: dst = (x > 0) - (x < 0), computed as selects
            w = self.ptx(ty)
            pp = self.em.reg("p")
            pn = self.em.reg("p")
            self.em.emit(f"setp.gt.{w} {pp}, {args[0]}, 0;")
            self.em.emit(f"setp.lt.{w} {pn}, {args[0]}, 0;")
            neg = self.em.reg("r")
            self.em.emit(f"selp.s32 {neg}, -1, 0, {pn};")
            if ty in ("i64", "u64"):
                t32 = self.em.reg("r")
                self.em.emit(f"selp.s32 {t32}, 1, {neg}, {pp};")
                self.em.emit(f"cvt.{self.ptx(ty)}.s32 {dst}, {t32};")
            elif ty in ("i32",):
                self.em.emit(f"selp.s32 {dst}, 1, {neg}, {pp};")
            else:
                self.em.emit(f"selp.u32 {dst}, 1, {neg}, {pp};")

    def _op_bitwise_not(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty.startswith("f") and op == "bitwise_not":
            raise NotImplementedError("bitwise_not on float")
        if op == "logical_not" and not ty.startswith("f"):
            w = self.ptx(ty)
            p = self.em.reg("p")
            self.em.emit(f"setp.eq.{w} {p}, {args[0]}, 0;")
            if ty in ("i64", "u64"):
                t32 = self.em.reg("r")
                self.em.emit(f"selp.u32 {t32}, 1, 0, {p};")
                self.em.emit(f"cvt.{w}.u32 {dst}, {t32};")
            else:
                self.em.emit(f"selp.u32 {dst}, 1, 0, {p};")
        elif ty == "bool":
            self.em.emit(f"xor.b32 {dst}, {args[0]}, 1;")
        elif ty.startswith("f"):
            p0 = self.em.reg("p")
            w = "f64" if ty == "f64" else "f32"
            z = self.fconst(0.0) if ty == "f32" else "0d0000000000000000"
            one = self.fconst(1.0) if ty == "f32" else "0d3FF0000000000000"
            self.em.emit(f"setp.eq.{w} {p0}, {args[0]}, {z};")
            self.em.emit(f"selp.{w} {dst}, {one}, {z}, {p0};")
        else:
            w = "b64" if ty in ("i64", "u64") else "b32"
            self.em.emit(f"not.{w} {dst}, {args[0]};")

    def _op_left_shift(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty.startswith("f"):
            raise NotImplementedError("shift on float")
        unsigned = "u" if ty.startswith("u") else "s"
        if ty in ("i64", "u64"):
            # 64-bit shifts take a .u32 amount operand
            amt = self.em.reg("r")
            self.em.emit(f"cvt.u32.{self.ptx(ty)} {amt}, {args[1]};")
            ins = "shl.b64" if op == "left_shift" else f"shr.{unsigned}64"
            self.em.emit(f"{ins} {dst}, {args[0]}, {amt};")
        else:
            ins = "shl.b32" if op == "left_shift" else f"shr.{unsigned}32"
            self.em.emit(f"{ins} {dst}, {args[0]}, {args[1]};")

    def _op_bitwise_count(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty.startswith("f"):
            raise NotImplementedError("popc on float")
        if ty in ("i64", "u64"):
            t32 = self.em.reg("r")
            self.em.emit(f"popc.b64 {t32}, {args[0]};")
            self.em.emit(f"cvt.{self.ptx(ty)}.u32 {dst}, {t32};")
        else:
            self.em.emit(f"popc.b32 {dst}, {args[0]};")

    def _op_fma(self, op, dst, ty, p, rp, rcls, args, arg_tys=None):
        if ty.startswith("f"):
            self.em.emit(f"fma.rn.{p} {dst}, {args[0]}, {args[1]}, {args[2]};")
        else:
            self.em.emit(f"mad.lo.{p} {dst}, {args[0]}, {args[1]}, {args[2]};")
    def emit_sigmoid(self, dst, src, mul_by_src, ty="f32"):
        w = self.ptx(ty)
        rcls = REG_CLASS[ty]
        e = self.em.reg(rcls)
        self.em.emit(f"mul.rn.{w} {e}, {src}, {self.float_literal(-1.4426950408889634, ty)};")
        self.em.emit(f"ex2.approx.{w} {e}, {e};")
        d = self.em.reg(rcls)
        self.em.emit(f"add.rn.{w} {d}, {e}, {self.float_literal(1.0, ty)};")
        self.em.emit(f"rcp.rn.{w} {d}, {d};")
        if mul_by_src:
            self.em.emit(f"mul.rn.{w} {dst}, {src}, {d};")
        else:
            self.em.emit(f"mov.{w} {dst}, {d};")

    def emit_log(self, dst, src, base2, ty="f32"):
        w = self.ptx(ty)
        self.em.emit(f"lg2.approx.{w} {dst}, {src};")
        if not base2:
            self.em.emit(f"mul.rn.{w} {dst}, {dst}, {self.float_literal(0.6931471805599453, ty)};")

    def emit_erf(self, dst, src, ty="f32"):
        w = self.ptx(ty)
        rcls = REG_CLASS[ty]
        C = lambda v: self.float_literal(v, ty)
        a = self.em.reg(rcls)
        p = self.em.reg("p")
        self.em.emit(f"abs.{w} {a}, {src};")
        t = self.em.reg(rcls)
        self.em.emit(f"fma.rn.{w} {t}, {a}, {C(0.3275911)}, {C(1.0)};")
        self.em.emit(f"rcp.rn.{w} {t}, {t};")
        poly = self.em.reg(rcls)
        self.em.emit(f"mul.rn.{w} {poly}, {t}, {C(1.061405429)};")
        self.em.emit(f"add.rn.{w} {poly}, {poly}, {C(-1.453152027)};")
        self.em.emit(f"mul.rn.{w} {poly}, {poly}, {t};")
        self.em.emit(f"add.rn.{w} {poly}, {poly}, {C(1.421413741)};")
        self.em.emit(f"mul.rn.{w} {poly}, {poly}, {t};")
        self.em.emit(f"add.rn.{w} {poly}, {poly}, {C(-0.284496736)};")
        self.em.emit(f"mul.rn.{w} {poly}, {poly}, {t};")
        self.em.emit(f"add.rn.{w} {poly}, {poly}, {C(0.254829592)};")
        self.em.emit(f"mul.rn.{w} {poly}, {poly}, {t};")
        e = self.em.reg(rcls)
        neg = self.em.reg(rcls)
        self.em.emit(f"mul.{w} {neg}, {a}, {a};")
        self.em.emit(f"neg.{w} {neg}, {neg};")
        self.fexp(e, neg, ty)
        self.em.emit(f"mul.rn.{w} {poly}, {poly}, {e};")
        self.em.emit(f"sub.rn.{w} {dst}, {C(1.0)}, {poly};")
        skip = self.em.label("ERF_SKIP")
        self.em.emit(f"setp.lt.{w} {p}, {src}, {C(0.0)};")
        self.em.emit(f"@!{p} bra {skip};")
        self.em.emit(f"neg.{w} {dst}, {dst};")
        self.em.mark(skip)

    def fexp(self, dst, src, ty="f32"):
        w = self.ptx(ty)
        rcls = REG_CLASS[ty]
        tmp = self.em.reg(rcls)
        self.em.emit(f"mul.rn.{w} {tmp}, {src}, {self.float_literal(1.4426950408889634, ty)};")
        self.em.emit(f"ex2.approx.{w} {dst}, {tmp};")
