#!/usr/bin/env python3.12
import json
import struct
import sys
from math import prod
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var

INT_TYPES = {"i32", "i64", "bool", "u8", "u16", "i16", "u32", "u64"}
REG_CLASS = {"f32": "f", "f64": "fd", "i32": "r", "i64": "rd", "bool": "r",
             "u8": "r", "u16": "r", "i16": "r", "u32": "r", "u64": "rd"}
PTX_TY = {"f32": "f32", "f64": "f64", "i32": "s32", "i64": "s64", "bool": "u32",
          "u8": "u8", "u16": "u16", "i16": "s16", "u32": "u32", "u64": "u64"}
WIDEN_AT_LOAD = {"f16", "bf16"}  # narrow types compute at f32
SMEM_NBYTES = {
    "float": 4, "float32": 4, "double": 8, "float64": 8,
    "half": 2, "float16": 2, "bfloat16": 2,
    "int": 4, "int32": 4, "unsigned int": 4, "uint32_t": 4,
    "long long": 8, "int64_t": 8, "unsigned long long": 8, "uint64_t": 8,
    "char": 1, "uint8_t": 1, "signed char": 1,
    "i64": 8, "bool": 1, "u8": 1, "u16": 2, "i16": 2, "u32": 4, "u64": 8, "bf16": 2,
    "short": 2, "unsigned short": 2, "long": 8, "unsigned long": 8,
    "int8_t": 1, "int16_t": 2, "uint16_t": 2, "int32_t": 4, "uint32_t": 4,
    "__half": 2, "__half2": 4, "__nv_bfloat16": 2,
    "__nv_fp8_e4m3": 1, "__nv_fp8_e5m2": 1,
}


class Emitter:
    def __init__(self):
        self.lines = []
        self.counts = {"p": 0, "f": 0, "fd": 0, "r": 0, "rd": 0, "h": 0}
        self.label_n = 0

    def reg(self, cls):
        self.counts[cls] += 1
        return f"%{cls}{self.counts[cls]}"

    def emit(self, text):
        self.lines.append("\t" + text)

    def mark(self, name):
        self.lines.append(f"{name}:")

    def label(self, stem):
        self.label_n += 1
        return f"{stem}_{self.label_n}"


class _Pair:
    """Emission-time marker: two scalar Loads merged into one vector load."""
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a = a
        self.b = b


# compile_speed_safe: cse/loadcache/addrcache/fold/pow2 physically removed.
# Zero measured gain on the official corpus (ablation, Oct 28) while carrying
# the two highest-risk bug classes (dyn-leak, Cond-else addr reuse).


class Compiler:
    def __init__(self, graph, meta, kernel):
        self.g = graph
        self.meta = meta
        self.node_name = kernel
        self.op = graph.nodes[kernel].op
        self.entry = self.op.name or meta["kernels"][0].get("entry") or kernel
        self.em = Emitter()
        self.values = {}
        self.ints = {}
        self.int_tys = {}
        self.bases = {}
        self.shapes = {}
        self.dtypes = {}
        self.expr_cache = {}
        self.shared = {}
        self.shared_bases = {}
        self.tid_x = None
        self.smem_used = 0
        self.max_smem_align = 16

        for node in graph.nodes.values():
            for t in node.outputs:
                self.shapes[t.name] = tuple(getattr(d, "value", d) for d in t.shape)
                self.dtypes[t.name] = t.dtype.name

        info = next((k for k in meta["kernels"] if k["entry"] == self.entry), None)
        if info is None:
            info = meta["kernels"][0]
        self.arg_order = list(info["arg_order"])
        self.scalar_tys = {
            a["name"]: a["dtype"] for a in info.get("args", []) if a.get("scalar")
        }
        self.grid = tuple(info["grid"])
        self.block = tuple(info["block"])
        self.smem_bytes = info.get("smem_bytes", 0)
        self.dynamic_smem = self.smem_bytes > 0

    def prologue(self):
        for name in self.arg_order:
            if name in self.scalar_tys:
                ty = self.scalar_tys[name]
                p = self.em.reg(REG_CLASS.get(ty, "r"))
                self.em.emit(f"ld.param.{self.ptx(ty)} {p}, [param_{name}];")
                self.bases[name] = None
                if ty in INT_TYPES:
                    self.ints[name] = ("reg", p)
                    self.int_tys[name] = ty
                continue
            p = self.em.reg("rd")
            self.em.emit(f"ld.param.u64 {p}, [param_{name}];")
            b = self.em.reg("rd")
            self.em.emit(f"cvta.to.global.u64 {b}, {p};")
            self.bases[name] = b

    def global_tid(self):
        gx, gy, gz = self.grid
        bx, by, bz = self.block
        tx = self.em.reg("r")
        self.em.emit(f"mov.u32 {tx}, %tid.x;")
        self.tid_x = tx
        tlin = tx
        if by * bz > 1:
            ty = self.em.reg("r")
            self.em.emit(f"mov.u32 {ty}, %tid.y;")
            t = self.em.reg("r")
            self.em.emit(f"mul.lo.s32 {t}, {ty}, {bx};")
            tlin = self.em.reg("r")
            self.em.emit(f"add.s32 {tlin}, {tx}, {t};")
        if bz > 1:
            tz = self.em.reg("r")
            self.em.emit(f"mov.u32 {tz}, %tid.z;")
            t = self.em.reg("r")
            self.em.emit(f"mul.lo.s32 {t}, {tz}, {bx * by};")
            t2 = self.em.reg("r")
            self.em.emit(f"add.s32 {t2}, {tlin}, {t};")
            tlin = t2
        ct = self.em.reg("r")
        self.em.emit(f"mov.u32 {ct}, %ctaid.x;")
        clin = ct
        if gy * gz > 1:
            cy = self.em.reg("r")
            self.em.emit(f"mov.u32 {cy}, %ctaid.y;")
            t = self.em.reg("r")
            self.em.emit(f"mul.lo.s32 {t}, {cy}, {gx};")
            c2 = self.em.reg("r")
            self.em.emit(f"add.s32 {c2}, {clin}, {t};")
            clin = c2
        if gz > 1:
            cz = self.em.reg("r")
            self.em.emit(f"mov.u32 {cz}, %ctaid.z;")
            t = self.em.reg("r")
            self.em.emit(f"mul.lo.s32 {t}, {cz}, {gx * gy};")
            c2 = self.em.reg("r")
            self.em.emit(f"add.s32 {c2}, {clin}, {t};")
            clin = c2
        threads = bx * by * bz
        if gx * gy * gz > 1:
            t = self.em.reg("r")
            self.em.emit(f"mul.lo.s32 {t}, {clin}, {threads};")
            gid = self.em.reg("r")
            self.em.emit(f"add.s32 {gid}, {t}, {tlin};")
        else:
            gid = tlin
        return gid

    def materialize(self, v):
        if v[0] == "reg":
            return v[1]
        if v[0] == "regimm":
            if v[2] == 0:
                return v[1]
            r = self.em.reg("r")
            self.em.emit(f"add.s32 {r}, {v[1]}, {v[2]};")
            return r
        r = self.em.reg("r")
        self.em.emit(f"mov.s32 {r}, {v[1]};")
        return r

    def iadd(self, a, b):
        if a[0] == "imm" and b[0] == "imm":
            return ("imm", a[1] + b[1])
        if a[0] == "imm":
            a, b = b, a
        if b[0] == "imm":
            if b[1] == 0:
                return a
            if a[0] == "regimm":
                return ("regimm", a[1], a[2] + b[1])
            if a[0] == "reg":
                return ("regimm", a[1], b[1])
        if a[0] == "regimm" and b[0] == "regimm":
            inner = self.iadd(("reg", a[1]), ("reg", b[1]))
            return ("regimm", inner[1], a[2] + b[2])
        # Reassociation: (r + c) + x  ==  (r + x) + c — keep the constant lazy
        # so per-copy offsets fold into addressing modes instead of registers.
        # The register side is copy-invariant, so VN dedupes the emitted add.
        if a[0] == "regimm" and b[0] == "reg":
            inner = self.iadd(("reg", a[1]), b)
            return ("regimm", inner[1], a[2])
        if b[0] == "regimm" and a[0] == "reg":
            inner = self.iadd(a, ("reg", b[1]))
            return ("regimm", inner[1], b[2])
        r = self.em.reg("r")
        self.em.emit(f"add.s32 {r}, {self.materialize(a)}, {self.materialize(b)};")
        return ("reg", r)

    def isub(self, a, b):
        if a[0] == "imm" and b[0] == "imm":
            return ("imm", a[1] - b[1])
        if b[0] == "imm":
            if b[1] == 0:
                return a
            if a[0] == "reg":
                return ("regimm", a[1], -b[1])
            if a[0] == "regimm":
                return ("regimm", a[1], a[2] - b[1])
        if a[0] == "regimm" and b[0] == "regimm" and a[1] == b[1]:
            return ("imm", a[2] - b[2])
        if a[0] == "regimm" and b[0] == "reg":
            r = self.em.reg("r")
            self.em.emit(f"sub.s32 {r}, {a[1]}, {b[1]};")
            return ("regimm", r, a[2])
        r = self.em.reg("r")
        self.em.emit(f"sub.s32 {r}, {self.materialize(a)}, {self.materialize(b)};")
        return ("reg", r)

    def imul(self, a, b):
        if a[0] == "imm" and b[0] == "imm":
            return ("imm", a[1] * b[1])
        if a[0] == "imm":
            a, b = b, a
        if b[0] == "imm":
            if b[1] == 0:
                return ("imm", 0)
            if b[1] == 1:
                return a
            if a[0] == "reg":
                return ("reg", self.emit_bin("mul.lo.s32", a, b))
            if a[0] == "regimm":
                r = self.em.reg("r")
                self.em.emit(f"mul.lo.s32 {r}, {a[1]}, {b[1]};")
                return ("regimm", r, a[2] * b[1])
        r = self.em.reg("r")
        self.em.emit(f"mul.lo.s32 {r}, {self.materialize(a)}, {self.materialize(b)};")
        return ("reg", r)

    def emit_bin(self, ins, a, b):
        r = self.em.reg("r")
        aa = a[1] if a[0] == "imm" else self.materialize(a)
        bb = b[1] if b[0] == "imm" else self.materialize(b)
        self.em.emit(f"{ins} {r}, {aa}, {bb};")
        return r

    def idiv(self, a, b):
        if a[0] == "imm" and b[0] == "imm":
            return ("imm", int(a[1] / b[1]))  # truncation toward zero, matches PTX div.s32
        return ("reg", self.emit_bin("div.s32", a, b))

    def irem(self, a, b):
        if a[0] == "imm" and b[0] == "imm":
            # C-style remainder (sign of dividend), matches PTX rem.s32
            return ("imm", a[1] - b[1] * int(a[1] / b[1]))
        return ("reg", self.emit_bin("rem.s32", a, b))

    def ekey(self, e):
        t = type(e).__name__
        if t == "Literal":
            return ("l", e.value)
        if t == "Var":
            return ("v", e.name)
        if t == "Builtin":
            return ("bi", e.name)
        if t == "CastExpr":
            return ("cast", e.dtype, self.ekey(e.expr))
        if t == "BinaryExpr":
            return ("b", e.op, self.ekey(e.left), self.ekey(e.right))
        if t == "TernaryExpr":
            return ("t", self.ekey(e.cond), self.ekey(e.if_true), self.ekey(e.if_false))
        raise NotImplementedError(f"expression key for {t}")

    def iexpr(self, e):
        key = self.ekey(e)
        if key in self.expr_cache:
            return self.expr_cache[key]
        t = type(e).__name__
        if t == "Literal":
            result = ("imm", int(e.value))
        elif t == "Builtin":
            # hardware specials read straight into an integer register
            # emmy's own passes only construct Builtin("thread_idx.x"); the
            # remaining hardware specials mirror its render table 1:1.
            PTX_SPECIAL = {
                "thread_idx.x": "%tid.x", "thread_idx.y": "%tid.y", "thread_idx.z": "%tid.z",
                "block_idx.x": "%ctaid.x", "block_idx.y": "%ctaid.y", "block_idx.z": "%ctaid.z",
                "block_dim.x": "%ntid.x", "block_dim.y": "%ntid.y", "block_dim.z": "%ntid.z",
                "grid_dim.x": "%nctaid.x", "grid_dim.y": "%nctaid.y", "grid_dim.z": "%nctaid.z",
            }
            r = self.em.reg("r")
            self.em.emit(f"mov.u32 {r}, {PTX_SPECIAL[e.name]};")
            result = ("reg", r)
        elif t == "CastExpr":
            # C cast semantics in index/cond position: truncate float -> int
            if e.dtype in ("i32", "i64"):
                src_e = e.expr
                if type(src_e).__name__ == "Var" and src_e.name in self.values:
                    reg, sty = self.values[src_e.name]
                    if sty.startswith("f"):
                        r = self.em.reg(REG_CLASS[e.dtype])
                        self.em.emit(f"cvt.rzi.{self.ptx(e.dtype)}.{sty} {r}, {reg};")
                        result = ("reg", r)
                    else:
                        result = self.iexpr(src_e)
                else:
                    result = self.iexpr(src_e)
            else:
                raise NotImplementedError(f"CastExpr to {e.dtype}")
        elif t == "Var":
            if e.name in self.ints:
                result = self.ints[e.name]
            elif e.name in self.values and self.values[e.name][1] in INT_TYPES:
                result = ("reg", self.values[e.name][0])
            else:
                raise KeyError(f"unknown integer variable {e.name!r}")
        elif t == "BinaryExpr":
            if e.op in ("==", "!=", "<", "<=", ">", ">="):
                kind = self.cmp_kind(e.left, e.right)
                av = self.cmp_operand(e.left, kind)
                bv = self.cmp_operand(e.right, kind)
                cmp = {"==": "eq", "!=": "eq", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[e.op]
                suffix = {"int": "s32", "u32": "u32", "s64": "s64", "u64": "u64",
                          "f32": "f32", "f64": "f64"}[kind]
                p = self.em.reg("p")
                self.em.emit(f"setp.{cmp}.{suffix} {p}, {av}, {bv};")
                if e.op == "!=":
                    self.em.emit(f"not.pred {p}, {p};")
                d = self.em.reg("r")
                self.em.emit(f"selp.u32 {d}, 1, 0, {p};")
                result = ("reg", d)
            elif e.op in ("&&", "||"):
                pa = self.predicate(e.left)
                pb = self.predicate(e.right)
                p = self.em.reg("p")
                self.em.emit(f"{'and' if e.op == '&&' else 'or'}.pred {p}, {pa}, {pb};")
                d = self.em.reg("r")
                self.em.emit(f"selp.u32 {d}, 1, 0, {p};")
                result = ("reg", d)
            else:
                a = self.iexpr(e.left)
                b = self.iexpr(e.right)
                if e.op == "//":
                    result = self.idiv(a, b)
                elif e.op in ("<<", ">>", "&", "|", "^"):
                    ins = {"<<": "shl.b32", ">>": "shr.s32", "&": "and.b32",
                           "|": "or.b32", "^": "xor.b32"}[e.op]
                    d = self.em.reg("r")
                    self.em.emit(f"{ins} {d}, {self.materialize(a)}, {self.materialize(b)};")
                    result = ("reg", d)
                else:
                    result = {
                        "+": self.iadd,
                        "-": self.isub,
                        "*": self.imul,
                        "/": self.idiv,
                        "%": self.irem,
                    }[e.op](a, b)
        elif t == "TernaryExpr":
            p = self.predicate(e.cond)
            a = self.iexpr(e.if_true)
            b = self.iexpr(e.if_false)
            d = self.em.reg("r")
            self.em.emit(
                f"selp.s32 {d}, {self.materialize(a)}, {self.materialize(b)}, {p};"
            )
            result = ("reg", d)
        else:
            raise NotImplementedError(f"integer expression {t}")
        self.expr_cache[key] = result
        return result

    def linear_index(self, index, shape):
        if not index:
            return ("imm", 0)
        if len(index) == 1:
            return self.iexpr(index[0])
        acc = ("imm", 0)
        for i, e in enumerate(index):
            suffix = prod(shape[i + 1 :]) if i + 1 < len(shape) else 1
            term = self.iexpr(e)
            if suffix != 1:
                term = self.imul(term, ("imm", suffix))
            acc = self.iadd(acc, term)
        return acc

    def address(self, buf, index, nbytes):
        off = self.linear_index(index, self.shapes.get(buf, ()))
        return self._addr_from_off(buf, off, nbytes)

    def _addr_from_off(self, buf, off, nbytes):
        """Stateless folded addressing: immediate offsets fold into the
        ld/st operand; regimm recomputes its base each time (no cache)."""
        base = self.bases[buf]
        if off[0] == "imm":
            k = off[1] * nbytes
            return base if k == 0 else f"{base}+{k}"
        if off[0] == "regimm" and off[2] * nbytes >= 0:
            addr = self.base_address(buf, off[1], nbytes)
            k = off[2] * nbytes
            return addr if k == 0 else f"{addr}+{k}"
        reg = self.materialize(off)
        return self.base_address(buf, reg, nbytes)

    def base_address(self, buf, reg, nbytes):
        key = (buf, reg, nbytes)
        base = self.bases[buf]
        rd = self.em.reg("rd")
        self.em.emit(f"mul.wide.s32 {rd}, {reg}, {nbytes};")
        addr = self.em.reg("rd")
        self.em.emit(f"add.s64 {addr}, {base}, {rd};")
        return addr

    def ptx(self, ty):
        return PTX_TY.get(ty, ty)

    def identity(self, op, ty):
        if ty.startswith("f"):
            value = {"add": 0.0, "multiply": 1.0, "maximum": -1e30, "minimum": 1e30}[op]
            return self.float_literal(value, ty)
        if ty == "i64":
            return {"add": "0", "multiply": "1", "maximum": "-9223372036854775808",
                    "minimum": "9223372036854775807"}[op]
        return {"add": "0", "multiply": "1", "maximum": "-2147483648", "minimum": "2147483647"}[op]

    def fconst(self, value):
        return self.float_literal(value, "f32")

    def fexp(self, dst, src, ty="f32"):
        w = self.ptx(ty)
        rcls = REG_CLASS[ty]
        tmp = self.em.reg(rcls)
        self.em.emit(f"mul.rn.{w} {tmp}, {src}, {self.float_literal(1.4426950408889634, ty)};")
        self.em.emit(f"ex2.approx.{w} {dst}, {tmp};")


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


    def emit_mask_compare(self, dst, op, rty, aty, a, b):
        cmp = {
            "equal": "eq", "not_equal": "eq", "greater": "gt",
            "less": "lt", "greater_equal": "ge", "less_equal": "le",
        }[op]
        p = self.em.reg("p")
        if aty.startswith("f"):
            suffix = "f64" if aty == "f64" else "f32"
            self.em.emit(f"setp.{cmp}.{suffix} {p}, {a}, {b};")
        elif aty == "u64":
            self.em.emit(f"setp.{cmp}.u64 {p}, {a}, {b};")
        elif aty == "i64":
            self.em.emit(f"setp.{cmp}.s64 {p}, {a}, {b};")
        elif aty == "u32":
            self.em.emit(f"setp.{cmp}.u32 {p}, {a}, {b};")
        else:
            self.em.emit(f"setp.{cmp}.s32 {p}, {a}, {b};")
        if op == "not_equal":
            # PTX setp is ordered; C '!=' is true on NaN
            self.em.emit(f"not.pred {p}, {p};")
        if rty.startswith("f"):
            one = self.float_literal(1.0, rty)
            zero = self.float_literal(0.0, rty)
            self.em.emit(f"selp.{self.ptx(rty)} {dst}, {one}, {zero}, {p};")
        elif rty in ("i64", "u64"):
            t32 = self.em.reg("r")
            self.em.emit(f"selp.u32 {t32}, 1, 0, {p};")
            self.em.emit(f"cvt.{self.ptx(rty)}.u32 {dst}, {t32};")
        else:
            self.em.emit(f"selp.u32 {dst}, 1, 0, {p};")


    def mask_pred(self, reg, ty):
        p = self.em.reg("p")
        if ty in ("i64", "u64"):
            self.em.emit(f"setp.ne.{self.ptx(ty)} {p}, {reg}, 0;")
            return p
        if ty.startswith("f"):
            # C truthiness (x != 0): true on NaN — build via eq + not
            pt = self.em.reg("p")
            w = "f64" if ty == "f64" else "f32"
            self.em.emit(f"setp.eq.{w} {pt}, {reg}, {self.float_literal(0.0, ty)};")
            self.em.emit(f"not.pred {p}, {pt};")
        else:
            self.em.emit(f"setp.ne.u32 {p}, {reg}, 0;")
        return p


    def bool_reg(self, reg, ty):
        p = self.em.reg("p")
        if ty.startswith("f"):
            w = "f64" if ty == "f64" else "f32"
            self.em.emit(f"setp.ne.{w} {p}, {reg}, {self.float_literal(0.0, ty)};")
        else:
            self.em.emit(f"setp.ne.s32 {p}, {reg}, 0;")
        r = self.em.reg("r")
        self.em.emit(f"selp.u32 {r}, 1, 0, {p};")
        return r

    F32_ONLY = {
        "sin", "cos",
    }

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
        if op == "relu":
            if ty == "f64":
                zero = "0d0000000000000000"
            elif ty.startswith("f"):
                zero = "0f00000000"
            else:
                zero = "0"
            self.em.emit(f"max.{p} {dst}, {args[0]}, {zero};")
        elif op == "add":
            self.em.emit(f"add.{rp} {dst}, {args[0]}, {args[1]};")
        elif op == "multiply":
            ins = f"mul.{rp}" if ty.startswith("f") else f"mul.lo.{p}"
            self.em.emit(f"{ins} {dst}, {args[0]}, {args[1]};")
        elif op == "subtract":
            self.em.emit(f"sub.{rp} {dst}, {args[0]}, {args[1]};")
        elif op == "divide":
            if ty.startswith("f"):
                self.em.emit(f"div.rn.{p} {dst}, {args[0]}, {args[1]};")
            else:
                self.em.emit(f"div.{p} {dst}, {args[0]}, {args[1]};")
        elif op in ("floor_divide", "remainder", "mod", "fmod"):
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
        elif op == "maximum":
            self.em.emit(f"max.{p} {dst}, {args[0]}, {args[1]};")
        elif op == "minimum":
            self.em.emit(f"min.{p} {dst}, {args[0]}, {args[1]};")
        elif op in ("neg", "negative"):
            if ty in ("u32", "u64", "bool", "u8", "u16"):
                self.em.emit(f"sub.{p} {dst}, 0, {args[0]};")
            else:
                self.em.emit(f"neg.{p} {dst}, {args[0]};")
        elif op in ("abs", "fabs"):
            if ty in ("u32", "u64", "u8", "u16", "bool"):
                self.em.emit(f"mov.{p} {dst}, {args[0]};")  # unsigned abs is identity
            else:
                self.em.emit(f"abs.{p} {dst}, {args[0]};")
        elif op in ("copy", "pad"):
            self.em.emit(f"mov.{p} {dst}, {args[0]};")
        elif op == "sqrt":
            self.em.emit(f"sqrt.rn.{p} {dst}, {args[0]};")
        elif op == "rsqrt":
            t = self.em.reg(rcls)
            self.em.emit(f"sqrt.rn.{p} {t}, {args[0]};")
            self.em.emit(f"rcp.rn.{p} {dst}, {t};")
        elif op == "reciprocal":
            self.em.emit(f"rcp.rn.{p} {dst}, {args[0]};")
        elif op in ("exp", "exp_fast"):
            if ty != "f32":
                raise NotImplementedError("exp only implemented for f32")
            self.fexp(dst, args[0])
        elif op == "exp2":
            self.em.emit(f"ex2.approx.f32 {dst}, {args[0]};")
        elif op in ("log", "log2"):
            self.emit_log(dst, args[0], op == "log2")
        elif op == "log1p":
            t = self.em.reg("f")
            self.em.emit(f"add.rn.f32 {t}, {args[0]}, {self.fconst(1.0)};")
            self.emit_log(dst, t, False)
        elif op in ("sin", "cos"):
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
        elif op == "tanh":
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
        elif op == "sigmoid":
            self.emit_sigmoid(dst, args[0], False, ty)
        elif op == "silu":
            self.emit_sigmoid(dst, args[0], True, ty)
        elif op == "softplus":
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
        elif op == "erf":
            self.emit_erf(dst, args[0], ty)
        elif op == "square":
            ins = f"mul.{p}" if ty.startswith("f") else f"mul.lo.{p}"
            self.em.emit(f"{ins} {dst}, {args[0]}, {args[0]};")
        elif op in ("floor", "ceil", "round", "trunc"):
            if ty not in ("f32", "f64"):
                self.em.emit(f"mov.{p} {dst}, {args[0]};")  # integers: exact identity
            else:
                mode = {"floor": "rmi", "ceil": "rpi", "round": "rni", "trunc": "rzi"}[op]
                self.em.emit(f"cvt.{mode}.{p}.{p} {dst}, {args[0]};")
        elif op == "frac":
            if ty not in ("f32", "f64"):
                self.em.emit(f"mov.{p} {dst}, 0;")  # integers have no fractional part
            else:
                t = self.em.reg(rcls)
                self.em.emit(f"cvt.rzi.{p}.{p} {t}, {args[0]};")
                self.em.emit(f"sub.rn.{p} {dst}, {args[0]}, {t};")
        elif op == "pow":
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
        elif op in ("equal", "not_equal", "greater", "less", "greater_equal", "less_equal"):
            self.emit_mask_compare(dst, op, ty, arg_tys[0] if arg_tys else ty, args[0], args[1])
        elif op == "where":
            p0 = self.mask_pred(args[0], arg_tys[0] if arg_tys else "f32")
            if p in ("u32", "s32") and arg_tys and len(arg_tys) > 2 and arg_tys[1].startswith("f"):
                b1 = self.bool_reg(args[1], arg_tys[1])
                b2 = self.bool_reg(args[2], arg_tys[2])
                self.em.emit(f"selp.{p} {dst}, {b1}, {b2}, {p0};")
            else:
                self.em.emit(f"selp.{p} {dst}, {args[1]}, {args[2]}, {p0};")
        elif op in ("bitwise_and", "bitwise_or", "logical_and", "logical_or"):
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
        elif op in ("bitwise_xor", "logical_xor"):
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
        elif op == "sign":
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
        elif op in ("bitwise_not", "logical_not"):
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
        elif op in ("left_shift", "right_shift"):
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
        elif op == "bitwise_count":
            if ty.startswith("f"):
                raise NotImplementedError("popc on float")
            if ty in ("i64", "u64"):
                t32 = self.em.reg("r")
                self.em.emit(f"popc.b64 {t32}, {args[0]};")
                self.em.emit(f"cvt.{self.ptx(ty)}.u32 {dst}, {t32};")
            else:
                self.em.emit(f"popc.b32 {dst}, {args[0]};")
        elif op in ("fma", "mul_add"):
            if ty.startswith("f"):
                self.em.emit(f"fma.rn.{p} {dst}, {args[0]}, {args[1]}, {args[2]};")
            else:
                self.em.emit(f"mad.lo.{p} {dst}, {args[0]}, {args[1]}, {args[2]};")
        else:
            raise NotImplementedError(f"elementwise op {op!r}")


    def accum_reg(self, s):
        if s.name in self.values:
            return self.values[s.name]
        ty = s.dtype.name if s.dtype is not None else "f32"
        if ty in WIDEN_AT_LOAD:
            ty = "f32"
        reg = self.em.reg(REG_CLASS[ty])
        self.em.emit(f"mov.{self.ptx(ty)} {reg}, {self.identity(s.op.name, ty)};")
        self.values[s.name] = (reg, ty)
        return reg, ty

    def init_accumulators(self, body):
        # Accums nested under a Cond inside a seeded loop body must still get
        # their identity before the loop, not lazily at first use inside the
        # guard (which would reset the accumulator every iteration).
        for s in body:
            k = type(s).__name__
            if k == "Accum":
                self.accum_reg(s)
            elif k == "Cond":
                for sub in (s.body, s.else_body):
                    if sub:
                        self.init_accumulators(sub)

    def define_specials(self):
        if self.tid_x is None:
            return
        lane = self.em.reg("r")
        self.em.emit(f"and.b32 {lane}, {self.tid_x}, 31;")
        self.ints["lane"] = ("reg", lane)
        self.int_tys["lane"] = "u32"
        warp = self.em.reg("r")
        self.em.emit(f"shr.u32 {warp}, {self.tid_x}, 5;")
        self.ints["warp"] = ("reg", warp)
        self.int_tys["warp"] = "u32"

    def int_operand(self, v):
        if isinstance(v, int):
            return str(v)
        r = self.iexpr(v)
        if r[0] == "imm":
            return str(r[1])
        return self.materialize(r)

    def smem_nbytes(self, dtype):
        if dtype in SMEM_NBYTES:
            return SMEM_NBYTES[dtype]
        return getattr(dtype, "nbytes", 4)

    def smem_blob_base(self):
        if "_blob" not in self.shared_bases:
            r = self.em.reg("r")
            self.em.emit(f"mov.u32 {r}, __emmy_smem;")
            self.shared_bases["_blob"] = r
        return self.shared_bases["_blob"]

    def smem_base(self, buf):
        if buf not in self.shared_bases:
            if self.dynamic_smem:
                blob = self.smem_blob_base()
                off = self.shared[buf]["offset"]
                if off:
                    r = self.em.reg("r")
                    self.em.emit(f"add.s32 {r}, {blob}, {off};")
                    base = r
                else:
                    base = blob
            else:
                base = self.em.reg("r")
                self.em.emit(f"mov.u32 {base}, {buf};")
            self.shared_bases[buf] = base
        return self.shared_bases[buf]

    def smem_address_off(self, buf, off):
        info = self.shared[buf]
        nbytes = info["nbytes"]
        base = self.smem_base(buf)
        if off[0] == "imm":
            k = off[1] * nbytes
            return base if k == 0 else f"{base}+{k}"
        if off[0] == "regimm" and off[2] * nbytes >= 0:
            k = off[2] * nbytes
            addr = self.smem_scaled_base(buf, off[1])
            return addr if k == 0 else f"{addr}+{k}"
        return self.smem_scaled_base(buf, self.materialize(off))

    def smem_scaled_base(self, buf, reg):
        nbytes = self.shared[buf]["nbytes"]
        key = (buf, reg, nbytes)
        base = self.smem_base(buf)
        addr = self.em.reg("r")
        if nbytes == 1:
            self.em.emit(f"add.s32 {addr}, {base}, {reg};")
        else:
            scaled = self.em.reg("r")
            self.em.emit(f"mul.lo.s32 {scaled}, {reg}, {nbytes};")
            self.em.emit(f"add.s32 {addr}, {base}, {scaled};")
        return addr

    def smem_address(self, buf, index):
        off = self.linear_index(index, self.shared[buf]["extents"])
        return self.smem_address_off(buf, off)

    def emit_barrier(self, barrier_id=0, count=None, warp=False):
        bid = 0 if barrier_id is None else int(barrier_id)
        if warp:
            if bid:
                raise NotImplementedError("warp sync takes no named barrier")
            self.em.emit("bar.warp.sync 0xffffffff;")
        elif bid and count is None:
            raise NotImplementedError("named barrier requires a thread count")
        elif bid:
            self.em.emit(f"bar.sync {bid}, {count};")
        else:
            # emmy renders __syncthreads() for barrier_id 0, discarding count —
            # a counted partial barrier here would break cross-warp ordering.
            self.em.emit("bar.sync 0;")

    def decode_axes(self, tile, gid, extents, split=None):
        # split = (cta, tid, block) when the flat cell id exceeds INT32_MAX:
        # gid/s = cta*(block//s) + (cta*(block%s) + tid)//s, every term s32-safe.
        strides = [1] * len(extents)
        acc = 1
        for i in range(len(extents) - 1, -1, -1):
            strides[i] = acc
            acc *= extents[i]
        # Grouped CTA rasterization (emmy kernel/ir.py): remap the flat block
        # sub-id so `g` block-tiles of the grouped axis iterate fastest within
        # each launch stripe. Only fires for adjacent raster axes with both
        # extents > 1; otherwise it is the identity and the plain decode is
        # emitted.
        raster = None
        rg = getattr(tile, "raster_group", None)
        ra = getattr(tile, "raster_axes", None)
        if rg and ra:
            names = [a.name for a in tile.axes]
            if ra[0] in names and ra[1] in names:
                mi, ni = names.index(ra[0]), names.index(ra[1])
                if ni == mi + 1:
                    orient = getattr(tile, "raster_orient", None)
                    gi, oi = (mi, ni) if orient == "m" else (ni, mi)
                    g_eff = min(rg, extents[gi])
                    if extents[gi] > 1 and g_eff > 1:
                        raster = (gi, oi, g_eff)
        a_expr = b_expr = None
        if raster is not None:
            gi, oi, g = raster
            ea, eb = extents[gi], extents[oi]
            mi, ni = min(gi, oi), max(gi, oi)
            sn = strides[ni]
            sub = ("reg", gid) if sn == 1 else self.idiv(("reg", gid), ("imm", sn))
            if mi != 0:
                sub = self.irem(sub, ("imm", ea * eb))
            sub = ("reg", self.materialize(sub))
            if ea % g == 0:
                a_expr = self.iadd(
                    self.imul(self.idiv(sub, ("imm", g * eb)), ("imm", g)),
                    self.irem(sub, ("imm", g)),
                )
                b_expr = self.idiv(self.irem(sub, ("imm", g * eb)), ("imm", g))
            else:
                rgrp = ("reg", self.materialize(self.idiv(sub, ("imm", g * eb))))
                rem = ("reg", self.materialize(self.irem(sub, ("imm", g * eb))))
                t = self.em.reg("r")
                self.em.emit(f"mul.lo.s32 {t}, {rgrp[1]}, {g};")
                d = self.em.reg("r")
                self.em.emit(f"sub.s32 {d}, {ea}, {t};")
                rsz = self.em.reg("r")
                self.em.emit(f"min.s32 {rsz}, {d}, {g};")
                a_expr = self.iadd(
                    self.imul(rgrp, ("imm", g)),
                    self.irem(rem, ("reg", rsz)),
                )
                b_expr = self.idiv(rem, ("reg", rsz))
        for i, ax in enumerate(tile.axes):
            if raster is not None and i in (min(gi, oi), max(gi, oi)):
                self.ints[ax.name] = a_expr if i == gi else b_expr
                self.int_tys[ax.name] = "i32"
                continue
            extent = extents[i]
            if extent == 1:
                self.ints[ax.name] = ("imm", 0)
                continue
            s = strides[i]
            if split is None:
                cur = ("reg", gid)
                if s != 1:
                    cur = self.idiv(cur, ("imm", s))
            else:
                cta, tid, block = split
                hi = self.imul(cta, ("imm", block // s))
                lo = self.iadd(self.imul(cta, ("imm", block % s)), tid)
                cur = hi if s == 1 else self.iadd(hi, self.idiv(lo, ("imm", s)))
            if i != 0:
                cur = self.irem(cur, ("imm", extent))
            self.ints[ax.name] = cur

    def load_dtype(self, s):
        if s.dtype is not None:
            return s.dtype.name, s.dtype.nbytes
        if s.input in self.shared:
            info = self.shared[s.input]
            ty = {"float": "f32", "double": "f64", "half": "f16"}.get(
                info["dtype"], info["dtype"]
            )
            return ty, info["nbytes"]
        ty = self.dtypes.get(s.input, "f32")
        return ty, {"f32": 4, "f64": 8, "i32": 4, "i64": 8, "f16": 2, "bool": 1,
                    "u8": 1, "u16": 2, "i16": 2, "u32": 4, "u64": 8, "bf16": 2}.get(ty, 4)

    def _check_rank(self, buf, index):
        shape = self.shared[buf]["extents"] if buf in self.shared else self.shapes.get(buf)
        if shape is not None and len(index) != len(shape):
            raise NotImplementedError(
                f"index rank {len(index)} != buffer rank {len(shape)} for '{buf}'")

    def _vec_align(self, e):
        """Static element-index alignment as (coef, const) when the index is
        provably of the form coef*var + const; None when not provable."""
        t = type(e).__name__
        if t == "Literal":
            return (0, int(e.value))
        if t == "Var":
            return (1, 0)
        if t == "BinaryExpr":
            if e.op in ("+", "-"):
                a = self._vec_align(e.left)
                b = self._vec_align(e.right)
                if a is None or b is None:
                    return None
                sg = -1 if e.op == "-" else 1
                return (a[0] + sg * b[0], a[1] + sg * b[1])
            if e.op == "*":
                a = self._vec_align(e.left)
                b = self._vec_align(e.right)
                if a is None or b is None:
                    return None
                if a[0] and b[0]:
                    return None  # var*var: not affine
                return (a[0] * b[1] + b[0] * a[1], a[1] * b[1])
        return None

    def _vec_ok(self, index, n, buf=None):
        """Vector of n consecutive elements needs the linearized base element
        index provably ≡ 0 mod n (P3). Row-major: non-last dims are covered by
        their strides being ≡ 0 mod n; the last dim needs term alignment.
        Conservative: scalarize when unprovable."""
        if n <= 1:
            return True
        if buf is not None:
            if buf in self.shared:
                shape = self.shared[buf]["extents"]
            else:
                shape = self.shapes.get(buf)
            if shape is not None and len(shape) == len(index) and len(index) >= 2:
                strides = [1] * len(shape)
                for d in range(len(shape) - 2, -1, -1):
                    strides[d] = strides[d + 1] * shape[d + 1]
                for d in range(len(index) - 1):
                    if strides[d] % n != 0:
                        return False
                a = self._vec_align(index[-1])
                return a is not None and a[0] % n == 0 and a[1] % n == 0
        if len(index) == 1:
            a = self._vec_align(index[0])
            return a is not None and a[0] % n == 0 and a[1] % n == 0
        return False

    def stmt_load(self, s):
        ty, nbytes = self.load_dtype(s)
        self._check_rank(s.input, s.index)
        val_ty = "f32" if ty in WIDEN_AT_LOAD else ty  # narrow types compute at f32
        is_shared = s.input in self.shared
        # Key loads on the FOLDED address tuple, not the structural index
        # trees: unrolled copies fold their axis literals into the offset, so
        # identical runtime addresses dedupe across copies and across sibling
        # loops (a01's four loops re-loading one kt column).
        if is_shared:
            off = self.linear_index(s.index, self.shared[s.input]["extents"])
        else:
            off = self.linear_index(s.index, self.shapes.get(s.input, ()))
        key = (s.input, off, len(s.names))
        if is_shared:
            addr = self.smem_address_off(s.input, off)
            scope = "shared"
        else:
            addr = self._addr_from_off(s.input, off, nbytes)
            scope = "global"
        vec_ok = self._vec_ok(s.index, len(s.names), s.input)
        if ty in WIDEN_AT_LOAD:
            # Load narrow floats into b16 registers, then widen to f32
            h_regs = [self.em.reg("h") for _ in s.names]
            if len(h_regs) == 1 or not vec_ok:
                for i, h in enumerate(h_regs):
                    a = addr if i == 0 else f"{addr}+{i * nbytes}"
                    self.em.emit(f"ld.{scope}.b16 {h}, [{a}];")
            else:
                self.em.emit(
                    f"ld.{scope}.v{len(h_regs)}.b16 {{{', '.join(h_regs)}}}, [{addr}];"
                )
            regs = []
            cvt = "cvt.f32.f16" if ty == "f16" else "cvt.f32.bf16"
            for h in h_regs:
                f = self.em.reg("f")
                self.em.emit(f"{cvt} {f}, {h};")
                regs.append(f)
        else:
            t = "u8" if ty == "bool" else self.ptx(ty)
            regs = [self.em.reg(REG_CLASS[ty]) for _ in s.names]
            if ty == "bool" and len(regs) > 1:
                raise NotImplementedError("vector bool load")
            if ty in ("f64", "i64", "u64") and len(regs) > 2 and vec_ok:
                # 256-bit vectors need PTX ISA 8.8; split 8-byte types into v2 pairs.
                for i in range(0, len(regs), 2):
                    pair = regs[i : i + 2]
                    a = addr if i == 0 else f"{addr}+{i * nbytes}"
                    if len(pair) == 1:
                        self.em.emit(f"ld.{scope}.{t} {pair[0]}, [{a}];")
                    else:
                        self.em.emit(f"ld.{scope}.v2.{t} {{{', '.join(pair)}}}, [{a}];")
            elif len(regs) == 1:
                self.em.emit(f"ld.{scope}.{t} {regs[0]}, [{addr}];")
            elif not vec_ok:
                # index alignment unprovable: scalar loads (always correct)
                for i, r in enumerate(regs):
                    a = addr if i == 0 else f"{addr}+{i * nbytes}"
                    self.em.emit(f"ld.{scope}.{t} {r}, [{a}];")
            else:
                self.em.emit(
                    f"ld.{scope}.v{len(regs)}.{t} {{{', '.join(regs)}}}, [{addr}];"
                )
        for name, reg in zip(s.names, regs):
            self.values[name] = (reg, val_ty)

    def stmt_assign(self, s):
        if s.dtype is not None:
            ty = s.dtype.name
        else:
            ty = self.values[s.args[0]][1]
        if ty in WIDEN_AT_LOAD:
            ty = "f32"  # narrow floats are widened to f32 for computation
        elif ty in ("i16", "u8", "u16"):
            ty = "i32" if ty == "i16" else "u32"
        dst = self.em.reg(REG_CLASS[ty])
        args = [self.values[a][0] for a in s.args]
        arg_tys = [self.values[a][1] for a in s.args]
        self.emit_op(dst, s.op.name, ty, args, arg_tys)
        self.values[s.name] = (dst, ty)

    def convert_reg(self, reg, src_ty, dst_ty):
        """C conversion semantics for a cross-dtype store boundary."""
        if src_ty == dst_ty:
            return reg
        scls = REG_CLASS.get(src_ty, "r")
        dcls = REG_CLASS.get(dst_ty, "r")
        if scls == dcls and not (src_ty == "f16" or dst_ty == "f16"):
            return reg
        r = self.em.reg(dcls if dcls != "h" else "r")
        if dst_ty in ("i32", "i64", "u32", "u64") and src_ty.startswith("f"):
            self.em.emit(f"cvt.rzi.{self.ptx(dst_ty)}.{src_ty} {r}, {reg};")
        elif dst_ty.startswith("f") and src_ty in ("i32", "i64", "u32", "u64", "bool"):
            self.em.emit(f"cvt.rn.{self.ptx(dst_ty)}.{self.ptx(src_ty)} {r}, {reg};")
        elif src_ty == "f64" and dst_ty == "f32":
            self.em.emit(f"cvt.rn.f32.f64 {r}, {reg};")
        elif src_ty == "f32" and dst_ty == "f64":
            self.em.emit(f"cvt.f64.f32 {r}, {reg};")
        elif src_ty == "i32" and dst_ty == "i64":
            self.em.emit(f"cvt.s64.s32 {r}, {reg};")
        elif src_ty == "i64" and dst_ty == "i32":
            self.em.emit(f"cvt.s32.s64 {r}, {reg};")
        elif dst_ty == "bool":
            r = self.bool_reg(reg, src_ty)
        else:
            raise NotImplementedError(f"store conversion {src_ty} -> {dst_ty}")
        return r

    def stmt_write(self, s):
        ty = s.value_dtype.name
        nbytes = s.value_dtype.nbytes
        self._check_rank(s.output, s.index)
        if getattr(s, 'swizzle', None) not in (None, "NONE"):
            raise NotImplementedError(f"Write swizzle {s.swizzle!r}")
        is_shared = s.output in self.shared
        if is_shared:
            addr = self.smem_address(s.output, s.index)
            scope = "shared"
        else:
            addr = self.address(s.output, s.index, nbytes)
            scope = "global"
        # f16 narrowing is handled by the f16 branch below (h_regs path);
        # every other cross-dtype store goes through C conversion semantics.
        regs = [self.values[v][0] if ty in WIDEN_AT_LOAD else
                self.convert_reg(self.values[v][0], self.values[v][1], ty)
                for v in s.values]
        vec_ok = self._vec_ok(s.index, len(s.values), s.output)
        if ty in WIDEN_AT_LOAD:
            # Narrow f32 computation results back for storage
            h_regs = []
            cvt = "cvt.rn.f16.f32" if ty == "f16" else "cvt.rn.bf16.f32"
            for reg in regs:
                h = self.em.reg("h")
                self.em.emit(f"{cvt} {h}, {reg};")
                h_regs.append(h)
            if s.atomic:
                if ty == "bf16":
                    raise NotImplementedError("atomic bf16 write")
                # red.add.f16 (sm_70+): scalar per element, no vector form
                for i, h in enumerate(h_regs):
                    a = addr if i == 0 else f"{addr}+{i * nbytes}"
                    self.em.emit(f"red.{scope}.add.noftz.f16 [{a}], {h};")
            elif len(h_regs) == 1 or not vec_ok:
                for i, h in enumerate(h_regs):
                    a = addr if i == 0 else f"{addr}+{i * nbytes}"
                    self.em.emit(f"st.{scope}.b16 [{a}], {h};")
            else:
                self.em.emit(
                    f"st.{scope}.v{len(h_regs)}.b16 [{addr}], {{{', '.join(h_regs)}}};"
                )
        else:
            t = "u8" if ty == "bool" else self.ptx(ty)
            if ty == "bool" and len(regs) > 1:
                raise NotImplementedError("vector bool write")
            if ty in ("f64", "i64", "u64") and len(regs) > 2 and vec_ok:
                # 256-bit vectors need PTX ISA 8.8; split 8-byte types into v2 pairs.
                for i in range(0, len(regs), 2):
                    pair = regs[i : i + 2]
                    a = addr if i == 0 else f"{addr}+{i * nbytes}"
                    if len(pair) == 1:
                        self.em.emit(f"st.{scope}.{t} [{a}], {pair[0]};")
                    else:
                        self.em.emit(f"st.{scope}.v2.{t} [{a}], {{{', '.join(pair)}}};")
            elif s.atomic:
                # red.add rejects .s64; .u64 is bitwise-identical for adds
                t = "u64" if ty == "i64" else t
                for i, reg in enumerate(regs):
                    a = addr if i == 0 else f"{addr}+{i * nbytes}"
                    self.em.emit(f"red.{scope}.add.{t} [{a}], {reg};")
            elif len(regs) == 1:
                self.em.emit(f"st.{scope}.{t} [{addr}], {regs[0]};")
            elif not vec_ok:
                # index alignment unprovable: scalar stores (always correct)
                for i, r in enumerate(regs):
                    a = addr if i == 0 else f"{addr}+{i * nbytes}"
                    self.em.emit(f"st.{scope}.{t} [{a}], {r};")
            else:
                self.em.emit(
                    f"st.{scope}.v{len(regs)}.{t} [{addr}], {{{', '.join(regs)}}};"
                )

    def stmt_let(self, s):
        ty = s.dtype.name
        if ty in INT_TYPES:
            self.ints[s.name] = self.iexpr(s.value)
            self.int_tys[s.name] = ty
        else:
            t = type(s.value).__name__
            if t == "Literal":
                lit = self.float_literal(s.value.value, ty)
                r = self.em.reg(REG_CLASS[ty])
                self.em.emit(f"mov.{self.ptx(ty)} {r}, {lit};")
                self.values[s.name] = (r, ty)
            elif t == "Var":
                self.values[s.name] = self.values[s.value.name]
            else:
                raise NotImplementedError(f"Let value {t}")

    def predicate(self, e):
        t = type(e).__name__
        if t == "BinaryExpr" and e.op in ("==", "!=", "<", "<=", ">", ">="):
            kind = self.cmp_kind(e.left, e.right)
            a = self.cmp_operand(e.left, kind)
            b = self.cmp_operand(e.right, kind)
            p = self.em.reg("p")
            op = {"==": "eq", "!=": "eq", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[e.op]
            suffix = {"int": "s32", "u32": "u32", "s64": "s64", "u64": "u64",
                      "f32": "f32", "f64": "f64"}[kind]
            self.em.emit(f"setp.{op}.{suffix} {p}, {a}, {b};")
            if e.op == "!=":
                self.em.emit(f"not.pred {p}, {p};")
            return p
        if t == "BinaryExpr" and e.op in ("&&", "||"):
            a = self.predicate(e.left)
            b = self.predicate(e.right)
            p = self.em.reg("p")
            self.em.emit(f"{'and' if e.op == '&&' else 'or'}.pred {p}, {a}, {b};")
            return p
        if t == "Var":
            p = self.em.reg("p")
            if e.name in self.values or e.name in self.ints:
                kind = self.cmp_kind(e)
                if e.name in self.values:
                    reg, ty = self.values[e.name]
                else:
                    reg = self.materialize(self.ints[e.name])
                    ty = self.int_tys.get(e.name, "i32")
                if kind.startswith("f"):
                    zero = self.float_literal(0.0, ty)
                    pt = self.em.reg("p")
                    self.em.emit(f"setp.eq.{kind} {pt}, {reg}, {zero};")
                    self.em.emit(f"not.pred {p}, {pt};")
                elif kind in ("s64", "u64"):
                    self.em.emit(f"setp.ne.{kind} {p}, {reg}, 0;")
                elif kind == "u32":
                    self.em.emit(f"setp.ne.u32 {p}, {reg}, 0;")
                else:
                    self.em.emit(f"setp.ne.s32 {p}, {reg}, 0;")
            else:
                self.em.emit(f"setp.ne.s32 {p}, {self.materialize(self.iexpr(e))}, 0;")
            return p
        if t == "Literal":
            p = self.em.reg("p")
            if isinstance(e.value, float):
                pt = self.em.reg("p")
                self.em.emit(
                    f"setp.eq.f32 {pt}, "
                    f"{self.float_literal(e.value, 'f32')}, "
                    f"{self.float_literal(0.0, 'f32')};"
                )
                self.em.emit(f"not.pred {p}, {pt};")
            else:
                self.em.emit(f"setp.ne.s32 {p}, {int(e.value)}, 0;")
            return p
        raise NotImplementedError(f"condition {t}")

    def cmp_kind(self, *exprs):
        kinds = set()
        for e in exprs:
            t = type(e).__name__
            ty = None
            if t == "Var":
                if e.name in self.values:
                    ty = self.values[e.name][1]
                elif e.name in self.ints:
                    ty = self.int_tys.get(e.name, "i32")
            elif t == "CastExpr":
                ty = e.dtype

            if ty == "f64":
                kinds.add("f64")
            elif ty and ty.startswith("f"):
                kinds.add("f32")
            elif ty == "u64":
                kinds.add("u64")
            elif ty == "i64":
                kinds.add("s64")
            elif ty in ("u32", "u8", "u16", "bool"):
                kinds.add("u32")
            elif ty in ("i32", "i16"):
                kinds.add("int")
            elif t == "Literal":
                kinds.add("f32" if isinstance(e.value, float) else "int")
        if "f64" in kinds:
            return "f64"
        if "f32" in kinds:
            return "f32"
        if "u64" in kinds:
            return "u64"
        if "s64" in kinds:
            return "s64"
        if "u32" in kinds:
            return "u32"
        return "int"

    def cmp_operand(self, e, kind):
        if kind in ("int", "u32", "s64", "u64"):
            if type(e).__name__ == "Var" and e.name in self.values:
                return self.values[e.name][0]
            r = self.iexpr(e)
            return str(r[1]) if r[0] == "imm" else self.materialize(r)
        t = type(e).__name__
        if t == "Var":
            if e.name in self.values:
                return self.values[e.name][0]
            if e.name in self.ints:
                iv = self.materialize(self.iexpr(e))
                r = self.em.reg("f" if kind == "f32" else "fd")
                src_ty = self.int_tys.get(e.name, "i32")
                src_ty = {
                    "bool": "u32",
                    "u8": "u32",
                    "u16": "u32",
                    "i16": "s32",
                }.get(src_ty, self.ptx(src_ty))
                self.em.emit(f"cvt.rn.{kind}.{src_ty} {r}, {iv};")
                return r
            raise KeyError(f"unknown comparison variable {e.name!r}")
        if t == "Literal":
            if kind == "f64":
                return f"0d{struct.unpack('<Q', struct.pack('<d', float(e.value)))[0]:016X}"
            bits = struct.unpack("<I", struct.pack("<f", float(e.value)))[0]
            return f"0f{bits:08X}"
        raise NotImplementedError(f"float operand {t}")

    def stmt_cond(self, s):
        p = self.predicate(s.cond)
        else_label = self.em.label("COND_ELSE")
        self.em.emit(f"@!{p} bra {else_label};")
        self.expr_cache.clear()
        self.emit_body(s.body)
        self.expr_cache.clear()
        if s.else_body:
            end_label = self.em.label("COND_END")
            self.em.emit(f"bra {end_label};")
            self.em.mark(else_label)
            self.emit_body(s.else_body)
            self.expr_cache.clear()
            self.em.mark(end_label)
        else:
            self.em.mark(else_label)

    UNROLL_CAP = 128  # PTX size guard: full-unroll only small static trips

    def _contains_nested_loop(self, body):
        # nvcc declines `#pragma unroll` when the body nests loops (a01's a1
        # loop: 64 copies x 4 inner loops exploded PTX 22x and ran 18x slower).
        # Mirror that refusal: unroll only flat bodies.
        stack = list(body)
        while stack:
            s = stack.pop()
            t = type(s).__name__
            if t in ("Loop", "StridedLoop"):
                return True
            for attr in ("body", "else_body"):
                sub = getattr(s, attr, None)
                if sub:
                    stack.extend(sub)
        return False

    def count_uses(self, body):
        """COMPLETE use census: every value reference in every statement field
        — including Cond.cond, Load.index and Let.value, which the old
        Assign/Accum/Write-only walk missed. A product read by a condition
        must count as a use, or FMA contraction aliases the condition onto
        the accumulator register."""
        import dataclasses
        uses = {}
        DEF_FIELDS = {"name", "output", "input"}  # defs/buffer refs, not uses
        stack = list(body)
        while stack:
            st = stack.pop()
            if st is None or isinstance(st, (int, float, bool)):
                continue
            if isinstance(st, str):
                uses[st] = uses.get(st, 0) + 1
                continue
            if hasattr(st, "free_vars"):  # Expr: authoritative reference set
                for n in st.free_vars():
                    uses[n] = uses.get(n, 0) + 1
                continue
            if dataclasses.is_dataclass(st):
                is_stmt = type(st).__module__.startswith("emmy.compiler.ir.stmt")
                for f in dataclasses.fields(st):
                    if is_stmt and f.name in DEF_FIELDS:
                        continue
                    stack.append(getattr(st, f.name))
            elif isinstance(st, (list, tuple)):
                stack.extend(st)
        return uses

    PURE_OPS = {"add.s32", "mul.lo.s32", "sub.s32", "and.b32", "or.b32",
                "xor.b32", "shl.b32", "shr.s32", "shr.u32", "min.s32", "max.s32",
                "mul.wide.s32", "add.s64"}
    COMMUTATIVE = {"add.s32", "mul.lo.s32", "and.b32", "or.b32", "xor.b32",
                   "add.s64"}

    @staticmethod
    def _tree_align(e):
        """Known power-of-two divisor (exponent) of the expression's value."""
        t = type(e).__name__
        if t == "Literal":
            v = abs(int(e.value))
            return 30 if v == 0 else (v & -v).bit_length() - 1
        if t == "Var":
            return 0
        if t == "BinaryExpr":
            if e.op in ("+", "-"):
                return min(Compiler._tree_align(e.left), Compiler._tree_align(e.right))
            if e.op == "*":
                for x, y in ((e.left, e.right), (e.right, e.left)):
                    if type(y).__name__ == "Literal":
                        m = abs(int(y.value))
                        if m == 0:
                            return 30
                        return Compiler._tree_align(x) + (m & -m).bit_length() - 1
                return min(Compiler._tree_align(e.left), Compiler._tree_align(e.right))
        return 0

    @classmethod
    def _leaf_step_diff(cls, k1, k2):
        """0 = identical, 1 = exactly one ('l', c) leaf became ('l', c+1), 2 = worse."""
        if isinstance(k1, tuple) and isinstance(k2, tuple):
            if k1[0] != k2[0] or len(k1) != len(k2):
                return 2
            if k1[0] == "l":
                return 1 if k2[1] == k1[1] + 1 else (0 if k1[1] == k2[1] else 2)
            total = 0
            for x, y in zip(k1, k2):
                if isinstance(x, tuple) or isinstance(y, tuple):
                    d = cls._leaf_step_diff(x, y)
                    if d == 2:
                        return 2
                    total += d
                    if total > 1:
                        return 2
                elif x != y:
                    return 2
            return total
        return 0 if k1 == k2 else 2

    def _pair_diff_elem(self, s, t):
        """Index-element position where the two loads differ by exactly +1."""
        if len(s.index) != len(t.index):
            return None
        diffs = []
        for p in range(len(s.index)):
            d = self._leaf_step_diff(self.ekey(s.index[p]), self.ekey(t.index[p]))
            if d == 2:
                return None
            if d == 1:
                diffs.append(p)
        return diffs[0] if len(diffs) == 1 else None

    def _index_strides(self, s):
        if s.input in self.shared:
            ext = list(self.shared[s.input]["extents"])
        elif s.input in self.shapes:
            ext = list(self.shapes[s.input])
        else:
            return None
        strides = [1] * len(s.index)
        acc = 1
        for k in range(len(strides) - 1, -1, -1):
            strides[k] = acc
            acc *= ext[k] if k < len(ext) else 1
        return strides

    def _index_align_exp(self, s):
        """Known alignment exponent of the flat element index."""
        strides = self._index_strides(s)
        if strides is None:
            return -1
        total = 30
        for e, st in zip(s.index, strides):
            total = min(total, self._tree_align(e) + (st.bit_length() - 1 if st > 1 else 0))
        return total

    _MEM_KINDS = {"Write", "CpAsyncCopy", "CpAsyncCommit", "CpAsyncWait", "Sync",
                  "Loop", "StridedLoop", "Cond", "TreeHalve", "WarpShuffle",
                  "Smem", "Pre", "Carry", "Select"}

    def _merge_loads(self, body):
        """Pair scalar loads of the same buffer whose indices differ by exactly
        one element in a stride-1 dimension and whose alignment admits a vector
        load; emission then issues one ld.v2 instead of two ld scalars."""
        out = []
        skip = set()
        for i in range(len(body)):
            if i in skip:
                continue
            s = body[i]
            if type(s).__name__ != "Load" or len(s.names) != 1:
                out.append(s)
                continue
            ty, _ = self.load_dtype(s)
            if ty == "bool":
                out.append(s)
                continue
            partner = None
            j = i + 1
            while j < len(body):
                t = body[j]
                kt = type(t).__name__
                if kt in self._MEM_KINDS:
                    break
                if kt == "Load" and len(t.names) == 1 and t.input == s.input \
                        and self.load_dtype(t)[0] == ty:
                    pos = self._pair_diff_elem(s, t)
                    strides = self._index_strides(s)
                    if pos is not None and strides is not None \
                            and strides[pos] == 1 and self._index_align_exp(s) >= 1:
                        partner = j
                    break
                j += 1
            if partner is not None:
                out.append(_Pair(s, body[partner]))
                skip.add(partner)
            else:
                out.append(s)
        return out

    def stmt_load_pair(self, a, b):
        ty, nbytes = self.load_dtype(a)
        val_ty = "f32" if ty in WIDEN_AT_LOAD else ty
        if a.input in self.shared:
            off = self.linear_index(a.index, self.shared[a.input]["extents"])
        else:
            off = self.linear_index(a.index, self.shapes.get(a.input, ()))
        if a.input in self.shared:
            addr = self.smem_address(a.input, a.index)
            scope = "shared"
        else:
            addr = self.address(a.input, a.index, nbytes)
            scope = "global"
        pair_ok = self._vec_ok(a.index, 2, a.input)
        if ty in WIDEN_AT_LOAD:
            h0, h1 = self.em.reg("h"), self.em.reg("h")
            if pair_ok:
                self.em.emit(f"ld.{scope}.v2.b16 {{{h0}, {h1}}}, [{addr}];")
            else:
                self.em.emit(f"ld.{scope}.b16 {h0}, [{addr}];")
                self.em.emit(f"ld.{scope}.b16 {h1}, [{addr}+{nbytes}];")
            cvt = "cvt.f32.f16" if ty == "f16" else "cvt.f32.bf16"
            f0, f1 = self.em.reg("f"), self.em.reg("f")
            self.em.emit(f"{cvt} {f0}, {h0};")
            self.em.emit(f"{cvt} {f1}, {h1};")
            regs = [f0, f1]
        else:
            tt = self.ptx(ty)
            r0, r1 = self.em.reg(REG_CLASS[ty]), self.em.reg(REG_CLASS[ty])
            if pair_ok:
                self.em.emit(f"ld.{scope}.v2.{tt} {{{r0}, {r1}}}, [{addr}];")
            else:
                self.em.emit(f"ld.{scope}.{tt} {r0}, [{addr}];")
                self.em.emit(f"ld.{scope}.{tt} {r1}, [{addr}+{nbytes}];")
            regs = [r0, r1]
        self.values[a.names[0]] = (regs[0], val_ty)
        self.values[b.names[0]] = (regs[1], val_ty)

    def _vectorize_loads(self, begin):
        # Line-level pass over the unrolled region: two scalar loads from the
        # same base register at byte offsets `step` apart become one v2 load
        # whose destinations are the two original registers — no renaming.
        # Fires only when the address's 8-byte alignment is provable from the
        # region's own multiply/add constants; otherwise a conservative no-op.
        lines = self.em.lines
        align = {}                      # register -> known alignment exponent
        hazards = ("st.", "red.", "atom.", "bar.", "membar", "call", "bra",
                   "@", "cp.async")
        out = lines[:begin]             # keep everything before the region
        pend = {}                       # (op, base) -> {off: (dst, out_index)}
        for ln in lines[begin:]:
            core = ln.strip()
            if not ln.startswith("\t") or not core or core.endswith(":"):
                pend.clear()
                out.append(ln)
                continue
            head = core.split(None, 1)[0]
            if any(head.startswith(h) for h in hazards):
                pend.clear()
                out.append(ln)
                continue
            core = core[:-1] if core.endswith(";") else core
            toks = [t.strip() for t in core.split(",")]
            op = toks[0].split()[0]
            dst = toks[0].split()[1] if len(toks[0].split()) > 1 else None
            # ---- alignment tracking through pure integer arithmetic ----
            if op in self.PURE_OPS and dst:
                srcs = toks[1:]
                if op == "mul.lo.s32" and len(srcs) == 2:
                    reg = next((s for s in srcs if s.startswith("%")), None)
                    imm = next((int(s) for s in srcs if s.lstrip("-").isdigit()), None)
                    if reg is not None and imm and align.get(reg) is not None:
                        align[dst] = min(30, align[reg] + ((imm & -imm).bit_length() - 1))
                    else:
                        align.pop(dst, None)
                elif op in ("add.s32", "sub.s32"):
                    exps = []
                    ok = True
                    for s in srcs:
                        if s.lstrip("-").isdigit():
                            v = abs(int(s))
                            exps.append((v & -v).bit_length() - 1 if v else 30)
                        elif align.get(s) is not None:
                            exps.append(align[s])
                        else:
                            ok = False
                            break
                    if ok:
                        align[dst] = min(exps)
                    else:
                        align.pop(dst, None)
                else:
                    align.pop(dst, None)
            # ---- the merge itself ----
            if op.startswith("ld.") and ".v" not in op and len(toks) == 2 \
                    and toks[1].startswith("[") and toks[1].endswith("]") \
                    and dst:
                inner = toks[1][1:-1]
                if "+" in inner:
                    base, _, imms = inner.rpartition("+")
                    try:
                        off = int(imms)
                    except ValueError:
                        off = None
                else:
                    base, off = inner, 0
                if base.startswith("%") and off is not None:
                    width = {"f32": 4, "f64": 8, "s32": 4, "u32": 4,
                             "s64": 8, "u64": 8}.get(op.split(".")[2], 4)
                    bucket = pend.setdefault((op, base), {})
                    prev = bucket.pop(off - width, None)
                    if (prev is not None and align.get(base, -1) >= 3
                            and off % (2 * width) == 0):
                        vec_op = op.rsplit(".", 1)[0] + ".v2." + op.split(".")[2]
                        out[prev[1]] = f"\t{vec_op} {{{prev[0]}, {dst}}}, [{base}+{off - width}];"
                        continue        # second scalar absorbed by the v2
                    bucket[off] = (dst, len(out))
                    out.append(ln)
                    continue
            if op.startswith("ld."):
                pend.clear()
            out.append(ln)
        self.em.lines = out

    def emit_body(self, body):
        """Statement walk: adjacent-load merge + mul->Accum fma contraction."""
        body = self._merge_loads(body)
        uses = self.count_uses(body)
        i = 0
        while i < len(body):
            s = body[i]
            if isinstance(s, _Pair):
                self.stmt_load_pair(s.a, s.b)
                i += 1
                continue
            nxt = body[i + 1] if i + 1 < len(body) else None
            if (type(s).__name__ == "Assign" and s.op.name == "multiply"
                    and uses.get(s.name, 0) == 1 and nxt is not None
                    and type(nxt).__name__ == "Accum" and nxt.value == s.name
                    and nxt.op.name == "add"
                    and nxt.dtype.name.startswith("f")):
                # fma: acc = acc + a*b in one rounding (nvcc -fmad parity)
                reg, ty = self.accum_reg(nxt)
                left = self.values[nxt.base][0] if nxt.base else reg
                self.em.emit(
                    f"fma.rn.{self.ptx(ty)} {reg}, {self.values[s.args[0]][0]}, "
                    f"{self.values[s.args[1]][0]}, {left};"
                )
                self.values[s.name] = (reg, ty)
                i += 2
                continue
            self.stmt(s)
            i += 1


    @staticmethod
    def _static_int(v):
        if isinstance(v, int):
            return v
        val = getattr(v, "value", None)
        return val if isinstance(val, int) else None

    @staticmethod
    def _ekey_stable(key, axis, stable):
        # True when an ekey tree mentions no variable that is rebound across
        # unrolled copies: the unrolled axis itself, or any name not bound
        # before the unroll began (body Lets, nested state, ...).
        if not isinstance(key, tuple):
            return True
        if key[0] == "v" and len(key) == 2:
            return key[1] != axis and key[1] in stable
        return all(Compiler._ekey_stable(x, axis, stable)
                   for x in key if isinstance(x, tuple))

    def emit_loop(self, axis, start, step, end, body, seed, unroll=False):
        if seed:
            self.init_accumulators(body)
        name = axis.name
        s_v = self._static_int(start)
        st_v = self._static_int(step)
        b_v = (self._static_int(end) if end is not None
               else self._static_int(getattr(axis, "extent", None)))
        trip = None
        if None not in (s_v, st_v, b_v) and st_v > 0 and b_v > s_v:
            trip = -(-(b_v - s_v) // st_v)
        # nvcc semantics: `#pragma unroll` forces, absence = auto-decide. We
        # mirror the auto part: every flat, static-trip, size-bounded loop
        # unrolls (a01's a2 loops: 16k runtime iterations x 4 instructions of
        # loop overhead on a 3-instruction body was half the hot path).
        if trip is not None and 1 < trip <= self.UNROLL_CAP \
                and not self._contains_nested_loop(body):
            # Full unroll: bind the axis to each constant and re-emit the body;
            # the lazy integer machinery folds every index on paper. The IR
            # carries `unroll` — emmy's renderer maps it to `#pragma unroll`.
            # Accumulators whose identity init lands *inside* the body (nested
            # seeded loops, or lazy init when this loop is unseeded) re-execute
            # that init on every runtime iteration of the loop form; each copy
            # must too, so drop their bindings and let init re-emit.
            reset = set()
            if not seed:
                self._body_accums(body, reset)
            self._nested_seed_accums(body, reset)
            # State pre-bound before this loop (e.g. by an `Init` statement)
            # is carried state: the loop form never re-initializes it, so the
            # copies must not either. Only bindings *created during body
            # emission* are re-executed per runtime iteration.
            reset -= {nm for nm in reset if nm in self.values}
            saved = self.ints.get(name)
            saved_ty = self.int_tys.get(name)
            cse_begin = len(self.em.lines)
            # Hoisting: expressions whose keys touch no copy-varying variable
            # compute once in copy 0 and every later copy reuses the result —
            # the cache discipline LLVM applies to unrolled bodies. Only keys
            # mentioning the axis (or names rebound per copy) are dropped.
            stable = set(self.ints) | {n for n, (r_, t_) in self.values.items()
                                       if t_ in INT_TYPES}
            for k in range(trip):
                self.expr_cache = {ek: v_ for ek, v_ in self.expr_cache.items()
                                   if self._ekey_stable(ek, name, stable)}
                for nm in reset:
                    self.values.pop(nm, None)
                self.ints[name] = ("imm", s_v + k * st_v)
                self.int_tys[name] = "i32"
                self.emit_body(body)
            if saved is None:
                self.ints.pop(name, None)
                self.int_tys.pop(name, None)
            else:
                self.ints[name] = saved
                if saved_ty is None:
                    self.int_tys.pop(name, None)
                else:
                    self.int_tys[name] = saved_ty
            self.expr_cache.clear()
            self._vectorize_loads(cse_begin)
            return
        start_op = self.int_operand(start)
        step_op = self.int_operand(step)
        bound_op = self.int_operand(end) if end is not None else str(axis.extent.value)
        saved = self.ints.get(name)
        saved_ty = self.int_tys.get(name)
        var = self.em.reg("r")
        self.em.emit(f"mov.s32 {var}, {start_op};")
        self.expr_cache.clear()
        self.ints[name] = ("reg", var)
        self.int_tys[name] = "i32"
        end_label = self.em.label("LOOP_END")
        p = self.em.reg("p")
        U = 16
        accums = set()
        self._body_accums(body, accums)
        # Profitability: the worst-case trip (bound/step, start-independent)
        # must admit at least one full group — otherwise the guard always
        # fails and the dead 8-copy body only bloats the instruction cache.
        max_trip = -(-b_v // st_v) if (b_v is not None and st_v) else None
        can_partial = (st_v is not None and st_v > 0
                       and not self._contains_nested_loop(body)
                       and (seed or not accums)
                       and max_trip is not None and max_trip > U)
        if can_partial:
            # nvcc-shaped: derive the exact group count from the runtime start
            # (groups = (bound − start) / (U·step), clamped ≥ 0), run a
            # countdown main loop with a single conditional back-edge and no
            # mid-loop exit, and drain the < U-stride tail one copy at a time.
            # Visitation order matches the plain loop exactly.
            rem_label = self.em.label("REM")
            t_reg = self.em.reg("r")
            self.em.emit(f"sub.s32 {t_reg}, {bound_op}, {var};")
            self.em.emit(f"max.s32 {t_reg}, {t_reg}, 0;")
            g_reg = self.em.reg("r")
            span = st_v * U
            if span & (span - 1) == 0:
                self.em.emit(f"shr.u32 {g_reg}, {t_reg}, {span.bit_length() - 1};")
            else:
                self.em.emit(f"div.s32 {g_reg}, {t_reg}, {span};")
            main_label = self.em.label("MAIN")
            self.em.mark(main_label)
            self.em.emit(f"setp.eq.s32 {p}, {g_reg}, 0;")
            self.em.emit(f"@{p} bra {rem_label};")
            cse_begin = len(self.em.lines)
            stable = (set(self.ints) | {n for n, (r_, t_) in self.values.items()
                                        if t_ in INT_TYPES})
            for j in range(U):
                self.expr_cache = {ek: v_ for ek, v_ in self.expr_cache.items()
                                   if self._ekey_stable(ek, name, stable)}
                self.ints[name] = ("regimm", var, j * st_v)
                self.int_tys[name] = "i32"
                self.emit_body(body)
            self.expr_cache.clear()
            self.em.emit(f"add.s32 {var}, {var}, {span};")
            self.em.emit(f"add.s32 {g_reg}, {g_reg}, -1;")
            self.em.emit(f"bra {main_label};")
            self.em.mark(rem_label)
            self.em.emit(f"setp.ge.s32 {p}, {var}, {bound_op};")
            self.em.emit(f"@{p} bra {end_label};")
            # The remainder loop re-executes its body: its address must be
            # recomputed from the incremented var each iteration. The group
            self.ints[name] = ("reg", var)
            self.emit_body(body)
            self.em.emit(f"add.s32 {var}, {var}, {step_op};")
            self.em.emit(f"bra {rem_label};")
            self.em.mark(end_label)
        else:
            loop_label = self.em.label("LOOP")
            self.em.mark(loop_label)
            self.em.emit(f"setp.ge.s32 {p}, {var}, {bound_op};")
            self.em.emit(f"@{p} bra {end_label};")
            self.emit_body(body)
            self.em.emit(f"add.s32 {var}, {var}, {step_op};")
            self.em.emit(f"bra {loop_label};")
            self.em.mark(end_label)
        if saved is None:
            del self.ints[name]
            self.int_tys.pop(name, None)
        else:
            self.ints[name] = saved
            if saved_ty is None:
                self.int_tys.pop(name, None)
            else:
                self.int_tys[name] = saved_ty
        self.expr_cache.clear()

    def _static_trip(self, s):
        s_v = self._static_int(getattr(s, "start", 0))
        st_v = self._static_int(getattr(s, "step", 1))
        end = getattr(s, "end", None)
        b_v = (self._static_int(end) if end is not None
               else self._static_int(getattr(s.axis, "extent", None)))
        if None in (s_v, st_v, b_v) or st_v <= 0 or b_v <= s_v:
            return None
        return -(-(b_v - s_v) // st_v)

    def _body_accums(self, body, out):
        # mirrors init_accumulators' traversal: direct accums + Cond-nested
        for s in body:
            k = type(s).__name__
            if k == "Accum":
                out.add(s.name)
            elif k == "Cond":
                for sub in (s.body, s.else_body):
                    if sub:
                        self._body_accums(sub, out)

    def _nested_seed_accums(self, body, out):
        # accumulators whose identity init is emitted by a seeded loop nested
        # in this body — the loop form re-executes those inits every iteration
        for s in body:
            k = type(s).__name__
            if k in ("Loop", "StridedLoop"):
                if getattr(s, "seed", False):
                    self._body_accums(s.body, out)
            elif k == "Cond":
                for sub in (s.body, s.else_body):
                    if sub:
                        self._nested_seed_accums(sub, out)

    def stmt_strided_loop(self, s):
        self.emit_loop(s.axis, s.start, s.step, s.end, s.body, s.seed,
                       unroll=bool(getattr(s, "unroll", False)))

    def stmt_loop(self, s):
        self.emit_loop(s.axis, 0, 1, None, s.body, s.seed,
                       unroll=bool(getattr(s, "unroll", False)))

    def stmt_accum(self, s):
        reg, ty = self.accum_reg(s)
        left = self.values[s.base][0] if s.base else reg
        val = self.values[s.value][0]
        self.emit_op(reg, s.op.name, ty, [left, val])

    def stmt_smem(self, s):
        extents = tuple(int(e) for e in s.extents)
        nbytes = self.smem_nbytes(s.dtype)
        size = prod(extents) * nbytes
        vec = 4 if nbytes <= 4 else 2  # widest vector the emitter can form
        align = max(s.align or 0, vec * nbytes)
        if self.dynamic_smem:
            off = -(-self.smem_used // align) * align
            self.smem_used = off + size
            self.max_smem_align = max(self.max_smem_align, align, nbytes)
        else:
            off = 0
        self.shared[s.name] = {
            "extents": extents,
            "dtype": s.dtype,
            "nbytes": nbytes,
            "align": s.align,
            "size": size,
            "offset": off,
        }
        self.smem_base(s.name)

    def float_literal(self, value, ty):
        value = float(value)
        if ty == "f64":
            bits = struct.unpack("<Q", struct.pack("<d", value))[0]
            return f"0d{bits:016X}"
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return f"0f{bits:08X}"

    def stmt_init(self, s):
        ty = s.dtype.name
        if ty in WIDEN_AT_LOAD:
            ty = "f32"
        reg = self.em.reg(REG_CLASS[ty])
        if ty.startswith("f"):
            lit = self.float_literal(s.identity, ty)
        else:
            lit = str(int(s.identity))
        self.em.emit(f"mov.{self.ptx(ty)} {reg}, {lit};")
        self.values[s.name] = (reg, ty)

    def apply_swizzle(self, flat, swizzle):
        """Apply shared-memory address swizzle to a flat element index.

        Formula: e ^ (((e >> shift) & mask) << 3)
        Modes: B128=(6,0x7), B64=(6,0x3), B32=(6,0x1), or '<mode>@<shift>'.
        """
        if swizzle in (None, "NONE"):
            return flat
        sw = str(swizzle)
        if "@" in sw:
            mode, shift_str = sw.split("@")
            shift = int(shift_str)
        else:
            shift = 6
            mode = sw
        mask = {"B128": 0x7, "B64": 0x3, "B32": 0x1}[mode]
        if flat[0] == "imm":
            e = flat[1]
            return ("imm", e ^ (((e >> shift) & mask) << 3))
        reg = self.materialize(flat)
        t1 = self.em.reg("r")
        self.em.emit(f"shr.u32 {t1}, {reg}, {shift};")
        t2 = self.em.reg("r")
        self.em.emit(f"and.b32 {t2}, {t1}, {mask};")
        t3 = self.em.reg("r")
        self.em.emit(f"shl.b32 {t3}, {t2}, 3;")
        result = self.em.reg("r")
        self.em.emit(f"xor.b32 {result}, {reg}, {t3};")
        return ("reg", result)

    def stmt_cpasync_copy(self, s):
        info = self.shared[s.smem]
        flat = self.linear_index(s.smem_index, info["extents"])
        if s.lane_index is not None:
            cols = info["extents"][-1]
            row = self.iexpr(s.lane_index[0])
            col = self.iexpr(s.lane_index[1])
            lane_flat = self.iadd(self.imul(row, ("imm", cols)), col)
            flat = self.iadd(flat, lane_flat)
        flat = self.apply_swizzle(flat, s.swizzle)
        saddr = self.smem_address_off(s.smem, flat)
        src_ty = self.dtypes.get(s.src, "f32")
        src_nbytes = {"f32": 4, "f64": 8, "f16": 2, "i32": 4}.get(src_ty, 4)
        gaddr = self.address(s.src, s.src_index, src_nbytes)
        if s.nbytes == 16:
            ins = "cp.async.cg.shared.global"
        else:
            ins = "cp.async.ca.shared.global"
        self.em.emit(f"{ins} [{saddr}], [{gaddr}], {s.nbytes};")

    def stmt_cpasync_commit(self, s):
        self.em.emit("cp.async.commit_group;")

    def stmt_cpasync_wait(self, s):
        self.em.emit(f"cp.async.wait_group {s.group};")

    def stmt_sync(self, s):
        self.emit_barrier(s.barrier_id, s.count, s.warp)

    def stmt_warp_shuffle(self, s):
        off = s.length // 2
        while off > 0:
            mask = self.em.reg("r")
            self.em.emit(f"activemask.b32 {mask};")
            for st, sb in zip(s.state, s.state_b):
                reg, ty = self.values[st]
                if REG_CLASS.get(ty, "f") in ("fd", "rd"):
                    # 64-bit state rides as two b32 shuffles (nvcc does the
                    # same for __shfl_xor_sync on double/long long).
                    lo = self.em.reg("r")
                    hi = self.em.reg("r")
                    self.em.emit(f"mov.b64 {{{lo}, {hi}}}, {reg};")
                    pl = self.em.reg("r")
                    ph = self.em.reg("r")
                    self.em.emit(f"shfl.sync.bfly.b32 {pl}, {lo}, {off}, 0x1f, {mask};")
                    self.em.emit(f"shfl.sync.bfly.b32 {ph}, {hi}, {off}, 0x1f, {mask};")
                    peer = self.em.reg(REG_CLASS[ty])
                    self.em.emit(f"mov.b64 {peer}, {{{pl}, {ph}}};")
                else:
                    b = self.em.reg("r")
                    self.em.emit(f"mov.b32 {b}, {reg};")
                    peer_b = self.em.reg("r")
                    self.em.emit(f"shfl.sync.bfly.b32 {peer_b}, {b}, {off}, 0x1f, {mask};")
                    peer = self.em.reg(REG_CLASS[ty])
                    self.em.emit(f"mov.b32 {peer}, {peer_b};")
                self.values[sb] = (peer, ty)
            for c in s.combine_states:
                self.stmt_assign(c)
            off //= 2

    def stmt_tree_halve(self, s):
        ty = s.dtype.name
        if ty in WIDEN_AT_LOAD:
            ty = "f32"
        if s.length & (s.length - 1):
            raise NotImplementedError("TreeHalve length must be a power of two")
        tid = self.materialize(self.iexpr(Var(s.tid_var)))
        scale = 1
        root = ("imm", 0)
        if s.inner is not None:
            iv_name, scale = s.inner
            root = self.iexpr(Var(iv_name))
        base = self.iadd(self.imul(("reg", tid), ("imm", scale)), root)
        level = s.length // 2
        while level > 0:
            p = self.em.reg("p")
            self.em.emit(f"setp.lt.s32 {p}, {tid}, {level};")
            skip = self.em.label("TREE_SKIP")
            self.em.emit(f"@!{p} bra {skip};")
            idx = base
            idx_s = self.iadd(idx, ("imm", level * scale))
            for buf, st, sb in zip(s.bufs, s.state, s.state_b):
                addr = self.smem_address_off(buf, idx)
                a = self.em.reg(REG_CLASS[ty])
                self.em.emit(f"ld.shared.{self.ptx(ty)} {a}, [{addr}];")
                addr_s = self.smem_address_off(buf, idx_s)
                b = self.em.reg(REG_CLASS[ty])
                self.em.emit(f"ld.shared.{self.ptx(ty)} {b}, [{addr_s}];")
                self.values[st] = (a, ty)
                self.values[sb] = (b, ty)
            for c in s.combine_states:
                self.stmt_assign(c)
            for buf, st in zip(s.bufs, s.state):
                reg = self.values[st][0]
                addr = self.smem_address_off(buf, idx)
                self.em.emit(f"st.shared.{self.ptx(ty)} [{addr}], {reg};")
            self.em.mark(skip)
            self.emit_barrier(s.barrier_id, s.barrier_count)
            level //= 2
        for buf, st in zip(s.bufs, s.state):
            addr = self.smem_address_off(buf, root)
            reg = self.em.reg(REG_CLASS[ty])
            self.em.emit(f"ld.shared.{self.ptx(ty)} {reg}, [{addr}];")
            self.values[st] = (reg, ty)

    CF_KINDS = {"Cond", "StridedLoop", "Loop", "TreeHalve"}

    def stmt(self, s):
        k = type(s).__name__
        safe = k in ("Load", "Assign", "Let", "Init")
        control = k in self.CF_KINDS
        if k == "Load":
            self.stmt_load(s)
        elif k == "Assign":
            self.stmt_assign(s)
        elif k == "Write":
            self.stmt_write(s)
        elif k == "Let":
            self.stmt_let(s)
        elif k == "StridedLoop":
            self.stmt_strided_loop(s)
        elif k == "Loop":
            self.stmt_loop(s)
        elif k == "Accum":
            self.stmt_accum(s)
        elif k == "Cond":
            self.stmt_cond(s)
        elif k == "Sync":
            self.stmt_sync(s)
        elif k == "Smem":
            self.stmt_smem(s)
        elif k == "WarpShuffle":
            self.stmt_warp_shuffle(s)
        elif k == "TreeHalve":
            self.stmt_tree_halve(s)
        elif k == "Init":
            self.stmt_init(s)
        elif k == "CpAsyncCopy":
            self.stmt_cpasync_copy(s)
        elif k == "CpAsyncCommit":
            self.stmt_cpasync_commit(s)
        elif k == "CpAsyncWait":
            self.stmt_cpasync_wait(s)
        else:
            raise NotImplementedError(f"statement {k}")

    def compile_tile(self, tile):
        extents = [ax.extent.value for ax in tile.axes]
        n_cells = prod(extents)
        total_threads = prod(self.grid) * prod(self.block)
        if n_cells > total_threads:
            raise NotImplementedError("fewer threads than cells")
        gid = self.global_tid()
        self.define_specials()
        # Flat cell ids past INT32_MAX wrap in 32-bit arithmetic; keep the
        # decode in s32 by splitting gid = cta*block + tid per axis (emmy
        # widens to long long for the same reason).
        split = None
        if n_cells > 0x7FFFFFFF:
            bd = prod(self.block)
            cta = self.em.reg("r")
            self.em.emit(f"div.s32 {cta}, {gid}, {bd};")
            tid = self.em.reg("r")
            self.em.emit(f"rem.s32 {tid}, {gid}, {bd};")
            split = (cta, tid, bd)
            gid = None
        # A warp-specialized producer band launches past the cooperative
        # cells: decode off `cta * block_threads + tid % block_threads` so an
        # aux thread wraps onto this CTA's cells (emmy kernel/ir.py).
        aux = int(getattr(tile, "aux_threads", 0) or 0)
        if aux:
            bt = int(tile.block_threads)
            bd = prod(self.block)
            c = self.em.reg("r")
            self.em.emit(f"div.s32 {c}, {gid}, {bd};")
            r = self.em.reg("r")
            self.em.emit(f"rem.s32 {r}, {gid}, {bd};")
            w = self.em.reg("r")
            self.em.emit(f"rem.s32 {w}, {r}, {bt};")
            t = self.em.reg("r")
            self.em.emit(f"mul.lo.s32 {t}, {c}, {bt};")
            g2 = self.em.reg("r")
            self.em.emit(f"add.s32 {g2}, {t}, {w};")
            gid = g2
        done = None
        if split is not None:
            cta, tid, bd = split
            full_ctas, tail = divmod(n_cells, bd)
            p = self.em.reg("p")
            over = self.em.reg("p")
            self.em.emit(f"setp.gt.s32 {over}, {cta}, {full_ctas};")
            last = self.em.reg("p")
            eqc = self.em.reg("p")
            self.em.emit(f"setp.eq.s32 {eqc}, {cta}, {full_ctas};")
            self.em.emit(f"setp.ge.s32 {last}, {tid}, {tail};")
            self.em.emit(f"and.pred {p}, {eqc}, {last};")
            self.em.emit(f"or.pred {p}, {p}, {over};")
            done = self.em.label("DONE")
            self.em.emit(f"@{p} bra {done};")
            self.decode_axes(tile, None, extents, split=split)
            self.emit_body(tile.body)
            self.em.mark(done)
            return
        if n_cells < total_threads or aux:
            p = self.em.reg("p")
            self.em.emit(f"setp.ge.s32 {p}, {gid}, {n_cells};")
            done = self.em.label("DONE")
            self.em.emit(f"@{p} bra {done};")
        self.decode_axes(tile, gid, extents)
        self.emit_body(tile.body)
        if done:
            self.em.mark(done)

    def compile(self):
        self.prologue()
        body = self.op.body
        if len(body) != 1 or type(body[0]).__name__ != "Tile":
            raise NotImplementedError("expected a single Tile at top level")
        self.compile_tile(body[0])
        self.em.emit("ret;")

    def render(self):
        out = [".version 8.0", ".target sm_89", ".address_size 64", ""]
        if self.dynamic_smem:
            if self.smem_used > self.smem_bytes:
                raise RuntimeError(
                    f"smem layout {self.smem_used} exceeds meta smem_bytes {self.smem_bytes}"
                )
            out.append(f".extern .shared .align {self.max_smem_align} .b8 __emmy_smem[];")
            out.append("")
        else:
            for name, info in self.shared.items():
                vec = 4 if info["nbytes"] <= 4 else 2
                align = max(info["align"] or 0, vec * info["nbytes"])
                out.append(f".shared .align {align} .b8 {name}[{info['size']}];")
            if self.shared:
                out.append("")
        out.append(f".visible .entry {self.entry}(")
        params = []
        for a in self.arg_order:
            if a in self.scalar_tys:
                pty = {"f16": "b16", "bool": "u8"}.get(self.scalar_tys[a], self.ptx(self.scalar_tys[a]))
                params.append(f"    .param .{pty} param_{a}")
            else:
                params.append(f"    .param .u64 param_{a}")
        out.append(",\n".join(params))
        out.append(")")
        out.append("{")
        c = self.em.counts
        for cls, ptx_ty in (("p", "pred"), ("f", "f32"), ("fd", "f64"), ("r", "b32"), ("rd", "b64"), ("h", "b16")):
            if c[cls]:
                out.append(f"    .reg .{ptx_ty} %{cls}<{c[cls] + 1}>;")
        out.extend(self.em.lines)
        out.append("}")
        return "\n".join(out) + "\n"


def find_kernel(graph):
    kernels = [
        name for name, node in graph.nodes.items()
        if type(node.op).__name__ == "KernelOp"
    ]
    if len(kernels) != 1:
        raise RuntimeError(f"expected exactly one KernelOp, found {kernels}")
    return kernels[0]


def main():
    if len(sys.argv) != 2:
        print("usage: python compile.py <kernel.ir.json>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    graph = Graph.from_dict(json.loads(path.read_text()))
    meta_path = path.parent / "meta.json"
    if not meta_path.exists():
        print(f"missing {meta_path}", file=sys.stderr)
        return 1
    meta = json.loads(meta_path.read_text())
    kernel = find_kernel(graph)
    compiler = Compiler(graph, meta, kernel)
    compiler.compile()
    sys.stdout.write(compiler.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
