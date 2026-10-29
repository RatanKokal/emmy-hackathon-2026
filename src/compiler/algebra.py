# Extracted verbatim from compile_speed_safe.py (validated checkpoint f289d05).
# integer/float expression algebra and folded addressing
import struct
import sys
from math import prod
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var

from .tables import (INT_TYPES, REG_CLASS, PTX_TY, WIDEN_AT_LOAD, SMEM_NBYTES,
                     F32_ONLY, UNROLL_CAP, PURE_OPS, COMMUTATIVE, _MEM_KINDS,
                     CF_KINDS)


class AlgebraMixin:
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

    def float_literal(self, value, ty):
        value = float(value)
        if ty == "f64":
            bits = struct.unpack("<Q", struct.pack("<d", value))[0]
            return f"0d{bits:016X}"
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return f"0f{bits:08X}"

    def address(self, buf, index, nbytes):
        off = self.linear_index(index, self.shapes.get(buf, ()))
        return self._addr_from_off(buf, off, nbytes)

    def _addr_from_off(self, buf, off, nbytes):
        """Stateless folded addressing: immediate offsets fold into the
        ld/st operand; regimm recomputes its b..."""
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
