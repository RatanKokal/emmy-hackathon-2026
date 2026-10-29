# Extracted verbatim from compile_speed_safe.py (validated checkpoint f289d05).
# per-statement emitters: load/write/assign/let/cond/predicates
import struct
import sys
from math import prod
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var

from .tables import (INT_TYPES, REG_CLASS, PTX_TY, WIDEN_AT_LOAD, SMEM_NBYTES,
                     F32_ONLY, UNROLL_CAP, PURE_OPS, COMMUTATIVE, _MEM_KINDS,
                     CF_KINDS)


class StmtsMixin:
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

    def stmt_accum(self, s):
        reg, ty = self.accum_reg(s)
        left = self.values[s.base][0] if s.base else reg
        val = self.values[s.value][0]
        self.emit_op(reg, s.op.name, ty, [left, val])

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
