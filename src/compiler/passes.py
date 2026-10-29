# Extracted verbatim from compile_speed_safe.py (validated checkpoint f289d05).
# speed passes: load merge, vectorization, body walk with FMA gate
import struct
import sys
from math import prod
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var

from .tables import (INT_TYPES, REG_CLASS, PTX_TY, WIDEN_AT_LOAD, SMEM_NBYTES,
                     F32_ONLY, UNROLL_CAP, PURE_OPS, COMMUTATIVE, _MEM_KINDS,
                     CF_KINDS)
from .emitter import _Pair


class PassesMixin:
    def _merge_loads(self, body):
        """Pair scalar loads of the same buffer whose indices differ by exactly
        one element in a stride-1 dime..."""
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
