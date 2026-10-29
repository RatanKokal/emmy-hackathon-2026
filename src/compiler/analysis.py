# Extracted verbatim from compile_speed_safe.py (validated checkpoint f289d05).
# pure IR queries and provability checks
import struct
import sys
from math import prod
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var

from .tables import (INT_TYPES, REG_CLASS, PTX_TY, WIDEN_AT_LOAD, SMEM_NBYTES,
                     F32_ONLY, UNROLL_CAP, PURE_OPS, COMMUTATIVE, _MEM_KINDS,
                     CF_KINDS)


class AnalysisMixin:
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
        — including Cond.cond, Load.ind..."""
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
                return min(AnalysisMixin._tree_align(e.left), AnalysisMixin._tree_align(e.right))
            if e.op == "*":
                for x, y in ((e.left, e.right), (e.right, e.left)):
                    if type(y).__name__ == "Literal":
                        m = abs(int(y.value))
                        if m == 0:
                            return 30
                        return AnalysisMixin._tree_align(x) + (m & -m).bit_length() - 1
                return min(AnalysisMixin._tree_align(e.left), AnalysisMixin._tree_align(e.right))
        return 0

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
        return all(AnalysisMixin._ekey_stable(x, axis, stable)
                   for x in key if isinstance(x, tuple))

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

    def _vec_align(self, e):
        """Static element-index alignment as (coef, const) when the index is
        provably of the form coef*var + c..."""
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
        index provably ≡ 0 mod n (P3)."""
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

    def _check_rank(self, buf, index):
        shape = self.shared[buf]["extents"] if buf in self.shared else self.shapes.get(buf)
        if shape is not None and len(index) != len(shape):
            raise NotImplementedError(
                f"index rank {len(index)} != buffer rank {len(shape)} for '{buf}'")

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
