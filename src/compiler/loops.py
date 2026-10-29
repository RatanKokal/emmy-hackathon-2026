# Extracted verbatim from compile_speed_safe.py (validated checkpoint f289d05).
# loop emission: full and partial unroll
import struct
import sys
from math import prod
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var

from .tables import (INT_TYPES, REG_CLASS, PTX_TY, WIDEN_AT_LOAD, SMEM_NBYTES,
                     F32_ONLY, UNROLL_CAP, PURE_OPS, COMMUTATIVE, _MEM_KINDS,
                     CF_KINDS)


class LoopMixin:
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

    def stmt_strided_loop(self, s):
        self.emit_loop(s.axis, s.start, s.step, s.end, s.body, s.seed,
                       unroll=bool(getattr(s, "unroll", False)))

    def stmt_loop(self, s):
        self.emit_loop(s.axis, 0, 1, None, s.body, s.seed,
                       unroll=bool(getattr(s, "unroll", False)))

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
