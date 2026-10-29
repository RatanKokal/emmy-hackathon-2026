# Extracted verbatim from compile_speed_safe.py (validated checkpoint f289d05).
# shared memory, barriers, cp.async, warp shuffle, tree halve
import struct
import sys
from math import prod
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var

from .tables import (INT_TYPES, REG_CLASS, PTX_TY, WIDEN_AT_LOAD, SMEM_NBYTES,
                     F32_ONLY, UNROLL_CAP, PURE_OPS, COMMUTATIVE, _MEM_KINDS,
                     CF_KINDS)


class SmemMixin:
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

    def apply_swizzle(self, flat, swizzle):
        """Apply shared-memory address swizzle to a flat element index."""
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
