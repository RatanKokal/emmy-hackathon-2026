# Extracted verbatim from compile_speed_safe.py (validated checkpoint f289d05).
# Compiler core: state, geometry, tile compilation, render, entry points.
import json
import struct
import sys
from math import prod
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var

from .tables import (INT_TYPES, REG_CLASS, PTX_TY, WIDEN_AT_LOAD, SMEM_NBYTES,
                     F32_ONLY, UNROLL_CAP, PURE_OPS, COMMUTATIVE, _MEM_KINDS,
                     CF_KINDS)

from .emitter import Emitter, _Pair
from .analysis import AnalysisMixin
from .algebra import AlgebraMixin
from .ops import OpsMixin
from .stmts import StmtsMixin
from .smem import SmemMixin
from .loops import LoopMixin
from .passes import PassesMixin
from . import tables


class Compiler(StmtsMixin, SmemMixin, LoopMixin, PassesMixin, OpsMixin,
               AlgebraMixin, AnalysisMixin):
    F32_ONLY = tables.F32_ONLY
    UNROLL_CAP = tables.UNROLL_CAP
    PURE_OPS = tables.PURE_OPS
    COMMUTATIVE = tables.COMMUTATIVE
    _MEM_KINDS = tables._MEM_KINDS
    CF_KINDS = tables.CF_KINDS

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
