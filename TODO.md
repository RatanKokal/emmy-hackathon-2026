# TODO

## Hidden-set hardening
- [ ] `cp.async` software swizzle (`CpAsyncCopy.swizzle` / `Write.swizzle` != `"NONE"`).
      Formula: flat smem element index `e ^ (((e >> shift) & mask) << 3)`; modes
      B128 `(6, 0x7)`, B64 `(6, 0x3)`, B32 `(6, 0x1)`, software override `"<mode>@<shift>"`.
      Emmy emits this only for slabs feeding an MMA/ldmatrix drain (excluded by TASK),
      and every public/schedule-tested IR uses `NONE`. Implement fill+drain together and
      verify with a hand-built swizzled IR against a torch/emmy-rendered reference.
- [x] `TreeHalve(inner=...)` (coop-t segmented combine) — implemented; flat path byte-identical.
- [ ] Runtime-validate a genuine coop-t IR (default emmy pipeline did not emit one for any
      probed reduction/matvec shape; `inner` path codegen-tested via a force-patched IR).

## Verification (see VERIFICATION.md)
- [x] Official pipeline 15/15 (all categories, incl. atomics + rasterized attention)
- [x] 85-problem regression: 78 GPU-pass, 7 documented non-defects
- [x] Differential vs emmy's own nvcc-compiled kernels: zero divergences, 11 bit-exact
- [x] Dispatcher x dtype x ptxas sweep: 0/333 invalid (was 95)
- [x] Exhaustive thread-decode proof: 236,480 gids vs emmy's formula
- [x] compile_speed audit: dispatcher matrix 408 emitted / 496 ptxas-clean incl.
      88 statement-sweep PTX; seed sweeps 1-3 all 77 PASS / 7 known non-defects
- [x] GPU-confirmed speed corner cases (findings.md "Oct 28"): dyn-loop load-cache
      leak, partial-unroll groups==0 (FAIL 143/2048), FMA cond alias (FAIL 2048/2048
      via fma_alias4 empty-else harness), Cond else addr_cache (CUresult 716, also
      in frozen; zero else-branches in corpus explains the 15/15 escape)

## Deferred (documented, unreachable per TASK.md)
- [ ] TMA/mbarrier statements (marked `*`; needs host tensor-map contract)
- [ ] fp8/fp4/bitcast, exotic buffer dtypes (marked `*`; loud raises in place)
- [ ] sin/cos sub-ulp floor, exp overflow-edge margin (tolerance-safe, speed trade)
