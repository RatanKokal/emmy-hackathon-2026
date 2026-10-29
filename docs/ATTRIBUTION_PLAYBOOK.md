# Attribution & Benchmarking Playbook

**Purpose:** the complete diagnostic methodology for comparing compiler outputs
(ours vs emmy+nvcc) on this project — every measurement technique, the exact
commands, the gotchas, and the findings register. Written for an agent taking
over the optimization work: read this before running any experiment.

The workflow this documents is: **diff first, build second, attribute third.**
Every optimization claim in this repo is backed by one of the experiments below.

---

## 1. The environments (what runs where, and what each is for)

| Environment | Hardware / toolchain | Role |
|---|---|---|
| **Official runner** (`http://runner.riftstack.ai:48701`) | RTX 5090 (sm_120), organizers' ptxas | **Benchmark of record.** The hidden set is evaluated here. Only place that matters for scoring. |
| **H200 box** (SSH key auth, flaky) | 4× NVIDIA H200 (sm_90), CUDA 12.0 ptxas/nvdisasm/cuobjdump/nvcc, python3.12 + emmy-ml | **Attribution lab.** SASS dumps, local CUDA-event timing, cross-toolchain experiments. GPU perf counters are BLOCKED (see §7). |
| **Dev machine** | python3.12 only (emmy-ml in `~/.local/lib/python3.12` — use `python3.12`, NOT `python` which is conda without emmy) | PTX generation, static comparison, corpus generation |

**Key asymmetry:** the two toolchains produce *different SASS from the same
PTX*. This is not a nuisance — it is the single most powerful attribution
tool (§4.4). The 15 official problems also live on the box at
`~/emmybench/problems/` (IR + meta + ref.cu + expected.npz).

---

## 2. Static comparison (free, do this FIRST for any gap)

### 2.1 Instruction histograms

```bash
# PTX opcode histogram — the "instruction mix" of two builds:
sed 's/^\t*//' ours.ptx | grep -oE "^[a-z][a-z0-9.]*" | sort | uniq -c | sort -rn
# same for theirs; diff the two histograms
```

Read it for: `ld.global` / `ld.shared` counts (memory ops), `fma` vs
`mul+add` pairs (contraction), `div/rem.s32` (emulated integer divide —
~20 SASS ops each, no HW int divide), `mul.lo` vs `shl` (strength
reduction), address-ALU volume (`add.s32/mul.lo.s32` — per-copy waste in
unrolled bodies).

### 2.2 Structure checks

```bash
# label/bra balance — every control-flow transform must keep this balanced:
grep -c bra f.ptx ; grep -cE "^[A-Za-z_]+[0-9]+:" f.ptx
# NOTE: partial-unrolled (countdown) loops legitimately have +1 bra over
# labels per loop (guard + group + remainder folded into one label set).
```

### 2.3 Load-address duplication (cross-loop redundancy)

```bash
grep -oE "ld\.global\.f32 %f[0-9]+, \[[^]]*\]" f.ptx | sed 's/%f[0-9]*//' | sort | uniq -c | sort -rn
# tokens are NOT enough: sibling loops get different base REGISTERS for the
# same runtime address. Count distinct (buffer, folded-offset) pairs instead
# by instrumenting stmt_load (§5.2). a01 had 192 redundant kt loads (4 loops
# x 64) invisible to token-grep.
```

### 2.4 emmy reference PTX generation

The reference answer for ANY IR — emmy's own renderer, then nvcc:

```python
from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.render import render_kernelop   # note: ir.kernel.render
g = Graph.from_dict(json.loads(open(ir).read()))
kern = next(n for n in g.nodes.values() if type(getattr(n,'op',None)).__name__ == "KernelOp")
tensors = {nid: n.output for nid, n in g.nodes.items() if n.output is not None}
cuda_src = render_kernelop(kern.op, tensors=tensors)
# then: nvcc -ptx -arch=sm_89 cu_file.cu -o emmy.ptx
```

Reading their PTX is the spec for what the lowering should produce
("diff first, build second"). Chaining it into a submission is forbidden.

### 2.5 SASS comparison

```bash
# CUDA 12 ptxas has NO -cubin flag — default output IS cubin:
ptxas -arch=sm_90 ours.ptx -o ours.cubin
nvdisasm -c ours.cubin | grep -E "LDG|FADD|IADD3|ISETP|BRA"
```

Read it for: LDG batching depth (loads-in-flight per iteration — with few
resident warps, loads-in-flight IS the performance model), FADD chain
placement (batched = latency-exposed per group; interleaved with the next
batch's LDGs = software-pipelined), per-iteration SASS size.

---

## 3. Runtime benchmarking

### 3.1 The official runner (benchmark of record)

```bash
python submit.py <problem> --compiler compile.py        # correctness + 1 iter
# direct API for controlled A/B:
POST /compile {ptx}            -> cubin_id, registers per entry
POST /run      {cubin_id, entry, grid, block, smem_bytes, zero_outputs,
                args, warmup<=20, iters<=200, max_return<=2^22}
                -> median_us
```

**Gotchas:** args must have the `intermediate` key STRIPPED; warmup/iters
caps are 20/200 (422 otherwise); 20 s per-launch limit; the runner is
403 from the dev machine — reachable only from the H200 box.

### 3.2 The H200 local runner (attribution lab)

`run_local.py` (ctypes CUDA driver API): loads a cubin, materializes
meta.json args from seeds, CUDA-event timing. Uses `cuCtxCreate` —
**NOT** `cuDevicePrimaryCtxRetain`+`cuCtxSetCurrent` (that path returns
CUDA error 201 on this box — see §7). Compile cubins with
`ptxas -arch=sm_90` (sm_89 cubins fail with error 209 on H200).
**Always `cuCtxDestroy` the context after each run** — 176 leaked
contexts exhaust the driver (CUDA error 2 at cuCtxCreate).

---

## 4. The attribution ladder (how to locate a runtime gap)

Apply in order; each level either localizes the gap or eliminates a layer.

### 4.1 Bounded-iteration decomposition

Set the dominant loop's trip to 1 (sed the `setp` immediate), measure both
builds, subtract:

```
per_iter_cost = (t_full − t_trip1) / (N − 1)
```

Fixed overhead (launch/prologue/init) vs loop body — separates "same kernel,
slower body" from "everything slower". Used on a02: overhead parity (8.96 vs
8.80), per-iteration 0.468 vs 0.336 µs → the whole gap was the loop body.

### 4.2 Latency modeling (for low-occupancy kernels)

Grid [1,1,1] kernels (the big reductions) run **4 warps total** — no
co-resident warps to hide memory latency. Model:

```
time ≈ (trip / loads_in_flight) × mem_latency + compute
```

Validated: r02 with 16 loads in flight predicts 11.4 µs; measured 10.94.
If a competitor is faster with the same batching depth, its SASS pipelines
across iterations — check §4.4 for interleaved LDG/FADD.

### 4.3 Grid-sensitivity probe

Run both builds at native grid and k× warps (ctypes runner makes this a
parameter). If the gap shrinks with more warps → per-warp latency exposure
(scheduling/MLP). If it persists → memory-system or compute bound.

### 4.4 Cross-toolchain reproduction (THE attribution experiment)

Same PTX pair, different toolchain:

| toolchain | result | conclusion |
|---|---|---|
| runner (their ptxas, sm_120) | gap appears | — |
| H200 (CUDA 12 ptxas, sm_90) | parity / reversed | **the gap is created downstream of the PTX** |

If the gap vanishes on identical toolchains, it is ptxas-version/GPU-
generation scheduling — outside the PTX we write, inside the layer the
rules delegate to NVIDIA. Document and stop optimizing that kernel.

### 4.5 ncu stall attribution — BLOCKED

`ncu` exists on the box but returns `ERR_NVGPUCTRPERM` (no sudo, driver
restricts profiling). The intended experiment — decompose cycles into
`long_scoreboard` / `not_selected` / `math_pipe_throttle` / `no_instruction`
stall ratios — would be the definitive attribution. If the box ever gets
profiling permissions (`NVreg_RestrictProfilingToAdminUsers=0`), run:

```bash
ncu --metrics gpu__time_duration.sum,smsp__inst_executed.sum,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio \
    python3 run_local.py <cubin> <meta>
```

---

## 5. Compiler-internal debugging (this codebase's specific hazards)

### 5.1 The cache taxonomy — what each cache keys on, what invalidates it

| cache | key | invalidation |
|---|---|---|
| `expr_cache` | structural expression tree (ekey) — does NOT encode variable *values* | cleared around control flow (dominance), after loops (zero-trip), per unrolled copy (binding changes) |
| `load_cache` | **folded address tuple** `(input, off, len(names))` — encodes per-copy literals | memory mutation (Write/Sync/CpAsync), branches (non-dominance), loop exits (trip=0); NOT cleared on unrolled-copy boundaries (the folded literals distinguish copies) |
| `addr_cache` | `(buffer, index register, elem size)` | control flow both sides; the loop-exit dominance rule; before remainder bodies (a loop-carried register's value changes across the back-edge) |

**The critical lesson:** structural (ekey) keys cannot distinguish unrolled
copies — a `Load` cache keyed on index ekeys will wrongly deduplicate
sibling copies and silently reuse the wrong values (this broke m01's pair
cache: err 165). Folded-value keys fix it. Conversely, folded keys embed
register names — any pass that RENAMES registers (value numbering) must
rewrite the cached tuples too (`_cse_lines._fix`).

**Cache-clear hygiene classification (the current stmt() discipline):**
entries go stale only on memory mutation (Write/Sync/CpAsync*), branches
(Cond), and loop exits (trip=0 dominance). Register-only statements
(Accum, WarpShuffle) and compile-time declarations (Smem) must NOT clear —
their clears killed cross-loop load sharing (a01's 4 loops × one kt column:
192 redundant loads, invisible to token-grep, found by the Shadow-dict).

### 5.2 The Shadow-dict trace (finding who clears a cache)

```python
class Shadow(dict):
    def clear(self):
        for fr in traceback.extract_stack():
            if "compile_speed" in fr.filename and fr.name != "clear":
                print(f"  {fr.name}:{fr.lineno}", file=sys.stderr)
        dict.clear(self)
c.load_cache = Shadow(c.load_cache)     # REPLACE the attribute (patching
                                        # dict.clear is read-only)
```

Run the compiler; every cache wipe prints its call site. This found the
cross-loop kt-clear in minutes after an hour of static reasoning.
**Print ALL compile_speed frames** (`[-2]` grabs the wrong caller when
stmt() nests through emit_body).

### 5.3 The pending-marker pattern (deferred dominance)

Loop-body definitions do not dominate post-loop code (trip may be zero), but
they DO dominate a *sibling loop with the identical static trip*. The
dispatcher sets `_after_loop = static_trip` at a loop exit; the next
statement keeps the cache only if it is a same-trip sibling loop, else
clears and resets the marker. **The marker must be consumed (set to None)
when the decision is made** — leaving it pending makes the NEXT statement's
first load re-trigger the clear (this exact bug cost a debug cycle).

### 5.4 Instrumented-compiler debug pattern

```python
import importlib.util
spec = importlib.util.spec_from_file_location("cs", "compile_speed.py")
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
orig = cs.Compiler.stmt_load
def traced(self, s): ...print keys/sizes...; return orig(self, s)
cs.Compiler.stmt_load = traced
# build Compiler(g, meta, kernel_name) directly; redirect stdout; run
```

Find the KernelOp by op type, not by name (node names vary). The
`find_kernel` logic: `[name for name, node in g.nodes.items() if
type(node.op).__name__ == "KernelOp"]`.

### 5.5 Diff-first workflow

For any optimization: generate emmy's PTX for the target IR FIRST (§2.4),
diff the histograms/structure against ours, read the delta as the
specification, implement the smallest transform that produces it, diff
again. Every skip of this step cost a wasted benchmark cycle.

---

## 6. Findings register (do not re-derive)

| finding | evidence | status |
|---|---|---|
| m01 matmul: exact parity with emmy+nvcc (15.04 µs), 2.63× over frozen | bench_all12 | closed |
| a01 attention: 1.34× FASTER than emmy+nvcc (47.78 vs 64.16) | bench_all12 | closed — at the 255-register ceiling, no headroom left |
| r03_mean: exact parity (6.848) | bench_all11 | closed |
| r02_big_sum: 1.23× residual — **toolchain-side** | H200 cross-toolchain: parity (15.73 vs 14.93), ours wins at 4× warps | attributed, closed |
| a02_attention: 1.42× residual — cross-iteration software pipelining | bounded-iteration: per-iter 0.468 vs 0.336 µs; emmy holds next-iteration loads in its 77 regs | open — the rotating-loop transform is the only lever (~100+ lines) |
| L2 prefetch (body-top, guarded): **proven dead end** | a01 7.4× slower, a02 13× slower, register collapse 255→48, reductions broke | do not retry without a profiler |
| 13/15 kernels at parity-or-ahead; all 15 bit-exact-verified | verify_speed13 + bench_all14/12 | current state |

**Gains vs frozen (final):** m01 2.63×, a01 2.41×, r02 1.94×, r03 1.89×,
a02 1.25×; everything else parity.

---

## 7. Failure archive (every error, its cause, its fix)

| error | cause | fix |
|---|---|---|
| `ModuleNotFoundError: emmy` | `python` is conda (no emmy-ml) | use `python3.12` (emmy-ml in `~/.local/lib/python3.12`) |
| runner HTTP 403 from dev machine | proxy | run from the H200 box |
| runner 422 on /run | `intermediate` key present; warmup/iters over caps (20/200) | strip the key; clamp to caps |
| `ptxas fatal: Unknown option 'cubin'` | CUDA 12 ptxas: cubin is the default output | drop the flag |
| CUDA error 209 | sm_89 cubin on an H200 (sm_90) | `ptxas -arch=sm_90` locally |
| CUDA error 201 at cuMemAlloc | `cuDevicePrimaryCtxRetain`+`SetCurrent` broken on this box | use `cuCtxCreate` |
| CUDA error 2 at cuCtxCreate (bulk runs) | leaked contexts (one per run_local invocation) | `cuCtxDestroy` after each run; serialize heavy chains |
| ctypes `cuMemAlloc` garbage sizes | no argtypes → 32-bit int in a 64-bit slot | set explicit argtypes for every driver call |
| `ERR_NVGPUCTRPERM` | profiling needs root | unavailable — use the attribution ladder instead |
| SSH to box flaky/timeouts | documented | retry loop, background watcher, `ConnectTimeout=10`, BatchMode |
| box /tmp cleaned | ephemeral | work in `~/emmybench`; **md5-verify every upload** (a stale-file launch cost one full bench cycle) |
| runner "busy" | shared queue | retry loop on `stage=="busy"`, sleep 3 |
| bench resume skipped fresh runs | resume logic read the OLD log | filter the done-set per problem with anchored patterns (`^name,`) — unanchored grep -v missed line-start names |

---

## 8. The corpora

| corpus | location | contents |
|---|---|---|
| official problems | `problems/` (15) + box `~/emmybench/problems/` | IR + meta + ref.cu + expected.npz |
| statement sweep | `.tools/stmt_sweep/` (88 dirs) + box `~/emmybench/sweep_corpus/` | `<entry>.ir.json` + `meta.json` per emitter path × dtype — NO ref.cu (generate via render_kernelop, §2.4) — NO expected (speed sweeps only) |
| pre-compiled sweep PTX | `.tools/sweep_frozen/`, `.tools/sweep_live/`, `.tools/sweep_out/` | frozen vs live-build outputs for static diffing |
| generator | `tools/generate.py` | synthetic problem generation |

---

## 9. Bench scripts (on the box, `~/emmybench/`)

| script | what |
|---|---|
| `run_local.py` | ctypes CUDA-driver runner: cubin + meta.json → CUDA-event medians; `cuCtxCreate`-based; argtypes-complete; **add cuCtxDestroy for bulk sweeps** |
| `bench_all.py` | 15 official problems × {frozen, speed, emmy+nvcc} through run_local; resume-aware via `bench_all.log` |
| `bench_sweep.py` | the 88-problem corpus × {ours, emmy+nvcc} on H200 |
| `verify_speed.sh` | 15-problem correctness sweep (submit.py) + m01/a01 pair bench |
| `bench_all*.log` | results archive — check the newest before re-running |

**Protocol discipline:** scp + md5-verify the compiler before every bench
run (a stale-file launch cost a full cycle); filter resume-logs with
line-anchored patterns (`^name,`); kill stale background chains
(`pkill -f verify_speed`) before relaunching; never run bulk sweeps
concurrently with the verification chains (context exhaustion).

---

## 10. The transform inventory (`compile_speed.py` vs the frozen baseline)

Eleven shipped transforms, one reverted failure. All are shape-triggered —
they fire automatically on hidden-set kernels of the same families.

### 10.1 Shipped

| # | transform | mechanism | kernels hit | measured effect |
|---|---|---|---|---|
| 1 | full auto-unroll | flat + static-trip ≤ 128 loops → N straight-line copies, axis bound to `("imm", k)`; nested-loop gate mirrors nvcc's refusal | matmul (m01 ×64, m02 ×8), attention (a2 loops ×64/×32) | m01: loop overhead + per-copy stalls deleted |
| 2 | accumulator-reset semantics | nested seeded loops' accumulators re-init per unrolled copy (`_body_accums`/`_nested_seed_accums` + `values.pop`) | correctness enabler for #1 (nested dot products) | without it unrolling corrupts nested reductions |
| 3 | fma contraction | `Assign(multiply)` + adjacent single-use `Accum(add)` → one `fma.rn` (nvcc -fmad parity) | 7 of 15 (a01 ×20, m01 ×4, m02 ×2, a02 ×3, f* ×1) | dependency chains halved |
| 4 | strength reduction | div/rem by power-of-2 → biased shifts (`_divmod_pow2`) | all 15 (axis decode) | 140 emulated divisions → 2 |
| 5 | value numbering (`_cse_lines`) | straight-line VN + renaming over unrolled regions; PURE_OPS incl. mul.wide.s32/add.s64 | unrolled kernels | m01 integer ALU ~1900 → ~150 |
| 6 | reassociation | iadd/isub fold `(r+c) ± x → (r±x) + c` | unrolled kernels | per-copy materializations collapsed (m01 adds 950→66) |
| 7 | folded-value load cache | load_cache keyed on the folded address tuple, not structural ekeys | all unrolled; a01 critically | a01: 514 → 322 loads (192 redundant kt cross-loop) |
| 8 | deferred dominance hygiene | dispatcher clears reclassified: mem-mutators/barriers clear; register-only don't; loop exits defer via the `_after_loop` marker | a01 (4 sibling a2 loops share one kt column) | enables #7's cross-loop dedup |
| 9 | partial unroll (countdown) | big flat static-trip loops → 16-copy groups + 1-copy remainder; runtime group count | r02 (trip 512), r03 (trip 256) | r02 21.2→10.9, r03 13.0→6.85 (parity) |
| 10 | alignment-tracked pair merge | sibling loads +1 element in a stride-1 dim, 8B-provable → v2 (`_tree_align`/`_leaf_step_diff`/`_index_align_exp`) | m01 (b-pairs), p03, a01 | m01: scalar b-loads → v2 |
| 11 | hygiene reclassification | Accum/WarpShuffle/Smem no longer clear the load cache (register-only/compile-time) | a01 | prerequisite for #7/#8 |

### 10.2 The reverted failure

| transform | result | lesson |
|---|---|---|
| L2 prefetch (body-top, guarded, nested-simulated) | a01 7.4× slower, a02 13× slower (register collapse 255→48), reduction sums corrupted via expression-cache leakage (two layers — the first fix was insufficient) | prefetch hints occupy the LSU pipeline real loads need and destabilize ptxas's allocation — do not retry without profiler access |

### 10.3 Impact by kernel category

| category | transforms that fire | gains vs frozen |
|---|---|---|
| pointwise (p00–p03) | #4, #10 (p03) | parity (memory-trivial) |
| reductions (r01–r04) | #4, #9, #11 | r02 1.94×, r03 1.89×, r01/r04 parity |
| matmul (m01/m02) | the full stack | m01 2.63× (exact nvcc parity), m02 parity |
| fused (f01–f03) | #3, #4 | parity (bandwidth-bound) |
| attention (a01/a02) | #1, #2, #3, #4, #7, #8 | a01 2.41× (beats nvcc), a02 1.25× |

### 10.4 The remaining (documented) gaps

- a02 (1.42×): cross-iteration software pipelining — the rotating-loop
  transform; register-feasible (128 → 255 headroom); a01 is register-blocked
  (at the 255 ceiling) — measured to the cycle (0.468 vs 0.336 µs/iter).
- r02 (1.23×): proven toolchain-side (H200 cross-toolchain parity, §4.4).

**Correctness invariant across every transform:** all 15 official kernels
bit-exact vs expected on the official runner, verified at every step.
Additionally, the 3-way corpus head-to-head (frozen vs ours vs emmy-render,
§8 sweep) resolved every sum deviation: **ours == frozen exactly on all 11
comparable disputed kernels** (7 narrow-dtype loadg kernels where emmy's
harness-run was the outlier, tree_f64 where ours==frozen==nan, 2 shuffles
all-agree); loadg_bf16 "differs" only because bf16's ~2-digit precision makes
1% sum thresholds meaningless; writev4_* were launch-blocked for all three.
**Zero kernels where our transforms deviate from the 100/100-verified frozen
baseline.** The 3-way head-to-head (frozen as the gold standard) is the
recommended correctness instrument for any corpus lacking expected values.
