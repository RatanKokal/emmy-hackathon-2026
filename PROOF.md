# Translation correctness: conditions, bounds, and residual risk

This document states what the compiler guarantees, under what conditions, with
what error bounds, and what remains unproven. It replaces the earlier version,
whose claims were stronger than the arguments supported.

**Theorem (conditional).** For a Kernel IR program P satisfying preconditions
P1–P6 (§4.1), compiled by this compiler to PTX for sm_89 and executed on the
target GPU, each output buffer matches emmy's CUDA rendering of P up to the
per-operation tolerances ε_f defined in §3.3, composed as described in §4.5.
Outside these preconditions or beyond the stated domains, the compiler either
raises `NotImplementedError` or the output is not covered by this document.

This is not a proof of absolute correctness. It is a conditional refinement
statement: the compiler's output is close to the reference implementation's
output, for inputs that satisfy stated preconditions, with stated error bounds.

---

## 1. What the compiler is (and is not)

The compiler reads a Kernel IR graph and emits PTX for sm_89. The translation
is statement-by-statement: each IR statement maps to a fixed sequence of PTX
instructions. There is no cross-kernel optimization.

However, within a statement sequence, the compiler performs:

- **Common subexpression caching** (expr_cache, load_cache, addr_cache) —
  prevents duplicate emission of identical pure expressions
- **Regimm folding** — `gid ± constant` folded into a base+offset register pair
- **Constant folding in unrolled copies** — axis values are compile-time constants
- **FMA contraction** — a single-use `multiply` feeding an `Accum(add)` becomes
  a single `fma.rn`

These are cross-statement optimizations. They are value-preserving under
stated conditions (§4.4), but the compiler is not a pure lookup table —
the PTX for a statement depends on what came before it (through the caches).

---

## 2. PTX semantics used

We emit a subset of PTX ISA 8.0 for sm_89. The hardware semantics we rely on:

**IEEE 754 float arithmetic (round-to-nearest):**
`add.rn`, `sub.rn`, `mul.rn`, `div.rn`, `fma.rn`, `sqrt.rn` — these compute
the correctly rounded result of the corresponding real-number operation.

**Integer arithmetic (two's complement, wraparound):**
`add.s32`, `sub.s32`, `mul.lo.s32`, `add.s64`, etc. — modular arithmetic.
The IR defines wraparound for signed overflow (this is an IR-level definition,
not a C guarantee, since signed overflow is UB in C).

**Integer division/remainder:**
`div.s32` — truncating division (C semantics). `rem.s32` — remainder with
sign of dividend (C semantics). These match emmy's rendering of `//` as C `/`
and `remainder`/`mod` as C `%`.

**Predication:**
`setp.{lt,le,gt,ge,eq,ne}.{f32,f64}` — ordered comparison (false on NaN).
`setp.ne.{f32,f64}` is also ordered (false on NaN), unlike C's `!=`.
The compiler emits `setp.eq + not.pred` for `!=` to match C semantics.
`selp.{type} dst, a, b, p` — `dst = p ? a : b`.

**Approximate transcendentals (f32 only):**
`ex2.approx.f32` — 2^x, rel err ≤ 2⁻²² (NVIDIA PTX ISA table).
`lg2.approx.f32` — log₂(x), abs err ≤ 2⁻²² for x > 0 (NVIDIA PTX ISA table).
`sin.approx.f32` — sin(x), abs err ≤ 2⁻²⁰·⁵ for |x| ≤ π (in-quadrant).
`cos.approx.f32` — cos(x), same envelope.
These instructions do NOT have f64 variants. All f64 transcendental ops
raise `NotImplementedError`.

**Conversion:**
`cvt.{dst}.{src}` — type conversion, with optional rounding modifier
(`.rn` = nearest, `.rzi` = toward zero, `.rmi` = toward −inf, `.rpi` = +inf).
Rounding modifiers apply to float source/dest only.

**Data movement:**
`ld.global`, `st.global` — global memory (32-bit address space via generic).
`ld.shared`, `st.shared` — shared memory (32-bit shared window).
`ld.global.v2/v4` — vectorized load (alignment: base + offset ≡ 0 mod width).
`red.global.add.{f32,u64}` — atomic add, result discarded, order nondeterministic.
`cp.async.{ca,cg}.shared.global` — asynchronous global→shared copy.

**Barriers:**
`bar.sync 0` — CTA-wide barrier (all threads).
`bar.sync id, count` — named barrier; `count` is the number of participating
threads (must be a multiple of the warp size). The count is required by emmy's
renderer for named barriers, not by PTX (where it is optional).
`bar.warp.sync 0xffffffff` — warp-synchronizing barrier (all lanes in the warp).

**Warp shuffle:**
`shfl.sync.bfly.b32 dst, src, laneMask, c, membermask` — butterfly shuffle.
`c` encodes clamp (bits 4:0) and segment (bits 12:8). 0x1f = clamp to 31,
no segment split. The source lane is `laneid XOR laneMask`, clamped to the
segment. Lanes outside the membermask produce undefined values.

---

## 3. Per-operation translation bounds

### 3.0 Obligation format

For an op f with reference f_ref (emmy's CUDA rendering: `expf`, `sinf`, …),
the translation is ε-refining on a stated domain D_f:

∀x ∈ D_f: |PTX_f(x) − f_ref(x)| ≤ ε_f(x)

where ε_f combines the PTX-vs-true error and the libm-vs-true error
(taken as ≤ 2 ulp ≈ 2.4e-7 relative for correctly rounded ops, ≤ 2 ulp for
the libm function). Outside D_f, either special values match f_ref exactly
(see the pow table) or the compiler raises.

Here u = 2⁻²⁴ ≈ 5.96e-8 (f32 machine epsilon / 2).

### 3.1 IEEE 754 float arithmetic

For f32/f64, IR `add`/`sub`/`multiply`/`divide` mean IEEE 754 arithmetic with
round-to-nearest. PTX `add.rn.f32` etc. implement the same standard.

**Proof:** `add.rn.f32 %dst, %a, %b` computes `round_nearest(a + b)`.
The IR `Assign(op="add", dtype="f32")` means the same. The compiler emits
exactly `add.rn.f32` for this statement. Same for `sub.rn`, `mul.rn`,
`div.rn` (with `-prec-div=true` default). No rounding-mode divergence
exists on any shared op. ∎

**Integer arithmetic:** for int dtypes, IR `add`/`subtract`/`multiply`/
`divide`/`remainder` mean two's complement wraparound and truncating division.
This is an **IR-level definition** (signed overflow is UB in C, so the IR
defines the behavior, it doesn't inherit it from C). PTX `add.s32`, `mul.lo.s32`,
`div.s32`, `rem.s32` implement exactly this. ∎

**FMA note:** the f32 `mul`→`Accum(add)` contraction emits `fma.rn.f32`,
which computes `round_nearest(a×b + c)` (single rounding). The non-contracted
path emits `mul.rn.f32` then `add.rn.f32` (two roundings). The difference
is ≤ 1 ulp of the product `a×b`, not of the sum. For reductions (sum of many
products), per-step differences can accumulate to O(n) ulps of the running
sum. The emmy/nvcc reference also contracts (default `-fmad=true`), so both
implementations contract. The difference between our contraction and nvcc's
is zero when the contraction pattern is the same (same multiply, same add).
When the patterns differ (e.g., nvcc contracts across an intervening
instruction), the results can differ by ≤ 1 ulp of the product. This is
within tolerance for all corpus problems. ∎

### 3.2 Integer arithmetic

**Signed overflow:** the IR defines signed integer arithmetic as two's
complement wraparound. This is NOT a C guarantee (signed overflow is UB in C).
It is an IR-level definition. PTX `add.s32` implements wraparound. ∎

**Unsigned operations:** `u32`/`u64` use the same wraparound. `div.u32` and
`rem.u32` are unsigned (no sign extension). The IR `u32` dtype maps to
unsigned semantics. ∎

### 3.3 Transcendental functions (f32)

Each f32 transcendental op has an error bound relative to the libm reference
(`expf`, `sinf`, etc.), on a stated domain. Outside the domain, either
special values match, or the compiler raises. All bounds are **analytical
derivations from the PTX ISA error tables**, not empirical measurements.
Empirical values are labeled as such.

**exp**

Sequence: `t = fl(x · c)` where c = fl(log₂e), then `ex2.approx.f32(t)`.

- c = fl(log₂e) has relative error ≈ 1.34e-8 (constant rounding)
- Product rounding: ≤ u ≈ 5.96e-8
- ex2.approx.f32: rel err ≤ 2⁻²² ≈ 2.4e-7

Total relative error vs true eˣ:
≤ 2.4e-7 + (5.96e-8 + 1.34e-8)·|x| = 2.4e-7 + 7.3e-8·|x|

At |x| = 88.7 (the f32 exp overflow threshold): ≤ 6.7e-6.
Combined with libm's ≤ 2 ulp: ≤ 7e-6. Passes rtol 1e-5 with ~1.4× margin.

Domain edges:
- Overflow: ex2 produces +inf at t ≥ 128, i.e., x ≳ 88.7228. Near this
  threshold, PTX and expf can disagree (finite vs inf). This is a
  tolerance-boundary issue, not a correctness issue (both are "very large").
- Subnormal results (x < −87.3): the relative bound doesn't apply because
  it depends on ex2's flushing behavior. The absolute difference is below
  1.2e-38, which passes any positive atol but fails pure-rtol checks. ∎

**sin / cos**

Sequence:
1. k = round(x · (1/2π)) in f64
2. r = fma.rn.f64(−k, 2π_f64, x) [single rounding, or sub after mul —
   verify which is emitted]
3. Narrow r to f32: cvt.rn.f32.f64
4. `sin.approx.f32(r)` or `cos.approx.f32(r)`

Error terms:

| Term | Bound |
|---|---|
| Reduction (with fma, single rounding of k·2π) | ≤ k · ulp(2π_f64) ≈ (|x|/2π) · 8.9e-16 ≈ \|x\| × 1.4e-16 |
| Reduction (without fma, product rounds separately) | ≈ \|x\| × 2.1e-16 |
| cvt.rn.f32.f64 of r | ≤ u·|r| ≤ 5.96e-8·π ≈ 1.9e-7 |
| sin.approx.f32 on [−π, π] | ≤ 6.9e-7 abs (PTX ISA table) |

Total for |x| ≤ 1e6: ≤ 1.4e-10 (reduction) + 1.9e-7 (cvt) + 6.9e-7 (approx)
≈ 8.9e-7 abs. Well inside atol 1e-4, with room for libm's 2 ulp.

**Domain limitation:** beyond |x| ≈ 2.5e12, the reduction error reaches 1e-4
(the typical tolerance). For larger inputs, CUDA's `sinf` uses Payne–Hanek
reduction and the f64 scheme is insufficient (mismatches up to O(1)). The
compiler does not implement Payne–Hanek reduction. **Inputs with |x| > 1e6
are outside the verified domain** for sin/cos. The earlier claims "≤ 7e-7 for
any f32 input" and "exact when |x| < 2.8e16" are retracted. Only the integer
representability of k holds at 2.8e16, not accuracy. cos is analogous. ∎

**tanh**

Sequence [verify exact emitted form]: the exp-identity chain
`E = ex2.approx(−2|x|·log₂e); result = (1 − E)/(1 + E)`, with sign restored
and saturation at |x| > 20.

**Saturation reason (corrected):** for |x| > 20, `2·e^(−2|x|) < 2⁻²⁵`, so
tanhf returns exactly ±1.0f. Clamping at |x| > 20 is therefore value-identical
to tanhf. The reason is **f32 precision** (the correction term falls below
the f32 rounding threshold), not overflow (e^{2x} overflows f32 only past
|x| ≈ 44). The earlier explanation is retracted.

**Error:** the ex2 chain gives E a relative error of
≤ 2.4e-7 + 7.3e-8·|2x| = 2.4e-7 + 1.5e-7·|x|. Since E ≤ 1, the absolute
error of (1−E)/(1+E) is ≲ 5e-7 (conservative). The measured value 1.2e-7
is empirical.

**Small-x cancellation:** for x ≈ 0, `1 − E` cancels (both ≈ 1), so the
relative error is amplified. The absolute error remains ≤ 5e-7, which is
fine for atol ≥ 1e-5. If any problem uses pure rtol on tanh, the small-x
regime needs an expm1-style path or the identity `tanh(x) ≈ x` for |x| < ε.
No corpus problem uses pure-rtol tanh. ∎

**sigmoid**

Sequence: `E = ex2.approx(−x · log₂e)`, then `add.rn 1`, then `rcp.rn`.
The table entry should read `ex2.approx(−x·log₂e)`, not `ex2.approx(−x)`
— the log₂e multiply is present in the emitted code.

**Error:** σ′ = σ(1−σ) ≤ ¼, so the error in E contributes
≤ ¼ · (2.4e-7 + 7.3e-8·|x|) ≤ 6e-8. The add and rcp.rn each contribute
≤ 6e-8. Total ≲ 2e-7 per side, ≲ 4e-7 difference vs the reference,
consistent with the measured 4e-7.

Saturation: x → −∞ gives E = +inf and rcp.rn(+inf) = 0 ✓.
x → +∞ gives E = 0 and rcp.rn(0) = +inf, then +1 gives 1 ✓.
NaN propagates. ∎

**pow (f32)**

Sequence: `L = lg2.approx(|x|)`, `z = fl(y · L)`, `mag = ex2.approx(z)`,
with sign correction for negative x (even/odd integer y), NaN for
non-integer y with negative x, and pow(0,0) = 1.

Relative error vs true result:
≤ ln2 · (|y| · ε_L + u · |z|) + 2⁻²²

where ε_L is the lg2.approx absolute error. Two regimes:

- **Large \|z\| (up to 128):** ≈ 2.1e-7·\|z\| + 2.4e-7, so ≤ 2.7e-5 at \|z\| = 128.
  Passes rtol 1e-4 but not 1e-5.
- **x ≈ 1, large \|y\|:** the \|y\| · ε_L term dominates. At x = 1+2⁻²³ and
  y = 10⁶, the true z ≈ 0.17 while the lg2 error alone can be ≈ 0.24,
  producing a wrong result. The error model must include this term.

**Domain:** |y| ≤ 64 and z ∈ [−126, 128], with a bound of ≤ 3e-5 relative.
Passes rtol 1e-4. Larger |y| is not covered. The earlier "≤ 1e-4 at
z ≥ 1e3" was vacuous because ex2 has already saturated by then. ∎

**Special values** (must match powf exactly; currently unproven — add a
sweep test over this table):

| Case | powf result |
|---|---|
| pow(x, ±0), any x including NaN | 1 |
| pow(1, y), any y including NaN | 1 |
| x < 0, y non-integer | NaN |
| x < 0, y odd integer | sign flips |
| x < 0, y even integer | \|x\|^y |
| pow(±0, y < 0) | ±inf if y is an odd integer, else +inf |
| pow(±0, y > 0) | ±0 if y is an odd integer, else +0 |
| pow(−1, ±inf) | 1 |
| pow(x, ±inf), pow(±inf, y) | by the standard table |
| z > 128 (after reduction) | +inf |
| z < −150 (after reduction) | 0 |

"Matches powf exactly" should be downgraded until this table passes
on the GPU. ∎ [for the special values; the magnitude bound is as above]

**log**

The absolute error is up to ≈ ln2 · 2⁻²² · |log₂(x)|, about 2.1e-5 at
x = 1e-38 (empirical). This is abs-only, because near x = 1 the relative
error is unbounded (log(1) = 0). Passes atol ≥ 1e-4. The earlier "1.6e-5"
was an empirical value, not a derived bound. ∎

**erf**

The A&S 7.1.26 rational approximation has max abs err 1.5e-7 in exact
arithmetic. In the f32 chain, the accumulated error is ≤ 3.2e-7
(**empirical measurement**, not a derived bound — the individual rounding
errors compound through 15+ operations). The libm `erff` has ≤ 2 ulp.
Both pass atol ≥ 1e-4. Odd symmetry and NaN propagation match. ∎

**sqrt, rsqrt, reciprocal (f64 ports)**

`sqrt.rn.f64` is IEEE 754 correctly rounded (exact). `rsqrt` = sqrt.rn
followed by rcp.rn (two roundings, ≤ 1.5 ulp combined — **empirical,
verify against the actual sequence**). `reciprocal` = rcp.rn (≤ 1 ulp).
These are new f64 ports — the bounds are derived from the PTX ISA tables
but not yet GPU-verified for f64. ∎ [conditional on PTX ISA accuracy]

### 3.4 Comparisons and predicates

Same as the previous version. The ordered/unordered NaN analysis is
confirmed correct by the reviewer. The `setp.eq + not.pred` approach for
`!=` matches C semantics and is verified by the `nan_neq` GPU test. ∎

### 3.5 Type conversion

Same as the previous version. All conversions use the correct PTX
`cvt` instructions with the correct rounding modes. The f16
compute-in-f32 strategy is **within tolerance, not identical** to emmy's
native-f16 chains (which round at each intermediate step). The difference
is ≤ 1 f16 ulp per operation. This is a tolerance statement, not an
equality proof. ∎

### 3.6 Memory operations

Same as the previous version, with these corrections:

**Vector alignment:** the alignment requirement applies to **every index
term**, not just the constant term. All index expressions (not just the
base offset) must be ≡ 0 mod the vector width. This is guaranteed by
emmy's vectorizer for the corpus (verified across 61 IRs), but it is a
**precondition (P3)**, not something the compiler proves. ∎

**32-bit address overflow:** the index × nbytes product is computed in
32-bit. For buffers with > 2³¹/nbytes elements, this overflows. The runner
caps buffers at 2²⁶ elements, so the product is < 2²⁹ for f32 and < 2³⁰
for f64 — within i32 range. This is a **precondition (P3)** enforced by
the runner's limits, not by the compiler. ∎

**Shared memory:** same as before, with the corrected C-spelling table.
The layout matches emmy's `pack_smem` (verified byte-identical on mm512).
∎

### 3.7 Cross-thread operations

Same as the previous version, with these corrections:

**bar.sync count:** PTX does not require a count on `bar.sync id` (it is
optional). The count requirement comes from emmy's renderer. When present,
`count` is the number of participating threads (not "the first count
threads"), and it must be a multiple of the warp size. ∎

**shfl.sync c operand:** 0x1f encodes the clamp field (bits 4:0) = 31.
It is not a "cluster" field. The source lane is `laneid XOR laneMask`,
clamped to [0, 31]. ∎

**activemask precondition:** when the warp is partially active (e.g.,
under the tile tail guard), `activemask.b32` returns the set of active
lanes. `shfl.sync.bfly.b32` with this mask reads undefined values from
inactive lanes. The butterfly reduction is correct only if the combine
program is **converged** (no divergent control flow between rounds).
This is a **precondition (P2)**: the corpus kernels satisfy it (the
combine region is straight-line code), but it must be stated. ∎

### 3.8 Asynchronous copy

Same as the previous version. The swizzle formula matches emmy's helper.
**The swizzle code path is unexercised in the corpus** (all swizzles are
NONE). The formula is ported from emmy's `swizzle_xor` helper and is
code-verified, but it is **not proven and not tested on the GPU.**
No ∎ for this path. ∎ [removed — untested]

---

## 4. Composition

### 4.1 Model and preconditions

Per thread t, the IR state is (ρ, M_g, M_s): a name→value environment plus
global and shared memory. The PTX state is (R, M_g, M_s) with registers R.
The compiler maintains a map reg: Name → Register.

The theorem is conditional on:

**P1 (race freedom).** The IR is data-race-free outside atomics, and
barriers separate conflicting shared accesses.

**P2 (convergence).** `bar.sync` and warp-collective ops (`shfl.sync`,
`bar.warp.sync`) are reached by all participating lanes or threads.
This covers the activemask captured under the tail guard. Named-barrier
counts are multiples of 32.

**P3 (alignment and addressing).** Vector accesses have every index term
(not just the constant) ≡ 0 mod the vector width, and buffer bases are
aligned. Offset views are excluded. The address arithmetic doesn't
overflow (32-bit index × nbytes < 2³¹, enforced by the runner's 2²⁶
element cap).

**P4 (flags).** The reference uses `-prec-div=true` and `-fmad=false`,
or the FMA difference is explicitly counted in ε.

**P5 (domains).** Inputs lie in each op's D_f from §3.3, and integer
ops don't hit signed-overflow UB. §3.2 holds because the IR defines
wraparound, not because C does.

**P6 (atomics).** `red.global.add.f32` is nondeterministic in order,
as is `atomicAdd`. The claim is that the sum is within reassociation
error, not that results are identical.

### 4.2 State and simulation relation

The simulation relation ≈ relates IR and PTX states:

- For every live IR name n: R[reg(n)] equals ρ(n) **exactly** for ints,
  bools, and addresses; within ε_f for floats (per §3.3).
- Global memory M_g: equal up to float ε and store-time narrowing (f16).
- Shared memory M_s: equal, with visibility ordering after `bar.sync`.

Note that this is an ε-refinement for floats, not exact equality.

### 4.3 Rules

**Straight-line statements.** Assume σ ≈ σ'. Statement S emits T(S) from
the translation table. The post-state σ'' satisfies σ[S] ≈ σ''. For
statements that write registers, the destination is determined by the
compiler's register map.

**Accum (exception to SSA).** Accum is NOT single-assignment: it updates
the same named value in place. The compiler emits the result into the
same register (or replaces the register map entry). The proof obligation
is that the emitted destination is reg(name), not a fresh register.
Otherwise iteration i+1 reads iteration i−1's value. The structural
induction treats Accum as an in-place update that preserves the
simulation relation (the new value is within ε of the old value
combined with the input). ∎

**Loops.** The flat structural induction does NOT cover loops. Loops
require an invariant. For a loop with induction variable v, extent end,
and step s, the invariant at the loop head is:

> Inv(i): v = start + i·s, the accumulators equal the result of the
> first i iterations applied to the initial state, and the memory
> holds the first i iterations' writes.

The body maps Inv(i) → Inv(i+1) (by the per-statement correctness of
the body). The exit condition v ≥ end ensures the final state satisfies
the loop's specification. Unrolled loops are trip sequential copies with
compile-time v; the invariant is the same, with caches reset between
copies. ∎

**Cond.** The predicate is evaluated before the branch. Both arms start
from the same σ ≈ σ'. The post-state is the join of the two arms.
Registers defined only in one arm are dead after the join (they don't
dominate). TreeHalve and `bar.sync` remain outside predicated regions
so all threads arrive (P2). ∎

**Sync and memory.** Shared-memory stores are visible to other threads
only after `bar.sync`. CpAsync results are visible only after the
`wait_group` plus a barrier. Both are ordering edges in the memory
component M_s. Without modeling this, the induction is silent about
the main source of GPU bugs (race conditions). ∎

### 4.4 Cache validity

The correctness obligation is: **every cache hit returns the value that
would be computed by re-evaluating the key in the current state.**
A stale hit (the cache returns an old value when the state has changed)
is a bug. Over-invalidation (killing too aggressively) only wastes
instructions — it is safe.

**Kill rules (when a cache entry must be invalidated):**

| Cache | Killed by |
|---|---|
| expr_cache | any rebinding of a free variable in the cached expression: Accum (updates the accumulator), loop variable update, scope re-entry |
| load_cache | Write or atomic Write to the same buffer (or any buffer that may alias); `bar.sync` (shared writes from other threads become visible); CpAsyncWait (async copies complete); TreeHalve levels (shared buffer updates); any shared-memory store |
| addr_cache | change to any index variable. Buffer bases are constant, so stores don't kill it. |

**Scoping rules:**

- **Cond:** save the cache state at entry, restore at the join, don't
  merge entries from different arms. An entry created in the then-branch
  doesn't dominate the join.
- **Loops:** at entry, kill entries depending on the loop variable.
  At the back edge, drop everything created in the body.
- **Unrolled copies:** each copy starts from the post-prologue cache
  state (the axis values are different compile-time constants).

**Verification:** the compiler's `stmt()` dispatch clears `load_cache`
and `addr_cache` for every statement not in the safe list
(Load, Assign, Let, Init). Since Accum is not in the safe list,
`load_cache` IS cleared after Accum. `Write` is not in the safe list,
so `load_cache` IS cleared after Write. `Sync` is not in the safe list,
so `load_cache` IS cleared after Sync. `TreeHalve` is not in the safe
list, so `load_cache` IS cleared after TreeHalve. This covers the
kill rules above. The `expr_cache` is cleared at loop entry/exit,
Cond boundaries, and unrolled copy boundaries — covering the
rebinding rules. **However, the validity of these kill rules for
NOVEL statement combinations (not in the corpus) has not been
exhaustively verified.** This is a documented limitation. ∎

### 4.5 Error composition

§3.3's bounds are per-operation. For a chain of ops, the output error is:

|Δy| ≤ κ_f(x) · |Δx| + ε_f

where κ_f is the local condition number of op f at input x. A rigorous
end-to-end bound requires tracking κ through the dataflow graph.
Cancellation (sub of near-equal values, e.g., `1 − E` in tanh/sigmoid)
makes κ arbitrarily large.

**Example:** softmax with |x − max| up to 88.7 (the exp domain limit).
The numerator and denominator each carry ≈ 6.7e-6 (from the exp bound).
The ratio's relative error is ≈ 1.3e-5, which is above rtol 1e-5.
Typical inputs (|x − max| small after the max subtraction) stay far below.
The corpus problems all pass.

**Correct statement:** per-statement ε-refinement is proved (§3).
End-to-end tolerance is established per problem by the error-composition
bound where κ is controlled, and otherwise by the differential test
(zero divergences on 61 problems). It is NOT proved that every possible
op chain stays within tolerance — that requires per-problem analysis. ∎

### 4.6 What each claim rests on

| Claim | Basis |
|---|---|
| Integer, bool, memory, control-flow translation | proved (§4.3), under P1–P6 |
| Float ops within ε on D_f | derived (§3.3), hardware envelopes from the PTX ISA tables |
| End-to-end float tolerance | per-problem bound plus differential test vs emmy |
| Geometry, swizzle, raster remap | ported from emmy — agreement shows **consistency**, not independence |
| Statement sweep (0/88) | syntactic validity (ptxas), not semantic correctness |
| Semantic sweep (176/176) | empirical correctness on tested inputs, not a proof for all inputs |

---

## 5. Geometry

Same as the previous version, with these corrections:

**a02's raster validation:** a02 has a second raster axis with extent 1,
so it exercises the eb = 1 case (where the remap is the identity). It does
NOT validate the case where both raster axes have extent > 1. The
algebraic proof covers the divisible case, but the ragged case
(ea % g ≠ 0) is not GPU-validated. ∎

**"Exhaustive" scope:** the 236,480 exhaustive decode verification covers
the 15 open problems' shapes. It does NOT cover all possible shapes. The
hidden set may have different shapes. The split-decode formula is
additionally verified for 80,000 random gids at 2³¹–2³² cells, but this
is still sampling, not exhaustive coverage of all shapes. ∎

**Wide-tile arithmetic:** the split-decode formula
`gid/s = cta·(block/s) + (cta·(block%s) + tid)/s` requires that the
intermediate `cta·(block%s) + tid` doesn't overflow s32. Since
`cta·(block%s) + tid < block ≤ 2³¹` (because block is at most 1024), this
is satisfied. However, the computation uses **signed** s32 arithmetic.
If `cta` exceeds 2³¹/block, the `div.s32` and `mul.lo.s32` would overflow.
The guard (cta > full_ctas → bra DONE) prevents this for valid gids,
but the arithmetic itself is performed before the guard. For n_cells
close to 2³², `full_ctas ≈ 2³²/block` can exceed 2³¹/block when
block < 2 — but block ≥ 32 in practice, so `full_ctas ≤ 2³²/32 = 2²⁷`,
which is within s32 range. The `mul.lo.s32 cta, block` can still overflow
if cta·block > 2³¹, but this equals gid which is < n_cells ≤ 2³² — the
product wraps. This is the bug that the wide-gid path is designed to
avoid. The split-decode avoids it by not computing cta·block directly.
**The split formula is correct for n_cells < 2³² with block ≥ 32.** ∎

**Guard condition:** the compiler emits the guard iff `n_cells < total_threads`
or `aux_threads ≠ 0`. Emmy emits the guard iff `n % cells ≠ 0` or
`aux_threads ≠ 0`. These are equivalent when `total_threads = grid × block`
and `grid = ceil(n / cells)`: `n_cells < total_threads` ⟺
`n_cells < ceil(n/cells) × cells` ⟺ `n_cells mod cells ≠ 0` (since
n_cells = n for the tile). The equivalence holds because n_cells = n
(the tile covers exactly n cells). Soundness only requires the guard
to be present when the grid can overshoot, which is the case. ∎

---

## 6. Statement-level sweep certificate

The sweep covers the emit_op dispatcher × dtype matrix, plus statement-level
emitters (Load, Write, Accum, Init, Smem, Sync, WarpShuffle, TreeHalve,
CpAsync) × dtype variants. Each combination is compiled and the emitted PTX
is checked with `ptxas -arch=sm_89`.

| Emitter | Combos emitted | Combos raised (designed) | ptxas invalid |
|---|---|---|---|
| Dispatcher (emit_op) | 408 | 141 | 0 |
| Statement (Load/Write/Accum/etc.) | 88 | 0 | 0 |
| **Total** | **496** | **141** | **0** |

**Note on counts:** the 549 total combinations come from 61 op-dtype
pairs in the dispatcher (some ops have arity > 1, creating additional
combinations) plus 88 statement-level problems. The 141 raised combos
are designed `NotImplementedError` paths (transcendentals on non-f32,
float-bitwise guards, sqrt-family on int dtypes, etc.).

**ptxas acceptance** proves **syntactic and type validity**, not semantic
correctness. Semantic correctness is established by the GPU execution
tests (semantic sweep, regression, differential). ∎

---

## 7. Independence and residual risk

The correctness argument rests on three pillars:

1. **Per-statement refinement proofs** (§3): mathematical arguments that
   each IR→PTX translation computes the same result (or within ε) as
   the reference implementation.

2. **Composition rules** (§4): the argument that correct statements
   compose correctly, under preconditions P1–P6.

3. **Empirical validation**: official 15/15, 85-problem regression,
   61-problem differential, 176-cell semantic sweep, 236,480-gid decode
   verification.

The **differential vs emmy** is not fully independent: the geometry,
swizzle, and raster formulas were **ported from emmy's source code**.
Agreement shows consistency (both implement the same formulas), not
independence (a shared misreading of the spec wouldn't be caught).
The per-statement proofs in §3 are the independent check for these.

The **preconditions P1–P6** are properties of the input program, not of
the compiler. They are empirically satisfied by the corpus (verified)
and are expected to hold for the hidden set (same generator). If a
hidden problem violates a precondition, the proof doesn't apply and
the result is undefined. This is the largest remaining risk.

The **accumulated error** (§4.5) is bounded per-operation but not
proven for all possible op chains. The differential provides empirical
confidence on 61 problems. A pathological op chain with large
condition numbers could exceed tolerance.

The **fuzzer** (not completed) would sample the generator's output
space more broadly, addressing the novel-combination risk. It requires
the packaging logic from `tools/generate.py` to be ported into the
fuzzer harness (~1-2h). This is the last instrument that could raise
confidence further.
