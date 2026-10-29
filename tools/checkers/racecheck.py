#!/usr/bin/env python3
"""Static racecheck shadow: symbolic shared-memory race analysis."""
import re
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/opencode")
from ptxcheck import parse_kernel  # noqa: E402

REG = re.compile(r"%[\w.]+")
IMM = re.compile(r"^-?\d+$")


class Sym:
    """a*tid + c, or None if not provably tid-linear."""
    __slots__ = ("a", "c", "buf")

    def __init__(self, a=0, c=0, buf=""):
        self.a, self.c, self.buf = a, c, buf

    def __repr__(self):
        return f"{self.buf}+{self.a}*tid+{self.c}" if self.a else f"{self.buf}+{self.c}"


def instr_ops(line):
    line = line.strip().rstrip(";").strip()
    if line.startswith("@"):
        line = re.sub(r"^@!?(%[\w.]+)\s+", "", line)
    return line.replace(",", " ").replace("{", " ").replace("}", " ").split()


def make_analyzer(blocks):
    # symbolic env: reg -> Sym; guards: block -> (a, cmp, K) on tid
    env = {}
    guards = {}
    labels = [b["label"] for b in blocks]
    by_label = {b["label"]: i for i, b in enumerate(blocks)}
    special_t = {"tid": (1, 0), "ctaid": (0, 0)}  # ctaid uniform => a=0

    def sym_of(tok):
        m = re.match(r"%(\w+)\.(\w+)", tok)
        if m and m.group(1) in special_t:
            a, c = special_t[m.group(1)]
            return Sym(a, c)
        return env.get(tok)

    # pass 1: linear propagation (single pass; compiler output is SSA-ish)
    for b in blocks:
        for (ln, p, op0, dsts, uses) in b["instrs"]:
            ops = instr_ops(ln)
            if not ops:
                continue
            op = ops[0].split(".")[0]
            try:
                if op == "mov" and len(ops) == 3:
                    s = sym_of(ops[2])
                    if s is not None:
                        env[dsts[0]] = Sym(s.a, s.c, s.buf)
                    elif not ops[2].startswith("%") and not IMM.match(ops[2]):
                        env[dsts[0]] = Sym(0, 0, buf=ops[2])      # bare smem symbol base
                    elif IMM.match(ops[2]) and (".u64" in ops[0] or ".s64" in ops[0]):
                        env[dsts[0]] = Sym(0, 0, buf=f"S{ops[2]}")
                    elif IMM.match(ops[2]):
                        env[dsts[0]] = Sym(0, int(ops[2]))
                elif op == "cvta" and len(ops) == 3:
                    s = sym_of(ops[2])
                    env[dsts[0]] = Sym(s.a, s.c, s.buf) if s is not None else None
                elif op in ("add", "sub") and len(ops) >= 4:
                    x, y = sym_of(ops[2]), sym_of(ops[3])
                    if x is not None and y is not None:
                        k = 1 if op == "add" else -1
                        if y.a and x.a:
                            env[dsts[0]] = None
                        elif y.a:
                            env[dsts[0]] = Sym(k * y.a, x.c + k * y.c, x.buf or y.buf)
                        else:
                            env[dsts[0]] = Sym(x.a, x.c + k * y.c, x.buf or y.buf)
                    elif x is not None and IMM.match(ops[3]):
                        env[dsts[0]] = Sym(x.a, x.c + (int(ops[3]) if op == "add" else -int(ops[3])), x.buf)
                    elif y is not None and IMM.match(ops[2]):
                        env[dsts[0]] = Sym(y.a, int(ops[2]) + k * y.c, y.buf)
                    else:
                        env[dsts[0]] = None
                elif op in ("mul", "mul.lo", "mul.wide", "shl") and len(ops) >= 4:
                    x, y = sym_of(ops[2]), sym_of(ops[3])
                    if x is not None and y is not None and (x.a == 0 or y.a == 0):
                        v, k = (x, y) if y.a == 0 else (y, x)
                        env[dsts[0]] = Sym(k.a * int(v.c), 0, k.buf)
                    elif x is not None and IMM.match(ops[3]):
                        env[dsts[0]] = Sym(x.a * int(ops[3]), 0, x.buf) if x.a else Sym(0, x.c * int(ops[3]), x.buf)
                    elif y is not None and IMM.match(ops[2]):
                        env[dsts[0]] = Sym(y.a * int(ops[2]), 0, y.buf) if y.a else Sym(0, y.c * int(ops[2]), y.buf)
                    else:
                        env[dsts[0]] = None
                elif op in ("mad", "mad.lo") and len(ops) >= 5:
                    # dst = x*y + z
                    x, y, z = sym_of(ops[2]), sym_of(ops[3]), sym_of(ops[4])
                    zc = int(ops[4]) if z is None and IMM.match(ops[4]) else (z.c if z is not None and z.a == 0 else None)
                    zbuf = z.buf if z is not None else ""
                    if zc is None:
                        env[dsts[0]] = None
                    elif x is not None and IMM.match(ops[3]):
                        prod_a = x.a * int(ops[3])
                        env[dsts[0]] = Sym(prod_a, zc, zbuf) if prod_a else Sym(0, x.c * int(ops[3]) + zc, zbuf)
                    elif y is not None and IMM.match(ops[2]):
                        prod_a = y.a * int(ops[2])
                        env[dsts[0]] = Sym(prod_a, zc, zbuf) if prod_a else Sym(0, y.c * int(ops[2]) + zc, zbuf)
                    else:
                        env[dsts[0]] = None
                elif op in ("ld", "cvt", "setp", "and", "or", "xor", "shl", "shr",
                            "max", "min", "selp", "fma", "rcp", "sqrt", "neg",
                            "abs", "not", "activemask", "shfl", "rem", "div"):
                    env[dsts[0]] = None
            except Exception:
                env[dsts[0]] = None

    # pass 2: guard constraints from predicated branches
    for b in blocks:
        if b["term"] and b["term"][0] == "bra" and b["term"][1].lstrip().startswith("@"):
            ln = b["term"][1]
            neg = ln.lstrip().startswith("@!")
            pred = REG.findall(ln.split("bra")[0])[0]
            # find setp defining pred
            for b2 in blocks:
                for (l2, p2, o2, d2, u2) in b2["instrs"]:
                    if d2 and d2[0] == pred and o2 == "setp":
                        ops = instr_ops(l2)
                        x = sym_of(ops[2]) if len(ops) > 2 else None
                        k = int(ops[3]) if len(ops) > 3 and IMM.match(ops[3]) else None
                        if x and k is not None and x.a == 1 and not x.buf:
                            cmp = ops[1].split(".")[-1] if "." in ops[1] else ops[1]
                            # pred = (x < k) for lt; branch taken when pred [or !pred]
                            side = not neg          # target side sees pred as-is
                            for tgt in b["succ"]:
                                if tgt == b["term"][1].split(";")[0].split()[-1]:
                                    is_target = True
                                else:
                                    is_target = False
                                take = side if is_target else not side
                                if cmp == "lt":
                                    guards[tgt] = ("lt", k) if take else ("ge", k)
                                elif cmp == "ge":
                                    guards[tgt] = ("ge", k) if take else ("lt", k)
    return env, guards


def smem_accesses(blocks, env, guards):
    """Yield (region_idx, kind, sym, imm, block, line) for ld/st.shared."""
    out = []
    cur = 0
    for b in blocks:
        g = guards.get(b["label"])
        for (ln, p, op0, dsts, uses) in b["instrs"]:
            if op0 in ("bar", "barrier"):
                cur += 1
                continue
            if op0 in ("ld", "st") and ".shared" in ln:
                kind = "R" if op0 == "ld" else "W"
                m = re.search(r"\[(%[\w.]+)([+-]\d+)?\]", ln)
                if not m:
                    continue
                base, imm = m.group(1), int(m.group(2) or 0)
                s = env.get(base)
                if s is None:
                    out.append((cur, kind, None, imm, b["label"], ln.strip(), g))
                else:
                    out.append((cur, kind, Sym(s.a, s.c, s.buf), imm, b["label"], ln.strip(), g))
    return out


def conflicts(accs):
    """Pairwise same-region conflicts."""
    races = []
    candidates = []
    safe = 0
    for i in range(len(accs)):
        for j in range(i + 1, len(accs)):
            r1, k1, s1, i1, b1, l1, g1 = accs[i]
            r2, k2, s2, i2, b2, l2, g2 = accs[j]
            if r1 != r2:
                continue                      # different barrier regions => ordered
            if "W" not in k1 + k2:
                continue                      # read-read never conflicts
            if s1 is None or s2 is None:
                candidates.append(("UNKNOWN-PROV", b1, l1, b2, l2))
                continue
            if s1.buf != s2.buf:
                continue                      # different buffers
            # solve a1*t + c1 + i1 == a2*t' + c2 + i2 with guards
            A, C = s1.a, s1.c + i1
            B, D = s2.a, s2.c + i2
            sols = []
            if A == B:
                if C == D:
                    sols = "same-thread-only"   # t == t'
                else:
                    dt = (C - D) // A if A else 0
                    sols = f"t'-t={dt}" if A else "unknown"
            else:
                sols = "cross-stride"           # conservative
            if sols == "same-thread-only":
                safe += 1
                continue
            if sols == "unknown" or sols == "cross-stride":
                candidates.append((sols, b1, l1, b2, l2))
                continue
            # integer shift: check guard compatibility (t' = t + dt)
            dt = int(sols.split("=")[1])
            if guards_compatible(g1, g2, dt):
                races.append((f"t'-t={dt}", b1, l1, b2, l2))
            else:
                safe += 1
    return races, candidates, safe


def guards_compatible(g1, g2, dt):
    """Can t and t'=t+dt both satisfy their guards (t in [0, 1024))?"""
    def ok(g, t):
        if g is None:
            return True
        cmp, K = g
        return t < K if cmp == "lt" else t >= K
    for t in range(0, 1100):
        if ok(g1, t) and ok(g2, t + dt):
            return True
    return False


def check(text):
    blocks = parse_kernel(text)
    env, guards = make_analyzer(blocks)
    accs = smem_accesses(blocks, env, guards)
    r, c, s = conflicts(accs)
    return (r, c, s), len(accs)


def main():
    args = sys.argv[1:]
    total = 0
    for arg in args:
        for p in Path(arg).rglob("*.ptx") if Path(arg).is_dir() else [Path(arg)]:
            try:
                (races, cands, safe), naccs = check(p.read_text())
            except Exception as e:
                print(f"{p}: SKIP {type(e).__name__} {e}")
                continue
            total += len(races)
            if races or cands:
                print(f"{p}: {len(races)} RACE(S), {len(cands)} candidate(s) ({naccs} accesses, {safe} proven-safe)")
                for f in races[:3]:
                    print(f"  RACE {f}")
                for f in cands[:2]:
                    print(f"  cand {f}")
    print(f"TOTAL {total} proven race(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
