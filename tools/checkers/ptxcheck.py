#!/usr/bin/env python3
"""Oracle 4 v1: PTX register definedness/dominance checker."""
import re
import sys
from pathlib import Path

SPECIAL = {
    "tid", "ctaid", "ntid", "nctaid", "laneid", "warpid", "nwarpid",
    "smid", "gridid", "clock", "clock64", "lanemask_eq", "lanemask_le",
    "lanemask_lt", "lanemask_ge", "lanemask_gt", "envreg", "totaltimer",
}
REG = re.compile(r"%[\w.]+")
PRED_PREFIX = re.compile(r"^@!?(%[\w.]+)\s+")
NO_DEF = ("st", "red", "bar", "bra", "ret")


def base(tok):
    m = re.match(r"%([a-zA-Z]+)", tok)
    return m.group(1) if m else ""


def strip_pred(line):
    p = None
    m = PRED_PREFIX.match(line)
    if m:
        p = m.group(1)
        line = line[m.end():]
    return p, line


def parse_instr(line):
    """Return (pred, opcode, dsts, uses) or None for non-instructions."""
    line = line.strip().rstrip(";").strip()
    p, line = strip_pred(line)
    toks = line.replace(",", " ").replace("{", " ").replace("}", " ").split()
    if not toks:
        return p, None, [], []
    op0 = toks[0].split(".")[0]
    # dst candidates: first %reg after opcode unless no-def op
    rest = toks[1:]
    if op0 in NO_DEF:
        dsts, uses = [], REG.findall(line)
        if p:
            uses = [u for u in uses if u != p]
        return p, op0, dsts, uses
    dsts, uses = [], []
    seen_dst = False
    brace = "{" in line.split(";")[0] and "{" in line
    if brace:
        # leading braced vector dst: {%f1, %f2}, [...]
        before_brack = line.split("[")[0]
        dsts = REG.findall(before_brack.split("}")[0]) if "}" in before_brack else []
    else:
        for t in rest:
            if t.startswith("%"):
                dsts = [t]
                break
    dstset = set(dsts)
    for u in REG.findall(line):
        if p and u == p:
            continue
        if u in dstset:
            continue
        uses.append(u)
    return p, op0, dsts, uses


def parse_kernel(text):
    """Split text into blocks {label: (idx0, instrs, succ_labels)} preserving order."""
    lines = []
    for raw in text.splitlines():
        line = raw.split("//")[0].strip()
        if not line or line.startswith("."):
            continue
        lines.append(line)
    blocks = []          # list of dicts
    cur = {"label": "ENTRY", "instrs": [], "term": None}
    for ln in lines:
        if ln.endswith(":") and ";" not in ln and " " not in ln:
            if cur["instrs"] or cur["term"]:
                blocks.append(cur)
            cur = {"label": ln[:-1], "instrs": [], "term": None}
            continue
        p, op0, dsts, uses = parse_instr(ln)
        cur["instrs"].append((ln, p, op0, dsts, uses))
        if op0 in ("bra", "ret"):
            cur["term"] = ("ret" if op0 == "ret" else "bra", ln)
            blocks.append(cur)
            cur = {"label": f"FALL_{len(blocks)}", "instrs": [], "term": None}
    if cur["instrs"] or cur["term"]:
        blocks.append(cur)
    # successors
    by_label = {b["label"]: i for i, b in enumerate(blocks)}
    for i, b in enumerate(blocks):
        succ = []
        if b["term"] is None:
            if i + 1 < len(blocks):
                succ.append(blocks[i + 1]["label"])
        elif b["term"][0] == "ret":
            pass
        else:
            ln = b["term"][1]
            tgt = ln.split(";")[0].split()[-1]
            cond = ln.lstrip().startswith("@")
            if cond and i + 1 < len(blocks):
                succ.append(blocks[i + 1]["label"])   # fallthrough
            if tgt in by_label:
                succ.append(tgt)
        b["succ"] = succ
    return blocks


def check(text, name="", stub_mode=False):
    blocks = parse_kernel(text)
    allregs = set()
    for b in blocks:
        for (ln, p, op0, dsts, uses) in b["instrs"]:
            allregs.update(REG.findall(ln))
    for r in list(allregs):
        if base(r) in SPECIAL:
            allregs.discard(r)
    U = allregs

    DEF = {}
    for b in blocks:
        d = set()
        for (ln, p, op0, dsts, uses) in b["instrs"]:
            d.update(x for x in dsts if base(x) not in SPECIAL)
        DEF[b["label"]] = d
    defined_anywhere = set().union(*DEF.values()) if DEF else set()

    preds = {b["label"]: [] for b in blocks}
    for b in blocks:
        for s in b["succ"]:
            preds[s].append(b["label"])

    labels = [b["label"] for b in blocks]
    OUT = {l: set(U) for l in labels}
    IN = {l: set(U) for l in labels}
    entry = labels[0]
    IN[entry] = set()
    OUT[entry] = set(DEF[entry])
    changed = True
    it = 0
    while changed and it < 100:
        changed = False
        it += 1
        for b in blocks:
            l = b["label"]
            if l == entry:
                continue
            if not preds[l]:
                new = set(U)           # unreachable: no constraint
            else:
                new = set.intersection(*(OUT[p] for p in preds[l]))
            if new != IN[l]:
                IN[l] = new
                changed = True
            o = IN[l] | DEF[l]
            if o != OUT[l]:
                OUT[l] = o
                changed = True

    violations = []
    ambient = set()
    for b in blocks:
        live = set(IN[b["label"]])
        for (ln, p, op0, dsts, uses) in b["instrs"]:
            for u in uses:
                if base(u) in SPECIAL or u in live:
                    continue
                if stub_mode and u not in defined_anywhere:
                    ambient.add(u)     # never defined: template input, not a bug
                    continue
                violations.append((b["label"], ln.strip(), u))
            for d in dsts:
                if base(d) not in SPECIAL:
                    live.add(d)
    return violations, ambient


def main():
    if len(sys.argv) < 2:
        print("usage: ptxcheck.py [--stub] <ptx...>", file=sys.stderr)
        return 2
    args = sys.argv[1:]
    stub = "--stub" in args
    args = [a for a in args if a != "--stub"]
    total = 0
    for arg in args:
        for p in Path(arg).rglob("*.ptx") if Path(arg).is_dir() else [Path(arg)]:
            v, amb = check(p.read_text(), p.name, stub_mode=stub)
            total += len(v)
            if v:
                print(f"{p}: {len(v)} violation(s)")
                for blk, ln, reg in v[:5]:
                    print(f"  [{blk}] use of {reg}: {ln}")
                if len(v) > 5:
                    print(f"  ... +{len(v) - 5} more")
            if amb:
                print(f"{p}: {len(amb)} ambient input(s) (stub mode): {sorted(amb)[:4]}")
    print(f"TOTAL {total} violations across {len(args)} path(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
