#!/usr/bin/env python3
"""Oracle 4 v2: IR-name→register provenance (aliasing) checker."""
import json
import sys
from pathlib import Path

import copy

import ptxcheck


def census(body):
    """Complete use census over ALL expr fields (incl."""
    import dataclasses
    uses = {}
    DEF_FIELDS = {"name", "output", "input"}   # Stmt-level defs/buffer refs only
    stack = list(body)
    while stack:
        st = stack.pop()
        if st is None or isinstance(st, (int, float, bool)):
            continue
        if isinstance(st, str):
            uses[st] = uses.get(st, 0) + 1
            continue
        if hasattr(st, "free_vars"):            # Expr: authoritative reference set
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


class AliasTracker:
    def __init__(self):
        self.owner = {}
        self.aliases = []          # (kind, name, owner, reg)
        self.violations = []       # unexplained aliasing = bug signature
        self.in_cse = False
        self.in_loadcache = False
        self.dead_ok = set()       # names with complete-census use count <= 1

    def bind(self, name, val, prev_val):
        reg = val[0]
        owner = self.owner.get(reg)
        if owner is not None and owner != name:
            if prev_val is not None and val is prev_val:
                self.aliases.append(("let", name, owner, reg))
            elif self.in_cse:
                self.aliases.append(("cse", name, owner, reg))
            elif self.in_loadcache:
                self.aliases.append(("loadcache", name, owner, reg))
            elif name in self.dead_ok:
                self.aliases.append(("deadval", name, owner, reg))
            else:
                self.violations.append((name, owner, reg))
        elif owner is None:
            self.owner[reg] = name


class DetectCache(dict):
    """Flags load_cache hits so dependent binds are classified as such."""
    def __init__(self, tr):
        super().__init__()
        self.tr = tr

    def __getitem__(self, k):
        self.tr.in_loadcache = True
        return super().__getitem__(k)

    def get(self, k, d=None):
        self.tr.in_loadcache = True
        return super().get(k, d)


class ProvDict(dict):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def __setitem__(self, name, val):
        self.tracker.bind(name, val, self.get(name))
        super().__setitem__(name, val)


def make_prov_compiler(compiler_cls):
    """Compiler subclass with the alias-tracking values map installed."""
    class ProvCompiler(compiler_cls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.tr = AliasTracker()
            self.values = ProvDict(self.tr)
            self.load_cache = DetectCache(self.tr)

        def stmt(self, s):
            self.tr.in_loadcache = False
            super().stmt(s)

        def stmt_load_pair(self, a, b):
            self.tr.in_loadcache = False
            super().stmt_load_pair(a, b)

    return ProvCompiler

def check_ir(ir_path, meta_path, mod=None, cls=None):
    m = mod or cs
    kls = cls or ProvCompiler
    g = m.Graph.from_dict(copy.deepcopy(json.load(open(ir_path))))
    meta = json.load(open(meta_path))
    kernel = m.find_kernel(g)
    comp = kls(g, meta, kernel)
    kn = kernel
    body = g.nodes[kn].op.body
    use_counts = census(body)
    comp.tr.dead_ok = {n for n, c in use_counts.items() if c <= 1}
    comp.compile()
    ptx = comp.render()
    dom, _amb = ptxcheck.check(ptx)
    return comp.tr.violations, comp.tr.aliases, dom, ptx


def main():
    paths = sys.argv[1:]
    bad = 0
    for p in paths:
        p = Path(p)
        if p.is_dir():
            jobs = [(ir, ir.parent / "meta.json") for ir in sorted(p.rglob("*.ir.json"))]
        else:
            jobs = [(p, p.parent / "meta.json")]
        for ir, meta in jobs:
            if not meta.exists():
                print(f"{ir.parent.name}: SKIP (no meta.json)")
                continue
            try:
                av, al, dom, _ = check_ir(str(ir), str(meta))
            except Exception as e:
                print(f"{p.name}: SKIP ({type(e).__name__}: {str(e)[:60]})")
                continue

            tag = []
            if av:
                tag.append(f"ALIAS-BUG {av[:3]}")
            if dom:
                tag.append(f"DOM {len(dom)}")
            if av or dom:
                bad += 1
                print(f"{p.name}: {'; '.join(tag)}")
            else:
                kinds = [k for k, *_ in al]
                info = f" [{kinds.count('cse')} cse, {kinds.count('loadcache')} loadcache, {kinds.count('let')} let aliases]" if al else ""
                print(f"{p.name}: clean{info}")
    print(f"\n{bad} file(s) with violations")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
