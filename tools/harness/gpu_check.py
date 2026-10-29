#!/usr/bin/env python3
import array, json, sys
from pathlib import Path
root = Path(sys.argv[1])
bad = 0
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    if not (d / "got.bin").exists():
        continue
    meta = json.load(open(d / "meta.json"))
    ref = array.array("f"); ref.fromfile(open(d / "ref.bin", "rb"), (d / "ref.bin").stat().st_size // 4)
    got = array.array("f"); got.fromfile(open(d / "got.bin", "rb"), (d / "got.bin").stat().st_size // 4)
    rtol, atol = meta["rtol"], meta["atol"]
    mabs = mrel = 0.0; nbad = 0
    for a, b in zip(got, ref):
        e = abs(a - b)
        if e > mabs: mabs = e
        if e / (abs(b) + 1e-12) > mrel: mrel = e / (abs(b) + 1e-12)
        if e > atol + rtol * abs(b): nbad += 1
    ok = nbad == 0
    bad += 0 if ok else 1
    print(f"{d.name:14s} {'PASS' if ok else 'FAIL'} abs {mabs:.3e} rel {mrel:.3e}" + ("" if ok else f" bad {nbad}/{len(ref)}"))
print(f"{bad} failing")
