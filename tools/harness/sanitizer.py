#!/usr/bin/env python3
"""sanrun: compute-sanitizer (memcheck/racecheck/synccheck/initcheck) per problem."""
import os
import re
import subprocess
import sys
from pathlib import Path

TOOLS = sys.argv[2:] or ["memcheck", "synccheck", "racecheck"]
ERR = re.compile(r"ERROR SUMMARY: (\d+) errors")


def run_one(d, tool, tmp_root):
    for old in tmp_root.iterdir():
        if old.is_dir():
            subprocess.run(["rm", "-rf", str(old)])
    subprocess.run(["cp", "-r", str(d), str(tmp_root / d.name)])
    r = subprocess.run(
        [os.environ.get("EMMY_SANITIZER", "compute-sanitizer"), f"--tool={tool}", "--error-exitcode=9", "--print-limit=8",
         "python3", "/tmp/emmygpu/gpurun2.py", str(tmp_root)],
        capture_output=True, text=True, timeout=300)
    m = ERR.findall(r.stdout + r.stderr)
    n = sum(int(x) for x in m)
    # pull one representative error line if any
    line = ""
    for ln in (r.stdout + r.stderr).splitlines():
        if "Invalid" in ln or "Race" in ln or "divergent" in ln or "barrier" in ln.lower():
            line = ln.strip()[:110]
            break
    return n, line, r.returncode


def main():
    root = Path(sys.argv[1])
    tmp_root = Path("/tmp/emmygpu/san_one")
    tmp_root.mkdir(exist_ok=True)
    bad = 0
    print(f"{'problem':22s}" + "".join(f"{t:>12s}" for t in TOOLS))
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (d / "k.ptx").exists():
            continue
        counts = []
        line = ""
        for t in TOOLS:
            n, line, rc = run_one(d, t, tmp_root)
            counts.append(n)
        bad += 1 if any(counts) else 0
        cells = "".join(f"{c:>12d}" for c in counts)
        extra = f"  | {line}" if any(counts) and line else ""
        print(f"{d.name:22s}{cells}{extra}")
    print(f"\n{bad} problem(s) flagged")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
