#!/usr/bin/env python3
"""Sync the repo to the GPU box and prepare the validation environment.

Requires EMMY_BOX_HOST (user@host). Box workspace: EMMY_BOX_DIR (default ~/emmy).
Extracts compute-sanitizer 13.4 if not present. No IPs or paths in code.
"""
import os
import subprocess
import sys

HOST = os.environ.get("EMMY_BOX_HOST")
BOX_DIR = os.environ.get("EMMY_BOX_DIR", "emmy")
SANITIZER_URL = ("https://developer.download.nvidia.com/compute/cuda/repos/"
                 "ubuntu2404/x86_64/cuda-sanitizer-13-4_13.4.92-1_amd64.deb")

def run(cmd, check=True):
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        sys.exit(f"command failed: {cmd}")

def main():
    if not HOST:
        print("set EMMY_BOX_HOST=user@host", file=sys.stderr)
        return 2
    box = f"{HOST}:{BOX_DIR}"
    run(f"ssh {HOST} 'mkdir -p {BOX_DIR}'")
    # sync only what validation needs, same structure as the repo
    run(f"rsync -a --delete "
        f"--exclude .git --exclude __pycache__ --exclude .testcorpus "
        f"--exclude learn --exclude slides --exclude examples --exclude .tools "
        f"--exclude tests/.gpu --exclude sweep_out --exclude stmt_sweep --exclude tests/.out "
        f"./ {box}/")
    run(f"ssh {HOST} 'cd {BOX_DIR} && "
        f"mkdir -p tests/.out /tmp/emmy-sanitizer && "
        f"python3 -m compileall -q tools src compile.py && "
        f"echo repo-syntax-ok'")
    # sanitizer: extract into the box workspace if missing (wrapper + libexec as a unit)
    run(f"ssh {HOST} 'cd {BOX_DIR} && "
        f"if [ ! -x .sanitizer/root/usr/local/cuda-13.4/bin/compute-sanitizer ]; then "
        f"mkdir -p .sanitizer && cd .sanitizer && "
        f"curl -sLO {SANITIZER_URL} && "
        f"dpkg -x cuda-sanitizer-13-4_13.4.92-1_amd64.deb root; "
        f"fi; .sanitizer/root/usr/local/cuda-13.4/bin/compute-sanitizer --version | head -1'")

if __name__ == "__main__":
    sys.exit(main())
