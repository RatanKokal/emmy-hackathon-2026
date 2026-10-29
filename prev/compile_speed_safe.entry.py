#!/usr/bin/env python3
"""emmy-safe compiler (trimmed speed variant: unroll, FMA, merge/vec).
Reads one Kernel IR json, writes PTX to stdout. Implementation: src/compiler/.
"""
import json
import sys
from pathlib import Path

from src.compiler import Compiler
from src.compiler.tables import INT_TYPES  # noqa: F401  (used via Compiler)


def find_kernel(graph):
    """Return the single KernelOp node name of the graph."""
    kernels = [
        name for name, node in graph.nodes.items()
        if type(node.op).__name__ == "KernelOp"
    ]
    if len(kernels) != 1:
        raise RuntimeError(f"expected exactly one KernelOp, found {kernels}")
    return kernels[0]


def main(argv=None):
    """Compile argv[0] IR to PTX; exit 0 ok, 1 unsupported, 2 usage."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python compile_speed_safe.py <kernel.ir.json>", file=sys.stderr)
        return 2
    path = Path(argv[0])
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 2
    graph = json.loads(path.read_text())
    meta_path = path.parent / "meta.json"
    if not meta_path.exists():
        print(f"missing {meta_path}", file=sys.stderr)
        return 2
    try:
        import copy
        from emmy.compiler.graph import Graph
        g = Graph.from_dict(json.loads(path.read_text()))
        meta = json.loads(meta_path.read_text())
        compiler = Compiler(g, meta, find_kernel(g))
        compiler.compile()
        sys.stdout.write(compiler.render())
    except NotImplementedError as e:
        print(f"unsupported construct: {e}", file=sys.stderr)
        import os
        if os.environ.get("COMPILER_TRACEBACK") == "1":
            raise
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
