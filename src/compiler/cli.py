"""Application layer: IR file in, PTX out. Used by compile.py and `python -m src.compiler`."""
import json
import sys
from pathlib import Path

from .core import Compiler


def find_kernel(graph):
    """Return the single KernelOp node name of the graph."""
    kernels = [
        name for name, node in graph.nodes.items()
        if type(node.op).__name__ == "KernelOp"
    ]
    if len(kernels) != 1:
        raise RuntimeError(f"expected exactly one KernelOp, found {kernels}")
    return kernels[0]


def compile_ir(ir_path):
    """Compile one IR file to PTX text; raises NotImplementedError on unsupported IR."""
    from emmy.compiler.graph import Graph
    path = Path(ir_path)
    graph = Graph.from_dict(json.loads(path.read_text()))
    meta_path = path.parent / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text())
    compiler = Compiler(graph, meta, find_kernel(graph))
    compiler.compile()
    return compiler.render()


def main(argv=None):
    """Compile argv[0] IR to PTX; exit 0 ok, 1 unsupported, 2 usage."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python compile.py <kernel.ir.json>", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(compile_ir(argv[0]))
    except NotImplementedError as e:
        print(f"unsupported construct: {e}", file=sys.stderr)
        import os
        if os.environ.get("COMPILER_TRACEBACK") == "1":
            raise
        return 1
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0
