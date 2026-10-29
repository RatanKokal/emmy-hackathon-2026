# Extracted verbatim from compile_speed_safe.py (validated checkpoint f289d05).
import json
import struct
import sys
from math import prod
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var

from .tables import (INT_TYPES, REG_CLASS, PTX_TY, WIDEN_AT_LOAD, SMEM_NBYTES,
                     F32_ONLY, UNROLL_CAP, PURE_OPS, COMMUTATIVE, _MEM_KINDS,
                     CF_KINDS)


class Emitter:
    def __init__(self):
        self.lines = []
        self.counts = {"p": 0, "f": 0, "fd": 0, "r": 0, "rd": 0, "h": 0}
        self.label_n = 0

    def reg(self, cls):
        self.counts[cls] += 1
        return f"%{cls}{self.counts[cls]}"

    def emit(self, text):
        self.lines.append("\t" + text)

    def mark(self, name):
        self.lines.append(f"{name}:")

    def label(self, stem):
        self.label_n += 1
        return f"{stem}_{self.label_n}"


class _Pair:
    """Emission-time marker: two scalar Loads merged into one vector load."""
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a = a
        self.b = b
