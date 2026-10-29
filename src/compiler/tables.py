# Extracted verbatim from compile_speed_safe.py (validated checkpoint f289d05).
# Shared constant tables.

INT_TYPES = {"i32", "i64", "bool", "u8", "u16", "i16", "u32", "u64"}
REG_CLASS = {"f32": "f", "f64": "fd", "i32": "r", "i64": "rd", "bool": "r",
             "u8": "r", "u16": "r", "i16": "r", "u32": "r", "u64": "rd"}
PTX_TY = {"f32": "f32", "f64": "f64", "i32": "s32", "i64": "s64", "bool": "u32",
          "u8": "u8", "u16": "u16", "i16": "s16", "u32": "u32", "u64": "u64"}
WIDEN_AT_LOAD = {"f16", "bf16"}  # narrow types compute at f32
SMEM_NBYTES = {
    "float": 4, "float32": 4, "double": 8, "float64": 8,
    "half": 2, "float16": 2, "bfloat16": 2,
    "int": 4, "int32": 4, "unsigned int": 4, "uint32_t": 4,
    "long long": 8, "int64_t": 8, "unsigned long long": 8, "uint64_t": 8,
    "char": 1, "uint8_t": 1, "signed char": 1,
    "i64": 8, "bool": 1, "u8": 1, "u16": 2, "i16": 2, "u32": 4, "u64": 8, "bf16": 2,
    "short": 2, "unsigned short": 2, "long": 8, "unsigned long": 8,
    "int8_t": 1, "int16_t": 2, "uint16_t": 2, "int32_t": 4, "uint32_t": 4,
    "__half": 2, "__half2": 4, "__nv_bfloat16": 2,
    "__nv_fp8_e4m3": 1, "__nv_fp8_e5m2": 1,
}



F32_ONLY = {
        "sin", "cos",
    }


UNROLL_CAP = 128  # PTX size guard: full-unroll only small static trips


PURE_OPS = {"add.s32", "mul.lo.s32", "sub.s32", "and.b32", "or.b32",
                "xor.b32", "shl.b32", "shr.s32", "shr.u32", "min.s32", "max.s32",
                "mul.wide.s32", "add.s64"}


COMMUTATIVE = {"add.s32", "mul.lo.s32", "and.b32", "or.b32", "xor.b32",
                   "add.s64"}


_MEM_KINDS = {"Write", "CpAsyncCopy", "CpAsyncCommit", "CpAsyncWait", "Sync",
                  "Loop", "StridedLoop", "Cond", "TreeHalve", "WarpShuffle",
                  "Smem", "Pre", "Carry", "Select"}


CF_KINDS = {"Cond", "StridedLoop", "Loop", "TreeHalve"}
