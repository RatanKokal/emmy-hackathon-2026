# VERIFICATION

Run everything through the validation engine. Every layer gates against
`tests/expectations.json`; any drift fails loudly.

## Quick (no GPU, ~30s, runs anywhere)

    python3 tools/validate.py --layer quick

Covers: package import, 23/23 golden byte-compare (15 official + 8 bug
fixtures), ptxcheck (dominance), racecheck (static races), provcheck
(register aliasing) with the documented whitelist.

## Full (GPU box)

    EMMY_BOX_HOST=user@host python3 tools/setup_box.py   # sync + sanitizer
    # then, on the box:
    python3 tools/gpu_layers.py fixtures corpus          # 8/8 fixtures, 15/15 official, 78/7 regression
    python3 tools/gpu_layers.py sweeps                   # semantic 176/0, matrix 408/0, stmt 88/0, seeds 77/7 x3
    python3 tools/gpu_layers.py differential             # vs emmy/nvcc, verdict rows frozen
    python3 tools/gpu_layers.py sanitizer                # memcheck over fixtures (EMMY_SANITIZER or auto)

## Layout

    tests/golden/            23 committed golden PTXs (--update-goldens to regenerate)
    tests/fixtures/          8 bug fixtures: IR + meta + inputs + frozen-oracle ref
    tests/hsx|synth|cases    85-problem regression corpus
    tests/expectations.json  frozen gates (counts, whitelists, verdict rows)
    tools/checkers/          ptxcheck / racecheck / provcheck
    tools/harness/           box-side runners (path-free, env-configured)
    tools/validate.py        layer engine (local)
    tools/gpu_layers.py      layer engine (box)
    tools/setup_box.py       box sync + sanitizer install (EMMY_BOX_HOST env)

## Historical results (2026-10-28)

All layers green on the modular bundle: official 15/15, regression 78/7
(7 documented non-defects), semantic 176/0, matrix 408/0 ptxas, statement
88/0, seeds 77/7 x3, differential verdict-identical to full speed,
sanitizer 8 fixtures 0 flagged. Bug-fixture corpus covers all five
confirmed bugs (findings.md): dyn-leak, groups==0, FMA alias, CSE
dropped-def, Cond-else addr reuse, plus the two pre-existing alignment
bugs.

## File map (post-restructure)

    compile.py               THE submission: bundled single-file safe variant
    src/compiler/            source of truth (10 modules); regenerate compile.py
                             via tools/bundle.py; dev entry: python3 -m src.compiler
    prev/                    prior compiler generations (frozen, speed, verified,
                             pre-safe compile.py, safe entry shim)
    backups/                 pre-modular validated monolith
