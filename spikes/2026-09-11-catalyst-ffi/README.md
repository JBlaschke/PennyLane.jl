# Spike 2026-09-11: Catalyst from Julia, without Python at runtime

Verifies the two riskiest assumptions of the plan (see `../../PLAN.md`):

1. Hand-written MLIR in Catalyst's `quantum`/`gradient` dialects can be compiled by the
   `catalyst` CLI shipped in the `pennylane-catalyst` wheel, linked with `clang`, and called
   from Julia through the generated C interface (memref descriptors).
2. The Catalyst runtime (`librt_capi`) and the Lightning device plugin can be driven directly
   from Julia via `ccall`, with no compiler and no Python interpreter involved.

## Setup (uv, no Conda)

```bash
uv venv --python 3.14 python/.venv          # from the repo root; 3.12/3.13 also fine
uv pip install --python python/.venv/bin/python pennylane pennylane-catalyst
```

## Run

```bash
julia spikes/2026-09-11-catalyst-ffi/catalyst_ffi_spike.jl
```

Set `PENNYLANE_JL_VENV=/path/to/.venv` if the venv lives elsewhere.

## Result on the reference machine (M4 Max, macOS, Julia 1.12.7, catalyst 0.15.0, pennylane 0.45.1)

```
catalyst CLI compile: 0.3 s, link: 0.1 s
== Path B: compiled Catalyst MLIR -> dylib -> ccall ==
expval = 0.955336489126  (cos θ  = 0.955336489126)
grad   = -0.295520206661  (-sin θ = -0.295520206661)
latency: ~10 µs / call (includes device init+release per call)
== Path A': Julia -> Catalyst runtime C API -> Lightning ==
expval = 0.955336489126  (cos θ  = 0.955336489126)
latency: ~11 µs / circuit
```
