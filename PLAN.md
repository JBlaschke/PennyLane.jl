# PennyLane.jl — research plan

*Status: plan v1.3, 2026-09-11 (decisions in §9 resolved; M0 and M1 implemented, see §7). Reference machine: Apple M4 Max (macOS, arm64), Julia 1.12.7, Python 3.14.7 + uv.*

## 0. Summary

**Thesis.** Julia can be a first-class frontend to PennyLane's compiler stack (Catalyst) with no Python
in the hot path, and Julia's type system and multiple dispatch give a more natural hybrid
quantum/classical programming model than the Python API does. The two ideas reinforce each other:
Catalyst's quantum dialect uses *value semantics* (a gate consumes `!quantum.bit` values and produces
new ones), which is exactly the Sturm.jl idea of "the type boundary is the quantum/classical
boundary". A Julia tracer that hands out SSA qubit values maps 1:1 onto Catalyst IR.

**Verified today** (spike in `spikes/2026-09-11-catalyst-ffi/`, reproducible):

| Claim | Result |
|---|---|
| Hand-written MLIR in Catalyst's `quantum` + `gradient` dialects compiles with the `catalyst` CLI shipped in the PyPI wheel (macOS arm64, Catalyst 0.15.0) | yes, 0.3 s for a 2-qubit circuit plus its adjoint-method gradient |
| The resulting object links with `clang` and is callable from plain Julia via the generated C interface (memref descriptors) | yes, expval = cos θ, gradient = −sin θ to 1e-12 |
| The Catalyst runtime + Lightning device plugin can be driven directly from Julia `ccall`s, no compiler and no Python | yes, ~11 µs per 2-qubit circuit |
| Reactant's bundled MLIR (LLVM 24) can parse, verify and re-print Catalyst's `quantum.*` ops as unregistered ops next to `func`/`arith` | yes |
| PythonCall can use a uv-managed venv with Conda fully disabled, and share arrays zero-copy in both directions | yes |

**Recommendation.** Build a pure-Julia core (operators, hybrid-program IR, tracer, statevector
simulator, gradients) with four pluggable execution backends: (B1) Julia simulator, (B2) direct
Catalyst-runtime FFI to Lightning, (B3) Julia → Catalyst MLIR → compiled shared library, and (B4)
PennyLane itself over PythonCall, used as a test oracle and a convenience bridge only. MLIR *text* is
the interchange format between Julia and Catalyst; Reactant's MLIR bindings are used for Julia-side IR
manipulation. Python is optional and managed with uv, never Conda.

## 1. Goals, non-goals, principles

**Goals**

1. A Julia programming model for hybrid quantum programs that feels like ordinary Julia: values, dispatch,
   operator algebra, `do`-blocks, native control flow, integration with the Julia AD ecosystem.
2. A Julia frontend for Catalyst: emit the quantum/gradient dialects, drive the CLI, execute compiled code,
   and reuse Catalyst's passes (gate cancellation, rotation merging, adjoint differentiation, ...).
3. A port of a useful subset of PennyLane features: gates and observables, `expval`/`var`/`probs`/`sample`/
   `counts`/`state`, mid-circuit measurement, adjoint/controlled modifiers, a handful of templates,
   gradient methods (parameter-shift, adjoint, backprop, finite differences), qchem Hamiltonian import,
   and a device abstraction covering simulators and Catalyst devices.
4. Research nuggets around Julia's LLVM/MLIR access (see §3.5).

**Non-goals (for now):** feature parity with PennyLane, hardware access, PennyLane's transform and
plugin ecosystems, datasets, pulse programming. When needed these are reachable through the PythonCall
bridge (B4).

**Principles**

- Python is never in an inner loop. Anything performance-relevant has a Julia or C-ABI path.
- One IR, many backends. Every backend consumes the same traced program.
- Test against PennyLane, do not depend on it. B4 is the oracle in CI; the package must load and run without Python.
- Prefer the documented textual MLIR format over binary linkage with Catalyst's LLVM (version pinning is impossible otherwise).
- Reuse the Julia ecosystem via package extensions (weak deps), keep the core light.

## 2. Landscape and verified facts

### 2.1 PennyLane and Catalyst (PennyLane 0.45.1, Catalyst 0.15.0)

- Wheels exist for Linux x86_64/aarch64 and macOS arm64, Python ≥ 3.11 (3.14 works; Intel macOS dropped after 0.11).
  PennyLane now depends on xDSL (a Python re-implementation of MLIR) for its experimental Python compiler.
- Architecture: Python/JAX frontend traces to jaxpr → MLIR (StableHLO for classical parts, `quantum`, `gradient`,
  `catalyst`, `mitigation`, `mbqc` dialects for the rest) → `catalyst` CLI (quantum-opt + mlir-translate + llc)
  → object file → linked with `clang` against the runtime → loaded with `dlopen`. The docs state the stack is
  modular and PennyLane is only "the primary frontend"; no Julia frontend exists yet.
- The CLI (`.venv/bin/catalyst`, LLVM 22.0.0git @ 8f26458) runs five pipeline stages:
  `quantum-compilation-stage`, `hlo-lowering-stage`, `gradient-lowering-stage`, `bufferization-stage`,
  `llvm-dialect-lowering-stage`. Input that uses `arith`/`tensor`/`scf`/`func` instead of StableHLO passes
  through unchanged (verified). Plugins: `--load-dialect-plugin`, `--load-pass-plugin` (must be built against
  Catalyst's LLVM commit).
- Runtime: `catalyst/lib/librt_capi.dylib` exports 76 `__catalyst__rt__*` / `__catalyst__qis__*` C functions
  (QIR-style; see Appendix B). Devices are plugins loaded at `quantum.device` / `__catalyst__rt__device_init`;
  `qml.device("lightning.qubit").get_c_interface()` returns `("LightningSimulator", <path>/liblightning_qubit_catalyst.dylib)`.
  The Lightning plugin depends only on libc++ and libSystem.
- Compiled entry points: for `func.func @jit_f` with `llvm.emit_c_interface`, the object exports `jit_f`,
  `_catalyst_ciface_jit_f(result*, arg_memref*...)`, `_catalyst_pyface_jit_f(result*, void**)`, plus `setup(argc, argv)`
  and `teardown()` (which wrap `quantum.init`/`quantum.finalize`). Result buffers belong to the runtime's memory
  manager until `_mlir_memory_transfer(ptr)` is called.
- Gradients: `gradient.grad "auto" @f(%x)` selects the method from the callee's `diff_method` attribute
  (`adjoint` on Lightning, `parameter-shift`, `finite-diff`); `gradient.jvp`/`vjp`/`value_and_grad` also exist.

### 2.2 Julia quantum ecosystem

| Package | Relevance |
|---|---|
| Yao.jl (Apache-2.0) | Mature block DSL + fast `ArrayReg` kernels, AD, CUDA, OpenQASM, tensor-network export. Candidate simulator backend and a reference for kernel design; its block IR is matrix-centric, not value-semantic. |
| QuantumClifford.jl | Stabilizer simulation; possible backend for Clifford subsets. |
| ITensors / PastaQ | Tensor-network simulation backend for larger circuits. |
| Braket.jl | AWS SDK, circuit DSL, OpenQASM 3 tooling. |
| QuantumOptics.jl, QuantumInterface.jl, QuantumSymbolics.jl | Shared operator/state interfaces worth conforming to (`expect`, `dagger`, `⊗`). |
| Sturm.jl (AGPL-3.0) | The programming-model inspiration: `QBool`/`QInt` registers, consuming measurements (`Bool(q)`), `when(q) do ... end` coherent control, `dual(q)`, `oracle(f, x)` (Bennett bridge), and three execution contexts (`eager`, `density`, `trace`) for the same program. **AGPL: adopt ideas, do not copy code.** |
| JuliaQuantum.github.io | Community page; last updated around JuliaCon 2024. Ecosystem lives in QuantumBFS / QuantumSavory / ITensor orgs. |

### 2.3 Julia ↔ MLIR / LLVM

- **MLIR.jl** (JuliaLabs) wraps the MLIR C API; maintainers work on a vendored copy inside **Reactant.jl**
  (`Reactant.MLIR.IR`, `Reactant.MLIR.Dialects`: func, arith, tensor, memref, scf/affine, llvm, stablehlo, enzyme, ...).
  Reactant v0.2.285 is already in this machine's default environment; its `libReactantExtra` is LLVM **24**.0.0git.
- Catalyst is LLVM **22**. Sharing an `MLIRContext` or loading Catalyst's dialect libraries into Reactant's MLIR
  is therefore out. Textual IR is the bridge: verified that Reactant's MLIR parses Catalyst ops in generic form
  (`"quantum.custom"(...) {gate_name = "RX", ...}`) with `allow_unregistered_dialects!`, verifies the module, and
  prints it back, and Catalyst's parser accepts the generic form.
- Julia's own LLVM path: `GPUCompiler.jl` (standalone LLVM modules from Julia functions), `LLVM.jl`, and
  Julia 1.12's experimental `juliac --trim` (standalone binaries). A Julia function that `ccall`s
  `__catalyst__qis__*` compiles to LLVM IR that *is* a QIR-style program; see §3.5.

### 2.4 Julia ↔ Python

- **PythonCall.jl** with `JULIA_CONDAPKG_BACKEND=Null` and `JULIA_PYTHONCALL_EXE=<venv>/bin/python` uses an
  externally managed interpreter and never touches Conda (verified: imported PennyLane 0.45.1 from a uv venv,
  ran a QNode, shared a Julia matrix with NumPy and a NumPy array with Julia without copies).
- `uv` creates the venv and resolves/locks packages in seconds; Python 3.14 wheels for both PennyLane and
  Catalyst installed without building anything.

## 3. Architecture

Four layers; each has a Julia-only implementation and optional extensions.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ L1 Frontend   @qnode / @qjit macros, wire-indexed API, value-semantic API,     │
│               operator algebra (PauliSum), control flow, adjoint/ctrl, measure │
├───────────────────────────────────────────────────────────────────────────────┤
│ L2 IR         value-semantic hybrid program IR (mirrors Catalyst's quantum     │
│               dialect + scf/arith), verifier, Julia-side passes               │
├───────────────────────────────────────────────────────────────────────────────┤
│ L3 Backends   B1 StateVector (Julia) │ B2 Runtime FFI (Lightning) │            │
│               B3 Catalyst compiled   │ B4 PennyLane via PythonCall (ext)       │
├───────────────────────────────────────────────────────────────────────────────┤
│ L4 Gradients  parameter-shift, finite-diff (any backend); adjoint (B2/B3);     │
│               backprop via Enzyme/Zygote (B1); ChainRulesCore glue             │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Programming model (what "natural Julia" buys us)

Two surface styles over one IR. The wire-indexed style exists to port PennyLane templates and to feel
familiar; the value style is the primary design and is what Catalyst IR looks like anyway.

```julia
using PennyLane                                  # working name, see §9

dev = LightningDevice(4)                         # B2; also StateVector(4), CatalystDevice(...), PyDevice("default.qubit", 4)

# Style 1 — wire-indexed, PennyLane-like. Wires are integers; the tracer keeps the current SSA value per wire.
@qnode dev function circuit(θ)
    RX(θ[1], 1);  RY(θ[2], 2)
    CNOT(1, 2)
    return expval(Z(2))
end

# Style 2 — value semantics (Sturm/Catalyst-like). Qubits are SSA values; using a consumed value is a trace-time error.
@qjit dev function bell(θ)
    a, b = qubits(2)
    a    = RX(θ, a)
    a, b = CNOT(a, b)
    return expval(Z(b))
end

# Operator algebra: Hamiltonians are Julia values with +, *, ⊗, adjoint, commutator; expval(H) lowers to
# quantum.hamiltonian / quantum.tensor / quantum.namedobs.
H = 0.5 * Z(1) * Z(2) + 0.3 * X(1) - 0.2 * Y(2)
U = exp(-im * t * H)                             # lazy; lowers to PauliRot/MultiRZ Trotter steps via a template

# Control flow and mid-circuit measurement (Catalyst for_loop / cond / measure).
@qjit dev function layered(θ::Vector{Float64}, n::Int)
    q = qubits(n)
    @qfor i in 1:n                               # traced loop -> scf.for; plain `for` unrolls at trace time
        q[i] = RY(θ[i], q[i])
    end
    m, q[1] = measure(q[1])                      # -> quantum.measure, m::Traced{Bool}
    @qif m  q[2] = X(q[2])  end                  # -> scf.if on a traced boolean
    adjoint() do                                 # -> quantum.adjoint region
        q[1], q[2] = CNOT(q[1], q[2])
    end
    return probs(q)
end

# Differentiation: one entry point, method chosen per device/diff_method; ChainRulesCore makes any Julia AD work.
gradient(bell, 0.3)                              # adjoint on B2/B3, parameter-shift or Enzyme backprop on B1
Zygote.gradient(θ -> bell(θ), 0.3)
```

Why this is better than a transliteration of the Python API:

- Multiple dispatch replaces PennyLane's device/gradient plumbing: `execute(::StateVector, prog)`,
  `execute(::CatalystDevice, prog)`, `gradient(::AdjointMethod, ...)` are just methods.
- Operators are first-class algebraic objects (`PauliString`, `PauliSum`, `exp`, `kron`), so Hamiltonians,
  Trotterization, commutator-based analysis and parameter-shift rules are ordinary generic code.
- Value semantics with trace-time use-after-consume checks give Sturm's "no-cloning by construction" without
  linear types, and they *are* the IR Catalyst wants.
- Tracing (Reactant-style: typed tracer values, `@qfor`/`@qif` for traced control flow, plain Julia control
  flow unrolled) needs no source rewriting. Julia 1.12's `Compiler` stdlib and `Base.Experimental.@overlay`
  are available if we later want abstract-interpretation-based capture like Reactant does.
- Three execution contexts for the same program, as in Sturm: `simulate` (B1), `execute` on a device (B2/B3),
  `trace` (return the IR for inspection, passes, MLIR export).

### 3.2 Hybrid program IR (L2)

A small Julia-native SSA IR whose ops mirror Catalyst's quantum dialect plus the classical subset we need
(`arith`, `tensor`, `scf.for`/`scf.if`/`scf.while`, `func`). Rationale for owning an IR instead of emitting MLIR
text directly during tracing: it gives every backend the same input, allows Julia-side passes and verification,
and keeps the MLIR printer a trivial, testable 1:1 lowering. It is also the natural analogue of PennyLane's own
new IR (plxpr) and of the xDSL mirror of the Catalyst dialect PennyLane is building.

Core node kinds: `Alloc`, `Extract`, `Insert`, `Dealloc`, `Custom(gate, params, qubits; adjoint, ctrls, ctrlvals)`,
`MultiRZ`, `PauliRot`, `Unitary`, `GPhase`, `NamedObs`, `Hermitian`, `Tensor`, `Hamiltonian`, `CompBasis`,
`Expval`, `Var`, `Probs`, `Sample`, `Counts`, `State`, `Measure(postselect)`, `Adjoint{region}`, `Ctrl{region}`,
`For`, `If`, `While`, and typed classical values. Verifier rules: each qubit value used exactly once, wires
consistent, observables built from live qubits.

Lowerings: `to_mlir(prog)` (Catalyst dialects, textual), `interpret(prog, dev)` (B1/B2), `to_qasm3(prog)` (later),
`to_pennylane(prog)` (B4, builds a `qml.tape.QuantumScript`).

### 3.3 Backends (L3)

| Backend | What it is | Needs | Use |
|---|---|---|---|
| B1 `StateVector{T}` | Pure-Julia statevector simulator; dispatch-based gate kernels, SIMD, threads; later GPU via KernelAbstractions (Metal on this machine, CUDA elsewhere) | nothing | default, teaching, backprop via Enzyme |
| B2 `LightningDevice` | `ccall`s into `librt_capi` + Lightning plugin while interpreting the IR | Catalyst + Lightning wheel files, no Python process | fast, shot-based, adjoint gradients via `__catalyst__qis__Gradient` |
| B3 `CatalystDevice` | `to_mlir` → `catalyst` CLI → `clang` link → `dlopen` → C interface | Catalyst wheel + clang | compiled control flow, `gradient.grad`, quantum passes, future hardware devices |
| B4 `PyDevice` (extension) | PennyLane over PythonCall | uv venv | oracle in tests, templates/qchem/datasets |

B2 and B3 share one FFI module: runtime C API bindings, memref descriptor structs (rank-0..n, f64/i64/complex),
ownership handling (`_mlir_memory_transfer`), device lifecycle (`initialize`/`device_init`/`device_release`/`finalize`),
and a library locator that finds the wheel files from a venv path or explicit env vars.

### 3.4 Differentiation (L4)

- `gradient(f, args...; method)` with methods `ParameterShift()`, `FiniteDiff()`, `Adjoint()`, `Backprop()`.
- B3: emit a second entry point with `gradient.grad`/`value_and_grad`; adjoint is compiled by Catalyst/Enzyme.
- B2: Lightning's adjoint Jacobian through the runtime `Gradient` call; parameter-shift as a fallback.
- B1: Enzyme reverse mode through the simulator kernels, or a hand-written adjoint-Jacobian (memory O(2ⁿ)).
- `ChainRulesCore.rrule(::typeof(execute), dev, prog, params)` so Zygote/Enzyme/Optimization.jl compose.

### 3.5 Compiler research track (the "nuggets")

1. **Julia as a Catalyst frontend via textual MLIR** (verified). Cheap, version-tolerant, gives us Catalyst's
   full pipeline and gradient lowering for free.
2. **Julia-side MLIR passes** on Catalyst IR using Reactant's MLIR bindings: parse Catalyst output with unregistered
   dialects, rewrite in Julia (e.g. gate cancellation, rotation merging, commutation-based reordering, Pauli-frame
   tracking), print, and feed back to the CLI. This is the Julia analogue of PennyLane's xDSL passes, with no C++.
   Ceiling: unregistered ops have no verifier and no canonicalization on the Julia side.
3. **Classical parts via Reactant.** Trace the classical portion of a hybrid function with Reactant into StableHLO,
   serialize as a versioned VHLO portable artifact targeting Catalyst's StableHLO (v1.13.7), and splice quantum
   ops in. Speculative; worth one experiment because it would give XLA-grade classical optimization for free.
4. **Julia → QIR without MLIR.** A Julia function calling `__catalyst__qis__*` through `ccall`, compiled to a
   standalone LLVM module with GPUCompiler.jl (or a whole program with `juliac --trim`), is a QIR-style hybrid
   program produced by Julia's own compiler. Compare against B3 on the same test problems: what Catalyst's
   quantum passes buy versus what LLVM alone does.
5. **Catalyst plugins.** If a pass needs registered-op power, write it as a C++ dialect/pass plugin against
   Catalyst's LLVM commit and load it with `--load-pass-plugin`; the Julia side only needs to pass the flag.
6. **Sturm-style reversible oracles.** `oracle(f, x)` compiling classical Julia functions to reversible circuits
   is a natural fit for Julia's compiler introspection (Bennett-style); park until the core exists.

## 4. Python dependency management (no Conda)

Decision: **uv-managed virtual environment, pinned with a lockfile, discovered by the Julia package; PythonCall
with the Null CondaPkg backend.** Python is a weak dependency (package extension), so `using PennyLane` never
requires it.

```
python/
  pyproject.toml        # pennylane==0.45.1, pennylane-catalyst==0.15.0, pennylane-lightning pinned; requires-python >=3.14,<3.15
  uv.lock               # committed; reproducible across machines
```

Setup is two commands, no Conda, no compilers:

```bash
uv venv --python 3.14 python/.venv
```

```bash
uv pip install --python python/.venv/bin/python pennylane pennylane-catalyst
```

or, from the repo root, `uv sync --project python` (uses the committed `uv.lock`).

Julia side:

- A `PythonEnv` module resolves the interpreter in this order: `ENV["PENNYLANE_JL_PYTHON"]`, `python/.venv/bin/python`
  next to the package, else a clear error with the two commands above. It also locates the Catalyst CLI, runtime
  libraries and the Lightning plugin for B2/B3 (via `sysconfig` and `get_c_interface()`, or from cached paths so
  B2/B3 do not need to start Python at all after the first discovery).
- For B4, users set `JULIA_CONDAPKG_BACKEND=Null` and `JULIA_PYTHONCALL_EXE` before `using PythonCall` (verified);
  we ship a `LocalPreferences.toml` template and a `PennyLane.setup_python!()` helper that writes it, so the env vars
  are not needed in day-to-day use. `JULIA_PYTHONCALL_EXE=@venv` is an alternative for `.venv` next to the project.
- CI: a Julia-only job (B1 tests), a "with Python" job (uv install, oracle tests), macOS arm64 and Linux x86_64.
- Later option: package the Catalyst and Lightning wheel contents as a Julia artifact (a wheel is a zip of dylibs plus
  the CLI), giving B2/B3 users a Python-free install. Needs a check that `Pkg.Artifacts` can unpack zips or a small
  custom downloader.

Why not the alternatives: CondaPkg's MicroMamba/Conda/Pixi backends drag in a second package manager and a 500 MB
environment; PyCall's build-time `PYTHON` variable is global per depot and fights with multiple projects; a subprocess
RPC to PennyLane would serialize arrays. PythonCall + uv keeps one interpreter per project, reproducible, fast, and
zero-copy.

## 5. Getting performance-critical work out of Python

Rules:

1. Python is used for *setup and verification*, never per shot, per gate or per gradient evaluation.
2. Every operation on the critical path (state evolution, expectation values, gradients, compilation) has a Julia
   (B1) or C-ABI (B2/B3) implementation before it is used in a benchmark.
3. When Python must be called, call it once with whole batches (a tape batch, a Hamiltonian, a dataset) and pass
   arrays zero-copy (`PyArray`, buffer protocol).
4. Every milestone ships a benchmark; the PennyLane number is the baseline to beat, not the implementation to wrap.

Concrete migrations, in order: statevector kernels and measurement statistics → parameter-shift and adjoint gradients →
Hamiltonian construction (`PauliSum` arithmetic, qchem import happens once through B4 and is cached as Julia data) →
circuit compilation (B3 replaces Python's compile driver entirely) → optimizers (Optim.jl / Optimization.jl, no autograd).

Latency reference from the spike: a compiled 2-qubit circuit costs ~10 µs per call from Julia, dominated by device
init/release inside the qnode; Python's ctypes path adds tens of µs on top. Hoisting `quantum.device` out of hot loops
(a Catalyst-side option) or keeping a persistent device in B2 will matter for VQE-style loops.

## 6. Test problems

| # | Problem | Exercises | Accept when |
|---|---|---|---|
| T0 | RX–CNOT–expval(Z) and its gradient | FFI, MLIR emission, setup/teardown | matches cos θ / −sin θ on B1, B2, B3, B4 (done manually in the spike) |
| T1 | VQE for H₂ (4 qubits, hardware-efficient and UCCSD-style ansatz) | `PauliSum` Hamiltonian imported once from `qml.qchem`, parameter-shift (B1), adjoint (B2/B3), Enzyme backprop (B1), Optim.jl loop | energy within 1e-6 Ha of PennyLane's; gradients agree across methods |
| T2 | QAOA MaxCut on a small graph (Graphs.jl) | templates, `@qfor` layers, `exp(-im t H)` lowering, sampling/counts | approximation ratio matches PennyLane on the same seed |
| T3 | Teleportation and a repeat-until-success loop | mid-circuit measurement, `@qif`/`@qwhile`, value-semantic API | correct output state; MLIR passes Catalyst verification; B1 and B3 agree |
| T4 | Random 20–26 qubit circuits | B1 kernels, threads, Metal; B2 Lightning | B1 within 2× of Lightning single-threaded; both beat `default.qubit` |
| T5 | Julia-side passes: cancel inverses, merge rotations | Reactant-MLIR round trip, pass framework | identical gate counts to Catalyst's `cancel-inverses`/`merge-rotations` on a circuit corpus |
| T6 | Julia → QIR via GPUCompiler / `juliac --trim` | §3.5 item 4 | standalone binary runs T0 against `librt_capi` |

## 7. Roadmap

**Status 2026-09-11 (evening): M0 and M1 done.** The package loads with no Python; the full suite
(simulators, Lightning FFI, compiled Catalyst, PennyLane bridge) passes on the reference machine.
Implemented: Pauli operator algebra (`X/Y/Z`, `*`, `+`, `⊗`, `commutator`, `matrix`, `exp`), the PennyLane
gate set incl. `SingleExcitation`/`DoubleExcitation`, the value-semantic tracer with `@qnode`, integer-wire
sugar, traced parameter arithmetic, `expval`/`var`/`probs`/`state`/`sample`, the IR and its Catalyst MLIR
printer, the `AbstractSimulator` interface, `StateVector` (B1, with shot-based sampling), `LightningDevice`
(B2, runtime C API, sampling), `CatalystDevice` (B3, CLI + clang + FFI, compiled adjoint gradients, sampling),
`PyDevice` (B4, PythonCall extension: any PennyLane device incl. hardware plugins, `molecular_hamiltonian`,
`draw`, `to_pennylane`), parameter-shift gradients (two- and four-term rules, chain rule through traced
arithmetic; default on simulators and `PyDevice`), finite differences, the built-in H₂ Hamiltonian, and T1
(VQE for H₂ converging to the exact energy within 1e-6 Ha with parameter-shift gradients on all backends,
gradients agreeing across methods to 1e-7). Not yet: control flow (`@qfor`/`@qif`), mid-circuit `measure`,
Enzyme backprop, Yao extension, Trotter (`exp` of a `PauliSum`), Julia-side MLIR passes.

| Phase | Deliverables | Effort |
|---|---|---|
| M0 Scaffold — **done** | `Project.toml`, module skeleton, `python/` env + lockfile, `PythonEnv` discovery, FFI module (runtime C API bindings, memref structs), CI, the spike turned into tests for T0 on B2/B3 | ~1 week |
| M1 Core — **done** | gate and observable types, `PauliSum` algebra, IR + verifier, tracer with both API styles, `StateVector` (B1), measurements, parameter-shift and finite-diff, PythonCall extension as oracle (and hardware bridge), shots/sampling, T1 on B1–B4 | 2–3 weeks |
| M2 Catalyst | `to_mlir`, CLI driver and linker, generalized memref FFI (rank-n, multiple results), `gradient.grad`, `@qfor`/`@qif`/`measure`, T1 and T3 on B3, benchmark harness | 2 weeks |
| M3 Performance and passes | SIMD/threaded kernels, Metal via KernelAbstractions, Lightning adjoint through B2, Enzyme through B1, Julia-side passes on Reactant MLIR, T2/T4/T5 | 3 weeks |
| M4 Research | Julia→QIR (T6), Reactant/StableHLO bridge experiment, Sturm-style oracles, optional Yao/QuantumClifford/ITensor backends, hardware via Catalyst devices | open-ended |

## 8. Risks and mitigations

- **Catalyst IR churn between releases.** Pin `pennylane-catalyst` in the lockfile; golden-file tests for `to_mlir`;
  emit only documented ops; keep the generic-form printer as fallback.
- **LLVM version gap (Catalyst 22 vs Reactant 24).** Never link the two; text only. Standard dialect syntax
  (`func`, `arith`, `scf`, `tensor`) is stable across these versions, and the spike confirms both parsers accept it.
- **Linker flags differ per OS.** Mirror Catalyst's `LinkerDriver` (`-no-as-needed`/`--disable-new-dtags` on Linux,
  `-arch_errors_fatal` on macOS) behind `Sys.isapple()`/`Sys.islinux()`; `libcustom_calls` is a `.so` on both.
- **Runtime memory ownership and thread safety.** Always `_mlir_memory_transfer` + `free` results; one device per
  task; no concurrent use of a runtime context.
- **Licensing.** PennyLane, Catalyst, Yao: Apache-2.0. Sturm.jl: AGPL-3.0, ideas only. Choose Apache-2.0 or MIT
  for this package so PennyLane code (templates, decompositions) can be ported with attribution.
- **Python 3.14 is new.** Works today; the lockfile can pin 3.12/3.13 if a dependency regresses.
- **Name.** "PennyLane" is Xanadu's project name; a port should probably carry a distinct name (see §9). The
  directory is currently spelled `PennlyLane.jl`.

## 9. Decisions (resolved 2026-09-11)

| # | Decision | Consequence |
|---|---|---|
| 1 | Package **PennyLane.jl**, **Apache-2.0** | Same license as PennyLane/Catalyst: templates and decompositions can be ported with attribution. Sturm.jl (AGPL) stays ideas-only. |
| 2 | **Value semantics is the primary API**; Julia operator algebra is a hard requirement; PennyLane names and conventions wherever they lower the onramp. Design test: "can an application designer reason about this program?" (Sturm.jl) | Qubits are SSA values, gates consume and return them (`a, b = CNOT(a, b)`), reuse of a consumed value is a loud trace-time error. Observables are algebraic objects built on wires **or on qubit values** (`Z(2)`, `Z(b)`, `0.5*Z(a)*Z(b) + 0.3*X(a)`); Pauli *gates* use PennyLane's names (`PauliX(q)`), Pauli *operators* are `X, Y, Z`. Integer wires are accepted everywhere a qubit is (`RX(θ, 1)`) and resolve to the wire's current value, so PennyLane circuits transliterate line by line. |
| 3 | Own small reference simulator in core; **Yao as an optional extension**; a **simulator interface** so other Julia simulators (QuantumClifford, ITensorMPS, GPU kernels, Lightning through FFI) plug in via dispatch | The interpreter over the IR is written once against `AbstractSimulator` (init, apply gate, expectation, probabilities, sample); every backend is an implementation. This is where Julia should shine, so the interface is a first-class deliverable, not an afterthought. |
| 4 | **Own IR mirroring Catalyst's dialect now; direct-to-MLIR later** once Reactant/MLIR.jl stabilise | IR nodes correspond 1:1 to `quantum`/`gradient`/`scf`/`arith` ops so the textual printer can later be swapped for an MLIR.jl builder without touching the tracer or the backends. |
| 5 | **Python 3.14** in the lockfile unless a dependency regresses | `python/pyproject.toml` pins `requires-python >=3.14,<3.15`, pennylane 0.45.1, pennylane-catalyst 0.15.0; `uv sync --project python` reproduces the environment. |

Open follow-ups: directory is still spelled `PennlyLane.jl` (rename with `mv` when convenient; the package name inside `Project.toml` is `PennyLane`); repository not yet under git.

---

## Appendix A — Spike log (2026-09-11)

Environment created in a scratch directory; reproducible script committed at
`spikes/2026-09-11-catalyst-ffi/catalyst_ffi_spike.jl` (see its README for the uv commands).

- `uv venv --python 3.14 .venv && uv pip install pennylane pennylane-catalyst` → pennylane 0.45.1, catalyst 0.15.0
  (pulls in jax 0.7.x, xdsl 0.59, pennylane-lightning). No compilers, no Conda.
- Wheel layout: CLI at `.venv/bin/catalyst`; runtime libs in `site-packages/catalyst/lib/` (`librt_capi`,
  `libmlir_c_runner_utils`, `libmlir_async_runtime`, `librt_rsdecomp`, `liblapacke.3`); `site-packages/catalyst/utils/libcustom_calls.so`;
  Lightning plugin `site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib`.
- MLIR emitted by `qjit(target="mlir")` for RX–CNOT–expval was copied verbatim, given a second `gradient.grad "auto"`
  entry point, compiled with `catalyst --module-name=jlprobe --workspace=ws -o ws/jlprobe.ll jlprobe.mlir` (0.3 s), linked with
  `clang -shared -rdynamic ... -lrt_capi -lmlir_c_runner_utils -llapacke.3 libcustom_calls.so -lmlir_async_runtime -lrt_rsdecomp`.
- From Julia (`Libdl` only): `setup(1, ["jitted-function"])`, `_catalyst_ciface_jit_circuit(res*, arg*)` with rank-0 memref
  descriptors, `_mlir_memory_transfer` + `free`, `teardown()`. Results: expval 0.955336489126 (cos 0.3),
  gradient −0.295520206661 (−sin 0.3). ~10 µs per call.
- Direct runtime path: `__catalyst__rt__initialize(NULL)`, `__catalyst__rt__device_init(lib, "LightningSimulator", kwargs, 0, false)`,
  `qubit_allocate_array(2)`, `__catalyst__qis__RX(θ, q0, NULL)`, `CNOT`, `NamedObs(3, q1)`, `Expval` → same value, ~11 µs per circuit.
- Reactant (LLVM 24) parsed and verified a module mixing `func`/`arith` with generic-form `quantum.*` ops and the opaque
  attribute `#quantum<named_observable PauliZ>`; round trip printed identically.
- PythonCall with `JULIA_CONDAPKG_BACKEND=Null`, `JULIA_PYTHONCALL_EXE=<venv>/bin/python`: `pyimport("pennylane")` 0.45.1,
  QNode on `default.qubit` returned cos 0.3; `Py(A).__array__()` and `pyconvert(PyArray, ...)` both shared memory.

## Appendix B — Catalyst IR and runtime cheat-sheet (0.15.0)

Types: `!quantum.reg`, `!quantum.bit`, `!quantum.obs`, `!quantum.res`. Named observables: Identity 0, PauliX 1, PauliY 2, PauliZ 3, Hadamard 4.

```mlir
quantum.device shots(%s) ["<plugin path>", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
%r  = quantum.alloc( 2) : !quantum.reg
%q  = quantum.extract %r[ 0] : !quantum.reg -> !quantum.bit
%q1 = quantum.custom "RX"(%theta) %q : !quantum.bit                 // (`adj`)? ; controls via ctrls(...) ctrlvals(...)
%a:2 = quantum.custom "CNOT"() %q1, %q2 : !quantum.bit, !quantum.bit
%o  = quantum.namedobs %a#1[ PauliZ] : !quantum.obs
%h  = quantum.hamiltonian(%coeffs : tensor<2xf64>) %o1, %o2 : !quantum.obs
%e  = quantum.expval %o : f64
%m, %q3 = quantum.measure %q2                                        // (postselect N)?
%r2 = quantum.insert %r[ 0], %a#0 : !quantum.reg, !quantum.bit
quantum.dealloc %r2 : !quantum.reg
quantum.device_release
%g  = gradient.grad "auto" @mod::@f(%x) {diffArgIndices = dense<0> : tensor<1xi64>} : (tensor<f64>) -> tensor<f64>
```

Runtime C API subset (all `extern "C"`, in `librt_capi`): `__catalyst__rt__initialize(uint32_t*)`,
`__catalyst__rt__device_init(int8_t* lib, int8_t* name, int8_t* kwargs, int64_t shots, bool auto_qubit_mgmt)`,
`__catalyst__rt__device_release()`, `__catalyst__rt__finalize()`, `QirArray* __catalyst__rt__qubit_allocate_array(int64_t)`,
`int8_t* __catalyst__rt__array_get_element_ptr_1d(QirArray*, int64_t)`, `__catalyst__rt__qubit_release_array(QirArray*)`,
gates `__catalyst__qis__{PauliX,PauliY,PauliZ,Hadamard,S,T,RX,RY,RZ,Rot,PhaseShift,CNOT,CY,CZ,SWAP,IsingXX,...}(params..., QUBIT*..., const Modifiers*)`,
`__catalyst__qis__MultiRZ`, `__catalyst__qis__PauliRot`, `__catalyst__qis__QubitUnitary`, observables
`ObsIdType __catalyst__qis__{NamedObs,HermitianObs,TensorObs,HamiltonianObs}(...)`, measurements
`__catalyst__qis__{Expval,Variance}(ObsIdType)`, `__catalyst__qis__{Probs,Sample,Counts,State}(MemRef*, int64_t n, ...)`,
`RESULT* __catalyst__qis__Measure(QUBIT*, int32_t postselect)`, `__catalyst__qis__Gradient(int64_t, ...)`.
`Modifiers* = NULL` means no adjoint and no controls.

## Appendix C — Proposed repository layout

```
Project.toml
src/
  PennyLane.jl              # module, exports, PythonEnv discovery
  operators/                # Pauli algebra, gate definitions and matrices, decompositions
  ir/                       # hybrid program IR, verifier, passes, printer
  frontend/                 # @qnode/@qjit, tracer, wire-indexed + value APIs, control flow, adjoint/ctrl
  devices/                  # AbstractDevice, StateVector (B1)
  gradients/                # parameter-shift, finite-diff, adjoint glue, ChainRules
  catalyst/                 # runtime C API bindings, memref FFI, CLI driver, linker, CatalystDevice (B3), LightningDevice (B2)
ext/
  PennyLanePythonCallExt.jl # PyDevice (B4), qchem import, tape export
  PennyLaneReactantExt.jl   # MLIR round trip and Julia-side passes
  PennyLaneYaoExt.jl        # optional Yao backend / cross-checks
python/                     # pyproject.toml, uv.lock
spikes/                     # dated, reproducible experiments (this one included)
test/  bench/  docs/
```
