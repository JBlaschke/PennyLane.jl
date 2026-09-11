# PennyLane.jl

Quantum programs as ordinary Julia, and a Julia frontend for [Catalyst](https://github.com/PennyLaneAI/catalyst),
PennyLane's MLIR-based compiler. Research project; see [PLAN.md](PLAN.md).

```julia
using PennyLane

dev = StateVector()                     # pure Julia; or LightningDevice(), CatalystDevice(), PyDevice(...)

@qnode dev function bell(θ)
    a, b = qubits(2)                    # qubits are values
    a    = RX(θ, a)                     # gates consume and return them
    a, b = CNOT(a, b)
    return expval(Z(b))                 # observables are algebraic: 0.5*Z(a)*Z(b) + 0.3*X(a)
end

bell(0.3)                               # 0.9553… = cos(0.3)
gradient(bell, 0.3)                     # -0.2955… = -sin(0.3), by parameter shift
println(mlir(bell, 0.3))                # Catalyst MLIR for the program
```

Integer wires work wherever a qubit does (`RX(θ, 1); CNOT(1, 2); expval(Z(2))`), so PennyLane
circuits transliterate line by line. Using a qubit value twice is a `QubitConsumedError`.

Operators are Julia values: `H = 0.5*Z(1)*Z(2) + 0.3*X(1)`, `commutator(H, X(1))`, `matrix(H)`,
`exp(-im*t*H)` (a gate), `evolve(H, t, q...; steps)` (Trotterised time evolution, `t` may be a
parameter). `h2_hamiltonian()` ships the H₂ Hamiltonian; `examples/vqe_h2.jl` runs the VQE on
every backend.

Mid-circuit measurements and control flow that must survive into the program:

```julia
m, q = measure(q)                       # m is a traced Bool; measure(q; postselect=1), measure(q; reset=true)
@trace if m                             # branch on it (scf.if); variables assigned inside are threaded through
    c = PauliX(c)
end
@trace while !done                      # repeat until success (scf.while)
    q = Hadamard(q); done, q = measure(q)
end
@trace for l in 1:p                     # traced loop (scf.for); θ[l] becomes a dynamic index
    q = collect(evolve(C, γ[l], q...))
end
```

Plain Julia `if`/`for` are unrolled at trace time; `@trace` keeps them (see `examples/control_flow.jl`).

## Backends

| Device | What runs | Needs |
|---|---|---|
| `StateVector(; shots, threads)` | Julia simulator: specialised 1-/2-qubit and diagonal kernels, single-pass Pauli expectations, multithreaded from 14 qubits (`julia -t auto`); with `shots` every result is sampled | nothing |
| `LightningDevice(; shots)` | PennyLane-Lightning via the Catalyst runtime C API | Catalyst binaries |
| `CatalystDevice(; shots)` | Julia → MLIR → `catalyst` CLI → shared library; compiled adjoint gradients | Catalyst binaries, `clang` |
| `PyDevice(name; shots, kwargs...)` | any PennyLane device, including hardware plugins, through PythonCall | `using PythonCall` |
| `YaoDevice(; shots)` | Yao.jl's `ArrayReg`/`instruct!` kernels through the simulator interface | `import Yao` |
| `JITDevice()` | Julia code generated from the program, calling the Catalyst runtime directly; compiled by Julia's JIT (static expval circuits) | Catalyst binaries |

Measurements: `expval`, `var`, `probs`, `state`, `sample` (needs shots), `measure` (mid-circuit).
`PyDevice` lowers `measure` and `@trace if` on measurement results to PennyLane's measurement values
(deferred measurement or dynamic one-shot on the device), unrolls `@trace for`/classical `while`, and rejects
repeat-until-success loops (dynamic control flow).
Gradients: `gradient(qn, args...; method)` with `:adjoint` (exact, one backward pass; the default on
analytic `StateVector`/`LightningDevice` for static circuits, and compiled on `CatalystDevice` where it also
works through `@trace for`/`if`), `:parameter_shift` (hardware compatible; default with shots and on
`PyDevice`/`YaoDevice`), `:finitediff`. `PyDevice` does not take `@trace`/`measure` programs yet.

Passes: `optimize(prog)` or `@qnode dev passes=[:cancel_inverses, :merge_rotations] function ... end`
removes gate/inverse pairs and fuses consecutive rotations on the value-semantic IR (the same gate counts
Catalyst's `cancel-inverses`/`merge-rotations` produce). `gate_count(prog)` counts gates.

### Catalyst binaries

They come from the `pennylane-catalyst` and `pennylane-lightning` wheels. Create the pinned
environment with [uv](https://docs.astral.sh/uv/) (no Conda; no Python runs when you use them):

```bash
uv sync --project python
```

### PennyLane and hardware through PythonCall

```julia
using PennyLane
PennyLane.setup_python!()               # once per project, BEFORE adding PythonCall: points it at python/.venv, Conda off
# ] add PythonCall
using PythonCall                        # activates PyDevice, molecular_hamiltonian, draw, to_pennylane

dev = PyDevice("default.qubit"; shots=1000)         # or e.g. PyDevice("qiskit.remote"; wires=5, shots=1000, backend=...)
H, n = molecular_hamiltonian(["H", "H"], [0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
println(draw(bell, 0.3))
```

Other simulators plug in by implementing the `AbstractSimulator` interface
(`sim_allocate`, `sim_apply!`, `sim_expval`, `sim_state`, optional `sim_var`, `sim_probs`, `sim_sample`,
`sim_measure!`, `sim_release!`); `ext/PennyLaneYaoExt.jl` is a 60-line example.

## Compiler research (optional extensions)

- **Julia → native, no MLIR** (`using GPUCompiler`): `native_llvm_ir(qn, args...)` shows the LLVM IR of the program
  as straight-line calls into the Catalyst runtime; `compile_native(qn, args...)` builds a standalone shared library
  whose only external symbols are the runtime's `__catalyst__*` functions, callable like the QNode.
- **Julia passes on Catalyst's IR** (`using Reactant`): `mlir_pass(text, :cancel_inverses)` runs a pass written in
  Julia against Reactant's MLIR bindings (Catalyst's ops as unregistered operations, exchanged as text), and
  `CatalystDevice(mlir_transform = s -> mlir_pass(s, :cancel_inverses))` compiles the result.
- `examples/compilers.jl` walks through both, next to the JIT and Catalyst paths.

## Tests

```bash
JULIA_CONDAPKG_BACKEND=Null julia --project -e 'using Pkg; Pkg.test()'
```

The variable keeps PythonCall (a test dependency) from creating a Conda environment while the test
project precompiles; the bridge tests use `python/.venv`. Add `-t auto` to exercise the threaded kernels.

Notes: `var` and `sample` are exported (as in PennyLane); qualify them if you also use `Statistics`/`StatsBase`. Load Yao with `import Yao`, since `using Yao` exports clashing names (`X`, `Y`, `Z`, `measure`, ...).

License: Apache-2.0.
