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
`exp(-im*t*Z(1)*Z(2))` (a gate). `h2_hamiltonian()` ships the H₂ Hamiltonian; `examples/vqe_h2.jl`
runs the VQE on every backend.

## Backends

| Device | What runs | Needs |
|---|---|---|
| `StateVector(; shots)` | reference Julia simulator; with `shots` every result is sampled | nothing |
| `LightningDevice(; shots)` | PennyLane-Lightning via the Catalyst runtime C API | Catalyst binaries |
| `CatalystDevice(; shots)` | Julia → MLIR → `catalyst` CLI → shared library; compiled adjoint gradients | Catalyst binaries, `clang` |
| `PyDevice(name; shots, kwargs...)` | any PennyLane device, including hardware plugins, through PythonCall | `using PythonCall` |

Measurements: `expval`, `var`, `probs`, `state`, `sample` (needs shots). Gradients:
`gradient(qn, args...; method)` with `:parameter_shift` (default, hardware compatible),
`:adjoint` (compiled, `CatalystDevice`), `:finitediff`.

### Catalyst binaries

They come from the `pennylane-catalyst` and `pennylane-lightning` wheels. Create the pinned
environment with [uv](https://docs.astral.sh/uv/) (no Conda; no Python runs when you use them):

```bash
uv sync --project python
```

### PennyLane and hardware through PythonCall

```julia
using PennyLane
PennyLane.setup_python!()               # once per project: points PythonCall at python/.venv, Conda off
using PythonCall                        # activates PyDevice, molecular_hamiltonian, draw, to_pennylane

dev = PyDevice("default.qubit"; shots=1000)         # or e.g. PyDevice("qiskit.remote"; wires=5, shots=1000, backend=...)
H, n = molecular_hamiltonian(["H", "H"], [0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
println(draw(bell, 0.3))
```

Other simulators plug in by implementing the `AbstractSimulator` interface
(`sim_allocate`, `sim_apply!`, `sim_expval`, `sim_state`, optional `sim_var`, `sim_probs`, `sim_sample`, `sim_release!`).

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Notes: `var` and `sample` are exported (as in PennyLane); qualify them if you also use `Statistics`/`StatsBase`.

License: Apache-2.0.
