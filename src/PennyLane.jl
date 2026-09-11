"""
PennyLane.jl — quantum programs as ordinary Julia, with a Julia frontend for the Catalyst compiler.

Quick start:

    using PennyLane
    dev = StateVector()                       # or LightningDevice(), CatalystDevice()
    @qnode dev function bell(θ)
        a, b = qubits(2)
        a = RX(θ, a)
        a, b = CNOT(a, b)
        return expval(Z(b))
    end
    bell(0.3)                                 # cos(0.3)
    gradient(bell, 0.3)
"""
module PennyLane

using LinearAlgebra
using Libdl
using Preferences
using Printf
using Random

include("operators/pauli.jl")
include("operators/gates.jl")
include("ir/program.jl")
include("devices/interface.jl")
include("ir/tracer.jl")
include("ir/mlir.jl")
include("devices/statevector.jl")
include("devices/adjoint.jl")
include("devices/yao.jl")
include("ir/passes.jl")
include("catalyst/env.jl")
include("catalyst/runtime.jl")
include("catalyst/compiled.jl")
include("gradients.jl")
include("datasets.jl")
include("python.jl")

# operators
export Operator, Observable, PauliString, PauliSum, X, Y, Z, Identity, ⊗, commutator, anticommutator, matrix, wires, terms
# gates
export Gate, ctrl, PauliX, PauliY, PauliZ, Hadamard, S, T, RX, RY, RZ, PhaseShift, Rot, CNOT, CY, CZ, SWAP, ISWAP,
       CRX, CRY, CRZ, CRot, ControlledPhaseShift, IsingXX, IsingYY, IsingZZ, Toffoli, CSWAP, MultiRZ, PauliRot,
       SingleExcitation, DoubleExcitation, BasisState
# programs
export @qnode, QNode, Program, Qubit, qubits, qubit, expval, var, probs, state, sample, program, mlir, to_mlir, gradient
export measure, @trace, @qif, @qfor, @qwhile, evolve, ApproxTimeEvolution, TracedReal, TracedInt, TracedBool
export optimize, gate_count
export QubitConsumedError, TraceError
# devices
export AbstractDevice, AbstractSimulator, StateVector, LightningDevice, CatalystDevice, PyDevice, YaoDevice, execute
export catalyst_env, has_catalyst, setup_python!, python_exe
# datasets and Python bridge
export h2_hamiltonian, H2_GROUND_ENERGY, molecular_hamiltonian, from_pennylane, to_pennylane, draw

end # module
