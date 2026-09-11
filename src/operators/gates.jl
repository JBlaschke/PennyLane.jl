# Gate registry and gate objects.
#
# Gates carry PennyLane's names and matrix conventions. `RX(θ)` is an (unbound) gate object;
# `RX(θ, q)` applies it to a qubit value and returns the new value; `RX(θ)(q)` does the same.
# Applying a gate to an integer wire acts on the wire's current qubit value (PennyLane style).

struct GateDef
    name::Symbol
    nqubits::Int         # -1 => variadic (MultiRZ)
    nparams::Int
    matrix::Function     # params -> Matrix (fixed arity) or (params, nqubits) -> Matrix (variadic)
end

const GATES = Dict{Symbol,GateDef}()
_reg(name, nq, np, f) = (GATES[name] = GateDef(name, nq, np, f))

const _I2 = ComplexF64[1 0; 0 1]
_rx(t) = ComplexF64[cos(t / 2) -im*sin(t / 2); -im*sin(t / 2) cos(t / 2)]
_ry(t) = ComplexF64[cos(t / 2) -sin(t / 2); sin(t / 2) cos(t / 2)]
_rz(t) = ComplexF64[exp(-im * t / 2) 0; 0 exp(im * t / 2)]
_phase(ϕ) = ComplexF64[1 0; 0 exp(im * ϕ)]
_rot(ϕ, θ, ω) = _rz(ω) * _ry(θ) * _rz(ϕ)
function _controlled(U::AbstractMatrix)
    n = size(U, 1)
    M = Matrix{ComplexF64}(I, 2n, 2n)
    M[n+1:end, n+1:end] .= U
    M
end
_ising(P, t) = cos(t / 2) * Matrix{ComplexF64}(I, 4, 4) - im * sin(t / 2) * kron(P, P)
const _SWAP = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
function _multirz(p, n)
    d = [exp(-im * p[1] / 2 * (isodd(count_ones(i)) ? -1 : 1)) for i in 0:(1<<n)-1]
    Matrix{ComplexF64}(Diagonal(d))
end

_reg(:PauliX, 1, 0, p -> PAULI_MATRICES[:X])
_reg(:PauliY, 1, 0, p -> PAULI_MATRICES[:Y])
_reg(:PauliZ, 1, 0, p -> PAULI_MATRICES[:Z])
_reg(:Hadamard, 1, 0, p -> ComplexF64[1 1; 1 -1] ./ sqrt(2))
_reg(:S, 1, 0, p -> ComplexF64[1 0; 0 im])
_reg(:T, 1, 0, p -> ComplexF64[1 0; 0 exp(im * π / 4)])
_reg(:RX, 1, 1, p -> _rx(p[1]))
_reg(:RY, 1, 1, p -> _ry(p[1]))
_reg(:RZ, 1, 1, p -> _rz(p[1]))
_reg(:PhaseShift, 1, 1, p -> _phase(p[1]))
_reg(:Rot, 1, 3, p -> _rot(p[1], p[2], p[3]))
_reg(:CNOT, 2, 0, p -> _controlled(PAULI_MATRICES[:X]))
_reg(:CY, 2, 0, p -> _controlled(PAULI_MATRICES[:Y]))
_reg(:CZ, 2, 0, p -> _controlled(PAULI_MATRICES[:Z]))
_reg(:SWAP, 2, 0, p -> _SWAP)
_reg(:ISWAP, 2, 0, p -> ComplexF64[1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1])
_reg(:CRX, 2, 1, p -> _controlled(_rx(p[1])))
_reg(:CRY, 2, 1, p -> _controlled(_ry(p[1])))
_reg(:CRZ, 2, 1, p -> _controlled(_rz(p[1])))
_reg(:CRot, 2, 3, p -> _controlled(_rot(p[1], p[2], p[3])))
_reg(:ControlledPhaseShift, 2, 1, p -> _controlled(_phase(p[1])))
_reg(:IsingXX, 2, 1, p -> _ising(PAULI_MATRICES[:X], p[1]))
_reg(:IsingYY, 2, 1, p -> _ising(PAULI_MATRICES[:Y], p[1]))
_reg(:IsingZZ, 2, 1, p -> _ising(PAULI_MATRICES[:Z], p[1]))
_reg(:Toffoli, 3, 0, p -> _controlled(_controlled(PAULI_MATRICES[:X])))
_reg(:CSWAP, 3, 0, p -> _controlled(_SWAP))
_reg(:MultiRZ, -1, 1, _multirz)
function _single_excitation(ϕ)          # |01⟩ ↦ c|01⟩ + s|10⟩,  |10⟩ ↦ c|10⟩ - s|01⟩
    c, s = cos(ϕ / 2), sin(ϕ / 2)
    ComplexF64[1 0 0 0; 0 c -s 0; 0 s c 0; 0 0 0 1]
end
function _double_excitation(ϕ)          # |0011⟩ ↦ c|0011⟩ + s|1100⟩,  |1100⟩ ↦ c|1100⟩ - s|0011⟩
    c, s = cos(ϕ / 2), sin(ϕ / 2)
    M = Matrix{ComplexF64}(I, 16, 16)
    M[4, 4] = c; M[4, 13] = -s; M[13, 4] = s; M[13, 13] = c
    M
end
_reg(:SingleExcitation, 2, 1, p -> _single_excitation(p[1]))
_reg(:DoubleExcitation, 4, 1, p -> _double_excitation(p[1]))

# Parameter-shift spectrum class of each parametrised gate (shared by all of its parameters):
#   :half     generator eigenvalues ±1/2  -> two-term rule (four-term when the gate is controlled)
#   :proj     generator eigenvalues {0,1} -> two-term rule, also when controlled
#   :ctrlhalf generator eigenvalues {0, ±1/2} -> four-term rule
const SHIFT_KIND = Dict{Symbol,Symbol}(
    :RX => :half, :RY => :half, :RZ => :half, :Rot => :half, :MultiRZ => :half,
    :IsingXX => :half, :IsingYY => :half, :IsingZZ => :half,
    :PhaseShift => :proj, :ControlledPhaseShift => :proj,
    :CRX => :ctrlhalf, :CRY => :ctrlhalf, :CRZ => :ctrlhalf, :CRot => :ctrlhalf,
    :SingleExcitation => :ctrlhalf, :DoubleExcitation => :ctrlhalf,
)

"""Dense matrix of the named gate for concrete parameters, acting on `nq` qubits."""
function gate_matrix(name::Symbol, params, nq::Integer)
    def = GATES[name]
    def.nqubits == -1 ? def.matrix(params, nq) : def.matrix(params)
end

"""
    Gate(name, params; adjoint=false)

An unbound gate. Apply it with `g(q...)` or `apply(g, q...)`; `g'` is its adjoint.
"""
struct Gate <: Operator
    name::Symbol
    params::Vector{Any}
    adjoint::Bool
    function Gate(name::Symbol, params, adjoint::Bool=false)
        haskey(GATES, name) || throw(ArgumentError("unknown gate $name"))
        def = GATES[name]
        length(params) == def.nparams ||
            throw(ArgumentError("$name expects $(def.nparams) parameter(s), got $(length(params))"))
        new(name, collect(Any, params), adjoint)
    end
end
Base.adjoint(g::Gate) = Gate(g.name, g.params, !g.adjoint)
nqubits(g::Gate) = GATES[g.name].nqubits
function matrix(g::Gate, nq::Integer=nqubits(g))
    nq >= 1 || throw(ArgumentError("matrix of a variadic gate needs the number of qubits: matrix(g, n)"))
    M = gate_matrix(g.name, Float64[Float64(x) for x in g.params], nq)
    g.adjoint ? Matrix(M') : M
end
Base.show(io::IO, g::Gate) = print(io, g.name, "(", join(string.(g.params), ", "), ")", g.adjoint ? "'" : "")

"""
    ctrl(gate, controls...; values=trues)

Controlled version of `gate`. Applying `ctrl(g, c)(t)` returns the new `(c, t)` values.
"""
struct Controlled <: Operator
    gate::Gate
    controls::Vector{Any}
    values::Vector{Bool}
end
function ctrl(g::Gate, controls...; values=nothing)
    isempty(controls) && throw(ArgumentError("ctrl needs at least one control"))
    vals = values === nothing ? fill(true, length(controls)) : collect(Bool, values)
    length(vals) == length(controls) || throw(ArgumentError("one control value per control qubit"))
    Controlled(g, collect(Any, controls), vals)
end

"""
    PauliRot(θ, word)  /  PauliRot(θ, word, q...)

`exp(-i θ/2 P)` for the Pauli word `word` (a string of I, X, Y, Z). Lowered to basis changes
and `MultiRZ`, exactly like PennyLane.
"""
struct PauliRotGate <: Operator
    theta::Any
    word::String
end
PauliRot(theta, word::AbstractString) = PauliRotGate(theta, String(word))
function matrix(g::PauliRotGate)
    θ = Float64(g.theta)
    P = fill(1.0 + 0im, 1, 1)
    for c in g.word
        P = kron(P, PAULI_MATRICES[Symbol(c)])
    end
    cos(θ / 2) * Matrix{ComplexF64}(I, size(P)) - im * sin(θ / 2) * P
end

# `exp(i a P)` for a Pauli string with purely imaginary coefficient is a Pauli rotation on P's wires.
struct BoundPauliRot <: Operator
    theta::Float64
    word::String
    wires::Vector{Int}
end
function Base.exp(p::PauliString)
    c = p.coeff
    abs(real(c)) <= 1e-14 * max(1.0, abs(c)) ||
        throw(ArgumentError("exp of a Pauli string needs a purely imaginary coefficient, e.g. exp(-im * t * Z(1) * Z(2))"))
    isempty(p.word) && throw(ArgumentError("exp of the identity is a global phase; not supported"))
    BoundPauliRot(-2 * imag(c), join(string(l) for l in last.(p.word)), first.(p.word))
end
matrix(u::BoundPauliRot) = matrix(PauliRotGate(u.theta, u.word))
wires(u::BoundPauliRot) = u.wires

"""exp(-i t H) for a real Hamiltonian, applied by Trotterisation (see `evolve`)."""
struct BoundEvolution <: Operator
    H::Observable
    t::Float64
    steps::Int
end
function Base.exp(s::PauliSum)
    ts = terms(s)
    isempty(ts) && throw(ArgumentError("exp of the zero operator is the identity"))
    for t in ts
        abs(real(t.coeff)) <= 1e-14 * max(1.0, abs(t.coeff)) ||
            throw(ArgumentError("exp of a Pauli sum needs purely imaginary coefficients, e.g. exp(-im * t * H)"))
    end
    BoundEvolution(PauliSum([PauliString(-imag(t.coeff), t.word) for t in ts if !isempty(t.word)]), 1.0, 1)
end
matrix(u::BoundEvolution, n::Integer=nwires(u.H)) = exp(-im * u.t * Matrix(matrix(u.H, n)))
wires(u::BoundEvolution) = wires(u.H)

# Application is defined by the tracer (ir/tracer.jl).
function apply end

# Named constructors: `RX(θ)` -> Gate, `RX(θ, q)` -> apply.
for def in collect(values(GATES))
    def.nqubits == -1 && continue
    name = def.name
    ps = [Symbol(:p, i) for i in 1:def.nparams]
    qs = [Symbol(:q, i) for i in 1:def.nqubits]
    @eval begin
        $name($(ps...)) = Gate($(QuoteNode(name)), Any[$(ps...)])
        $name($(ps...), $(qs...)) = apply(Gate($(QuoteNode(name)), Any[$(ps...)]), $(qs...))
    end
end
MultiRZ(theta) = Gate(:MultiRZ, Any[theta])
MultiRZ(theta, q1, qs...) = apply(Gate(:MultiRZ, Any[theta]), q1, qs...)
