# Yao.jl backend: `YaoDevice` runs programs on Yao's ArrayReg with `instruct!`. Activated by `import Yao`
# (prefer `import` over `using`: Yao and PennyLane both export X, Y, Z, measure, state, ...).
module PennyLaneYaoExt

using PennyLane
import Yao
using PennyLane: YaoDevice, PauliString, PauliSum, Observable, terms, gate_matrix, marginal_probs,
                 sample_pauli_eigenvalues, sample_bits, sample_probs, SVState, StateVector
import PennyLane: sim_allocate, sim_apply!, sim_expval, sim_var, sim_state, sim_probs, sim_sample, sim_measure!

# Our wire w is bit n-w of the basis index (wire 1 = most significant); Yao's qubit j is bit j-1.
# Hence wire w ↔ Yao qubit n-w+1 and both state vectors share the same amplitude order.
_loc(n, w) = n - w + 1

sim_allocate(::YaoDevice, n::Int) = Yao.zero_state(n)
sim_state(::YaoDevice, reg) = copy(vec(Yao.state(reg)))

function sim_apply!(::YaoDevice, reg, name::Symbol, params::Vector{Float64}, wires::Vector{Int};
                    adjoint::Bool=false, ctrl_wires::Vector{Int}=Int[], ctrl_values::Vector{Bool}=Bool[])
    n = Yao.nqubits(reg)
    U = gate_matrix(name, params, length(wires))
    adjoint && (U = Matrix(U'))
    locs = Tuple(_loc(n, w) for w in reverse(wires))            # our first wire is the most significant bit of U
    if isempty(ctrl_wires)
        Yao.instruct!(reg, U, locs)
    else
        Yao.instruct!(reg, U, locs, Tuple(_loc(n, w) for w in ctrl_wires), Tuple(Int(v) for v in ctrl_values))
    end
    reg
end

_yaopauli(l::Symbol) = l === :X ? Yao.X : l === :Y ? Yao.Y : Yao.Z
function _exact(reg, p::PauliString)
    isempty(p.word) && return real(p.coeff)
    n = Yao.nqubits(reg)
    op = Yao.kron(n, (_loc(n, w) => _yaopauli(l) for (w, l) in p.word)...)
    real(p.coeff * Yao.expect(op, reg))
end
function sim_expval(dev::YaoDevice, reg, p::PauliString)
    (dev.shots == 0 || isempty(p.word)) && return _exact(reg, p)
    real(p.coeff) * sum(sample_pauli_eigenvalues(dev.rng, sim_state(dev, reg), Yao.nqubits(reg), p.word, dev.shots)) / dev.shots
end
sim_expval(dev::YaoDevice, reg, s::PauliSum) = sum(sim_expval(dev, reg, t) for t in s.terms; init=0.0)
function sim_var(dev::YaoDevice, reg, p::PauliString)
    isempty(p.word) && return 0.0
    e = sim_expval(dev, reg, PauliString(1, p.word)) 
    real(p.coeff)^2 * (1 - e^2)
end
sim_probs(dev::YaoDevice, reg, wires::Vector{Int}) =
    dev.shots == 0 ? marginal_probs(abs2.(sim_state(dev, reg)), wires) : sample_probs(dev.rng, sim_state(dev, reg), wires, dev.shots)
function sim_sample(dev::YaoDevice, reg, wires::Vector{Int})
    dev.shots > 0 || throw(ArgumentError("sample() needs shots: use YaoDevice(shots=n)"))
    sample_bits(dev.rng, sim_state(dev, reg), wires, dev.shots)
end
function sim_measure!(dev::YaoDevice, reg, wire::Int, postselect::Int)
    ψ = vec(Yao.state(reg))                                      # shares memory with the register
    n = Yao.nqubits(reg)
    sim_measure!(StateVector{Float64}(n, 0, dev.rng, false), SVState{Float64}(n, ψ), wire, postselect)
end

end # module
