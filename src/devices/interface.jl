# Devices and the simulator interface.
#
# `AbstractDevice`: anything that can `execute(dev, prog, args)`.
# `AbstractSimulator`: a device driven by the generic interpreter below through a small
# interface. Implementing it is how another state-vector (or stabiliser, or tensor-network)
# simulator plugs in — the interpreter, tracer and gradients never see the backend.

abstract type AbstractDevice end
abstract type AbstractSimulator <: AbstractDevice end

"""Number of wires a device is fixed to, or `nothing` if it adapts to the program."""
device_wires(dev::AbstractDevice) = dev.nwires
"""Number of shots a device uses (0 = exact/analytic)."""
device_shots(dev::AbstractDevice) = hasproperty(dev, :shots) ? dev.shots : 0

# ---- simulator interface --------------------------------------------------------------------
"""
    sim_allocate(sim, nqubits) -> state

Create the |0…0⟩ state of `nqubits` qubits (required)."""
function sim_allocate end

"""
    sim_apply!(sim, state, name::Symbol, params::Vector{Float64}, wires::Vector{Int};
               adjoint::Bool, ctrl_wires::Vector{Int}, ctrl_values::Vector{Bool})

Apply the named gate (see `PennyLane.GATES`) in place (required)."""
function sim_apply! end

"""`sim_expval(sim, state, obs::Observable) -> Float64` (required)."""
function sim_expval end

"""`sim_state(sim, state) -> Vector{Complex}`: full state vector (required for default probs/state)."""
function sim_state end

"""`sim_sample(sim, state, wires::Vector{Int}) -> Matrix{Int}` (shots × wires): computational-basis samples (needed for `sample`)."""
sim_sample(sim::AbstractSimulator, state, wires::Vector{Int}) =
    throw(ArgumentError("$(typeof(sim)) cannot sample(); use a device constructed with shots > 0"))

"""`sim_release!(sim, state)`: free backend resources (optional)."""
sim_release!(sim::AbstractSimulator, state) = nothing

"""`sim_var(sim, state, obs)`: variance; default uses the operator algebra ⟨O²⟩ - ⟨O⟩²."""
sim_var(sim::AbstractSimulator, state, o::Observable) = sim_expval(sim, state, o * o) - sim_expval(sim, state, o)^2

"""`sim_probs(sim, state, wires::Vector{Int})`: marginal probabilities in the wire order given."""
sim_probs(sim::AbstractSimulator, state, wires::Vector{Int}) =
    marginal_probs(abs2.(sim_state(sim, state)), wires)

"""Marginal of a full probability vector (wire 1 = most significant bit) onto `wires`."""
function marginal_probs(p::AbstractVector{<:Real}, wires::Vector{Int})
    n = round(Int, log2(length(p)))
    k = length(wires)
    out = zeros(Float64, 1 << k)
    pos = [n - w for w in wires]
    @inbounds for i in 0:length(p)-1
        sub = 0
        for (b, q) in enumerate(pos)
            ((i >> q) & 1) == 1 && (sub |= 1 << (k - b))
        end
        out[sub+1] += p[i+1]
    end
    out
end

"""
    record_observable!(obs, node) -> Bool

Rebuild the observable (as a `PauliString`/`PauliSum`, or the wire list of a computational-basis
measurement) defined by an observable node into `obs[node.obs]`. Returns `false` for other nodes.
"""
function record_observable!(obs::Dict{Int,Any}, node::Node)
    if node isa NamedObsNode
        obs[node.obs] = node.kind === :Identity ? Identity() : pauli(Symbol(last(string(node.kind))), node.wire)
    elseif node isa TensorObsNode
        obs[node.obs] = prod(obs[t] for t in node.terms)
    elseif node isa HamiltonianNode
        obs[node.obs] = sum(c * obs[t] for (c, t) in zip(node.coeffs, node.terms))
    elseif node isa CompBasisNode
        obs[node.obs] = node.wires
    else
        return false
    end
    true
end

# ---- generic interpreter --------------------------------------------------------------------
"""
    execute(dev, prog::Program, args::Vector{Any})

Run a traced program on a device with concrete arguments (`Float64` / `Vector{Float64}`).
"""
function execute(dev::AbstractSimulator, prog::Program, args::Vector{Any})
    st = sim_allocate(dev, prog.nqubits)
    obs = Dict{Int,Any}()
    res = Dict{Int,Any}()
    try
        for node in prog.nodes
            if node isa GateNode
                params = Float64[ceval(p, args) for p in node.params]
                sim_apply!(dev, st, node.name, params, node.wires;
                           adjoint=node.adjoint, ctrl_wires=node.ctrl_wires, ctrl_values=node.ctrl_values)
            elseif record_observable!(obs, node)
            elseif node isa ExpvalNode
                res[node.result] = sim_expval(dev, st, obs[node.obs])
            elseif node isa VarNode
                res[node.result] = sim_var(dev, st, obs[node.obs])
            elseif node isa ProbsNode
                res[node.result] = sim_probs(dev, st, obs[node.obs])
            elseif node isa StateNode
                res[node.result] = sim_state(dev, st)
            elseif node isa SampleNode
                res[node.result] = sim_sample(dev, st, obs[node.obs])
            end
        end
    finally
        sim_release!(dev, st)
    end
    out = Any[res[r] for r in prog.results]
    prog.scalar_return ? out[1] : Tuple(out)
end
