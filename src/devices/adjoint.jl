# Adjoint differentiation on the reference simulator (Jones & Gacon 2020): one forward pass, then
# a backward sweep undoing gates on the state and on λ = O|ψ⟩, with ∂f/∂θ = Im⟨λ|G U ψ⟩ for gates
# U(θ) = exp(-iθ/2 G). Memory O(2ⁿ), cost O(#gates · 2ⁿ) independent of the number of parameters.

const _P1 = ComplexF64[0 0; 0 1]
function _excitation_generator(dim::Int, i::Int, j::Int)
    G = zeros(ComplexF64, dim, dim)
    G[i, j] = -im
    G[j, i] = im
    G
end
"""Hermitian generator G of a single-parameter gate, U(θ) = exp(-iθ/2 G)."""
const GENERATORS = Dict{Symbol,Function}(
    :RX => nq -> PAULI_MATRICES[:X],
    :RY => nq -> PAULI_MATRICES[:Y],
    :RZ => nq -> PAULI_MATRICES[:Z],
    :PhaseShift => nq -> ComplexF64[0 0; 0 -2],
    :ControlledPhaseShift => nq -> Matrix{ComplexF64}(Diagonal([0, 0, 0, -2])),
    :CRX => nq -> kron(_P1, PAULI_MATRICES[:X]),
    :CRY => nq -> kron(_P1, PAULI_MATRICES[:Y]),
    :CRZ => nq -> kron(_P1, PAULI_MATRICES[:Z]),
    :IsingXX => nq -> kron(PAULI_MATRICES[:X], PAULI_MATRICES[:X]),
    :IsingYY => nq -> kron(PAULI_MATRICES[:Y], PAULI_MATRICES[:Y]),
    :IsingZZ => nq -> kron(PAULI_MATRICES[:Z], PAULI_MATRICES[:Z]),
    :MultiRZ => nq -> Matrix{ComplexF64}(Diagonal([isodd(count_ones(i)) ? -1.0 : 1.0 for i in 0:(1<<nq)-1])),
    :SingleExcitation => nq -> _excitation_generator(4, 2, 3),
    :DoubleExcitation => nq -> _excitation_generator(16, 4, 13),
)

# μ = (P_ctrl ⊗ G) ψ: apply G to the targets where the controls match, zero elsewhere
function _apply_generator(ψ::Vector{Complex{T}}, G::AbstractMatrix, node::GateNode, n::Int, threaded::Bool) where {T}
    μ = copy(ψ)
    apply_matrix!(μ, G, node.wires, n, node.ctrl_wires, node.ctrl_values; threaded=threaded)
    if !isempty(node.ctrl_wires)
        cmask, cval = _control_masks(node.ctrl_wires, node.ctrl_values, n)
        @inbounds for i in 0:(1<<n)-1
            (i & cmask) == cval || (μ[i+1] = 0)
        end
    end
    μ
end

function _check_adjoint_program(prog::Program, what)
    _check_scalar_expval(prog, what)
    (has_control_flow(prog) || has_mcm(prog)) &&
        throw(ArgumentError("$what needs a static circuit (no @trace control flow or mid-circuit measurements)"))
end

"""Derivatives of a single-expval program with respect to its arguments, by the adjoint method."""
function adjoint_gradient(sim::StateVector{T}, prog::Program, args::Vector{Any}) where {T}
    _check_adjoint_program(prog, "the adjoint gradient")
    sim.shots == 0 || throw(ArgumentError("the adjoint gradient needs an analytic device (shots = 0)"))
    n = prog.nqubits
    threaded = _threaded(sim, n)
    st = sim_allocate(sim, n)
    obs = Dict{Int,Any}()
    gates = GateNode[]
    O = nothing
    for node in prog.nodes
        if node isa GateNode
            params = Float64[ceval(p, args) for p in node.params]
            sim_apply!(sim, st, node.name, params, node.wires; adjoint=node.adjoint, ctrl_wires=node.ctrl_wires, ctrl_values=node.ctrl_values)
            push!(gates, node)
        elseif record_observable!(obs, node)
        elseif node isa ExpvalNode
            O = obs[node.obs]
        end
    end
    ψ = st.ψ
    λ = apply_observable(ψ, O, n, threaded)
    dgate = zeros(Float64, length(gates))          # ∂f/∂(parameter of gate k)
    for k in length(gates):-1:1
        g = gates[k]
        params = Float64[ceval(p, args) for p in g.params]
        if !isempty(g.params) && any(hasarg, g.params)
            length(g.params) == 1 || throw(ArgumentError("adjoint differentiation of multi-parameter gate $(g.name) is not supported"))
            gen = get(GENERATORS, g.name, nothing)
            gen === nothing && throw(ArgumentError("no generator known for $(g.name); use method=:parameter_shift"))
            μ = _apply_generator(ψ, gen(length(g.wires)), g, n, threaded)
            d = imag(dot(λ, μ))
            dgate[k] = g.adjoint ? -d : d
        end
        U = Matrix(gate_matrix(g.name, params, length(g.wires))')      # undo the gate
        g.adjoint && (U = Matrix(U'))
        apply_matrix!(ψ, U, g.wires, n, g.ctrl_wires, g.ctrl_values; threaded=threaded)
        apply_matrix!(λ, U, g.wires, n, g.ctrl_wires, g.ctrl_values; threaded=threaded)
    end
    grads = _zero_grads(args)
    slots = _arg_slots(args)
    for (k, g) in enumerate(gates)
        (isempty(g.params) || dgate[k] == 0) && continue
        for (a, i) in slots
            d = cdual(g.params[1], args, a, i)[2]
            d == 0 && continue
            i == 0 ? (grads[a] += dgate[k] * d) : (grads[a][i] += dgate[k] * d)
        end
    end
    _pack(grads)
end
adjoint_gradient(dev::AbstractDevice, prog::Program, args::Vector{Any}) =
    throw(ArgumentError("the :adjoint gradient is available on StateVector, LightningDevice and CatalystDevice, not on $(typeof(dev))"))
