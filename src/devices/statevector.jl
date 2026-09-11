# Reference state-vector simulator (pure Julia).
#
# Wire 1 is the most significant bit of the computational-basis index, matching PennyLane's
# convention (wire 0 there), so `probs`/`state` agree element-wise with PennyLane and Lightning.
# With `shots > 0` every measurement is estimated from samples, like on hardware.

"""
    StateVector([nwires]; shots=0, rng=Random.default_rng(), T=Float64)

Dense state-vector simulator. Without a wire count it adapts to the program. With `shots > 0`,
expectation values, variances and probabilities are estimated from `shots` samples (drawn
from `rng`) and `sample()` is available.
"""
struct StateVector{T<:AbstractFloat} <: AbstractSimulator
    nwires::Union{Nothing,Int}
    shots::Int
    rng::Random.AbstractRNG
end
StateVector(n::Integer; shots::Integer=0, rng::Random.AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float64) =
    StateVector{T}(Int(n), Int(shots), rng)
StateVector(; shots::Integer=0, rng::Random.AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float64) =
    StateVector{T}(nothing, Int(shots), rng)
Base.show(io::IO, d::StateVector{T}) where {T} =
    print(io, "StateVector", T === Float64 ? "" : "{$T}", "(", d.nwires === nothing ? "" : d.nwires,
          d.shots == 0 ? "" : (d.nwires === nothing ? "" : "; ") * "shots=$(d.shots)", ")")

mutable struct SVState{T}
    n::Int
    ψ::Vector{Complex{T}}
end

function sim_allocate(::StateVector{T}, n::Int) where {T}
    ψ = zeros(Complex{T}, 1 << n)
    ψ[1] = 1
    SVState{T}(n, ψ)
end
sim_state(::StateVector, st::SVState) = copy(st.ψ)

function sim_apply!(::StateVector{T}, st::SVState{T}, name::Symbol, params::Vector{Float64}, wires::Vector{Int};
                    adjoint::Bool=false, ctrl_wires::Vector{Int}=Int[], ctrl_values::Vector{Bool}=Bool[]) where {T}
    U = gate_matrix(name, params, length(wires))
    adjoint && (U = Matrix(U'))
    apply_matrix!(st.ψ, U, wires, st.n, ctrl_wires, ctrl_values)
    st
end

"""
    apply_matrix!(ψ, U, wires, n, ctrl_wires=[], ctrl_values=[])

Apply the 2ᵏ×2ᵏ matrix `U` to `wires` of an `n`-qubit state (first wire = most significant bit
of `U`'s index), conditioned on the control wires having the given values.
"""
function apply_matrix!(ψ::Vector{Complex{T}}, U::AbstractMatrix, wires::Vector{Int}, n::Int,
                       ctrl_wires::Vector{Int}=Int[], ctrl_values::Vector{Bool}=Bool[]) where {T}
    k = length(wires)
    K = 1 << k
    size(U) == (K, K) || throw(DimensionMismatch("gate matrix is $(size(U)) but acts on $k wire(s)"))
    tpos = [n - w for w in wires]
    tmask = 0
    for p in tpos
        tmask |= 1 << p
    end
    cmask = 0
    cval = 0
    for (w, v) in zip(ctrl_wires, ctrl_values)
        p = n - w
        cmask |= 1 << p
        v && (cval |= 1 << p)
    end
    offs = Vector{Int}(undef, K)
    for j in 0:K-1
        o = 0
        for (b, p) in enumerate(tpos)
            ((j >> (k - b)) & 1) == 1 && (o |= 1 << p)
        end
        offs[j+1] = o
    end
    Uc = Matrix{Complex{T}}(U)
    buf = Vector{Complex{T}}(undef, K)
    @inbounds for base in 0:(1<<n)-1
        (base & tmask) == 0 || continue
        (base & cmask) == cval || continue
        for j in 1:K
            buf[j] = ψ[base+offs[j]+1]
        end
        for j in 1:K
            acc = zero(Complex{T})
            for l in 1:K
                acc += Uc[j, l] * buf[l]
            end
            ψ[base+offs[j]+1] = acc
        end
    end
    ψ
end

# ---- exact quantities ------------------------------------------------------------------------
function _exact_expval(st::SVState, p::PauliString)
    isempty(p.word) && return real(p.coeff)
    ϕ = copy(st.ψ)
    for (w, l) in p.word
        apply_matrix!(ϕ, PAULI_MATRICES[l], [w], st.n)
    end
    real(p.coeff * dot(st.ψ, ϕ))
end

# ---- sampling ----------------------------------------------------------------------------------
"""Draw `shots` indices (0-based) from the probability vector `p`."""
function sample_indices(rng::Random.AbstractRNG, p::AbstractVector{<:Real}, shots::Int)
    cdf = cumsum(p)
    n = length(p)
    idx = Vector{Int}(undef, shots)
    for k in 1:shots
        idx[k] = min(searchsortedfirst(cdf, rand(rng) * cdf[end]), n) - 1
    end
    idx
end

# eigenvalue samples (±1) of a Pauli word: rotate to the Z basis, sample bitstrings, take parities
function _sample_pauli(sim::StateVector, st::SVState, p::PauliString)
    ϕ = copy(st.ψ)
    for (w, l) in p.word
        if l === :X
            apply_matrix!(ϕ, gate_matrix(:Hadamard, Float64[], 1), [w], st.n)
        elseif l === :Y
            apply_matrix!(ϕ, Matrix(gate_matrix(:S, Float64[], 1)'), [w], st.n)
            apply_matrix!(ϕ, gate_matrix(:Hadamard, Float64[], 1), [w], st.n)
        end
    end
    ws = first.(p.word)
    idx = sample_indices(sim.rng, marginal_probs(abs2.(ϕ), ws), sim.shots)
    Float64[isodd(count_ones(i)) ? -1.0 : 1.0 for i in idx]
end
_mean(v) = sum(v) / length(v)

function sim_expval(sim::StateVector, st::SVState, p::PauliString)
    (sim.shots == 0 || isempty(p.word)) && return _exact_expval(st, p)
    real(p.coeff) * _mean(_sample_pauli(sim, st, p))
end
sim_expval(sim::StateVector, st::SVState, s::PauliSum) = sum(sim_expval(sim, st, t) for t in s.terms; init=0.0)

function sim_var(sim::StateVector, st::SVState, p::PauliString)
    sim.shots == 0 && return _exact_expval(st, p * p) - _exact_expval(st, p)^2
    isempty(p.word) && return 0.0
    m = _mean(_sample_pauli(sim, st, p))
    real(p.coeff)^2 * (1 - m^2)
end

function sim_probs(sim::StateVector, st::SVState, wires::Vector{Int})
    p = marginal_probs(abs2.(st.ψ), wires)
    sim.shots == 0 && return p
    out = zeros(Float64, length(p))
    for i in sample_indices(sim.rng, p, sim.shots)
        out[i+1] += 1 / sim.shots
    end
    out
end

function sim_sample(sim::StateVector, st::SVState, wires::Vector{Int})
    sim.shots > 0 || throw(ArgumentError("sample() needs shots: use StateVector(shots=n)"))
    k = length(wires)
    idx = sample_indices(sim.rng, marginal_probs(abs2.(st.ψ), wires), sim.shots)
    out = Matrix{Int}(undef, sim.shots, k)
    for (s, i) in enumerate(idx), b in 1:k
        out[s, b] = (i >> (k - b)) & 1
    end
    out
end
