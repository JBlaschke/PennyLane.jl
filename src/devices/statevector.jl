# Reference state-vector simulator (pure Julia).
#
# Wire 1 is the most significant bit of the computational-basis index, matching PennyLane's
# convention (wire 0 there), so `probs`/`state` agree element-wise with PennyLane and Lightning.
# Kernels are specialised for 1- and 2-qubit dense gates and for diagonal gates, run
# multithreaded from `THREAD_MIN_QUBITS` qubits on, and Pauli expectation values are single
# passes over the state. With `shots > 0` every measurement is estimated from samples.

"""
    StateVector([nwires]; shots=0, rng=Random.default_rng(), threads=Threads.nthreads() > 1, T=Float64)

Dense state-vector simulator. Without a wire count it adapts to the program. With `shots > 0`,
expectation values, variances and probabilities are estimated from `shots` samples (drawn
from `rng`) and `sample()` is available. Kernels use all Julia threads for large states when
`threads` is true (start Julia with `-t auto`).
"""
struct StateVector{T<:AbstractFloat} <: AbstractSimulator
    nwires::Union{Nothing,Int}
    shots::Int
    rng::Random.AbstractRNG
    threads::Bool
end
StateVector(n::Integer; shots::Integer=0, rng::Random.AbstractRNG=Random.default_rng(),
            threads::Bool=Threads.nthreads() > 1, T::Type{<:AbstractFloat}=Float64) =
    StateVector{T}(Int(n), Int(shots), rng, threads)
StateVector(; shots::Integer=0, rng::Random.AbstractRNG=Random.default_rng(),
            threads::Bool=Threads.nthreads() > 1, T::Type{<:AbstractFloat}=Float64) =
    StateVector{T}(nothing, Int(shots), rng, threads)
Base.show(io::IO, d::StateVector{T}) where {T} =
    print(io, "StateVector", T === Float64 ? "" : "{$T}", "(", d.nwires === nothing ? "" : d.nwires,
          d.shots == 0 ? "" : (d.nwires === nothing ? "" : "; ") * "shots=$(d.shots)", ")")

mutable struct SVState{T}
    n::Int
    ψ::Vector{Complex{T}}
end

const THREAD_MIN_QUBITS = 14
_threaded(sim::StateVector, n::Int) = sim.threads && n >= THREAD_MIN_QUBITS && Threads.nthreads() > 1

function sim_allocate(::StateVector{T}, n::Int) where {T}
    ψ = zeros(Complex{T}, 1 << n)
    ψ[1] = 1
    SVState{T}(n, ψ)
end
sim_state(::StateVector, st::SVState) = copy(st.ψ)

function sim_apply!(sim::StateVector{T}, st::SVState{T}, name::Symbol, params::Vector{Float64}, wires::Vector{Int};
                    adjoint::Bool=false, ctrl_wires::Vector{Int}=Int[], ctrl_values::Vector{Bool}=Bool[]) where {T}
    U = gate_matrix(name, params, length(wires))
    adjoint && (U = Matrix(U'))
    apply_matrix!(st.ψ, U, wires, st.n, ctrl_wires, ctrl_values; threaded=_threaded(sim, st.n))
    st
end

# ---- index helpers -----------------------------------------------------------------------------
@inline _insert_zero(x::Int, p::Int) = ((x >> p) << (p + 1)) | (x & ((1 << p) - 1))
@inline function _expand(base::Int, fixed::Vector{Int})      # insert zero bits at ascending positions
    idx = base
    @inbounds for p in fixed
        idx = _insert_zero(idx, p)
    end
    idx
end
function _control_masks(ctrl_wires::Vector{Int}, ctrl_values::Vector{Bool}, n::Int)
    cmask = 0
    cval = 0
    for (w, v) in zip(ctrl_wires, ctrl_values)
        p = n - w
        cmask |= 1 << p
        v && (cval |= 1 << p)
    end
    cmask, cval
end

# run `body(k)` for k in 0:len-1, threaded when asked
macro _parfor(threaded, k, len, body)
    quote
        if $(esc(threaded))
            Threads.@threads :static for $(esc(k)) in 0:($(esc(len)))-1
                $(esc(body))
            end
        else
            for $(esc(k)) in 0:($(esc(len)))-1
                $(esc(body))
            end
        end
    end
end

# ---- kernels -----------------------------------------------------------------------------------
function _apply_1q!(ψ::Vector{Complex{T}}, U::AbstractMatrix, wire::Int, n::Int, threaded::Bool) where {T}
    t = n - wire
    s = 1 << t
    u11, u12, u21, u22 = Complex{T}(U[1, 1]), Complex{T}(U[1, 2]), Complex{T}(U[2, 1]), Complex{T}(U[2, 2])
    mask = s - 1
    # flat index space over all pairs: balanced across threads for every wire position
    # (a contiguous hi/lo loop variant measured slower in bench/latency.jl, see PLAN.md)
    @_parfor threaded k (1 << (n - 1)) begin
        @inbounds begin
            i0 = (((k >> t) << (t + 1)) | (k & mask)) + 1
            i1 = i0 + s
            a = ψ[i0]
            b = ψ[i1]
            ψ[i0] = u11 * a + u12 * b
            ψ[i1] = u21 * a + u22 * b
        end
    end
    ψ
end

function _apply_2q!(ψ::Vector{Complex{T}}, U::AbstractMatrix, w1::Int, w2::Int, n::Int, threaded::Bool) where {T}
    s1, s2 = 1 << (n - w1), 1 << (n - w2)               # w1 is the most significant bit of U's index
    plo, phi = minmax(n - w1, n - w2)
    u = ntuple(i -> Complex{T}(U[i]), 16)              # column-major: U[j, l] = u[j + 4(l-1)]
    @_parfor threaded k (1 << (n - 2)) begin
        @inbounds begin
            i00 = _insert_zero(_insert_zero(k, plo), phi) + 1
            i01 = i00 + s2
            i10 = i00 + s1
            i11 = i10 + s2
            a, b, c, d = ψ[i00], ψ[i01], ψ[i10], ψ[i11]
            ψ[i00] = u[1] * a + u[5] * b + u[9] * c + u[13] * d
            ψ[i01] = u[2] * a + u[6] * b + u[10] * c + u[14] * d
            ψ[i10] = u[3] * a + u[7] * b + u[11] * c + u[15] * d
            ψ[i11] = u[4] * a + u[8] * b + u[12] * c + u[16] * d
        end
    end
    ψ
end

function _apply_diag!(ψ::Vector{Complex{T}}, d::Vector{Complex{T}}, wires::Vector{Int}, n::Int,
                      cmask::Int, cval::Int, threaded::Bool) where {T}
    k = length(wires)
    pos = [n - w for w in wires]
    @_parfor threaded i (1 << n) begin
        @inbounds if (i & cmask) == cval
            sub = 0
            for b in 1:k
                ((i >> pos[b]) & 1) == 1 && (sub |= 1 << (k - b))
            end
            ψ[i+1] *= d[sub+1]
        end
    end
    ψ
end

# general k-target, c-control dense kernel
function _apply_general!(ψ::Vector{Complex{T}}, U::AbstractMatrix, wires::Vector{Int}, n::Int,
                         ctrl_wires::Vector{Int}, ctrl_values::Vector{Bool}, threaded::Bool) where {T}
    k = length(wires)
    K = 1 << k
    tpos = [n - w for w in wires]
    cmask, cval = _control_masks(ctrl_wires, ctrl_values, n)
    fixed = sort!([tpos; [n - w for w in ctrl_wires]])
    offs = Vector{Int}(undef, K)
    for j in 0:K-1
        o = 0
        for (b, p) in enumerate(tpos)
            ((j >> (k - b)) & 1) == 1 && (o |= 1 << p)
        end
        offs[j+1] = o
    end
    Uc = Matrix{Complex{T}}(U)
    bufs = [Vector{Complex{T}}(undef, K) for _ in 1:Threads.maxthreadid()]
    @_parfor threaded base (1 << (n - length(fixed))) begin
        @inbounds begin
            idx = _expand(base, fixed) | cval
            buf = bufs[Threads.threadid()]
            for j in 1:K
                buf[j] = ψ[idx+offs[j]+1]
            end
            for j in 1:K
                acc = zero(Complex{T})
                for l in 1:K
                    acc += Uc[j, l] * buf[l]
                end
                ψ[idx+offs[j]+1] = acc
            end
        end
    end
    ψ
end

"""
    apply_matrix!(ψ, U, wires, n, ctrl_wires=[], ctrl_values=[]; threaded=false)

Apply the 2ᵏ×2ᵏ matrix `U` to `wires` of an `n`-qubit state (first wire = most significant bit
of `U`'s index), conditioned on the control wires having the given values.
"""
function apply_matrix!(ψ::Vector{Complex{T}}, U::AbstractMatrix, wires::Vector{Int}, n::Int,
                       ctrl_wires::Vector{Int}=Int[], ctrl_values::Vector{Bool}=Bool[]; threaded::Bool=false) where {T}
    k = length(wires)
    size(U) == (1 << k, 1 << k) || throw(DimensionMismatch("gate matrix is $(size(U)) but acts on $k wire(s)"))
    if isdiag(U)
        cmask, cval = _control_masks(ctrl_wires, ctrl_values, n)
        return _apply_diag!(ψ, Complex{T}[U[i, i] for i in 1:(1<<k)], wires, n, cmask, cval, threaded)
    end
    if isempty(ctrl_wires)
        k == 1 && return _apply_1q!(ψ, U, wires[1], n, threaded)
        k == 2 && return _apply_2q!(ψ, U, wires[1], wires[2], n, threaded)
    end
    _apply_general!(ψ, U, wires, n, ctrl_wires, ctrl_values, threaded)
end

# ---- Pauli words in one pass -----------------------------------------------------------------------
function _pauli_masks(word::Vector{Pair{Int,Symbol}}, n::Int)
    xmask = zmask = 0
    ny = 0
    for (w, l) in word
        bit = 1 << (n - w)
        (l === :X || l === :Y) && (xmask |= bit)
        (l === :Z || l === :Y) && (zmask |= bit)
        l === :Y && (ny += 1)
    end
    xmask, zmask, ny
end
_yphase(ny) = im^ny

"""⟨ψ|P|ψ⟩ for a Pauli word without copying the state."""
function pauli_expval(ψ::Vector{Complex{T}}, word::Vector{Pair{Int,Symbol}}, n::Int, threaded::Bool=false) where {T}
    isempty(word) && return 1.0
    xmask, zmask, ny = _pauli_masks(word, n)
    nt = threaded ? Threads.nthreads() : 1
    partial = zeros(Complex{T}, nt)
    len = 1 << n
    if threaded
        Threads.@threads :static for tid in 1:nt
            lo = div(len * (tid - 1), nt)
            hi = div(len * tid, nt) - 1
            acc = zero(Complex{T})
            @inbounds for i in lo:hi
                j = i ⊻ xmask
                s = isodd(count_ones(j & zmask)) ? -1 : 1
                acc += conj(ψ[i+1]) * ψ[j+1] * s
            end
            partial[tid] = acc
        end
    else
        acc = zero(Complex{T})
        @inbounds for i in 0:len-1
            j = i ⊻ xmask
            s = isodd(count_ones(j & zmask)) ? -1 : 1
            acc += conj(ψ[i+1]) * ψ[j+1] * s
        end
        partial[1] = acc
    end
    real(_yphase(ny) * sum(partial))
end

"""P|ψ⟩ for a Pauli word (new vector)."""
function apply_pauli(ψ::Vector{Complex{T}}, word::Vector{Pair{Int,Symbol}}, n::Int, threaded::Bool=false) where {T}
    isempty(word) && return copy(ψ)
    xmask, zmask, ny = _pauli_masks(word, n)
    ph = Complex{T}(_yphase(ny))
    out = similar(ψ)
    @_parfor threaded i (1 << n) begin
        @inbounds begin
            j = i ⊻ xmask
            out[i+1] = ψ[j+1] * (isodd(count_ones(j & zmask)) ? -ph : ph)
        end
    end
    out
end

"""O|ψ⟩ for an observable with real coefficients."""
function apply_observable(ψ::Vector{Complex{T}}, o::Observable, n::Int, threaded::Bool=false) where {T}
    out = zeros(Complex{T}, length(ψ))
    for t in terms(o)
        out .+= Complex{T}(t.coeff) .* apply_pauli(ψ, t.word, n, threaded)
    end
    out
end

_exact_expval(sim::StateVector, st::SVState, p::PauliString) =
    real(p.coeff) * pauli_expval(st.ψ, p.word, st.n, _threaded(sim, st.n))

# ---- mid-circuit measurement ---------------------------------------------------------------------
function sim_measure!(sim::StateVector, st::SVState, wire::Int, postselect::Int)
    n = st.n
    mask = 1 << (n - wire)
    p1 = 0.0
    @inbounds for i in 0:(1<<n)-1
        (i & mask) != 0 && (p1 += abs2(st.ψ[i+1]))
    end
    outcome = postselect >= 0 ? postselect == 1 : rand(sim.rng) < p1
    pkeep = outcome ? p1 : 1 - p1
    pkeep > 1e-14 || throw(ArgumentError("postselecting outcome $(Int(outcome)) on wire $wire has zero probability"))
    scale = 1 / sqrt(pkeep)
    @inbounds for i in 0:(1<<n)-1
        st.ψ[i+1] = ((i & mask) != 0) == outcome ? st.ψ[i+1] * scale : zero(eltype(st.ψ))
    end
    outcome
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

"""Eigenvalue samples (±1) of a Pauli word: rotate to the Z basis, sample bitstrings, take parities."""
function sample_pauli_eigenvalues(rng::Random.AbstractRNG, ψ::Vector{<:Complex}, n::Int, word::Vector{Pair{Int,Symbol}}, shots::Int)
    ϕ = copy(ψ)
    for (w, l) in word
        if l === :X
            apply_matrix!(ϕ, gate_matrix(:Hadamard, Float64[], 1), [w], n)
        elseif l === :Y
            apply_matrix!(ϕ, Matrix(gate_matrix(:S, Float64[], 1)'), [w], n)
            apply_matrix!(ϕ, gate_matrix(:Hadamard, Float64[], 1), [w], n)
        end
    end
    idx = sample_indices(rng, marginal_probs(abs2.(ϕ), first.(word)), shots)
    Float64[isodd(count_ones(i)) ? -1.0 : 1.0 for i in idx]
end
"""Computational-basis samples (shots × wires matrix of 0/1) of a state on the given wires."""
function sample_bits(rng::Random.AbstractRNG, ψ::Vector{<:Complex}, wires::Vector{Int}, shots::Int)
    k = length(wires)
    idx = sample_indices(rng, marginal_probs(abs2.(ψ), wires), shots)
    out = Matrix{Int}(undef, shots, k)
    for (s, i) in enumerate(idx), b in 1:k
        out[s, b] = (i >> (k - b)) & 1
    end
    out
end
"""Sampled probabilities of a state on the given wires."""
function sample_probs(rng::Random.AbstractRNG, ψ::Vector{<:Complex}, wires::Vector{Int}, shots::Int)
    p = marginal_probs(abs2.(ψ), wires)
    out = zeros(Float64, length(p))
    for i in sample_indices(rng, p, shots)
        out[i+1] += 1 / shots
    end
    out
end
_sample_pauli(sim::StateVector, st::SVState, p::PauliString) = sample_pauli_eigenvalues(sim.rng, st.ψ, st.n, p.word, sim.shots)
_mean(v) = sum(v) / length(v)

function sim_expval(sim::StateVector, st::SVState, p::PauliString)
    (sim.shots == 0 || isempty(p.word)) && return _exact_expval(sim, st, p)
    real(p.coeff) * _mean(_sample_pauli(sim, st, p))
end
sim_expval(sim::StateVector, st::SVState, s::PauliSum) = sum(sim_expval(sim, st, t) for t in s.terms; init=0.0)

function sim_var(sim::StateVector, st::SVState, p::PauliString)
    isempty(p.word) && return 0.0
    if sim.shots == 0
        e = pauli_expval(st.ψ, p.word, st.n, _threaded(sim, st.n))
        return real(p.coeff)^2 * (1 - e^2)
    end
    m = _mean(_sample_pauli(sim, st, p))
    real(p.coeff)^2 * (1 - m^2)
end

sim_probs(sim::StateVector, st::SVState, wires::Vector{Int}) =
    sim.shots == 0 ? marginal_probs(abs2.(st.ψ), wires) : sample_probs(sim.rng, st.ψ, wires, sim.shots)

function sim_sample(sim::StateVector, st::SVState, wires::Vector{Int})
    sim.shots > 0 || throw(ArgumentError("sample() needs shots: use StateVector(shots=n)"))
    sample_bits(sim.rng, st.ψ, wires, sim.shots)
end
