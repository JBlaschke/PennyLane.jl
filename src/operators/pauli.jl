# Operator algebra: Pauli strings and Pauli sums.
#
# Observables are ordinary Julia values. `X(w)`, `Y(w)`, `Z(w)` are Pauli operators on wire `w`
# (an integer) or on a qubit *value* (a `Qubit` obtained inside a `@qnode`). They compose with
# `*`, `+`, `-`, scalar multiplication, `⊗` (= `kron`), `adjoint`, `commutator`, and `exp`.

abstract type Operator end
abstract type Observable <: Operator end

const PAULI_MATRICES = Dict{Symbol,Matrix{ComplexF64}}(
    :I => ComplexF64[1 0; 0 1],
    :X => ComplexF64[0 1; 1 0],
    :Y => ComplexF64[0 -im; im 0],
    :Z => ComplexF64[1 0; 0 -1],
)

# A "site" is either a wire index (Integer) or a qubit value. The tracer adds methods for `Qubit`.
sitewire(w::Integer) = Int(w)
siteref(::Integer) = nothing
sitewire(x) = throw(ArgumentError("expected a wire index or a qubit, got $(typeof(x)). Note: X(q), Y(q), Z(q) build Pauli *operators*; the gates are PauliX(q), PauliY(q), PauliZ(q)."))
siteref(x) = sitewire(x)

"""
    PauliString(coeff, word)

A Pauli word `coeff * P₁ ⊗ P₂ ⊗ …` where `word` maps wires to letters `:X`, `:Y`, `:Z`
(identity is implicit). Usually built with `X(w)`, `Y(w)`, `Z(w)` and the operator algebra.
"""
struct PauliString <: Observable
    coeff::ComplexF64
    word::Vector{Pair{Int,Symbol}}     # sorted by wire; no identity letters
    refs::Dict{Int,Any}                # wire => qubit value the letter was built from (empty when built on wires)
    function PauliString(coeff, word, refs::Dict{Int,Any}=Dict{Int,Any}())
        w = sort!(collect(Pair{Int,Symbol}, word); by=first)
        for i in 2:length(w)
            w[i-1].first == w[i].first && throw(ArgumentError("duplicate wire $(w[i].first) in Pauli word"))
        end
        all(p -> p.second in (:X, :Y, :Z), w) || throw(ArgumentError("Pauli letters must be :X, :Y or :Z"))
        new(ComplexF64(coeff), w, refs)
    end
end

function pauli(letter::Symbol, site)
    w = sitewire(site)
    r = siteref(site)
    refs = r === nothing ? Dict{Int,Any}() : Dict{Int,Any}(w => r)
    PauliString(1, [w => letter], refs)
end

"""`X(site)`, `Y(site)`, `Z(site)`: Pauli operators on a wire (integer) or on a qubit value."""
X(site) = pauli(:X, site)
Y(site) = pauli(:Y, site)
Z(site) = pauli(:Z, site)
"""`Identity()`: the identity operator (the empty Pauli word)."""
Identity(args...) = PauliString(1, Pair{Int,Symbol}[])

"""
    PauliSum(terms)

A real or complex linear combination of Pauli words. Produced by `+`/`-` on observables.
Terms with equal words are merged and zero terms dropped.
"""
struct PauliSum <: Observable
    terms::Vector{PauliString}
    PauliSum(terms::AbstractVector{PauliString}) = new(_simplify(terms))
end

function _simplify(terms::AbstractVector{PauliString})
    acc = Dict{Vector{Pair{Int,Symbol}},PauliString}()
    order = Vector{Vector{Pair{Int,Symbol}}}()
    for t in terms
        if haskey(acc, t.word)
            s = acc[t.word]
            acc[t.word] = PauliString(s.coeff + t.coeff, t.word, _mergerefs(s.refs, t.refs))
        else
            acc[t.word] = t
            push!(order, t.word)
        end
    end
    PauliString[acc[w] for w in order if acc[w].coeff != 0]
end

function _mergerefs(a::Dict{Int,Any}, b::Dict{Int,Any})
    isempty(b) && return copy(a)
    isempty(a) && return copy(b)
    r = copy(a)
    for (w, q) in b
        haskey(r, w) && r[w] !== q &&
            throw(ArgumentError("observable combines two different qubit values on wire $w"))
        r[w] = q
    end
    r
end

# single-site product table: returns (phase, letter or nothing for identity)
function _mulletters(a::Symbol, b::Symbol)
    a === b && return (1.0 + 0.0im, nothing)
    (a, b) === (:X, :Y) && return (im, :Z)
    (a, b) === (:Y, :Z) && return (im, :X)
    (a, b) === (:Z, :X) && return (im, :Y)
    (a, b) === (:Y, :X) && return (-im, :Z)
    (a, b) === (:Z, :Y) && return (-im, :X)
    (a, b) === (:X, :Z) && return (-im, :Y)
    error("invalid Pauli letters $a, $b")
end

terms(p::PauliString) = [p]
terms(s::PauliSum) = s.terms
wires(p::PauliString) = first.(p.word)
wires(s::PauliSum) = sort!(unique!(reduce(vcat, wires.(s.terms); init=Int[])))
nwires(o::Observable) = isempty(wires(o)) ? 0 : maximum(wires(o))

function Base.:*(a::PauliString, b::PauliString)
    coeff = a.coeff * b.coeff
    da, db = Dict(a.word), Dict(b.word)
    word = Pair{Int,Symbol}[]
    for w in sort!(unique!([first.(a.word); first.(b.word)]))
        la, lb = get(da, w, nothing), get(db, w, nothing)
        if la === nothing
            push!(word, w => lb)
        elseif lb === nothing
            push!(word, w => la)
        else
            ph, l = _mulletters(la, lb)
            coeff *= ph
            l === nothing || push!(word, w => l)
        end
    end
    PauliString(coeff, word, _mergerefs(a.refs, b.refs))
end
Base.:*(a::Observable, b::Observable) = PauliSum([ta * tb for ta in terms(a) for tb in terms(b)])
Base.:*(c::Number, p::PauliString) = PauliString(c * p.coeff, p.word, p.refs)
Base.:*(p::PauliString, c::Number) = c * p
Base.:*(c::Number, s::PauliSum) = PauliSum([c * t for t in s.terms])
Base.:*(s::PauliSum, c::Number) = c * s
Base.:/(o::Observable, c::Number) = (1 / c) * o
Base.:-(o::Observable) = (-1) * o
Base.:+(a::Observable, b::Observable) = PauliSum([terms(a); terms(b)])
Base.:-(a::Observable, b::Observable) = a + (-1) * b
Base.:+(a::Observable, c::Number) = a + c * Identity()
Base.:+(c::Number, a::Observable) = a + c
Base.:-(a::Observable, c::Number) = a + (-c) * Identity()
Base.:-(c::Number, a::Observable) = c * Identity() - a
Base.adjoint(p::PauliString) = PauliString(conj(p.coeff), p.word, p.refs)
Base.adjoint(s::PauliSum) = PauliSum(adjoint.(s.terms))

"""`kron(a, b)` / `a ⊗ b`: tensor product of observables on disjoint wires."""
function Base.kron(a::Observable, b::Observable)
    isempty(intersect(wires(a), wires(b))) ||
        throw(ArgumentError("⊗ requires disjoint wires; use * for products acting on the same wires"))
    a * b
end
const ⊗ = kron

commutator(a::Observable, b::Observable) = a * b - b * a
anticommutator(a::Observable, b::Observable) = a * b + b * a

Base.:(==)(a::PauliString, b::PauliString) = a.coeff == b.coeff && a.word == b.word
Base.:(==)(a::PauliSum, b::PauliSum) = _sorted(a.terms) == _sorted(b.terms)
Base.:(==)(a::PauliString, b::PauliSum) = PauliSum([a]) == b
Base.:(==)(a::PauliSum, b::PauliString) = b == a
_sorted(ts) = sort(ts; by=t -> string(t.word))
Base.hash(p::PauliString, h::UInt) = hash(p.coeff, hash(p.word, h))
Base.hash(s::PauliSum, h::UInt) = hash(_sorted(s.terms), h)
function Base.isapprox(a::Observable, b::Observable; kwargs...)
    ta, tb = _sorted(terms(PauliSum(terms(a)))), _sorted(terms(PauliSum(terms(b))))
    length(ta) == length(tb) || return false
    all(x.word == y.word && isapprox(x.coeff, y.coeff; kwargs...) for (x, y) in zip(ta, tb))
end
Base.iszero(s::PauliSum) = isempty(s.terms)
Base.iszero(p::PauliString) = p.coeff == 0

"""
    matrix(o::Observable, nwires = maximum wire)

Dense matrix of an observable on wires `1:nwires` (wire 1 is the most significant bit).
"""
function matrix(p::PauliString, n::Integer=nwires(p))
    n >= nwires(p) || throw(ArgumentError("observable acts on wire $(nwires(p)) > $n"))
    d = Dict(p.word)
    M = fill(p.coeff, 1, 1)
    for w in 1:n
        M = kron(M, PAULI_MATRICES[get(d, w, :I)])
    end
    M
end
matrix(s::PauliSum, n::Integer=nwires(s)) = isempty(s.terms) ? zeros(ComplexF64, 1 << n, 1 << n) :
                                              sum(matrix(t, n) for t in s.terms)

# --- printing -------------------------------------------------------------------------------
function _fmtcoeff(c::ComplexF64)
    if imag(c) == 0
        r = real(c)
        return r == round(r) && abs(r) < 1e15 ? string(Int(r)) : string(r)
    end
    "(" * string(c) * ")"
end
function Base.show(io::IO, p::PauliString)
    letters = isempty(p.word) ? "Identity()" : join(("$(l)($(w))" for (w, l) in p.word), " * ")
    if p.coeff == 1
        print(io, letters)
    elseif p.coeff == -1
        print(io, "-", letters)
    else
        print(io, _fmtcoeff(p.coeff), " * ", letters)
    end
end
function Base.show(io::IO, s::PauliSum)
    isempty(s.terms) && return print(io, "0 * Identity()")
    for (i, t) in enumerate(s.terms)
        if i > 1
            if imag(t.coeff) == 0 && real(t.coeff) < 0
                print(io, " - ", -t)
                continue
            end
            print(io, " + ")
        end
        print(io, t)
    end
end
