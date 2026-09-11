# Tracing: turns a Julia function into a Program.
#
# Qubits are SSA values (`Qubit`). A gate consumes its input values and returns fresh ones;
# touching a consumed value is a loud error. Integer wires are accepted wherever a qubit is and
# resolve to the wire's current value, so PennyLane-style circuits work unchanged.
# Classical parameters are `TracedReal`s carrying a small expression tree.

mutable struct Builder
    prog::Program
    reg::Int                       # current register value id (0 = not allocated yet)
    current::Dict{Int,Any}         # wire => live Qubit
    measured::Bool
    device_wires::Union{Nothing,Int}
    results::Dict{Int,ResultSpec}
end

mutable struct Qubit
    const builder::Builder
    const wire::Int
    const id::Int
    consumed::Bool
    consumer::String
end
Base.show(io::IO, q::Qubit) = print(io, "Qubit(wire=", q.wire, q.consumed ? ", consumed by $(q.consumer))" : ")")
sitewire(q::Qubit) = q.wire
siteref(q::Qubit) = q

struct QubitConsumedError <: Exception
    msg::String
end
struct TraceError <: Exception
    msg::String
end
Base.showerror(io::IO, e::QubitConsumedError) = print(io, "QubitConsumedError: ", e.msg)
Base.showerror(io::IO, e::TraceError) = print(io, "TraceError: ", e.msg)

# ---- traced classical values -----------------------------------------------------------------
"""A real-valued function argument (or expression of arguments) seen during tracing."""
struct TracedReal <: Real
    expr::CExpr
end
struct TracedVector <: AbstractVector{TracedReal}
    arg::Int
    len::Int
end
Base.size(v::TracedVector) = (v.len,)
Base.getindex(v::TracedVector, i::Int) = (checkbounds(v, i); TracedReal(CArg(v.arg, i)))
Base.show(io::IO, x::TracedReal) = print(io, "TracedReal(", _fmt(x.expr), ")")

toexpr(x::TracedReal) = x.expr
toexpr(x::Real) = CConst(Float64(x))
toexpr(x) = throw(ArgumentError("gate parameters must be real numbers or traced values, got $(typeof(x))"))

_fold(op, a::CExpr, b::CExpr) = (a isa CConst && b isa CConst) ? CConst(ceval(CCall(op, [a, b]), Any[])) : CCall(op, [a, b])
_fold(op, a::CExpr) = a isa CConst ? CConst(ceval(CCall(op, [a]), Any[])) : CCall(op, [a])
for (f, op) in ((:+, :add), (:-, :sub), (:*, :mul), (:/, :div))
    @eval begin
        Base.$f(a::TracedReal, b::TracedReal) = TracedReal(_fold($(QuoteNode(op)), a.expr, b.expr))
        Base.$f(a::TracedReal, b::Real) = TracedReal(_fold($(QuoteNode(op)), a.expr, toexpr(b)))
        Base.$f(a::Real, b::TracedReal) = TracedReal(_fold($(QuoteNode(op)), toexpr(a), b.expr))
    end
end
Base.:-(a::TracedReal) = TracedReal(_fold(:neg, a.expr))
Base.:+(a::TracedReal) = a
for (f, op) in ((:sin, :sin), (:cos, :cos), (:sqrt, :sqrt), (:exp, :exp))
    @eval Base.$f(a::TracedReal) = TracedReal(_fold($(QuoteNode(op)), a.expr))
end
function Base.:^(a::TracedReal, n::Integer)
    n >= 1 || throw(ArgumentError("only positive integer powers of traced values are supported"))
    r = a
    for _ in 2:n
        r = r * a
    end
    r
end
Base.promote_rule(::Type{TracedReal}, ::Type{<:Real}) = TracedReal
Base.convert(::Type{TracedReal}, x::Real) = TracedReal(toexpr(x))
Base.convert(::Type{TracedReal}, x::TracedReal) = x
Base.convert(::Type{T}, x::TracedReal) where {T<:Number} =
    throw(ArgumentError("a traced parameter cannot be converted to $T: its value is only known when the circuit runs. Keep it as a gate parameter, or compute the quantity outside the @qnode."))
(::Type{T})(x::TracedReal) where {T<:AbstractFloat} = convert(T, x)
Base.zero(::TracedReal) = TracedReal(CConst(0.0))
Base.one(::TracedReal) = TracedReal(CConst(1.0))
const _BRANCH_MSG = "cannot branch on a traced parameter; classical control flow on parameters is not yet supported"
Base.isless(::TracedReal, ::TracedReal) = throw(ArgumentError(_BRANCH_MSG))
Base.isless(::TracedReal, ::Real) = throw(ArgumentError(_BRANCH_MSG))
Base.isless(::Real, ::TracedReal) = throw(ArgumentError(_BRANCH_MSG))
Base.:<(::TracedReal, ::TracedReal) = throw(ArgumentError(_BRANCH_MSG))
Base.:<=(::TracedReal, ::TracedReal) = throw(ArgumentError(_BRANCH_MSG))
Base.:(==)(a::TracedReal, b::TracedReal) = a.expr == b.expr

# ---- builder context -------------------------------------------------------------------------
const BUILDER_KEY = :__pennylane_builder__
current_builder() = get(task_local_storage(), BUILDER_KEY, nothing)
function with_builder(f, b::Builder)
    tls = task_local_storage()
    old = get(tls, BUILDER_KEY, nothing)
    tls[BUILDER_KEY] = b
    try
        f()
    finally
        old === nothing ? delete!(tls, BUILDER_KEY) : (tls[BUILDER_KEY] = old)
    end
end
function require_builder()
    b = current_builder()
    b === nothing && throw(TraceError("quantum operations can only be used inside a @qnode function"))
    b
end
newvalue!(b::Builder) = (b.prog.nvalues += 1)

function ensure_register!(b::Builder, n::Union{Nothing,Int}=nothing)
    if b.reg != 0
        n === nothing || n == b.prog.nqubits ||
            throw(TraceError("qubits($n) requested but this program already has $(b.prog.nqubits) qubits"))
        return b
    end
    nq = n === nothing ? b.device_wires : n
    nq === nothing && throw(TraceError("cannot infer the number of qubits: call qubits(n) first, or give the device a wire count such as StateVector(2)"))
    nq >= 1 || throw(TraceError("a program needs at least one qubit"))
    b.device_wires === nothing || nq <= b.device_wires ||
        throw(TraceError("qubits($nq) exceeds the device's $(b.device_wires) wires"))
    b.prog.nqubits = nq
    b.reg = newvalue!(b)
    push!(b.prog.nodes, AllocNode(b.reg, nq))
    for w in 1:nq
        id = newvalue!(b)
        push!(b.prog.nodes, ExtractNode(id, b.reg, w))
        b.current[w] = Qubit(b, w, id, false, "")
    end
    b
end

"""
    qubits(n) -> Vector{Qubit}

The qubit values on wires `1:n` (allocating the register on first use)."""
function qubits(n::Integer)
    b = require_builder()
    ensure_register!(b, Int(n))
    Qubit[b.current[w] for w in 1:n]
end
"""`qubit(w=1)`: the current qubit value on wire `w`."""
qubit(w::Integer=1) = resolve(require_builder(), w)

function resolve(b::Builder, w::Integer)
    ensure_register!(b)
    1 <= w <= b.prog.nqubits || throw(TraceError("wire $w is out of range 1:$(b.prog.nqubits)"))
    b.current[Int(w)]::Qubit
end
function resolve(b::Builder, q::Qubit)
    q.builder === b || throw(TraceError("this qubit belongs to a different @qnode trace"))
    q
end
resolve(::Builder, x) = throw(ArgumentError("expected a qubit or a wire index, got $(typeof(x)). Note: X(q), Y(q), Z(q) are Pauli *operators*; the gates are PauliX(q), PauliY(q), PauliZ(q)."))

function consume!(q::Qubit, by::String)
    q.consumed && throw(QubitConsumedError("the qubit value on wire $(q.wire) was already consumed by $(q.consumer); use the value returned by that operation instead (qubits have value semantics)"))
    q.consumed = true
    q.consumer = by
    q.id
end
function fresh!(b::Builder, wire::Int)
    q = Qubit(b, wire, newvalue!(b), false, "")
    b.current[wire] = q
    q
end
function builder_for(sites)
    for s in sites
        s isa Qubit && return s.builder
    end
    require_builder()
end

# ---- gate application ------------------------------------------------------------------------
# Rot/CRot are kept as gates (matrices, PennyLane names) but applied as RZ·RY·RZ so that every
# backend, including compiled adjoint differentiation, sees only generator-carrying rotations.
function _decompose_rot(g::Gate, sites...)
    ϕ, θ, ω = g.params
    f = g.name === :Rot ? (RZ, RY, RZ) : (CRZ, CRY, CRZ)
    seq = g.adjoint ? ((f[3], ω), (f[2], θ), (f[1], ϕ)) : ((f[1], ϕ), (f[2], θ), (f[3], ω))
    qs = sites
    for (F, p) in seq
        gate = g.adjoint ? F(p)' : F(p)
        qs = apply(gate, qs...)
        qs isa Tuple || (qs = (qs,))
    end
    length(qs) == 1 ? qs[1] : qs
end

function apply(g::Gate, sites...)
    g.name in (:Rot, :CRot) && return _decompose_rot(g, sites...)
    b = builder_for(sites)
    b.measured && throw(TraceError("cannot apply $(g.name) after a terminal measurement"))
    def = GATES[g.name]
    nq = length(sites)
    (def.nqubits == -1 ? nq >= 1 : nq == def.nqubits) ||
        throw(ArgumentError("$(g.name) acts on $(def.nqubits) qubit(s), got $nq"))
    ins = Qubit[resolve(b, s) for s in sites]
    length(unique(q.wire for q in ins)) == nq || throw(TraceError("$(g.name) applied to the same wire twice"))
    ids = [consume!(q, string(g.name)) for q in ins]
    outs = [fresh!(b, q.wire) for q in ins]
    push!(b.prog.nodes, GateNode(g.name, CExpr[toexpr(p) for p in g.params], ids, [o.id for o in outs],
                                 [q.wire for q in ins], g.adjoint, Int[], Int[], Int[], Bool[]))
    nq == 1 ? outs[1] : Tuple(outs)
end
(g::Gate)(sites...) = apply(g, sites...)

function apply(c::Controlled, sites...)
    if c.gate.name in (:Rot, :CRot)
        ϕ, θ, ω = c.gate.params
        f = c.gate.name === :Rot ? (RZ, RY, RZ) : (CRZ, CRY, CRZ)
        seq = c.gate.adjoint ? ((f[3], ω), (f[2], θ), (f[1], ϕ)) : ((f[1], ϕ), (f[2], θ), (f[3], ω))
        ctrls = c.controls
        qs = sites
        for (F, p) in seq
            out = apply(Controlled(c.gate.adjoint ? F(p)' : F(p), collect(Any, ctrls), c.values), qs...)
            ctrls = out[1:length(ctrls)]
            qs = out[length(ctrls)+1:end]
        end
        return Tuple([ctrls...; qs...])
    end
    b = builder_for((c.controls..., sites...))
    b.measured && throw(TraceError("cannot apply $(c.gate.name) after a terminal measurement"))
    def = GATES[c.gate.name]
    nq = length(sites)
    (def.nqubits == -1 ? nq >= 1 : nq == def.nqubits) ||
        throw(ArgumentError("$(c.gate.name) acts on $(def.nqubits) qubit(s), got $nq"))
    ctrls = Qubit[resolve(b, s) for s in c.controls]
    ins = Qubit[resolve(b, s) for s in sites]
    allw = vcat([q.wire for q in ctrls], [q.wire for q in ins])
    length(unique(allw)) == length(allw) || throw(TraceError("controlled $(c.gate.name) uses a wire twice"))
    cids = [consume!(q, "ctrl($(c.gate.name))") for q in ctrls]
    ids = [consume!(q, "ctrl($(c.gate.name))") for q in ins]
    couts = [fresh!(b, q.wire) for q in ctrls]
    outs = [fresh!(b, q.wire) for q in ins]
    push!(b.prog.nodes, GateNode(c.gate.name, CExpr[toexpr(p) for p in c.gate.params], ids, [o.id for o in outs],
                                 [q.wire for q in ins], c.gate.adjoint, cids, [o.id for o in couts],
                                 [q.wire for q in ctrls], copy(c.values)))
    Tuple([couts; outs])
end
(c::Controlled)(sites...) = apply(c, sites...)

"""`PauliRot(θ, word, q...)`: apply exp(-iθ/2 P) for the Pauli word (basis change + MultiRZ)."""
function PauliRot(theta, word::AbstractString, sites...)
    letters = [Symbol(c) for c in uppercase(String(word))]
    length(letters) == length(sites) || throw(ArgumentError("PauliRot word \"$word\" has $(length(letters)) letters for $(length(sites)) qubits"))
    all(l -> l in (:I, :X, :Y, :Z), letters) || throw(ArgumentError("PauliRot word must consist of I, X, Y, Z"))
    b = builder_for(sites)
    qs = Any[resolve(b, s) for s in sites]
    active = [i for i in eachindex(letters) if letters[i] !== :I]
    isempty(active) && return length(qs) == 1 ? qs[1] : Tuple(qs)     # identity word: global phase only
    for i in active
        letters[i] === :X && (qs[i] = Hadamard(qs[i]))
        letters[i] === :Y && (qs[i] = RX(π / 2, qs[i]))
    end
    outs = MultiRZ(theta, qs[active]...)
    outs = outs isa Tuple ? collect(outs) : [outs]
    for (k, i) in enumerate(active)
        qs[i] = outs[k]
    end
    for i in active
        letters[i] === :X && (qs[i] = Hadamard(qs[i]))
        letters[i] === :Y && (qs[i] = RX(-π / 2, qs[i]))
    end
    length(qs) == 1 ? qs[1] : Tuple(qs)
end
(g::PauliRotGate)(sites...) = PauliRot(g.theta, g.word, sites...)
(u::BoundPauliRot)() = PauliRot(u.theta, u.word, u.wires...)
function (u::BoundPauliRot)(sites...)
    length(sites) == length(u.wires) || throw(ArgumentError("expected $(length(u.wires)) qubits"))
    PauliRot(u.theta, u.word, sites...)
end

# ---- measurements ----------------------------------------------------------------------------
struct MeasurementToken
    builder::Builder
    id::Int
end
Base.show(io::IO, t::MeasurementToken) = print(io, "MeasurementToken(%m", t.id, ")")

# observable for one Pauli word (coefficient handled by the caller)
function word_value!(b::Builder, p::PauliString)
    ensure_register!(b)
    if isempty(p.word)
        q = resolve(b, 1)
        id = newvalue!(b)
        push!(b.prog.nodes, NamedObsNode(id, q.id, 1, :Identity))
        return id
    end
    ids = Int[]
    for (w, l) in p.word
        q = haskey(p.refs, w) ? p.refs[w] : resolve(b, w)
        q isa Qubit || throw(TraceError("observable on wire $w does not refer to a qubit"))
        q.builder === b || throw(TraceError("observable refers to a qubit from a different @qnode trace"))
        (q.consumed || b.current[w] !== q) &&
            throw(QubitConsumedError("observable $(l)($(w)) refers to a qubit value that was later consumed by $(q.consumer); build observables from the qubit values you hold at measurement time"))
        id = newvalue!(b)
        push!(b.prog.nodes, NamedObsNode(id, q.id, w, Symbol(:Pauli, l)))
        push!(ids, id)
    end
    length(ids) == 1 && return ids[1]
    id = newvalue!(b)
    push!(b.prog.nodes, TensorObsNode(id, ids))
    id
end
function obs_value!(b::Builder, o::Observable)
    ts = terms(o)
    isempty(ts) && throw(ArgumentError("cannot measure the zero observable"))
    length(ts) == 1 && ts[1].coeff == 1 && return word_value!(b, ts[1])
    coeffs = Float64[]
    for t in ts
        abs(imag(t.coeff)) <= 1e-12 * max(1.0, abs(t.coeff)) ||
            throw(ArgumentError("observable coefficients must be real, got $(t.coeff)"))
        push!(coeffs, real(t.coeff))
    end
    ids = [word_value!(b, t) for t in ts]
    id = newvalue!(b)
    push!(b.prog.nodes, HamiltonianNode(id, coeffs, ids))
    id
end

function _measurement!(b::Builder, spec::ResultSpec, mk)
    id = newvalue!(b)
    push!(b.prog.nodes, mk(id))
    b.measured = true
    b.results[id] = spec
    MeasurementToken(b, id)
end

"""`expval(obs)`: expectation value of an observable (terminal measurement)."""
function expval(o::Observable)
    b = require_builder()
    obs = obs_value!(b, o)
    _measurement!(b, ResultSpec(:expval, 0), id -> ExpvalNode(id, obs))
end
"""`var(obs)`: variance of an observable (terminal measurement)."""
function var(o::Observable)
    b = require_builder()
    obs = obs_value!(b, o)
    _measurement!(b, ResultSpec(:var, 0), id -> VarNode(id, obs))
end
"""`probs(sites...)`: computational-basis probabilities of the given wires/qubits (all if none)."""
function probs(sites...)
    b = builder_for(sites)
    ensure_register!(b)
    qs = isempty(sites) ? Qubit[b.current[w] for w in 1:b.prog.nqubits] : Qubit[resolve(b, s) for s in sites]
    for q in qs
        (q.consumed || b.current[q.wire] !== q) &&
            throw(QubitConsumedError("probs refers to a qubit value on wire $(q.wire) that was consumed by $(q.consumer)"))
    end
    obs = newvalue!(b)
    push!(b.prog.nodes, CompBasisNode(obs, [q.id for q in qs], [q.wire for q in qs]))
    _measurement!(b, ResultSpec(:probs, 1 << length(qs)), id -> ProbsNode(id, obs, length(qs)))
end
"""`sample(sites...)`: computational-basis samples of the given wires/qubits (all if none) as a
`shots × wires` matrix of 0/1; needs a device with shots."""
function sample(sites...)
    b = builder_for(sites)
    ensure_register!(b)
    qs = isempty(sites) ? Qubit[b.current[w] for w in 1:b.prog.nqubits] : Qubit[resolve(b, s) for s in sites]
    for q in qs
        (q.consumed || b.current[q.wire] !== q) &&
            throw(QubitConsumedError("sample refers to a qubit value on wire $(q.wire) that was consumed by $(q.consumer)"))
    end
    obs = newvalue!(b)
    push!(b.prog.nodes, CompBasisNode(obs, [q.id for q in qs], [q.wire for q in qs]))
    _measurement!(b, ResultSpec(:sample, length(qs)), id -> SampleNode(id, obs, length(qs)))
end

"""`BasisState(bits, q...)`: prepare |bits⟩ on the given qubits (applies `PauliX` where `bit == 1`)."""
function BasisState(bits::AbstractVector{<:Integer}, sites...)
    length(bits) == length(sites) || throw(ArgumentError("BasisState: $(length(bits)) bits for $(length(sites)) qubits"))
    all(x -> x in (0, 1), bits) || throw(ArgumentError("BasisState bits must be 0 or 1"))
    b = builder_for(sites)
    qs = Any[resolve(b, s) for s in sites]
    for (i, bit) in enumerate(bits)
        bit == 1 && (qs[i] = PauliX(qs[i]))
    end
    length(qs) == 1 ? qs[1] : Tuple(qs)
end

"""`state()`: the full state vector (simulators only)."""
function state()
    b = require_builder()
    ensure_register!(b)
    qs = Qubit[b.current[w] for w in 1:b.prog.nqubits]
    obs = newvalue!(b)
    push!(b.prog.nodes, CompBasisNode(obs, [q.id for q in qs], [q.wire for q in qs]))
    _measurement!(b, ResultSpec(:state, 1 << length(qs)), id -> StateNode(id, obs, length(qs)))
end

function finish!(b::Builder, ret)
    toks = ret isa MeasurementToken ? [ret] : ret isa Tuple ? collect(ret) :
           throw(TraceError("a @qnode function must return a measurement (expval, var, probs, state) or a tuple of measurements, got $(typeof(ret))"))
    isempty(toks) && throw(TraceError("a @qnode function must return at least one measurement"))
    all(t -> t isa MeasurementToken && t.builder === b, toks) ||
        throw(TraceError("returned values must be measurements of this @qnode"))
    ensure_register!(b)
    b.prog.results = [t.id for t in toks]
    b.prog.result_specs = [b.results[t.id] for t in toks]
    b.prog.scalar_return = ret isa MeasurementToken
    reg = b.reg
    for w in 1:b.prog.nqubits
        q = b.current[w]::Qubit
        newreg = newvalue!(b)
        push!(b.prog.nodes, InsertNode(newreg, reg, w, q.id))
        reg = newreg
    end
    push!(b.prog.nodes, DeallocNode(reg))
    b.prog.diff_method = all(s -> s.kind === :expval, b.prog.result_specs) ? "adjoint" : "parameter-shift"
    b.prog
end

# ---- QNode -----------------------------------------------------------------------------------
"""
    QNode(f, dev; name=:circuit)

A quantum function bound to a device. Calling it traces `f` once per argument signature and
executes the resulting `Program` on the device. Usually created with `@qnode`.
"""
mutable struct QNode{F,D<:AbstractDevice}
    f::F
    dev::D
    name::Symbol
    programs::Dict{Any,Program}
end
QNode(f, dev::AbstractDevice; name::Symbol=:circuit) = QNode{typeof(f),typeof(dev)}(f, dev, name, Dict{Any,Program}())
Base.show(io::IO, qn::QNode) = print(io, "QNode ", qn.name, " on ", qn.dev)

argsig(x::Real) = (:scalar, 0)
argsig(x::AbstractVector{<:Real}) = (:vector, length(x))
argsig(x) = throw(ArgumentError("QNode arguments must be real numbers or real vectors, got $(typeof(x))"))
normalize_args(args) = Any[a isa Real ? Float64(a) : Vector{Float64}(a) for a in args]

"""`program(qn, args...)`: the traced `Program` for this argument signature (cached)."""
function program(qn::QNode, args...)
    sig = map(argsig, args)
    get!(qn.programs, sig) do
        trace(qn.f, qn.dev, qn.name, collect(sig))
    end
end

function trace(f, dev::AbstractDevice, name::Symbol, sig)
    prog = Program(name, ArgSpec[ArgSpec(s[1], s[2]) for s in sig])
    b = Builder(prog, 0, Dict{Int,Any}(), false, device_wires(dev), Dict{Int,ResultSpec}())
    targs = Any[s[1] === :scalar ? TracedReal(CArg(i, 0)) : TracedVector(i, s[2]) for (i, s) in enumerate(sig)]
    with_builder(b) do
        finish!(b, f(targs...))
    end
    prog
end

(qn::QNode)(args...) = execute(qn.dev, program(qn, args...), normalize_args(args))

"""
    @qnode dev function name(args...) ... end

Define `name` as a `QNode` running on `dev`.
"""
macro qnode(dev, fdef)
    if fdef isa Expr && (fdef.head === :function || fdef.head === :(=)) && fdef.args[1] isa Expr && fdef.args[1].head === :call
        sig, body = fdef.args
        name = sig.args[1]
        anon = Expr(:function, Expr(:tuple, sig.args[2:end]...), body)
        return esc(:($name = $(QNode)($anon, $dev; name=$(QuoteNode(name)))))
    end
    error("usage: @qnode dev function name(args...) ... end")
end
