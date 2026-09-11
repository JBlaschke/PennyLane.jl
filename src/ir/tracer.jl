# Tracing: turns a Julia function into a Program.
#
# Qubits are SSA values (`Qubit`). A gate consumes its input values and returns fresh ones;
# touching a consumed value is a loud error. Integer wires are accepted wherever a qubit is and
# resolve to the wire's current value, so PennyLane-style circuits work unchanged.
# Classical parameters are traced values (`TracedReal`, `TracedInt`, `TracedBool`) carrying
# expression trees. Control flow that must survive into the program (branching on mid-circuit
# measurements, loops with a traced index) is written with `@trace if/for/while`; plain Julia
# control flow is unrolled at trace time.

mutable struct Builder
    prog::Program
    reg::Int                       # register value id (0 = not allocated yet)
    current::Dict{Int,Any}         # wire => live Qubit
    measured::Bool
    device_wires::Union{Nothing,Int}
    results::Dict{Int,ResultSpec}
    nodes::Vector{Node}            # node list being appended to (a region body while tracing control flow)
    depth::Int                     # region nesting depth
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
"""A real-valued traced quantity (function argument, or an expression of traced values)."""
struct TracedReal <: Real
    expr::CExpr
end
"""An integer traced quantity (loop index of `@trace for`, or arithmetic on one)."""
struct TracedInt <: Integer
    expr::CExpr
end
"""A boolean traced quantity (mid-circuit measurement result, or a comparison of traced values)."""
struct TracedBool
    expr::CExpr
end
struct TracedVector <: AbstractVector{TracedReal}
    arg::Int
    len::Int
end
Base.size(v::TracedVector) = (v.len,)
Base.getindex(v::TracedVector, i::Int) = (checkbounds(v, i); TracedReal(CArg(v.arg, i)))
Base.getindex(v::TracedVector, i::TracedInt) = TracedReal(CArgDyn(v.arg, _fold(:isub, i.expr, CIConst(1))))
Base.show(io::IO, x::TracedReal) = print(io, "TracedReal(", _fmt(x.expr), ")")
Base.show(io::IO, x::TracedInt) = print(io, "TracedInt(", _fmt(x.expr), ")")
Base.show(io::IO, x::TracedBool) = print(io, "TracedBool(", _fmt(x.expr), ")")

const AnyTraced = Union{TracedReal,TracedInt,TracedBool}
_wrap(kind::Symbol, e::CExpr) = kind === :f64 ? TracedReal(e) : kind === :int ? TracedInt(e) : TracedBool(e)

# gate parameters are f64
toexpr(x::TracedReal) = x.expr
toexpr(x::TracedInt) = _fold(:itof, x.expr)
toexpr(x::Bool) = throw(ArgumentError("gate parameters must be real numbers, got Bool"))
toexpr(x::Real) = CConst(Float64(x))
toexpr(x) = throw(ArgumentError("gate parameters must be real numbers or traced values, got $(typeof(x))"))
# any classical value (carried through control flow)
cexpr(x::TracedReal) = x.expr
cexpr(x::TracedInt) = x.expr
cexpr(x::TracedBool) = x.expr
cexpr(x::Bool) = CBConst(x)
cexpr(x::Integer) = CIConst(Int(x))
cexpr(x::Real) = CConst(Float64(x))
toboolexpr(x::TracedBool) = x.expr
toboolexpr(x::Bool) = CBConst(x)
toboolexpr(x) = throw(ArgumentError("expected a Bool or a traced boolean (measurement result / comparison), got $(typeof(x))"))
isclassical(x) = x isa AnyTraced || x isa Real

# constant folding
function _fold(op::Symbol, args::CExpr...)
    if all(isconst, args)
        v = ceval(CCall(op, collect(CExpr, args)), Any[])
        return mkconst(CLASSICAL_OPS[op], v)
    end
    # (x + c) - c  and  (x - c) + c  simplify (loop indices)
    if op === :isub && args[1] isa CCall && args[1].op === :iadd && args[1].args[2] == args[2]
        return args[1].args[1]
    end
    CCall(op, collect(CExpr, args))
end

for (f, op) in ((:+, :add), (:-, :sub), (:*, :mul), (:/, :div))
    @eval begin
        Base.$f(a::TracedReal, b::TracedReal) = TracedReal(_fold($(QuoteNode(op)), a.expr, b.expr))
        Base.$f(a::TracedReal, b::Real) = TracedReal(_fold($(QuoteNode(op)), a.expr, toexpr(b)))
        Base.$f(a::Real, b::TracedReal) = TracedReal(_fold($(QuoteNode(op)), toexpr(a), b.expr))
        Base.$f(a::TracedInt, b::TracedReal) = TracedReal(_fold($(QuoteNode(op)), toexpr(a), b.expr))
        Base.$f(a::TracedReal, b::TracedInt) = TracedReal(_fold($(QuoteNode(op)), a.expr, toexpr(b)))
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
for (f, op) in ((:<, :lt), (:<=, :le), (:>, :gt), (:>=, :ge), (:(==), :eq), (:(!=), :ne))
    @eval begin
        Base.$f(a::TracedReal, b::TracedReal) = TracedBool(_fold($(QuoteNode(op)), a.expr, b.expr))
        Base.$f(a::TracedReal, b::Real) = TracedBool(_fold($(QuoteNode(op)), a.expr, toexpr(b)))
        Base.$f(a::Real, b::TracedReal) = TracedBool(_fold($(QuoteNode(op)), toexpr(a), b.expr))
    end
end
Base.promote_rule(::Type{TracedReal}, ::Type{<:Real}) = TracedReal
Base.convert(::Type{TracedReal}, x::Real) = TracedReal(toexpr(x))
Base.convert(::Type{TracedReal}, x::TracedReal) = x
const _NOCONVERT = "a traced value cannot be converted to a concrete number: it is only known when the circuit runs. Keep it as a gate parameter, use it in `@trace if/for/while`, or compute the quantity outside the @qnode."
Base.convert(::Type{T}, x::TracedReal) where {T<:Number} = throw(ArgumentError(_NOCONVERT))
(::Type{T})(x::TracedReal) where {T<:AbstractFloat} = convert(T, x)
Base.zero(::TracedReal) = TracedReal(CConst(0.0))
Base.one(::TracedReal) = TracedReal(CConst(1.0))
Base.isless(::TracedReal, ::Real) = throw(ArgumentError("cannot order traced values with isless/sort; use comparisons inside `@trace if`"))
Base.isless(::Real, ::TracedReal) = throw(ArgumentError("cannot order traced values with isless/sort; use comparisons inside `@trace if`"))
Base.isless(::TracedReal, ::TracedReal) = throw(ArgumentError("cannot order traced values with isless/sort; use comparisons inside `@trace if`"))

# integers: +, -, * with Int, comparisons, conversion to float
for (f, op) in ((:+, :iadd), (:-, :isub), (:*, :imul))
    @eval begin
        Base.$f(a::TracedInt, b::TracedInt) = TracedInt(_fold($(QuoteNode(op)), a.expr, b.expr))
        Base.$f(a::TracedInt, b::Integer) = TracedInt(_fold($(QuoteNode(op)), a.expr, CIConst(Int(b))))
        Base.$f(a::Integer, b::TracedInt) = TracedInt(_fold($(QuoteNode(op)), CIConst(Int(a)), b.expr))
    end
end
Base.:-(a::TracedInt) = TracedInt(_fold(:isub, CIConst(0), a.expr))
for (f, op) in ((:<, :ilt), (:<=, :ile), (:(==), :ieq))
    @eval begin
        Base.$f(a::TracedInt, b::TracedInt) = TracedBool(_fold($(QuoteNode(op)), a.expr, b.expr))
        Base.$f(a::TracedInt, b::Integer) = TracedBool(_fold($(QuoteNode(op)), a.expr, CIConst(Int(b))))
        Base.$f(a::Integer, b::TracedInt) = TracedBool(_fold($(QuoteNode(op)), CIConst(Int(a)), b.expr))
    end
end
Base.:>(a::TracedInt, b::Integer) = b < a
Base.:>(a::Integer, b::TracedInt) = b < a
Base.:>=(a::TracedInt, b::Integer) = b <= a
Base.:>=(a::Integer, b::TracedInt) = b <= a
Base.:>(a::TracedInt, b::TracedInt) = b < a
Base.:>=(a::TracedInt, b::TracedInt) = b <= a
Base.float(a::TracedInt) = TracedReal(toexpr(a))
Base.Float64(a::TracedInt) = TracedReal(toexpr(a))
Base.convert(::Type{T}, x::TracedInt) where {T<:Number} = throw(ArgumentError("a traced loop index cannot be converted to a concrete integer (dynamic wire indices such as q[i] are not supported yet: use a plain `for` loop, which is unrolled, or index parameters θ[i]). " * _NOCONVERT))
Base.promote_rule(::Type{TracedInt}, ::Type{<:Integer}) = TracedInt
Base.promote_rule(::Type{TracedInt}, ::Type{<:AbstractFloat}) = TracedReal
Base.convert(::Type{TracedInt}, x::Integer) = TracedInt(CIConst(Int(x)))
Base.convert(::Type{TracedInt}, x::TracedInt) = x
Base.convert(::Type{TracedReal}, x::TracedInt) = TracedReal(toexpr(x))
Base.isless(::TracedInt, ::Integer) = throw(ArgumentError("cannot order traced values with isless/sort; use comparisons inside `@trace if`"))
Base.isless(::Integer, ::TracedInt) = throw(ArgumentError("cannot order traced values with isless/sort; use comparisons inside `@trace if`"))
Base.isless(::TracedInt, ::TracedInt) = throw(ArgumentError("cannot order traced values with isless/sort; use comparisons inside `@trace if`"))

# booleans
Base.:!(a::TracedBool) = TracedBool(_fold(:not, a.expr))
Base.:&(a::TracedBool, b::TracedBool) = TracedBool(_fold(:and, a.expr, b.expr))
Base.:|(a::TracedBool, b::TracedBool) = TracedBool(_fold(:or, a.expr, b.expr))
Base.:&(a::TracedBool, b::Bool) = TracedBool(_fold(:and, a.expr, CBConst(b)))
Base.:&(a::Bool, b::TracedBool) = b & a
Base.:|(a::TracedBool, b::Bool) = TracedBool(_fold(:or, a.expr, CBConst(b)))
Base.:|(a::Bool, b::TracedBool) = b | a
Base.convert(::Type{Bool}, ::TracedBool) = throw(ArgumentError("a traced boolean (measurement result) is only known when the circuit runs; branch on it with `@trace if m ... end`"))

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
emit!(b::Builder, node::Node) = push!(b.nodes, node)

function ensure_register!(b::Builder, n::Union{Nothing,Int}=nothing)
    if b.reg != 0
        n === nothing || n == b.prog.nqubits ||
            throw(TraceError("qubits($n) requested but this program already has $(b.prog.nqubits) qubits"))
        return b
    end
    b.depth == 0 || throw(TraceError("allocate qubits (qubits(n)) before entering control flow"))
    nq = n === nothing ? b.device_wires : n
    nq === nothing && throw(TraceError("cannot infer the number of qubits: call qubits(n) first, or give the device a wire count such as StateVector(2)"))
    nq >= 1 || throw(TraceError("a program needs at least one qubit"))
    b.device_wires === nothing || nq <= b.device_wires ||
        throw(TraceError("qubits($nq) exceeds the device's $(b.device_wires) wires"))
    b.prog.nqubits = nq
    b.reg = newvalue!(b)
    emit!(b, AllocNode(b.reg, nq))
    for w in 1:nq
        id = newvalue!(b)
        emit!(b, ExtractNode(id, b.reg, w))
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
    emit!(b, GateNode(g.name, CExpr[toexpr(p) for p in g.params], ids, [o.id for o in outs],
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
    emit!(b, GateNode(c.gate.name, CExpr[toexpr(p) for p in c.gate.params], ids, [o.id for o in outs],
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

"""
    evolve(H, t, q...; steps=1)   (alias `ApproxTimeEvolution(H, t, steps, q...)`)

Apply exp(-i t H) for a `PauliSum`/`PauliString` `H` with real coefficients by first-order
Trotterisation with `steps` steps: each step applies `PauliRot(2 t c/steps, P)` for every term
`c·P`. Exact when the terms commute (QAOA cost and mixer Hamiltonians). `t` may be traced.
The qubits are passed in the order of `sort(wires(H))`; identity terms are a global phase and skipped.
"""
function evolve(H::Observable, t, sites...; steps::Integer=1)
    steps >= 1 || throw(ArgumentError("evolve needs steps ≥ 1"))
    ws = wires(H)
    isempty(sites) && (sites = Tuple(ws))
    length(sites) == length(ws) || throw(ArgumentError("evolve: H acts on $(length(ws)) wire(s) $(ws) but $(length(sites)) qubit(s) were given"))
    b = builder_for(sites)
    qs = Any[resolve(b, s) for s in sites]
    pos = Dict(w => i for (i, w) in enumerate(ws))
    for term in terms(H)
        abs(imag(term.coeff)) <= 1e-12 * max(1.0, abs(term.coeff)) || throw(ArgumentError("evolve needs a Hamiltonian with real coefficients"))
    end
    for _ in 1:steps, term in terms(H)
        isempty(term.word) && continue
        idx = [pos[w] for (w, _) in term.word]
        word = join(string(l) for (_, l) in term.word)
        outs = PauliRot((2 * real(term.coeff) / steps) * t, word, qs[idx]...)
        outs isa Tuple || (outs = (outs,))
        for (k, i) in enumerate(idx)
            qs[i] = outs[k]
        end
    end
    length(qs) == 1 ? qs[1] : Tuple(qs)
end
ApproxTimeEvolution(H::Observable, t, steps::Integer, sites...) = evolve(H, t, sites...; steps=steps)
(u::BoundEvolution)(sites...) = evolve(u.H, u.t, sites...; steps=u.steps)

# ---- mid-circuit measurement -----------------------------------------------------------------
"""
    measure(q; postselect=nothing, reset=false) -> (m::TracedBool, q′)

Measure a qubit in the computational basis mid-circuit. `m` can be used in `@trace if` /
`@trace while`; `postselect` fixes the outcome (0 or 1); `reset=true` returns the qubit in |0⟩.
"""
function measure(site; postselect=nothing, reset::Bool=false)
    b = builder_for((site,))
    b.measured && throw(TraceError("cannot measure after a terminal measurement"))
    q = resolve(b, site)
    ps = postselect === nothing ? -1 : Int(postselect)
    ps in (-1, 0, 1) || throw(ArgumentError("postselect must be 0, 1 or nothing"))
    id = consume!(q, "measure")
    out = fresh!(b, q.wire)
    m = newvalue!(b)
    emit!(b, MeasureNode(m, id, out.id, q.wire, ps))
    mb = TracedBool(CValue(m, :bool))
    if reset
        out = qif(mb, (out,), t -> (PauliX(t[1]),))[1]
    end
    (mb, out)
end

# ---- control flow ----------------------------------------------------------------------------
struct NotCarried end                 # placeholder for variables not defined before a traced block

# structures carried through regions: tuples and vectors of qubits / classical values
map_leaves(f, x::Tuple) = map(y -> map_leaves(f, y), x)
map_leaves(f, x::AbstractVector) = map(y -> map_leaves(f, y), x)
map_leaves(f, x) = f(x)
function leaves(x, acc=Any[])
    if x isa Tuple || x isa AbstractVector
        for y in x
            leaves(y, acc)
        end
    else
        push!(acc, x)
    end
    acc
end
qubit_leaves(x) = Qubit[l for l in leaves(x) if l isa Qubit]
classical_leaves(x) = Any[l for l in leaves(x) if isclassical(l)]
_ckind(x) = x isa AnyTraced ? ckind(cexpr(x)) : x isa Bool ? :bool : x isa Integer ? :int : :f64

function validate_carried!(b::Builder, carried)
    for q in qubit_leaves(carried)
        q.builder === b || throw(TraceError("a qubit from another trace was carried into control flow"))
        (q.consumed || b.current[q.wire] !== q) &&
            throw(QubitConsumedError("a stale qubit value for wire $(q.wire) (consumed by $(q.consumer)) was carried into control flow; carry the current value"))
    end
end

# Trace `f(extra..., carried′)` in a nested node list where every wire is represented by the
# Qubit in `entry` and classical carried leaves are replaced by `cvals` (same order).
function trace_region(b::Builder, entry::Dict{Int,Any}, carried, cvals, f, extra...)
    saved = (b.nodes, b.current)
    b.nodes = Node[]
    b.current = copy(entry)
    b.depth += 1
    try
        k = 0
        carried′ = map_leaves(carried) do x
            x isa Qubit && return entry[x.wire]
            isclassical(x) || return x
            k += 1
            cvals[k]
        end
        ret = f(extra..., carried′)
        return (nodes=b.nodes, final=copy(b.current), ret=ret)
    finally
        b.nodes, b.current = saved
        b.depth -= 1
    end
end

function _check_structure(carried, ret, what)
    lin, lout = leaves(carried), leaves(ret)
    length(lin) == length(lout) || throw(TraceError("the body of $what must keep the shape of the carried variables"))
    for (x, y) in zip(lin, lout)
        if x isa Qubit
            y isa Qubit || throw(TraceError("a qubit variable was reassigned to a $(typeof(y)) inside $what"))
        elseif isclassical(x)
            isclassical(y) || throw(TraceError("a classical variable was reassigned to a $(typeof(y)) inside $what"))
            _ckind(x) == _ckind(y) || throw(TraceError("a classical variable changed type ($(_ckind(x)) → $(_ckind(y))) inside $what; use the same kind of value (e.g. 0.0 instead of 0)"))
        elseif x isa NotCarried
            y isa Qubit || y isa NotCarried || throw(TraceError("a variable is defined only inside $what; define it before the block (e.g. m = false)"))
        else
            y === x || throw(TraceError("only qubits and numbers can be reassigned inside $what (got $(typeof(y)))"))
        end
    end
end

# rebuild the carried structure after a region: qubit leaves -> current value of their wire,
# classical leaves -> region results (in order), NotCarried -> the wire value of the qubit that was assigned
function _rebuild(b::Builder, carried, ret, cout::Vector{Int}, kinds::Vector{Symbol})
    k = 0
    lret = leaves(ret)
    j = 0
    map_leaves(carried) do x
        j += 1
        if x isa Qubit
            return b.current[x.wire]
        elseif isclassical(x)
            k += 1
            return _wrap(kinds[k], CValue(cout[k], kinds[k]))
        elseif x isa NotCarried
            y = lret[j]
            return y isa Qubit ? b.current[y.wire] : x
        end
        x
    end
end

"""
    qif(cond, carried::Tuple, then_f[, else_f]) -> carried′

Functional form of `@trace if`: traces both branches with the carried values and returns the
values after the branch. Prefer the macro.
"""
function qif(cond, carried::Tuple, then_f, else_f=nothing)
    b = require_builder()
    ce = toboolexpr(cond)
    if ce isa CBConst
        return ce.value ? then_f(carried) : (else_f === nothing ? carried : else_f(carried))
    end
    ensure_register!(b)
    validate_carried!(b, carried)
    pre = copy(b.current)
    nq = b.prog.nqubits
    cl_in = classical_leaves(carried)
    kinds = Symbol[_ckind(x) for x in cl_in]
    # branches capture the outer values (same ids), with their own consumed flags
    view() = Dict{Int,Any}(w => Qubit(b, w, pre[w].id, false, "") for w in 1:nq)
    cvals = Any[x isa AnyTraced ? x : _wrap(_ckind(x), cexpr(x)) for x in cl_in]
    tr = trace_region(b, view(), carried, cvals, then_f)
    _check_structure(carried, tr.ret, "@trace if")
    er = else_f === nothing ? nothing : trace_region(b, view(), carried, cvals, else_f)
    er === nothing || _check_structure(carried, er.ret, "@trace if")
    touched(r) = Set{Int}(w for w in 1:nq if r.final[w].id != pre[w].id)
    W = sort!(collect(union(touched(tr), er === nothing ? Set{Int}() : touched(er), Set{Int}(q.wire for q in qubit_leaves(carried)))))
    in_ids = [consume!(pre[w], "@trace if") for w in W]
    then_yield = [tr.final[w].id for w in W]
    else_yield = er === nothing ? [pre[w].id for w in W] : [er.final[w].id for w in W]
    outs = [fresh!(b, w) for w in W]
    cout = [newvalue!(b) for _ in cl_in]
    then_cy = CExpr[cexpr(x) for x in classical_leaves(tr.ret)]
    else_cy = er === nothing ? CExpr[cexpr(x) for x in cvals] : CExpr[cexpr(x) for x in classical_leaves(er.ret)]
    emit!(b, IfNode(ce, W, in_ids, [o.id for o in outs], tr.nodes, er === nothing ? Node[] : er.nodes,
                    then_yield, else_yield, cout, kinds, then_cy, else_cy))
    _rebuild(b, carried, tr.ret, cout, kinds)
end

"""
    qfor(range, carried::Tuple, f) -> carried′

Functional form of `@trace for`: `f(i, carried)` is traced once with a traced index. Prefer the macro.
"""
function qfor(r::AbstractRange, carried::Tuple, f)
    b = require_builder()
    isempty(r) && return carried
    step(r) > 0 || throw(ArgumentError("@trace for needs a range with a positive step"))
    ensure_register!(b)
    validate_carried!(b, carried)
    pre = copy(b.current)
    nq = b.prog.nqubits
    cl_in = classical_leaves(carried)
    kinds = Symbol[_ckind(x) for x in cl_in]
    idx = newvalue!(b)
    entry = Dict{Int,Any}(w => Qubit(b, w, newvalue!(b), false, "") for w in 1:nq)     # block arguments
    cargs = [newvalue!(b) for _ in cl_in]
    cvals = Any[_wrap(kinds[k], CValue(cargs[k], kinds[k])) for k in eachindex(cl_in)]
    res = trace_region(b, entry, carried, cvals, f, TracedInt(CValue(idx, :int)))
    _check_structure(carried, res.ret, "@trace for")
    W = sort!(collect(union(Set{Int}(w for w in 1:nq if res.final[w].id != entry[w].id), Set{Int}(q.wire for q in qubit_leaves(carried)))))
    in_ids = [consume!(pre[w], "@trace for") for w in W]
    outs = [fresh!(b, w) for w in W]
    cout = [newvalue!(b) for _ in cl_in]
    emit!(b, ForNode(Int(first(r)), Int(last(r)), Int(step(r)), idx, W, in_ids, [entry[w].id for w in W],
                     [res.final[w].id for w in W], [o.id for o in outs], CExpr[cexpr(x) for x in cl_in], cargs,
                     CExpr[cexpr(x) for x in classical_leaves(res.ret)], cout, kinds, res.nodes))
    _rebuild(b, carried, res.ret, cout, kinds)
end

"""
    qwhile(cond_f, body_f, carried::Tuple) -> carried′

Functional form of `@trace while`: `cond_f(carried)` must be classical and return a traced boolean. Prefer the macro.
"""
function qwhile(cond_f, body_f, carried::Tuple)
    b = require_builder()
    ensure_register!(b)
    validate_carried!(b, carried)
    pre = copy(b.current)
    nq = b.prog.nqubits
    cl_in = classical_leaves(carried)
    kinds = Symbol[_ckind(x) for x in cl_in]
    entry = Dict{Int,Any}(w => Qubit(b, w, newvalue!(b), false, "") for w in 1:nq)
    cargs = [newvalue!(b) for _ in cl_in]
    cvals = Any[_wrap(kinds[k], CValue(cargs[k], kinds[k])) for k in eachindex(cl_in)]
    cr = trace_region(b, entry, carried, cvals, cond_f)
    isempty(cr.nodes) || throw(TraceError("the condition of @trace while must be classical (no gates or measurements)"))
    ce = toboolexpr(cr.ret)
    ce isa CBConst && !ce.value && return carried
    ce isa CBConst && throw(TraceError("the condition of @trace while is constantly true"))
    br = trace_region(b, Dict{Int,Any}(w => Qubit(b, w, entry[w].id, false, "") for w in 1:nq), carried, cvals, body_f)
    _check_structure(carried, br.ret, "@trace while")
    W = sort!(collect(union(Set{Int}(w for w in 1:nq if br.final[w].id != entry[w].id), Set{Int}(q.wire for q in qubit_leaves(carried)))))
    in_ids = [consume!(pre[w], "@trace while") for w in W]
    outs = [fresh!(b, w) for w in W]
    cout = [newvalue!(b) for _ in cl_in]
    emit!(b, WhileNode(W, in_ids, [entry[w].id for w in W], [br.final[w].id for w in W], [o.id for o in outs],
                       CExpr[cexpr(x) for x in cl_in], cargs, CExpr[cexpr(x) for x in classical_leaves(br.ret)], cout, kinds, ce, br.nodes))
    _rebuild(b, carried, br.ret, cout, kinds)
end

# -- macro plumbing: variables assigned in a block become the carried tuple
function _lhs_vars!(lhs, acc::Vector{Symbol})
    if lhs isa Symbol
        lhs in acc || push!(acc, lhs)
    elseif lhs isa Expr
        if lhs.head === :tuple
            foreach(x -> _lhs_vars!(x, acc), lhs.args)
        elseif lhs.head === :ref || lhs.head === :(::)
            _lhs_vars!(lhs.args[1], acc)
        end
    end
    acc
end
function assigned_vars(ex, acc::Vector{Symbol}=Symbol[])
    ex isa Expr || return acc
    if ex.head === :(=) || ex.head in (:+=, :-=, :*=, :/=)
        _lhs_vars!(ex.args[1], acc)
        assigned_vars(ex.args[2], acc)
    elseif ex.head in (:function, :(->), :let, :quote)
        return acc
    elseif ex.head === :for
        assigned_vars(ex.args[2], acc)          # the loop variable itself is local
    else
        for a in ex.args
            assigned_vars(a, acc)
        end
    end
    acc
end
function used_symbols(ex, acc::Vector{Symbol}=Symbol[])
    if ex isa Symbol
        ex in acc || push!(acc, ex)
    elseif ex isa Expr && ex.head !== :quote
        for a in (ex.head === :call ? ex.args[2:end] : ex.args)
            used_symbols(a, acc)
        end
    end
    acc
end
_carried_expr(vars) = Expr(:tuple, (:(@isdefined($v) ? $v : $(NotCarried)()) for v in vars)...)
_lambda(vars, body, extra...) = Expr(:->, Expr(:tuple, extra..., :__carried__),
                                      Expr(:block, Expr(:(=), Expr(:tuple, vars...), :__carried__), body, Expr(:tuple, vars...)))
_cond_lambda(vars, cond) = Expr(:->, Expr(:tuple, :__carried__),
                                Expr(:block, Expr(:(=), Expr(:tuple, vars...), :__carried__), cond))
_assign_back(vars, r) = isempty(vars) ? r : Expr(:(=), Expr(:tuple, vars...), r)

function _trace_expr(ex)
    ex isa Expr || error("@trace expects an if, for or while expression")
    if ex.head === :if || ex.head === :elseif
        cond, thenb = ex.args[1], ex.args[2]
        elseb = length(ex.args) == 3 ? ex.args[3] : nothing
        elseb isa Expr && elseb.head === :elseif && (elseb = _trace_expr(elseb))
        vars = assigned_vars(thenb)
        elseb === nothing || assigned_vars(elseb, vars)
        call = elseb === nothing ? :($(qif)($cond, $(_carried_expr(vars)), $(_lambda(vars, thenb)))) :
               :($(qif)($cond, $(_carried_expr(vars)), $(_lambda(vars, thenb)), $(_lambda(vars, elseb))))
        return _assign_back(vars, call)
    elseif ex.head === :for
        spec, body = ex.args
        (spec isa Expr && spec.head === :(=)) || error("@trace for expects `for i in range`")
        i, range = spec.args
        vars = filter(!=(i), assigned_vars(body))
        return _assign_back(vars, :($(qfor)($range, $(_carried_expr(vars)), $(_lambda(vars, body, i)))))
    elseif ex.head === :while
        cond, body = ex.args
        vars = assigned_vars(body)
        for s in used_symbols(cond)
            s in vars || push!(vars, s)
        end
        return _assign_back(vars, :($(qwhile)($(_cond_lambda(vars, cond)), $(_lambda(vars, body)), $(_carried_expr(vars)))))
    end
    error("@trace expects an if, for or while expression")
end

"""
    @trace if m ... [else ...] end
    @trace for i in 1:n ... end
    @trace while cond ... end

Control flow that is captured into the program instead of being unrolled at trace time:
branch on mid-circuit measurement results or traced comparisons, loop with a traced index
(`θ[i]` becomes a dynamic index; `q[i]` is not supported), repeat until a measured condition.
Variables assigned in the block are threaded through the region; qubits keep value semantics.
"""
macro trace(ex)
    esc(_trace_expr(ex))
end
"""`@qif cond block`: shorthand for `@trace if cond block end`."""
macro qif(cond, body)
    esc(_trace_expr(Expr(:if, cond, body)))
end
"""`@qfor i in range block`: shorthand for `@trace for i in range block end`."""
macro qfor(spec, body)
    (spec isa Expr && spec.head === :call && spec.args[1] === :in) || error("usage: @qfor i in range begin ... end")
    esc(_trace_expr(Expr(:for, Expr(:(=), spec.args[2], spec.args[3]), body)))
end
"""`@qwhile cond block`: shorthand for `@trace while cond block end`."""
macro qwhile(cond, body)
    esc(_trace_expr(Expr(:while, cond, body)))
end

# ---- terminal measurements -------------------------------------------------------------------
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
        emit!(b, NamedObsNode(id, q.id, 1, :Identity))
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
        emit!(b, NamedObsNode(id, q.id, w, Symbol(:Pauli, l)))
        push!(ids, id)
    end
    length(ids) == 1 && return ids[1]
    id = newvalue!(b)
    emit!(b, TensorObsNode(id, ids))
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
    emit!(b, HamiltonianNode(id, coeffs, ids))
    id
end

function _measurement!(b::Builder, spec::ResultSpec, mk)
    b.depth == 0 || throw(TraceError("terminal measurements (expval, var, probs, sample, state) must be outside control flow; use measure() inside"))
    id = newvalue!(b)
    emit!(b, mk(id))
    b.measured = true
    b.results[id] = spec
    MeasurementToken(b, id)
end

"""`expval(obs)`: expectation value of an observable (terminal measurement)."""
function expval(o::Observable)
    b = require_builder()
    b.depth == 0 || throw(TraceError("terminal measurements (expval, var, probs, sample, state) must be outside control flow; use measure() inside"))
    obs = obs_value!(b, o)
    _measurement!(b, ResultSpec(:expval, 0), id -> ExpvalNode(id, obs))
end
"""`var(obs)`: variance of an observable (terminal measurement)."""
function var(o::Observable)
    b = require_builder()
    b.depth == 0 || throw(TraceError("terminal measurements (expval, var, probs, sample, state) must be outside control flow; use measure() inside"))
    obs = obs_value!(b, o)
    _measurement!(b, ResultSpec(:var, 0), id -> VarNode(id, obs))
end
function _compbasis!(b::Builder, sites, what)
    ensure_register!(b)
    qs = isempty(sites) ? Qubit[b.current[w] for w in 1:b.prog.nqubits] : Qubit[resolve(b, s) for s in sites]
    for q in qs
        (q.consumed || b.current[q.wire] !== q) &&
            throw(QubitConsumedError("$what refers to a qubit value on wire $(q.wire) that was consumed by $(q.consumer)"))
    end
    obs = newvalue!(b)
    emit!(b, CompBasisNode(obs, [q.id for q in qs], [q.wire for q in qs]))
    (obs, length(qs))
end
"""`probs(sites...)`: computational-basis probabilities of the given wires/qubits (all if none)."""
function probs(sites...)
    b = builder_for(sites)
    obs, k = _compbasis!(b, sites, "probs")
    _measurement!(b, ResultSpec(:probs, 1 << k), id -> ProbsNode(id, obs, k))
end
"""`sample(sites...)`: computational-basis samples of the given wires/qubits (all if none) as a
`shots × wires` matrix of 0/1; needs a device with shots."""
function sample(sites...)
    b = builder_for(sites)
    obs, k = _compbasis!(b, sites, "sample")
    _measurement!(b, ResultSpec(:sample, k), id -> SampleNode(id, obs, k))
end
"""`state()`: the full state vector (simulators only)."""
function state()
    b = require_builder()
    obs, k = _compbasis!(b, (), "state")
    _measurement!(b, ResultSpec(:state, 1 << k), id -> StateNode(id, obs, k))
end

function finish!(b::Builder, ret)
    toks = ret isa MeasurementToken ? [ret] : ret isa Tuple ? collect(ret) :
           throw(TraceError("a @qnode function must return a measurement (expval, var, probs, sample, state) or a tuple of measurements, got $(typeof(ret))"))
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
        emit!(b, InsertNode(newreg, reg, w, q.id))
        reg = newreg
    end
    emit!(b, DeallocNode(reg))
    b.prog.diff_method = all(s -> s.kind === :expval, b.prog.result_specs) && !has_mcm(b.prog) ? "adjoint" : "parameter-shift"
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
    b = Builder(prog, 0, Dict{Int,Any}(), false, device_wires(dev), Dict{Int,ResultSpec}(), prog.nodes, 0)
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
