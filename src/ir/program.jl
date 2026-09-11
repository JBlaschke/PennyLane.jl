# Hybrid program IR.
#
# A `Program` is a value-semantic SSA program whose nodes mirror Catalyst's `quantum` dialect
# (alloc/extract/insert/dealloc, gates, observables, measurements) plus a small classical
# expression language for gate parameters. Every backend consumes this IR; `to_mlir` prints it
# in Catalyst's textual format. When MLIR.jl stabilises, the printer can be replaced by a
# builder without touching the tracer or the backends (see PLAN.md §3.2, decision 4).

# ---- classical expressions -------------------------------------------------------------------
abstract type CExpr end
struct CConst <: CExpr
    value::Float64
end
struct CArg <: CExpr           # function argument `arg` (1-based); index 0 = scalar, ≥1 = element of a vector
    arg::Int
    index::Int
end
struct CCall <: CExpr          # :add :sub :mul :div :neg :sin :cos :sqrt :exp
    op::Symbol
    args::Vector{CExpr}
end

ceval(e::CConst, args) = e.value
ceval(e::CArg, args) = e.index == 0 ? Float64(args[e.arg]) : Float64(args[e.arg][e.index])
function ceval(e::CCall, args)
    v = [ceval(a, args) for a in e.args]
    op = e.op
    op === :add && return v[1] + v[2]
    op === :sub && return v[1] - v[2]
    op === :mul && return v[1] * v[2]
    op === :div && return v[1] / v[2]
    op === :neg && return -v[1]
    op === :sin && return sin(v[1])
    op === :cos && return cos(v[1])
    op === :sqrt && return sqrt(v[1])
    op === :exp && return exp(v[1])
    error("unknown classical op $op")
end

hasarg(::CConst) = false
hasarg(::CArg) = true
hasarg(e::CCall) = any(hasarg, e.args)

"""Value and derivative of a classical expression with respect to argument slot `(a, i)` (forward mode)."""
cdual(e::CConst, args, a, i) = (e.value, 0.0)
cdual(e::CArg, args, a, i) = (ceval(e, args), (e.arg == a && e.index == i) ? 1.0 : 0.0)
function cdual(e::CCall, args, a, i)
    vd = [cdual(x, args, a, i) for x in e.args]
    v1, d1 = vd[1]
    op = e.op
    if op === :add
        v2, d2 = vd[2]; return (v1 + v2, d1 + d2)
    elseif op === :sub
        v2, d2 = vd[2]; return (v1 - v2, d1 - d2)
    elseif op === :mul
        v2, d2 = vd[2]; return (v1 * v2, d1 * v2 + v1 * d2)
    elseif op === :div
        v2, d2 = vd[2]; return (v1 / v2, (d1 * v2 - v1 * d2) / v2^2)
    elseif op === :neg
        return (-v1, -d1)
    elseif op === :sin
        return (sin(v1), cos(v1) * d1)
    elseif op === :cos
        return (cos(v1), -sin(v1) * d1)
    elseif op === :sqrt
        r = sqrt(v1); return (r, d1 / (2r))
    elseif op === :exp
        r = exp(v1); return (r, r * d1)
    end
    error("unknown classical op $op")
end

# ---- nodes -----------------------------------------------------------------------------------
abstract type Node end
struct AllocNode <: Node
    reg::Int
    n::Int
end
struct ExtractNode <: Node
    qubit::Int
    reg::Int
    wire::Int
end
struct InsertNode <: Node
    reg_out::Int
    reg_in::Int
    wire::Int
    qubit::Int
end
struct DeallocNode <: Node
    reg::Int
end
struct GateNode <: Node
    name::Symbol
    params::Vector{CExpr}
    in_qubits::Vector{Int}
    out_qubits::Vector{Int}
    wires::Vector{Int}
    adjoint::Bool
    in_ctrls::Vector{Int}
    out_ctrls::Vector{Int}
    ctrl_wires::Vector{Int}
    ctrl_values::Vector{Bool}
end
struct NamedObsNode <: Node      # kind ∈ (:Identity, :PauliX, :PauliY, :PauliZ)
    obs::Int
    qubit::Int
    wire::Int
    kind::Symbol
end
struct TensorObsNode <: Node
    obs::Int
    terms::Vector{Int}
end
struct HamiltonianNode <: Node
    obs::Int
    coeffs::Vector{Float64}
    terms::Vector{Int}
end
struct CompBasisNode <: Node
    obs::Int
    qubits::Vector{Int}
    wires::Vector{Int}
end
struct ExpvalNode <: Node
    result::Int
    obs::Int
end
struct VarNode <: Node
    result::Int
    obs::Int
end
struct ProbsNode <: Node
    result::Int
    obs::Int
    nwires::Int
end
struct StateNode <: Node
    result::Int
    obs::Int
    nqubits::Int
end
struct SampleNode <: Node
    result::Int
    obs::Int
    nwires::Int
end

# ---- program ---------------------------------------------------------------------------------
struct ArgSpec
    kind::Symbol      # :scalar or :vector
    length::Int
end
struct ResultSpec
    kind::Symbol      # :expval, :var, :probs, :state, :sample
    size::Int         # vector length for :probs / :state, number of wires for :sample
end

mutable struct Program
    name::Symbol
    nqubits::Int
    args::Vector{ArgSpec}
    nodes::Vector{Node}
    results::Vector{Int}
    result_specs::Vector{ResultSpec}
    nvalues::Int
    diff_method::String
    scalar_return::Bool
end
Program(name::Symbol, args::Vector{ArgSpec}) =
    Program(name, 0, args, Node[], Int[], ResultSpec[], 0, "adjoint", true)

# ---- printing --------------------------------------------------------------------------------
_fmt(e::CConst) = string(e.value)
_fmt(e::CArg) = e.index == 0 ? "arg$(e.arg)" : "arg$(e.arg)[$(e.index)]"
_fmt(e::CCall) = string(e.op, "(", join(_fmt.(e.args), ", "), ")")
_q(ids) = join(("%q$i" for i in ids), ", ")

function Base.show(io::IO, prog::Program)
    args = join(("arg$i::" * (a.kind == :scalar ? "Float64" : "Vector{Float64}($(a.length))") for (i, a) in enumerate(prog.args)), ", ")
    println(io, "Program $(prog.name)($args) on $(prog.nqubits) qubits, diff_method = $(prog.diff_method)")
    for n in prog.nodes
        print(io, "  ")
        if n isa AllocNode
            println(io, "%r$(n.reg) = alloc($(n.n))")
        elseif n isa ExtractNode
            println(io, "%q$(n.qubit) = extract %r$(n.reg)[$(n.wire)]")
        elseif n isa InsertNode
            println(io, "%r$(n.reg_out) = insert %r$(n.reg_in)[$(n.wire)], %q$(n.qubit)")
        elseif n isa DeallocNode
            println(io, "dealloc %r$(n.reg)")
        elseif n isa GateNode
            outs = _q([n.out_ctrls; n.out_qubits])
            mods = (n.adjoint ? " adj" : "") * (isempty(n.in_ctrls) ? "" : " ctrls(" * _q(n.in_ctrls) * ") ctrlvals(" * join(n.ctrl_values, ", ") * ")")
            println(io, "$outs = $(n.name)(", join(_fmt.(n.params), ", "), ") ", _q(n.in_qubits), mods)
        elseif n isa NamedObsNode
            println(io, "%o$(n.obs) = namedobs %q$(n.qubit)[$(n.kind)]")
        elseif n isa TensorObsNode
            println(io, "%o$(n.obs) = tensor ", join(("%o$t" for t in n.terms), ", "))
        elseif n isa HamiltonianNode
            println(io, "%o$(n.obs) = hamiltonian(", join(n.coeffs, ", "), ") ", join(("%o$t" for t in n.terms), ", "))
        elseif n isa CompBasisNode
            println(io, "%o$(n.obs) = compbasis ", _q(n.qubits))
        elseif n isa ExpvalNode
            println(io, "%m$(n.result) = expval %o$(n.obs)")
        elseif n isa VarNode
            println(io, "%m$(n.result) = var %o$(n.obs)")
        elseif n isa ProbsNode
            println(io, "%m$(n.result) = probs %o$(n.obs)")
        elseif n isa StateNode
            println(io, "%m$(n.result) = state %o$(n.obs)")
        elseif n isa SampleNode
            println(io, "%m$(n.result) = sample %o$(n.obs)")
        end
    end
    print(io, "  return ", join(("%m$r" for r in prog.results), ", "))
end
