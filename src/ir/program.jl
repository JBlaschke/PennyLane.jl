# Hybrid program IR.
#
# A `Program` is a value-semantic SSA program whose nodes mirror Catalyst's `quantum` dialect
# (alloc/extract/insert/dealloc, gates, observables, measurements, mid-circuit measurement)
# plus structured control flow (`scf.if` / `scf.for` / `scf.while`) and a small typed classical
# expression language (f64, index/int, bool). Every backend consumes this IR; `to_mlir` prints
# it in Catalyst's textual format. When MLIR.jl stabilises the printer can be swapped for a
# builder without touching the tracer or the backends (PLAN.md §3.2, decision 4).

# ---- classical expressions -------------------------------------------------------------------
abstract type CExpr end
struct CConst <: CExpr            # f64 constant
    value::Float64
end
struct CIConst <: CExpr           # integer (index) constant
    value::Int
end
struct CBConst <: CExpr           # boolean constant
    value::Bool
end
struct CArg <: CExpr              # function argument `arg` (1-based); index 0 = scalar, ≥1 = static element of a vector
    arg::Int
    index::Int
end
struct CArgDyn <: CExpr           # element of a vector argument at a dynamic 0-based index expression
    arg::Int
    index::CExpr
end
struct CValue <: CExpr            # classical SSA value defined by a node or region (:f64, :int, :bool)
    id::Int
    kind::Symbol
end
struct CCall <: CExpr
    op::Symbol
    args::Vector{CExpr}
end

# op => result kind
const CLASSICAL_OPS = Dict{Symbol,Symbol}(
    :add => :f64, :sub => :f64, :mul => :f64, :div => :f64, :neg => :f64,
    :sin => :f64, :cos => :f64, :sqrt => :f64, :exp => :f64,
    :lt => :bool, :le => :bool, :gt => :bool, :ge => :bool, :eq => :bool, :ne => :bool,
    :not => :bool, :and => :bool, :or => :bool,
    :iadd => :int, :isub => :int, :imul => :int, :ilt => :bool, :ile => :bool, :ieq => :bool,
    :itof => :f64,
)
ckind(::CConst) = :f64
ckind(::CIConst) = :int
ckind(::CBConst) = :bool
ckind(::CArg) = :f64
ckind(::CArgDyn) = :f64
ckind(e::CValue) = e.kind
ckind(e::CCall) = CLASSICAL_OPS[e.op]
isconst(e::CExpr) = e isa CConst || e isa CIConst || e isa CBConst
constval(e::CConst) = e.value
constval(e::CIConst) = e.value
constval(e::CBConst) = e.value
mkconst(kind::Symbol, v) = kind === :f64 ? CConst(Float64(v)) : kind === :int ? CIConst(Int(v)) : CBConst(Bool(v))

const NOENV = Dict{Int,Any}()
ceval(e::CConst, args, env=NOENV) = e.value
ceval(e::CIConst, args, env=NOENV) = e.value
ceval(e::CBConst, args, env=NOENV) = e.value
ceval(e::CArg, args, env=NOENV) = e.index == 0 ? Float64(args[e.arg]) : Float64(args[e.arg][e.index])
ceval(e::CArgDyn, args, env=NOENV) = Float64(args[e.arg][ceval(e.index, args, env)+1])
ceval(e::CValue, args, env=NOENV) = env[e.id]
function ceval(e::CCall, args, env=NOENV)
    v = [ceval(a, args, env) for a in e.args]
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
    (op === :lt || op === :ilt) && return v[1] < v[2]
    (op === :le || op === :ile) && return v[1] <= v[2]
    op === :gt && return v[1] > v[2]
    op === :ge && return v[1] >= v[2]
    (op === :eq || op === :ieq) && return v[1] == v[2]
    op === :ne && return v[1] != v[2]
    op === :not && return !v[1]
    op === :and && return v[1] && v[2]
    op === :or && return v[1] || v[2]
    op === :iadd && return v[1] + v[2]
    op === :isub && return v[1] - v[2]
    op === :imul && return v[1] * v[2]
    op === :itof && return Float64(v[1])
    error("unknown classical op $op")
end

hasarg(::CConst) = false
hasarg(::CIConst) = false
hasarg(::CBConst) = false
hasarg(::CArg) = true
hasarg(::CArgDyn) = true
hasarg(::CValue) = false
hasarg(e::CCall) = any(hasarg, e.args)

"""Value and derivative of a classical f64 expression with respect to argument slot `(a, i)` (forward mode)."""
cdual(e::CConst, args, a, i) = (e.value, 0.0)
cdual(e::CArg, args, a, i) = (ceval(e, args), (e.arg == a && e.index == i) ? 1.0 : 0.0)
cdual(e::CExpr, args, a, i) = throw(ArgumentError("cannot differentiate classical values produced inside control flow; use finite differences or Catalyst's adjoint"))
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
    throw(ArgumentError("cannot differentiate through classical op $op"))
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
"""Mid-circuit measurement: `result` is a boolean classical value; `postselect` is -1, 0 or 1."""
struct MeasureNode <: Node
    result::Int
    in_qubit::Int
    out_qubit::Int
    wire::Int
    postselect::Int
end
# Control flow. Qubit values enter a region for the touched `wires` and leave as fresh values;
# classical carried values (`c*`) have the given kinds.
struct IfNode <: Node
    cond::CExpr
    wires::Vector{Int}
    in_qubits::Vector{Int}
    out_qubits::Vector{Int}
    then_body::Vector{Node}
    else_body::Vector{Node}
    then_yield::Vector{Int}
    else_yield::Vector{Int}
    cout::Vector{Int}
    ckinds::Vector{Symbol}
    then_cyield::Vector{CExpr}
    else_cyield::Vector{CExpr}
end
struct ForNode <: Node           # iterates start:step:stop (inclusive, step > 0); `index` is the :int loop value
    start::Int
    stop::Int
    step::Int
    index::Int
    wires::Vector{Int}
    in_qubits::Vector{Int}
    args::Vector{Int}
    yields::Vector{Int}
    out_qubits::Vector{Int}
    cinit::Vector{CExpr}
    cargs::Vector{Int}
    cyield::Vector{CExpr}
    cout::Vector{Int}
    ckinds::Vector{Symbol}
    body::Vector{Node}
end
struct WhileNode <: Node         # cond is evaluated on the block arguments before each iteration
    wires::Vector{Int}
    in_qubits::Vector{Int}
    args::Vector{Int}
    yields::Vector{Int}
    out_qubits::Vector{Int}
    cinit::Vector{CExpr}
    cargs::Vector{Int}
    cyield::Vector{CExpr}
    cout::Vector{Int}
    ckinds::Vector{Symbol}
    cond::CExpr
    body::Vector{Node}
end

"""All nodes of a node list including nested regions (depth first)."""
function all_nodes(nodes::Vector{Node}, acc::Vector{Node}=Node[])
    for n in nodes
        push!(acc, n)
        n isa IfNode && (all_nodes(n.then_body, acc); all_nodes(n.else_body, acc))
        n isa ForNode && all_nodes(n.body, acc)
        n isa WhileNode && all_nodes(n.body, acc)
    end
    acc
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

all_nodes(prog::Program) = all_nodes(prog.nodes)
has_control_flow(prog::Program) = any(n -> n isa IfNode || n isa ForNode || n isa WhileNode, all_nodes(prog))
has_mcm(prog::Program) = any(n -> n isa MeasureNode, all_nodes(prog))

# ---- printing --------------------------------------------------------------------------------
_fmt(e::CConst) = string(e.value)
_fmt(e::CIConst) = string(e.value)
_fmt(e::CBConst) = string(e.value)
_fmt(e::CArg) = e.index == 0 ? "arg$(e.arg)" : "arg$(e.arg)[$(e.index)]"
_fmt(e::CArgDyn) = "arg$(e.arg)[" * _fmt(e.index) * "+1]"
_fmt(e::CValue) = "%v$(e.id)"
_fmt(e::CCall) = string(e.op, "(", join(_fmt.(e.args), ", "), ")")
_q(ids) = join(("%q$i" for i in ids), ", ")
_v(ids) = join(("%v$i" for i in ids), ", ")

function _show_nodes(io::IO, nodes::Vector{Node}, ind::String)
    for n in nodes
        print(io, ind)
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
            mods = (n.adjoint ? " adj" : "") *
                   (isempty(n.in_ctrls) ? "" : " ctrls(" * _q(n.in_ctrls) * ") ctrlvals(" * join(n.ctrl_values, ", ") * ")")
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
        elseif n isa MeasureNode
            println(io, "%v$(n.result), %q$(n.out_qubit) = measure %q$(n.in_qubit)", n.postselect < 0 ? "" : " postselect $(n.postselect)")
        elseif n isa IfNode
            println(io, join([("%q$i" for i in n.out_qubits)..., ("%v$i" for i in n.cout)...], ", "), isempty(n.out_qubits) && isempty(n.cout) ? "" : " = ",
                    "if ", _fmt(n.cond), " (", _q(n.in_qubits), ") {")
            _show_nodes(io, n.then_body, ind * "  ")
            println(io, ind, "  yield ", join([("%q$i" for i in n.then_yield)..., _fmt.(n.then_cyield)...], ", "))
            println(io, ind, "} else {")
            _show_nodes(io, n.else_body, ind * "  ")
            println(io, ind, "  yield ", join([("%q$i" for i in n.else_yield)..., _fmt.(n.else_cyield)...], ", "))
            println(io, ind, "}")
        elseif n isa ForNode
            println(io, join([("%q$i" for i in n.out_qubits)..., ("%v$i" for i in n.cout)...], ", "), isempty(n.out_qubits) && isempty(n.cout) ? "" : " = ",
                    "for %v$(n.index) in $(n.start):$(n.step):$(n.stop) iter_args(",
                    join([("%q$a = %q$i" for (a, i) in zip(n.args, n.in_qubits))..., ("%v$a = " * _fmt(e) for (a, e) in zip(n.cargs, n.cinit))...], ", "), ") {")
            _show_nodes(io, n.body, ind * "  ")
            println(io, ind, "  yield ", join([("%q$i" for i in n.yields)..., _fmt.(n.cyield)...], ", "))
            println(io, ind, "}")
        elseif n isa WhileNode
            println(io, join([("%q$i" for i in n.out_qubits)..., ("%v$i" for i in n.cout)...], ", "), isempty(n.out_qubits) && isempty(n.cout) ? "" : " = ",
                    "while ", _fmt(n.cond), " iter_args(",
                    join([("%q$a = %q$i" for (a, i) in zip(n.args, n.in_qubits))..., ("%v$a = " * _fmt(e) for (a, e) in zip(n.cargs, n.cinit))...], ", "), ") {")
            _show_nodes(io, n.body, ind * "  ")
            println(io, ind, "  yield ", join([("%q$i" for i in n.yields)..., _fmt.(n.cyield)...], ", "))
            println(io, ind, "}")
        end
    end
end

function Base.show(io::IO, prog::Program)
    args = join(("arg$i::" * (a.kind == :scalar ? "Float64" : "Vector{Float64}($(a.length))") for (i, a) in enumerate(prog.args)), ", ")
    println(io, "Program $(prog.name)($args) on $(prog.nqubits) qubits, diff_method = $(prog.diff_method)")
    _show_nodes(io, prog.nodes, "  ")
    print(io, "  return ", join(("%m$r" for r in prog.results), ", "))
end
