# Gradients.
#
#   :parameter_shift  hardware-compatible shift rules (default on simulators and PyDevice)
#   :adjoint          Catalyst's compiled adjoint differentiation (default on CatalystDevice)
#   :finitediff       central differences (any device; useful for cross-checks)
#
# Parameter-shift works at the level of gate parameters: for every parametrised gate the
# derivative ∂f/∂p is obtained from shifted executions, and the chain rule through the traced
# classical expression p(args) is applied with `cdual`.

const _C4P = (sqrt(2) + 1) / (4 * sqrt(2))
const _C4M = (sqrt(2) - 1) / (4 * sqrt(2))
const TWO_TERM_RULE = [(0.5, π / 2), (-0.5, -π / 2)]
const FOUR_TERM_RULE = [(_C4P, π / 2), (-_C4P, -π / 2), (-_C4M, 3π / 2), (_C4M, -3π / 2)]

"""Shift rule `[(coefficient, shift), ...]` for the parameters of a gate node: ∂f/∂p = Σ c·f(p + s)."""
function shift_rule(node::GateNode)
    kind = get(SHIFT_KIND, node.name, nothing)
    kind === nothing && throw(ArgumentError("no parameter-shift rule for $(node.name); use gradient(...; method=:finitediff)"))
    kind === :proj && return TWO_TERM_RULE
    kind === :ctrlhalf && return FOUR_TERM_RULE
    isempty(node.in_ctrls) ? TWO_TERM_RULE : FOUR_TERM_RULE
end

"""Copy of `prog` with parameter `pi` of gate node `ni` shifted by `s`."""
function shifted(prog::Program, ni::Int, pi::Int, s::Float64)
    node = prog.nodes[ni]::GateNode
    params = copy(node.params)
    params[pi] = CCall(:add, CExpr[params[pi], CConst(s)])
    nodes = copy(prog.nodes)
    nodes[ni] = GateNode(node.name, params, node.in_qubits, node.out_qubits, node.wires, node.adjoint,
                         node.in_ctrls, node.out_ctrls, node.ctrl_wires, node.ctrl_values)
    Program(prog.name, prog.nqubits, prog.args, nodes, prog.results, prog.result_specs, prog.nvalues,
            prog.diff_method, prog.scalar_return)
end

_zero_grads(args) = Any[a isa Float64 ? 0.0 : zeros(Float64, length(a)) for a in args]
_arg_slots(args) = [(a, i) for (a, arg) in enumerate(args) for i in (arg isa Float64 ? (0:0) : (1:length(arg)))]
_pack(grads) = length(grads) == 1 ? grads[1] : Tuple(grads)

function _check_scalar_expval(prog::Program, what)
    prog.scalar_return && prog.result_specs[1].kind === :expval ||
        throw(ArgumentError("$what needs a QNode returning a single expval"))
end

function parameter_shift_gradient(dev::AbstractDevice, prog::Program, args::Vector{Any})
    _check_scalar_expval(prog, "the parameter-shift gradient")
    grads = _zero_grads(args)
    slots = _arg_slots(args)
    for (ni, node) in enumerate(prog.nodes)
        node isa GateNode || continue
        for (pi, e) in enumerate(node.params)
            hasarg(e) || continue
            deps = [(a, i, cdual(e, args, a, i)[2]) for (a, i) in slots]
            filter!(t -> t[3] != 0, deps)
            isempty(deps) && continue
            dfdp = sum(c * (execute(dev, shifted(prog, ni, pi, s), args)::Float64) for (c, s) in shift_rule(node))
            for (a, i, d) in deps
                i == 0 ? (grads[a] += dfdp * d) : (grads[a][i] += dfdp * d)
            end
        end
    end
    _pack(grads)
end

function finitediff_gradient(dev::AbstractDevice, prog::Program, args::Vector{Any}, h::Float64)
    prog.scalar_return && prog.result_specs[1].kind in (:expval, :var) ||
        throw(ArgumentError("the finite-difference gradient needs a QNode returning a single expval or var"))
    f(x) = execute(dev, prog, x)::Float64
    grads = _zero_grads(args)
    for (a, i) in _arg_slots(args)
        ap = copy(args); am = copy(args)
        if i == 0
            ap[a] = args[a] + h; am[a] = args[a] - h
            grads[a] = (f(ap) - f(am)) / (2h)
        else
            ap[a] = copy(args[a]); am[a] = copy(args[a])
            ap[a][i] += h; am[a][i] -= h
            grads[a][i] = (f(ap) - f(am)) / (2h)
        end
    end
    _pack(grads)
end

"""
    gradient(qn::QNode, args...; method=:auto, h=1e-6)

Gradient of a single-expval QNode with respect to all of its arguments (one entry per argument,
a tuple when there are several). Methods: `:parameter_shift` (default on simulators and
`PyDevice`, hardware compatible), `:adjoint` (default on `CatalystDevice`, compiled by
Catalyst), `:finitediff` (any device, step `h`).
"""
function gradient(qn::QNode, args...; method::Symbol=:auto, h::Real=1e-6)
    a = normalize_args(args)
    prog = program(qn, a...)
    method === :auto && (method = qn.dev isa CatalystDevice ? :adjoint : :parameter_shift)
    method === :adjoint && return compiled_gradient(qn, a...)
    method === :parameter_shift && return parameter_shift_gradient(qn.dev, prog, a)
    method === :finitediff && return finitediff_gradient(qn.dev, prog, a, Float64(h))
    throw(ArgumentError("unknown gradient method $method (expected :auto, :parameter_shift, :adjoint or :finitediff)"))
end
