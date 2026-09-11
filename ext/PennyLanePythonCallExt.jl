# PennyLane (Python) bridge: PyDevice execution, qchem import, drawing. Activated by `using PythonCall`.
module PennyLanePythonCallExt

using PennyLane
using PythonCall
using PennyLane: Program, Node, GateNode, ExpvalNode, VarNode, ProbsNode, StateNode, SampleNode,
                 MeasureNode, IfNode, ForNode, WhileNode, CExpr, CCall, CValue,
                 PauliString, PauliSum, Observable, ResultSpec, QNode, PyDevice,
                 ceval, terms, program, normalize_args, record_observable!

const _qml = Ref{Any}(nothing)
function qml()
    _qml[] === nothing && (_qml[] = pyimport("pennylane"))
    _qml[]::Py
end
pywires(ws) = pylist([w - 1 for w in ws])

"""PennyLane operator for an observable of ours."""
function pyobs(o::Observable)
    q = qml()
    function pyterm(t::PauliString)
        c = t.coeff
        abs(imag(c)) <= 1e-12 * max(1.0, abs(c)) || throw(ArgumentError("observable coefficients must be real"))
        ops = [getproperty(q, Symbol(:Pauli, l))(w - 1) for (w, l) in t.word]
        base = isempty(ops) ? q.Identity(0) : length(ops) == 1 ? ops[1] : q.prod(ops...)
        real(c) == 1 ? base : q.s_prod(real(c), base)
    end
    ts = terms(o)
    length(ts) == 1 ? pyterm(ts[1]) : q.sum([pyterm(t) for t in ts]...)
end

# Mid-circuit measurement results become PennyLane `MeasurementValue`s; conditions on them map to
# `Conditional` operations (PennyLane then applies deferred measurement or dynamic one-shot execution).
# Purely classical conditions and static loops are evaluated at tape-construction time.
struct _TapeCtx
    q::Py
    ops::Vector{Py}
    obs::Dict{Int,Any}
    meas::Dict{Int,Py}           # result id => terminal measurement
    args::Vector{Any}
    env::Dict{Int,Any}            # classical values; measurement results hold a Py MeasurementValue
    depth::Int
    cond::Union{Nothing,Py}       # enclosing measurement condition (MeasurementValue) for Conditional ops
end
_TapeCtx(q, args) = _TapeCtx(q, Py[], Dict{Int,Any}(), Dict{Int,Py}(), args, Dict{Int,Any}(), 0, nothing)
_with_cond(c::_TapeCtx, cond) = _TapeCtx(c.q, c.ops, c.obs, c.meas, c.args, c.env, c.depth + 1, cond)

_ismv(x) = x isa Py
_identity() = pybuiltins.eval("lambda v: v", pydict())
# evaluate a classical expression; measurement values propagate as Py MeasurementValues
function _pyeval(c::_TapeCtx, e::CExpr)
    e isa CValue && haskey(c.env, e.id) && _ismv(c.env[e.id]) && return c.env[e.id]
    if e isa CCall
        a = [_pyeval(c, x) for x in e.args]
        if any(_ismv, a)
            e.op === :not && return ~a[1]
            e.op === :and && return a[1] & a[2]
            e.op === :or && return a[1] | a[2]
            e.op === :eq && return a[1] == a[2]
            e.op === :ne && return a[1] != a[2]
            throw(ArgumentError("PyDevice cannot lower classical op $(e.op) on measurement results"))
        end
    end
    ceval(e, c.args, c.env)
end

function _push_op!(c::_TapeCtx, op::Py)
    c.cond === nothing || (op = c.q.ops.op_math.Conditional(c.cond, op))
    push!(c.ops, op)
end

function emit_nodes!(c::_TapeCtx, nodes::Vector{Node})
    q = c.q
    for node in nodes
        if node isa GateNode
            params = [_pyeval(c, p) for p in node.params]
            any(_ismv, params) && throw(ArgumentError("PyDevice cannot use measurement results as gate parameters"))
            op = getproperty(q, node.name)(params...; wires=pywires(node.wires))
            node.adjoint && (op = q.adjoint(op))
            isempty(node.in_ctrls) || (op = q.ctrl(op; control=pywires(node.ctrl_wires), control_values=pylist(node.ctrl_values)))
            _push_op!(c, op)
        elseif node isa MeasureNode
            c.cond === nothing || throw(ArgumentError("PyDevice does not support measurements inside conditional branches"))
            mp = q.measurements.MidMeasureMP(; wires=q.wires.Wires(pylist([node.wire - 1])),
                                             postselect=node.postselect >= 0 ? node.postselect : pybuiltins.None,
                                             meas_uid=string("m", node.result))               # unique ids: deferred measurement keys on them
            push!(c.ops, mp)
            c.env[node.result] = q.measurements.MeasurementValue(pylist([mp]), _identity())
        elseif node isa IfNode
            cond = _pyeval(c, node.cond)
            if _ismv(cond)
                isempty(node.cout) || throw(ArgumentError("PyDevice cannot carry classical values out of a branch on a measurement result"))
                emit_nodes!(_with_cond(c, c.cond === nothing ? cond : (c.cond & cond)), node.then_body)
                isempty(node.else_body) || emit_nodes!(_with_cond(c, c.cond === nothing ? ~cond : (c.cond & ~cond)), node.else_body)
            else
                taken = cond::Bool
                emit_nodes!(c, taken ? node.then_body : node.else_body)
                for (k, id) in enumerate(node.cout)
                    c.env[id] = ceval(taken ? node.then_cyield[k] : node.else_cyield[k], c.args, c.env)
                end
            end
        elseif node isa ForNode
            for (k, e) in enumerate(node.cinit)
                c.env[node.cargs[k]] = ceval(e, c.args, c.env)
            end
            for i in node.start:node.step:node.stop
                c.env[node.index] = i
                emit_nodes!(c, node.body)
                vals = [_pyeval(c, e) for e in node.cyield]
                any(_ismv, vals) && throw(ArgumentError("PyDevice cannot carry measurement results across loop iterations"))
                for (k, v) in enumerate(vals)
                    c.env[node.cargs[k]] = v
                end
            end
            for (k, id) in enumerate(node.cout)
                c.env[id] = c.env[node.cargs[k]]
            end
        elseif node isa WhileNode
            for (k, e) in enumerate(node.cinit)
                c.env[node.cargs[k]] = ceval(e, c.args, c.env)
            end
            iters = 0
            while true
                cond = _pyeval(c, node.cond)
                _ismv(cond) && throw(ArgumentError("PyDevice cannot run a @trace while loop whose condition depends on a measurement result (repeat-until-success needs dynamic control flow)"))
                cond::Bool || break
                (iters += 1) > 100_000 && throw(ArgumentError("@trace while loop did not terminate within 100000 iterations"))
                emit_nodes!(c, node.body)
                for (k, e) in enumerate(node.cyield)
                    c.env[node.cargs[k]] = _pyeval(c, e)
                end
            end
            for (k, id) in enumerate(node.cout)
                c.env[id] = c.env[node.cargs[k]]
            end
        elseif record_observable!(c.obs, node)
        elseif node isa ExpvalNode
            c.meas[node.result] = q.expval(pyobs(c.obs[node.obs]))
        elseif node isa VarNode
            c.meas[node.result] = q.var(pyobs(c.obs[node.obs]))
        elseif node isa ProbsNode
            c.meas[node.result] = q.probs(; wires=pywires(c.obs[node.obs]))
        elseif node isa SampleNode
            c.meas[node.result] = q.sample(; wires=pywires(c.obs[node.obs]))
        elseif node isa StateNode
            c.meas[node.result] = q.state()
        end
    end
end

"""PennyLane `QuantumScript` for a program with concrete arguments."""
function to_tape(prog::Program, args::Vector{Any}, shots::Int)
    q = qml()
    c = _TapeCtx(q, args)
    emit_nodes!(c, prog.nodes)
    q.tape.QuantumScript(pylist(c.ops), pylist([c.meas[r] for r in prog.results]); shots=shots == 0 ? pybuiltins.None : shots)
end

function pydevice(dev::PyDevice, nqubits::Int)
    n = dev.nwires === nothing ? nqubits : dev.nwires
    n >= nqubits || throw(ArgumentError("PyDevice has $(dev.nwires) wires but the program uses $nqubits qubits"))
    dev.handle[] === nothing && (dev.handle[] = Dict{Int,Py}())
    cache = dev.handle[]::Dict{Int,Py}
    get!(cache, n) do
        qml().device(dev.name; wires=n, Pair{Symbol,Any}[k => v for (k, v) in dev.kwargs]...)   # shots travel with the tape
    end
end

function convert_result(r::Py, spec::ResultSpec)
    spec.kind in (:expval, :var) && return pyconvert(Float64, r)
    spec.kind === :probs && return pyconvert(Vector{Float64}, r)
    spec.kind === :sample && return pyconvert(Matrix{Int}, r)
    pyconvert(Vector{ComplexF64}, r)
end

function PennyLane._py_execute(dev::PyDevice, prog::Program, args::Vector{Any})
    tape = to_tape(prog, args, dev.shots)
    res = qml().execute(pylist([tape]), pydevice(dev, prog.nqubits))[0]
    items = prog.scalar_return ? Py[res] : Py[r for r in res]
    out = Any[convert_result(r, spec) for (r, spec) in zip(items, prog.result_specs)]
    prog.scalar_return ? out[1] : Tuple(out)
end

PennyLane.to_pennylane(qn::QNode, args...) = to_tape(program(qn, args...), normalize_args(args), PennyLane.device_shots(qn.dev))

PennyLane.draw(qn::QNode, args...; decimals::Integer=2) =
    pyconvert(String, qml().drawer.tape_text(PennyLane.to_pennylane(qn, args...); decimals=decimals))

function PennyLane.from_pennylane(op::Py)
    ps = qml().pauli.pauli_sentence(op)
    ts = PauliString[]
    for (pw, c) in ps.items()
        word = Pair{Int,Symbol}[pyconvert(Int, w) + 1 => Symbol(pyconvert(String, l)) for (w, l) in pw.items()]
        push!(ts, PauliString(pyconvert(Float64, pybuiltins.float(c)), word))
    end
    PauliSum(ts)
end

function PennyLane.molecular_hamiltonian(symbols::AbstractVector{<:AbstractString}, coordinates::AbstractVector{<:Real};
                                         charge::Integer=0, mult::Integer=1, basis::AbstractString="sto-3g",
                                         unit::AbstractString="bohr", method::AbstractString="dhf", kwargs...)
    q = qml()
    pnp = pyimport("pennylane.numpy")
    mol = q.qchem.Molecule(pylist(String.(symbols)), pnp.array(pylist(Float64.(coordinates)));
                           charge=charge, mult=mult, basis_name=basis, unit=unit)
    H, n = q.qchem.molecular_hamiltonian(mol; method=method, kwargs...)
    (PennyLane.from_pennylane(H), pyconvert(Int, n))
end

end # module
