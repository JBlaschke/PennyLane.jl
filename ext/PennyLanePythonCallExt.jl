# PennyLane (Python) bridge: PyDevice execution, qchem import, drawing. Activated by `using PythonCall`.
module PennyLanePythonCallExt

using PennyLane
using PythonCall
using PennyLane: Program, Node, GateNode, ExpvalNode, VarNode, ProbsNode, StateNode, SampleNode,
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

"""PennyLane `QuantumScript` for a program with concrete arguments."""
function to_tape(prog::Program, args::Vector{Any}, shots::Int)
    q = qml()
    ops = Py[]
    meas = Py[]
    obs = Dict{Int,Any}()
    for node in prog.nodes
        if node isa GateNode
            params = [ceval(p, args) for p in node.params]
            op = getproperty(q, node.name)(params...; wires=pywires(node.wires))
            node.adjoint && (op = q.adjoint(op))
            isempty(node.in_ctrls) || (op = q.ctrl(op; control=pywires(node.ctrl_wires), control_values=pylist(node.ctrl_values)))
            push!(ops, op)
        elseif record_observable!(obs, node)
        elseif node isa ExpvalNode
            push!(meas, q.expval(pyobs(obs[node.obs])))
        elseif node isa VarNode
            push!(meas, q.var(pyobs(obs[node.obs])))
        elseif node isa ProbsNode
            push!(meas, q.probs(; wires=pywires(obs[node.obs])))
        elseif node isa SampleNode
            push!(meas, q.sample(; wires=pywires(obs[node.obs])))
        elseif node isa StateNode
            push!(meas, q.state())
        end
    end
    q.tape.QuantumScript(pylist(ops), pylist(meas); shots=shots == 0 ? pybuiltins.None : shots)
end

# one Python device per wire count (or the fixed `wires` given by the user)
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
