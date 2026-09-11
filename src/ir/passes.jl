# Program-level optimisation passes on the value-semantic IR.
#
# Because qubits are SSA values, "consecutive gates on the same qubits" is a purely local
# pattern: gate g₂ consumes exactly the values gate g₁ produced. Passes run on every backend's
# program; Catalyst applies its own `cancel-inverses` / `merge-rotations` as well.

const SELF_INVERSE_GATES = Set([:PauliX, :PauliY, :PauliZ, :Hadamard, :CNOT, :CY, :CZ, :SWAP, :Toffoli, :CSWAP])
const MERGEABLE_ROTATIONS = Set([:RX, :RY, :RZ, :PhaseShift, :ControlledPhaseShift, :CRX, :CRY, :CRZ,
                                 :IsingXX, :IsingYY, :IsingZZ, :MultiRZ])

Base.:(==)(a::CCall, b::CCall) = a.op == b.op && a.args == b.args
Base.hash(e::CCall, h::UInt) = hash(e.args, hash(e.op, h))
Base.:(==)(a::CArgDyn, b::CArgDyn) = a.arg == b.arg && a.index == b.index
Base.hash(e::CArgDyn, h::UInt) = hash(e.index, hash(e.arg, h))

# rename qubit ids (removed gates: outputs -> inputs)
_r(id::Int, m::Dict{Int,Int}) = get(m, id, id)
_r(ids::Vector{Int}, m::Dict{Int,Int}) = Int[_r(i, m) for i in ids]
rename_ids(n::Node, m) = n
rename_ids(n::InsertNode, m) = InsertNode(n.reg_out, n.reg_in, n.wire, _r(n.qubit, m))
rename_ids(n::GateNode, m) = GateNode(n.name, n.params, _r(n.in_qubits, m), n.out_qubits, n.wires, n.adjoint,
                                      _r(n.in_ctrls, m), n.out_ctrls, n.ctrl_wires, n.ctrl_values)
rename_ids(n::NamedObsNode, m) = NamedObsNode(n.obs, _r(n.qubit, m), n.wire, n.kind)
rename_ids(n::CompBasisNode, m) = CompBasisNode(n.obs, _r(n.qubits, m), n.wires)
rename_ids(n::MeasureNode, m) = MeasureNode(n.result, _r(n.in_qubit, m), n.out_qubit, n.wire, n.postselect)
rename_ids(n::IfNode, m) = IfNode(n.cond, n.wires, _r(n.in_qubits, m), n.out_qubits,
                                  Node[rename_ids(x, m) for x in n.then_body], Node[rename_ids(x, m) for x in n.else_body],
                                  _r(n.then_yield, m), _r(n.else_yield, m), n.cout, n.ckinds, n.then_cyield, n.else_cyield)
rename_ids(n::ForNode, m) = ForNode(n.start, n.stop, n.step, n.index, n.wires, _r(n.in_qubits, m), n.args, n.yields, n.out_qubits,
                                    n.cinit, n.cargs, n.cyield, n.cout, n.ckinds, n.body)
rename_ids(n::WhileNode, m) = WhileNode(n.wires, _r(n.in_qubits, m), n.args, n.yields, n.out_qubits,
                                        n.cinit, n.cargs, n.cyield, n.cout, n.ckinds, n.cond, n.body)
function rename_ids!(nodes::Vector{Node}, m::Dict{Int,Int})
    for i in eachindex(nodes)
        nodes[i] = rename_ids(nodes[i], m)
    end
    nodes
end

_same_wiring(g1::GateNode, g2::GateNode) =
    g2.in_qubits == g1.out_qubits && g2.in_ctrls == g1.out_ctrls && g2.ctrl_values == g1.ctrl_values
_inverse_pair(g1::GateNode, g2::GateNode) =
    g1.name === g2.name && _same_wiring(g1, g2) &&
    ((g1.name in SELF_INVERSE_GATES && isempty(g1.params) && g1.adjoint == g2.adjoint) ||
     (g1.adjoint != g2.adjoint && g1.params == g2.params))

# the GateNode that defines a qubit value, by position in `nodes`
function _definers(nodes::Vector{Node})
    defs = Dict{Int,Int}()
    for (i, n) in enumerate(nodes)
        n isa GateNode || continue
        for q in n.out_qubits
            defs[q] = i
        end
        for q in n.out_ctrls
            defs[q] = i
        end
    end
    defs
end

"""
    cancel_inverses!(nodes) -> Bool

Remove pairs of consecutive gates that compose to the identity (self-inverse gates applied
twice, or a gate followed by its adjoint) on the same qubit values.
"""
function cancel_inverses!(nodes::Vector{Node})
    changed = false
    for n in nodes
        n isa IfNode && (changed |= cancel_inverses!(n.then_body) | cancel_inverses!(n.else_body))
        n isa ForNode && (changed |= cancel_inverses!(n.body))
        n isa WhileNode && (changed |= cancel_inverses!(n.body))
    end
    while true
        defs = _definers(nodes)
        found = false
        for (i, n) in enumerate(nodes)
            n isa GateNode && !isempty(n.in_qubits) || continue
            j = get(defs, n.in_qubits[1], 0)
            j == 0 && continue
            g1 = nodes[j]::GateNode
            _inverse_pair(g1, n) || continue
            m = Dict{Int,Int}(zip([n.out_qubits; n.out_ctrls], [g1.in_qubits; g1.in_ctrls]))
            deleteat!(nodes, (j, i))
            rename_ids!(nodes, m)
            found = changed = true
            break
        end
        found || break
    end
    changed
end

"""
    merge_rotations!(nodes) -> Bool

Fuse consecutive rotations of the same kind on the same qubit values into one rotation with the
summed angle.
"""
function merge_rotations!(nodes::Vector{Node})
    changed = false
    for n in nodes
        n isa IfNode && (changed |= merge_rotations!(n.then_body) | merge_rotations!(n.else_body))
        n isa ForNode && (changed |= merge_rotations!(n.body))
        n isa WhileNode && (changed |= merge_rotations!(n.body))
    end
    while true
        defs = _definers(nodes)
        found = false
        for (i, n) in enumerate(nodes)
            n isa GateNode && n.name in MERGEABLE_ROTATIONS || continue
            j = get(defs, n.in_qubits[1], 0)
            j == 0 && continue
            g1 = nodes[j]::GateNode
            (g1.name === n.name && g1.adjoint == n.adjoint && _same_wiring(g1, n)) || continue
            nodes[i] = GateNode(n.name, CExpr[_fold(:add, g1.params[1], n.params[1])], g1.in_qubits, n.out_qubits, n.wires,
                                n.adjoint, g1.in_ctrls, n.out_ctrls, n.ctrl_wires, n.ctrl_values)
            deleteat!(nodes, j)
            found = changed = true
            break
        end
        found || break
    end
    changed
end

const PASSES = Dict{Symbol,Function}(:cancel_inverses => cancel_inverses!, :merge_rotations => merge_rotations!)

"""
    optimize(prog::Program; passes=[:cancel_inverses, :merge_rotations]) -> Program

Optimised copy of a program. Passes run repeatedly until nothing changes.
"""
function optimize(prog::Program; passes=[:cancel_inverses, :merge_rotations])
    p = deepcopy(prog)
    fs = [PASSES[s] for s in passes]
    while true
        any(f -> f(p.nodes), fs) || break
    end
    p
end

"""Number of gates in a program (including gates inside regions, counted once)."""
gate_count(prog::Program) = count(n -> n isa GateNode, all_nodes(prog))
