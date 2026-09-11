# Julia-implemented MLIR passes on Catalyst's IR through Reactant's MLIR bindings (PLAN.md §3.5, item 2).
#
# Reactant bundles a newer MLIR than Catalyst, so Catalyst's dialects are unregistered there: the
# module travels as text, Catalyst's own CLI prints it in the generic op form, Reactant parses that,
# the pass rewrites operations through the MLIR C API, and the printed result goes back to the
# Catalyst compiler. Activated by `using Reactant`.
module PennyLaneReactantExt

using PennyLane
using Reactant
using PennyLane: catalyst_env, SELF_INVERSE_GATES
import PennyLane: mlir_pass
const IR = Reactant.MLIR.IR
const API = Reactant.MLIR.API

"""Catalyst MLIR text in the generic operation form (parseable without Catalyst's dialects)."""
function generic_form(text::AbstractString)
    env = catalyst_env()
    dir = mktempdir()
    path = joinpath(dir, "in.mlir")
    write(path, text)
    read(pipeline(`$(env.cli) --tool=opt --catalyst-pipeline=pipe\(symbol-dce\) --mlir-print-op-generic $path`; stderr=devnull), String)
end

function _attr(op::IR.Operation, name::String)
    a = API.mlirOperationGetAttributeByName(op, name)
    IR.mlirIsNull(a) ? nothing : IR.Attribute(a)          # mlirAttributeIsNull is header-only
end
function _string_attr(op, name)
    a = _attr(op, name)
    (a === nothing || !API.mlirAttributeIsAString(a)) ? nothing : String(a)
end
_unit_attr(op, name) = (a = _attr(op, name); a !== nothing && API.mlirAttributeIsAUnit(a))
_isf64(v::IR.Value) = string(IR.type(v)) == "f64"

# the operation defining a value, or nothing for block arguments
_definer(v::IR.Value) = API.mlirValueIsAOpResult(v) ? IR.op_owner(v) : nothing

# Does `op` undo `prev` (same gate on the same qubit values, adjoint pair or self-inverse)?
function _inverse_pair(prev::IR.Operation, op::IR.Operation)
    gate = _string_attr(op, "gate_name")
    gate === nothing && return false
    _string_attr(prev, "gate_name") == gate || return false
    nops = API.mlirOperationGetNumOperands(op)
    nres = API.mlirOperationGetNumResults(op)
    (nops == API.mlirOperationGetNumOperands(prev) && nres == API.mlirOperationGetNumResults(prev)) || return false
    nparams = count(i -> _isf64(IR.operand(op, i)), 1:nops)
    nops - nparams == nres || return false                       # no control values (i1 operands)
    for i in 1:nparams
        IR.operand(op, i) == IR.operand(prev, i) || return false   # same parameter SSA values
    end
    for i in 1:nres
        IR.operand(op, nparams + i) == IR.result(prev, i) || return false
    end
    adj, padj = _unit_attr(op, "adjoint"), _unit_attr(prev, "adjoint")
    (Symbol(gate) in SELF_INVERSE_GATES && nparams == 0 && adj == padj) || adj != padj
end

function _cancel_in_block!(block::IR.Block)
    changed = false
    while true
        found = false
        for op in collect(block)
            for region in op, inner in region          # nested regions first
                changed |= _cancel_in_block!(inner)
            end
            IR.name(op) == "quantum.custom" || continue
            nops = API.mlirOperationGetNumOperands(op)
            nops == 0 && continue
            nres = API.mlirOperationGetNumResults(op)
            nparams = nops - nres
            nparams >= 0 || continue
            prev = _definer(IR.operand(op, nparams + 1))
            (prev === nothing || IR.name(prev) != "quantum.custom" || !_inverse_pair(prev, op)) && continue
            for i in 1:nres
                API.mlirValueReplaceAllUsesOfWith(IR.result(op, i), IR.operand(prev, nparams + i))
            end
            API.mlirOperationDestroy(IR.rmfromparent!(op))
            API.mlirOperationDestroy(IR.rmfromparent!(prev))
            found = changed = true
            break
        end
        found || break
    end
    changed
end

const MLIR_PASSES = Dict{Symbol,Function}(:cancel_inverses => mod -> begin
    for region in IR.Operation(mod), block in region
        _cancel_in_block!(block)
    end
end)

function PennyLane.mlir_pass(text::AbstractString, passes::Symbol...)
    src = generic_form(text)
    ctx = Reactant.ReactantContext()
    IR.allow_unregistered_dialects!(true; context=ctx)
    IR.activate(ctx)
    try
        mod = parse(IR.Module, src)
        for p in passes
            haskey(MLIR_PASSES, p) || throw(ArgumentError("unknown MLIR pass $p (available: $(collect(keys(MLIR_PASSES))))"))
            MLIR_PASSES[p](mod)
        end
        IR.verifyall(IR.Operation(mod)) || error("module failed verification after passes $(passes)")
        return sprint(show, mod)
    finally
        IR.deactivate(ctx)
    end
end

end # module
