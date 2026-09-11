# Textual MLIR emission in Catalyst's dialects (format verified against catalyst 0.15.0 output).

const LIGHTNING_KWARGS = "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"

function _mlirfloat(v::Float64)
    isfinite(v) || throw(ArgumentError("non-finite constant $v in circuit"))
    s = @sprintf("%.17g", v)
    (occursin('.', s) || occursin('e', s)) ? s : s * ".0"
end
_argtype(a::ArgSpec) = a.kind === :scalar ? "tensor<f64>" : "tensor<$(a.length)xf64>"
function _restype(r::ResultSpec, shots::Integer)
    r.kind in (:expval, :var) && return "tensor<f64>"
    r.kind === :probs && return "tensor<$(r.size)xf64>"
    r.kind === :state && return "tensor<$(r.size)xcomplex<f64>>"
    shots > 0 || throw(ArgumentError("sample() needs a device with shots > 0, e.g. CatalystDevice(shots=1000)"))
    "tensor<$(shots)x$(r.size)xf64>"
end
_paren(types) = length(types) == 1 ? types[1] : "(" * join(types, ", ") * ")"
_mlirkind(k::Symbol) = k === :f64 ? "f64" : k === :int ? "index" : "i1"

mutable struct _Emitter
    io::IOBuffer
    counter::Int
    exprs::Dict{CExpr,String}
    indent::String
end
_fresh(em::_Emitter, prefix) = (em.counter += 1; "%$(prefix)$(em.counter)")
_line(em::_Emitter, s) = println(em.io, em.indent, s)

const _F64_OPS = Dict(:add => "arith.addf", :sub => "arith.subf", :mul => "arith.mulf", :div => "arith.divf",
                      :neg => "arith.negf", :sin => "math.sin", :cos => "math.cos", :sqrt => "math.sqrt", :exp => "math.exp")
const _CMPF = Dict(:lt => "olt", :le => "ole", :gt => "ogt", :ge => "oge", :eq => "oeq", :ne => "one")
const _CMPI = Dict(:ilt => "slt", :ile => "sle", :ieq => "eq")
const _INT_OPS = Dict(:iadd => "arith.addi", :isub => "arith.subi", :imul => "arith.muli")

function _emit_expr!(em::_Emitter, prog::Program, e::CExpr)
    e isa CValue && return "%v$(e.id)"
    get!(em.exprs, e) do
        if e isa CConst
            n = _fresh(em, "c")
            _line(em, "$n = arith.constant $(_mlirfloat(e.value)) : f64")
            n
        elseif e isa CIConst
            n = _fresh(em, "i")
            _line(em, "$n = arith.constant $(e.value) : index")
            n
        elseif e isa CBConst
            n = _fresh(em, "b")
            _line(em, "$n = arith.constant $(e.value)")
            n
        elseif e isa CArg
            n = _fresh(em, "c")
            if e.index == 0
                _line(em, "$n = tensor.extract %arg$(e.arg-1)[] : tensor<f64>")
            else
                idx = _emit_expr!(em, prog, CIConst(e.index - 1))
                _line(em, "$n = tensor.extract %arg$(e.arg-1)[$idx] : $(_argtype(prog.args[e.arg]))")
            end
            n
        elseif e isa CArgDyn
            idx = _emit_expr!(em, prog, e.index)
            n = _fresh(em, "c")
            _line(em, "$n = tensor.extract %arg$(e.arg-1)[$idx] : $(_argtype(prog.args[e.arg]))")
            n
        else
            ops = [_emit_expr!(em, prog, a) for a in e.args]
            op = e.op
            if haskey(_F64_OPS, op)
                n = _fresh(em, "c")
                _line(em, "$n = $(_F64_OPS[op]) $(join(ops, ", ")) : f64")
                n
            elseif haskey(_CMPF, op)
                n = _fresh(em, "b")
                _line(em, "$n = arith.cmpf $(_CMPF[op]), $(ops[1]), $(ops[2]) : f64")
                n
            elseif haskey(_CMPI, op)
                n = _fresh(em, "b")
                _line(em, "$n = arith.cmpi $(_CMPI[op]), $(ops[1]), $(ops[2]) : index")
                n
            elseif haskey(_INT_OPS, op)
                n = _fresh(em, "i")
                _line(em, "$n = $(_INT_OPS[op]) $(ops[1]), $(ops[2]) : index")
                n
            elseif op === :not
                t = _emit_expr!(em, prog, CBConst(true))
                n = _fresh(em, "b")
                _line(em, "$n = arith.xori $(ops[1]), $t : i1")
                n
            elseif op === :and || op === :or
                n = _fresh(em, "b")
                _line(em, "$n = arith.$(op === :and ? "andi" : "ori") $(ops[1]), $(ops[2]) : i1")
                n
            elseif op === :itof
                t = _fresh(em, "i")
                _line(em, "$t = arith.index_cast $(ops[1]) : index to i64")
                n = _fresh(em, "c")
                _line(em, "$n = arith.sitofp $t : i64 to f64")
                n
            else
                error("cannot emit classical op $op")
            end
        end
    end
end

# emit a region body: constants defined inside are not visible afterwards
function _emit_region!(f, em::_Emitter)
    saved = copy(em.exprs)
    em.indent *= "  "
    try
        f()
    finally
        em.indent = em.indent[1:end-2]
        em.exprs = saved
    end
end

_qtypes(n) = join(fill("!quantum.bit", n), ", ")
_regiontypes(nq, kinds) = join([fill("!quantum.bit", nq); _mlirkind.(kinds)], ", ")

function _emit_nodes!(em::_Emitter, prog::Program, nodes::Vector{Node}, ctx)
    for node in nodes
        if node isa AllocNode
            _line(em, "quantum.device shots(%shots) [\"$(ctx.device_lib)\", \"$(ctx.device_name)\", \"$(ctx.device_kwargs)\"]")
            _line(em, "%r$(node.reg) = quantum.alloc( $(node.n)) : !quantum.reg")
        elseif node isa ExtractNode
            _line(em, "%q$(node.qubit) = quantum.extract %r$(node.reg)[ $(node.wire-1)] : !quantum.reg -> !quantum.bit")
        elseif node isa InsertNode
            _line(em, "%r$(node.reg_out) = quantum.insert %r$(node.reg_in)[ $(node.wire-1)], %q$(node.qubit) : !quantum.reg, !quantum.bit")
        elseif node isa DeallocNode
            _line(em, "quantum.dealloc %r$(node.reg) : !quantum.reg")
            _line(em, "quantum.device_release")
        elseif node isa GateNode
            params = [_emit_expr!(em, prog, p) for p in node.params]
            outs = join(("%q$i" for i in [node.out_qubits; node.out_ctrls]), ", ")
            ins = join(("%q$i" for i in node.in_qubits), ", ")
            adj = node.adjoint ? " adj" : ""
            ctrl = ""
            ctrltypes = ""
            if !isempty(node.in_ctrls)
                vals = [_emit_expr!(em, prog, CBConst(v)) for v in node.ctrl_values]
                ctrl = " ctrls(" * join(("%q$i" for i in node.in_ctrls), ", ") * ") ctrlvals(" * join(vals, ", ") * ")"
                ctrltypes = " ctrls " * _qtypes(length(node.in_ctrls))
            end
            if node.name === :MultiRZ
                isempty(node.in_ctrls) || throw(ArgumentError("controlled MultiRZ is not supported in MLIR emission"))
                _line(em, "$outs = quantum.multirz($(params[1])) $ins$adj : $(_qtypes(length(node.in_qubits)))")
            else
                _line(em, "$outs = quantum.custom \"$(node.name)\"($(join(params, ", "))) $ins$adj$ctrl : $(_qtypes(length(node.in_qubits)))$ctrltypes")
            end
        elseif node isa NamedObsNode
            _line(em, "%o$(node.obs) = quantum.namedobs %q$(node.qubit)[ $(node.kind)] : !quantum.obs")
        elseif node isa TensorObsNode
            _line(em, "%o$(node.obs) = quantum.tensor " * join(("%o$t" for t in node.terms), ", ") * " : !quantum.obs")
        elseif node isa HamiltonianNode
            n = length(node.coeffs)
            _line(em, "%h$(node.obs) = arith.constant dense<[$(join(_mlirfloat.(node.coeffs), ", "))]> : tensor<$(n)xf64>")
            _line(em, "%o$(node.obs) = quantum.hamiltonian(%h$(node.obs) : tensor<$(n)xf64>) " * join(("%o$t" for t in node.terms), ", ") * " : !quantum.obs")
        elseif node isa CompBasisNode
            _line(em, "%o$(node.obs) = quantum.compbasis qubits " * join(("%q$q" for q in node.qubits), ", ") * " : !quantum.obs")
        elseif node isa ExpvalNode || node isa VarNode
            op = node isa ExpvalNode ? "expval" : "var"
            _line(em, "%m$(node.result) = quantum.$op %o$(node.obs) : f64")
            _line(em, "%t$(node.result) = tensor.from_elements %m$(node.result) : tensor<f64>")
            ctx.returns[node.result] = "%t$(node.result)"
        elseif node isa ProbsNode
            _line(em, "%t$(node.result) = quantum.probs %o$(node.obs) : tensor<$(1 << node.nwires)xf64>")
            ctx.returns[node.result] = "%t$(node.result)"
        elseif node isa StateNode
            _line(em, "%t$(node.result) = quantum.state %o$(node.obs) : tensor<$(1 << node.nqubits)xcomplex<f64>>")
            ctx.returns[node.result] = "%t$(node.result)"
        elseif node isa SampleNode
            _line(em, "%t$(node.result) = quantum.sample %o$(node.obs) : tensor<$(ctx.shots)x$(node.nwires)xf64>")
            ctx.returns[node.result] = "%t$(node.result)"
        elseif node isa MeasureNode
            ps = node.postselect < 0 ? "" : " postselect $(node.postselect)"
            _line(em, "%v$(node.result), %q$(node.out_qubit) = quantum.measure %q$(node.in_qubit)$ps : i1, !quantum.bit")
        elseif node isa IfNode
            c = _emit_expr!(em, prog, node.cond)
            results = [("%q$i" for i in node.out_qubits)..., ("%v$i" for i in node.cout)...]
            types = _regiontypes(length(node.wires), node.ckinds)
            head = isempty(results) ? "scf.if $c {" : "$(join(results, ", ")) = scf.if $c -> ($types) {"
            _line(em, head)
            _emit_region!(em) do
                _emit_nodes!(em, prog, node.then_body, ctx)
                cy = [_emit_expr!(em, prog, e) for e in node.then_cyield]
                ys = [("%q$i" for i in node.then_yield)..., cy...]
                _line(em, isempty(ys) ? "scf.yield" : "scf.yield $(join(ys, ", ")) : $types")
            end
            _line(em, "} else {")
            _emit_region!(em) do
                _emit_nodes!(em, prog, node.else_body, ctx)
                cy = [_emit_expr!(em, prog, e) for e in node.else_cyield]
                ys = [("%q$i" for i in node.else_yield)..., cy...]
                _line(em, isempty(ys) ? "scf.yield" : "scf.yield $(join(ys, ", ")) : $types")
            end
            _line(em, "}")
        elseif node isa ForNode
            lo = _emit_expr!(em, prog, CIConst(node.start))
            hi = _emit_expr!(em, prog, CIConst(node.stop + 1))
            st = _emit_expr!(em, prog, CIConst(node.step))
            inits = [_emit_expr!(em, prog, e) for e in node.cinit]
            iter = [("%q$a = %q$i" for (a, i) in zip(node.args, node.in_qubits))..., ("%v$a = $v" for (a, v) in zip(node.cargs, inits))...]
            results = [("%q$i" for i in node.out_qubits)..., ("%v$i" for i in node.cout)...]
            types = _regiontypes(length(node.wires), node.ckinds)
            head = (isempty(results) ? "" : "$(join(results, ", ")) = ") * "scf.for %v$(node.index) = $lo to $hi step $st" *
                   (isempty(iter) ? "" : " iter_args($(join(iter, ", "))) -> ($types)") * " {"
            _line(em, head)
            _emit_region!(em) do
                _emit_nodes!(em, prog, node.body, ctx)
                cy = [_emit_expr!(em, prog, e) for e in node.cyield]
                ys = [("%q$i" for i in node.yields)..., cy...]
                _line(em, isempty(ys) ? "scf.yield" : "scf.yield $(join(ys, ", ")) : $types")
            end
            _line(em, "}")
        elseif node isa WhileNode
            inits = [_emit_expr!(em, prog, e) for e in node.cinit]
            iter = [("%q$a = %q$i" for (a, i) in zip(node.args, node.in_qubits))..., ("%v$a = $v" for (a, v) in zip(node.cargs, inits))...]
            results = [("%q$i" for i in node.out_qubits)..., ("%v$i" for i in node.cout)...]
            types = _regiontypes(length(node.wires), node.ckinds)
            argnames = [("%q$a" for a in node.args)..., ("%v$a" for a in node.cargs)...]
            _line(em, (isempty(results) ? "" : "$(join(results, ", ")) = ") * "scf.while ($(join(iter, ", "))) : ($types) -> ($types) {")
            _emit_region!(em) do
                c = _emit_expr!(em, prog, node.cond)
                _line(em, "scf.condition($c) $(join(argnames, ", ")) : $types")
            end
            _line(em, "} do {")
            typed = [("%q$a: !quantum.bit" for a in node.args)..., ("%v$a: $(_mlirkind(k))" for (a, k) in zip(node.cargs, node.ckinds))...]
            _line(em, "^bb0($(join(typed, ", "))):")
            _emit_region!(em) do
                _emit_nodes!(em, prog, node.body, ctx)
                cy = [_emit_expr!(em, prog, e) for e in node.cyield]
                ys = [("%q$i" for i in node.yields)..., cy...]
                _line(em, "scf.yield $(join(ys, ", ")) : $types")
            end
            _line(em, "}")
        end
    end
end

"""
    to_mlir(prog; device_lib, device_name="LightningSimulator", device_kwargs, shots=0, grad=false)

Catalyst MLIR for a program: an entry point `jit_<name>` (C interface), the qnode kernel, and
optionally `jit_<name>_grad` using `gradient.grad` (single scalar result only).
"""
function to_mlir(prog::Program; device_lib::AbstractString="liblightning_qubit_catalyst.so",
                 device_name::AbstractString="LightningSimulator", device_kwargs::AbstractString=LIGHTNING_KWARGS,
                 shots::Integer=0, grad::Bool=false)
    em = _Emitter(IOBuffer(), 0, Dict{CExpr,String}(), "")
    io = em.io
    name = string(prog.name)
    argtypes = [_argtype(a) for a in prog.args]
    restypes = [_restype(r, shots) for r in prog.result_specs]
    arglist = join(("%arg$(i-1): $(argtypes[i])" for i in eachindex(argtypes)), ", ")
    argnames = join(("%arg$(i-1)" for i in eachindex(argtypes)), ", ")
    argtys = join(argtypes, ", ")
    println(io, "module @$name {")

    println(io, "  func.func public @jit_$name($arglist) -> $(_paren(restypes)) attributes {llvm.emit_c_interface} {")
    if length(restypes) == 1
        println(io, "    %0 = catalyst.launch_kernel @module_$name::@$name($argnames) : ($argtys) -> $(restypes[1])")
        println(io, "    return %0 : $(restypes[1])")
    else
        println(io, "    %0:$(length(restypes)) = catalyst.launch_kernel @module_$name::@$name($argnames) : ($argtys) -> $(_paren(restypes))")
        println(io, "    return ", join(("%0#$(i-1)" for i in eachindex(restypes)), ", "), " : ", join(restypes, ", "))
    end
    println(io, "  }")

    if grad
        length(restypes) == 1 && prog.result_specs[1].kind === :expval ||
            throw(ArgumentError("compiled gradients need a single expval result"))
        isempty(argtypes) && throw(ArgumentError("nothing to differentiate: the circuit has no arguments"))
        idx = length(argtypes) == 1 ? "dense<0> : tensor<1xi64>" : "dense<[$(join(0:length(argtypes)-1, ", "))]> : tensor<$(length(argtypes))xi64>"
        println(io, "  func.func public @jit_$(name)_grad($arglist) -> $(_paren(argtypes)) attributes {llvm.emit_c_interface} {")
        if length(argtypes) == 1
            println(io, "    %0 = gradient.grad \"auto\" @module_$name::@$name($argnames) {diffArgIndices = $idx} : ($argtys) -> $(argtypes[1])")
            println(io, "    return %0 : $(argtypes[1])")
        else
            println(io, "    %0:$(length(argtypes)) = gradient.grad \"auto\" @module_$name::@$name($argnames) {diffArgIndices = $idx} : ($argtys) -> $(_paren(argtypes))")
            println(io, "    return ", join(("%0#$(i-1)" for i in eachindex(argtypes)), ", "), " : ", join(argtypes, ", "))
        end
        println(io, "  }")
    end

    println(io, "  module @module_$name {")
    println(io, "    func.func public @$name($arglist) -> $(_paren(restypes)) attributes {diff_method = \"$(prog.diff_method)\", llvm.linkage = #llvm.linkage<internal>, quantum.node} {")
    em.indent = "      "
    _line(em, "%shots = arith.constant $(shots) : i64")
    ctx = (device_lib=String(device_lib), device_name=String(device_name), device_kwargs=String(device_kwargs), shots=Int(shots), returns=Dict{Int,String}())
    _emit_nodes!(em, prog, prog.nodes, ctx)
    _line(em, "return " * join((ctx.returns[r] for r in prog.results), ", ") * " : " * join(restypes, ", "))
    println(io, "    }")
    println(io, "  }")
    println(io, "  func.func @setup() {\n    quantum.init\n    return\n  }")
    println(io, "  func.func @teardown() {\n    quantum.finalize\n    return\n  }")
    println(io, "}")
    String(take!(io))
end

"""`mlir(qn, args...)`: Catalyst MLIR for the program traced for these arguments."""
function mlir(qn::QNode, args...; grad::Bool=false)
    prog = program(qn, args...)
    lib = has_catalyst() ? catalyst_env().lightning_plugin : "liblightning_qubit_catalyst." * Libdl.dlext
    dev = qn.dev
    to_mlir(prog; device_lib=lib, device_kwargs=hasproperty(dev, :kwargs) && dev.kwargs isa AbstractString ? dev.kwargs : LIGHTNING_KWARGS,
            shots=device_shots(dev), grad=grad)
end
