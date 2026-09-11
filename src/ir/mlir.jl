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

mutable struct _Emitter
    io::IOBuffer
    counter::Int
    exprs::Dict{CExpr,String}
end
_fresh(em::_Emitter, prefix) = (em.counter += 1; "%$(prefix)$(em.counter)")

const _CLASSICAL_OPS = Dict(:add => "arith.addf", :sub => "arith.subf", :mul => "arith.mulf", :div => "arith.divf",
                            :neg => "arith.negf", :sin => "math.sin", :cos => "math.cos", :sqrt => "math.sqrt", :exp => "math.exp")

function _emit_expr!(em::_Emitter, prog::Program, e::CExpr)
    get!(em.exprs, e) do
        if e isa CConst
            n = _fresh(em, "c")
            println(em.io, "      $n = arith.constant $(_mlirfloat(e.value)) : f64")
            n
        elseif e isa CArg
            n = _fresh(em, "c")
            if e.index == 0
                println(em.io, "      $n = tensor.extract %arg$(e.arg-1)[] : tensor<f64>")
            else
                idx = _fresh(em, "i")
                println(em.io, "      $idx = arith.constant $(e.index-1) : index")
                println(em.io, "      $n = tensor.extract %arg$(e.arg-1)[$idx] : $(_argtype(prog.args[e.arg]))")
            end
            n
        else
            ops = [_emit_expr!(em, prog, a) for a in e.args]
            n = _fresh(em, "c")
            println(em.io, "      $n = $(_CLASSICAL_OPS[e.op]) $(join(ops, ", ")) : f64")
            n
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
    em = _Emitter(IOBuffer(), 0, Dict{CExpr,String}())
    io = em.io
    name = string(prog.name)
    argtypes = [_argtype(a) for a in prog.args]
    restypes = [_restype(r, shots) for r in prog.result_specs]
    arglist = join(("%arg$(i-1): $(argtypes[i])" for i in eachindex(argtypes)), ", ")
    argnames = join(("%arg$(i-1)" for i in eachindex(argtypes)), ", ")
    argtys = join(argtypes, ", ")
    println(io, "module @$name {")

    # entry point
    println(io, "  func.func public @jit_$name($arglist) -> $(_paren(restypes)) attributes {llvm.emit_c_interface} {")
    if length(restypes) == 1
        println(io, "    %0 = catalyst.launch_kernel @module_$name::@$name($argnames) : ($argtys) -> $(restypes[1])")
        println(io, "    return %0 : $(restypes[1])")
    else
        println(io, "    %0:$(length(restypes)) = catalyst.launch_kernel @module_$name::@$name($argnames) : ($argtys) -> $(_paren(restypes))")
        println(io, "    return ", join(("%0#$(i-1)" for i in eachindex(restypes)), ", "), " : ", join(restypes, ", "))
    end
    println(io, "  }")

    # gradient entry point (all arguments differentiated)
    if grad
        length(restypes) == 1 && prog.result_specs[1].kind === :expval ||
            throw(ArgumentError("compiled gradients need a single expval result"))
        isempty(argtypes) && throw(ArgumentError("nothing to differentiate: the circuit has no arguments"))
        gtypes = argtypes
        idx = length(argtypes) == 1 ? "dense<0> : tensor<1xi64>" : "dense<[$(join(0:length(argtypes)-1, ", "))]> : tensor<$(length(argtypes))xi64>"
        println(io, "  func.func public @jit_$(name)_grad($arglist) -> $(_paren(gtypes)) attributes {llvm.emit_c_interface} {")
        if length(gtypes) == 1
            println(io, "    %0 = gradient.grad \"auto\" @module_$name::@$name($argnames) {diffArgIndices = $idx} : ($argtys) -> $(gtypes[1])")
            println(io, "    return %0 : $(gtypes[1])")
        else
            println(io, "    %0:$(length(gtypes)) = gradient.grad \"auto\" @module_$name::@$name($argnames) {diffArgIndices = $idx} : ($argtys) -> $(_paren(gtypes))")
            println(io, "    return ", join(("%0#$(i-1)" for i in eachindex(gtypes)), ", "), " : ", join(gtypes, ", "))
        end
        println(io, "  }")
    end

    # qnode kernel
    println(io, "  module @module_$name {")
    println(io, "    func.func public @$name($arglist) -> $(_paren(restypes)) attributes {diff_method = \"$(prog.diff_method)\", llvm.linkage = #llvm.linkage<internal>, quantum.node} {")
    println(io, "      %true = arith.constant true")
    println(io, "      %false = arith.constant false")
    println(io, "      %shots = arith.constant $(shots) : i64")
    returns = Dict{Int,String}()
    for node in prog.nodes
        if node isa AllocNode
            println(io, "      quantum.device shots(%shots) [\"$device_lib\", \"$device_name\", \"$device_kwargs\"]")
            println(io, "      %r$(node.reg) = quantum.alloc( $(node.n)) : !quantum.reg")
        elseif node isa ExtractNode
            println(io, "      %q$(node.qubit) = quantum.extract %r$(node.reg)[ $(node.wire-1)] : !quantum.reg -> !quantum.bit")
        elseif node isa InsertNode
            println(io, "      %r$(node.reg_out) = quantum.insert %r$(node.reg_in)[ $(node.wire-1)], %q$(node.qubit) : !quantum.reg, !quantum.bit")
        elseif node isa DeallocNode
            println(io, "      quantum.dealloc %r$(node.reg) : !quantum.reg")
            println(io, "      quantum.device_release")
        elseif node isa GateNode
            params = [_emit_expr!(em, prog, p) for p in node.params]
            outs = join(("%q$i" for i in [node.out_qubits; node.out_ctrls]), ", ")
            ins = join(("%q$i" for i in node.in_qubits), ", ")
            qtypes = join(fill("!quantum.bit", length(node.in_qubits)), ", ")
            adj = node.adjoint ? " adj" : ""
            ctrl = ""
            ctrltypes = ""
            if !isempty(node.in_ctrls)
                ctrl = " ctrls(" * join(("%q$i" for i in node.in_ctrls), ", ") * ") ctrlvals(" * join((v ? "%true" : "%false" for v in node.ctrl_values), ", ") * ")"
                ctrltypes = " ctrls " * join(fill("!quantum.bit", length(node.in_ctrls)), ", ")
            end
            if node.name === :MultiRZ
                isempty(node.in_ctrls) || throw(ArgumentError("controlled MultiRZ is not supported in MLIR emission"))
                println(io, "      $outs = quantum.multirz($(params[1])) $ins$adj : $qtypes")
            else
                println(io, "      $outs = quantum.custom \"$(node.name)\"($(join(params, ", "))) $ins$adj$ctrl : $qtypes$ctrltypes")
            end
        elseif node isa NamedObsNode
            println(io, "      %o$(node.obs) = quantum.namedobs %q$(node.qubit)[ $(node.kind)] : !quantum.obs")
        elseif node isa TensorObsNode
            println(io, "      %o$(node.obs) = quantum.tensor ", join(("%o$t" for t in node.terms), ", "), " : !quantum.obs")
        elseif node isa HamiltonianNode
            n = length(node.coeffs)
            println(io, "      %h$(node.obs) = arith.constant dense<[$(join(_mlirfloat.(node.coeffs), ", "))]> : tensor<$(n)xf64>")
            println(io, "      %o$(node.obs) = quantum.hamiltonian(%h$(node.obs) : tensor<$(n)xf64>) ", join(("%o$t" for t in node.terms), ", "), " : !quantum.obs")
        elseif node isa CompBasisNode
            println(io, "      %o$(node.obs) = quantum.compbasis qubits ", join(("%q$q" for q in node.qubits), ", "), " : !quantum.obs")
        elseif node isa ExpvalNode || node isa VarNode
            op = node isa ExpvalNode ? "expval" : "var"
            println(io, "      %m$(node.result) = quantum.$op %o$(node.obs) : f64")
            println(io, "      %t$(node.result) = tensor.from_elements %m$(node.result) : tensor<f64>")
            returns[node.result] = "%t$(node.result)"
        elseif node isa ProbsNode
            println(io, "      %t$(node.result) = quantum.probs %o$(node.obs) : tensor<$(1 << node.nwires)xf64>")
            returns[node.result] = "%t$(node.result)"
        elseif node isa StateNode
            println(io, "      %t$(node.result) = quantum.state %o$(node.obs) : tensor<$(1 << node.nqubits)xcomplex<f64>>")
            returns[node.result] = "%t$(node.result)"
        elseif node isa SampleNode
            println(io, "      %t$(node.result) = quantum.sample %o$(node.obs) : tensor<$(shots)x$(node.nwires)xf64>")
            returns[node.result] = "%t$(node.result)"
        end
    end
    println(io, "      return ", join((returns[r] for r in prog.results), ", "), " : ", join(restypes, ", "))
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
    to_mlir(prog; device_lib=lib, device_kwargs=hasproperty(dev, :kwargs) ? dev.kwargs : LIGHTNING_KWARGS,
            shots=hasproperty(dev, :shots) ? dev.shots : 0, grad=grad)
end
