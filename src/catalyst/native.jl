# Program -> Julia code that calls the Catalyst runtime C API directly.
#
# The generated function is straight-line Julia: it allocates qubits, applies gates through
# `ccall(:__catalyst__qis__<Gate>, ...)`, registers observables and returns expectation values.
# Julia's own compiler then produces native code for it — the same shape of program Catalyst
# emits through MLIR and LLVM, without MLIR (see PLAN.md §3.5, item 4). `JITDevice` executes
# programs this way; the GPUCompiler extension compiles the same function to a standalone
# object file / shared library (T6).

const GENERATED = Module(:PennyLaneGenerated)

_native_param(e::CConst, argnames) = e.value
function _native_param(e::CArg, argnames)
    e.index == 0 ? argnames[e.arg] : :(unsafe_load($(argnames[e.arg]), $(e.index)))
end
function _native_param(e::CCall, argnames)
    a = [_native_param(x, argnames) for x in e.args]
    op = e.op
    op === :add && return :($(a[1]) + $(a[2]))
    op === :sub && return :($(a[1]) - $(a[2]))
    op === :mul && return :($(a[1]) * $(a[2]))
    op === :div && return :($(a[1]) / $(a[2]))
    op === :neg && return :(-$(a[1]))
    op in (:sin, :cos, :sqrt, :exp) && return :($(op)($(a[1])))
    throw(ArgumentError("native code generation does not support classical op $op"))
end
_native_param(e::CExpr, argnames) = throw(ArgumentError("native code generation needs a static circuit (no traced control flow or measurement values)"))

# How generated code reaches a runtime symbol. `:ccall` is resolved lazily by Julia at run time
# (fine for the JIT); `:llvmcall` declares the symbol as a plain LLVM extern so the object file
# produced by GPUCompiler needs nothing but the runtime library at link time.
const _LLVM_TYPES = Dict(:Float64 => "double", :Int64 => "i64", :Bool => "i8", :Cvoid => "void")
_llvmtype(t) = t isa Expr ? "ptr" : get(_LLVM_TYPES, t, "ptr")     # Ptr{...} expressions -> opaque pointer

# `Base.llvmcall((module_ir, entry), ...)`: a small module per call site with a wrapper that calls
# the declared extern. Entry names are unique within the generated function (`counter`).
function _extern_call(mode::Symbol, sym::Symbol, ret, argtypes::Vector, args::Vector; nfixed::Int=length(argtypes), counter::Ref{Int}=Ref(0))
    if mode === :ccall
        types = nfixed == length(argtypes) ? Expr(:tuple, argtypes...) :
                Expr(:tuple, argtypes[1:nfixed]..., Expr(:..., argtypes[end]))
        return :(ccall($(QuoteNode(sym)), $ret, $types, $(args...)))
    end
    lt = [_llvmtype(t) for t in argtypes]
    rt = _llvmtype(ret)
    fixed = join(lt[1:nfixed], ", ")
    decl = nfixed == length(argtypes) ? "declare $rt @$sym($fixed)" : "declare $rt @$sym($fixed, ...)"
    params = join(("$(lt[i]) %a$(i-1)" for i in eachindex(argtypes)), ", ")
    callee = nfixed == length(argtypes) ? "$rt @$sym" : "$rt ($fixed, ...) @$sym"     # variadic calls name the function type
    entry = string("__pl_", sym, "_", counter[] += 1)
    inner = rt == "void" ? "  call $callee($params)\n  ret void" :
                           "  %r = call $callee($params)\n  ret $rt %r"
    mod = "$decl\ndefine $rt @$entry($params) {\n$inner\n}"
    :(Base.llvmcall(($mod, $entry), $ret, Tuple{$(argtypes...)}, $(args...)))
end

"""
    native_expr(prog::Program; name=prog.name, extern=:ccall) -> Expr

Julia function definition for a static program with expval results. Signature:
`(lib::Ptr{UInt8}, device::Ptr{UInt8}, kwargs::Ptr{UInt8}, shots::Int64, args...)` with scalar
arguments as `Float64` and vector arguments as `Ptr{Float64}`; returns a `Float64` or a tuple.
"""
function native_expr(prog::Program; name::Symbol=Symbol("native_", prog.name), extern::Symbol=:ccall)
    extern in (:ccall, :llvmcall) || throw(ArgumentError("extern must be :ccall or :llvmcall"))
    (has_control_flow(prog) || has_mcm(prog)) &&
        throw(ArgumentError("native code generation needs a static circuit (no @trace control flow or mid-circuit measurements)"))
    all(r -> r.kind === :expval, prog.result_specs) ||
        throw(ArgumentError("native code generation supports expval results only (use LightningDevice for var/probs/state/sample)"))
    argnames = [Symbol("arg", i) for i in eachindex(prog.args)]
    sig = Any[:(lib::Ptr{UInt8}), :(device::Ptr{UInt8}), :(kwargs::Ptr{UInt8}), :(shots::Int64)]
    for (i, a) in enumerate(prog.args)
        push!(sig, a.kind === :scalar ? :($(argnames[i])::Float64) : :($(argnames[i])::Ptr{Float64}))
    end
    counter = Ref(0)
    X(sym, ret, types, args; kw...) = _extern_call(extern, sym, ret, Any[types...], Any[args...]; counter=counter, kw...)
    body = Expr[]
    push!(body, X(:__catalyst__rt__device_init, :Cvoid, (:(Ptr{UInt8}), :(Ptr{UInt8}), :(Ptr{UInt8}), :Int64, :Bool), (:lib, :device, :kwargs, :shots, false)))
    push!(body, :(qreg = $(X(:__catalyst__rt__qubit_allocate_array, :(Ptr{Cvoid}), (:Int64,), (prog.nqubits,)))))
    qsym = Dict{Int,Symbol}()          # wire => variable holding the QUBIT*
    for w in 1:prog.nqubits
        s = Symbol("q", w)
        qsym[w] = s
        push!(body, :($s = unsafe_load($(X(:__catalyst__rt__array_get_element_ptr_1d, :(Ptr{Ptr{Cvoid}}), (:(Ptr{Cvoid}), :Int64), (:qreg, w - 1))))))
    end
    obs = Dict{Int,Any}()
    results = Dict{Int,Symbol}()
    nres = 0
    for node in prog.nodes
        if node isa GateNode
            params = [_native_param(p, argnames) for p in node.params]
            qs = [qsym[w] for w in node.wires]
            if node.adjoint || !isempty(node.ctrl_wires)
                extern === :llvmcall && throw(ArgumentError("static native compilation does not support gate modifiers (adjoint/ctrl) yet: they need a heap-allocated Modifiers struct"))
                cq = [qsym[w] for w in node.ctrl_wires]
                mods = :(Ref(Modifiers($(node.adjoint), $(length(cq)), pointer(ctrls), pointer(vals))))
                call = _native_gate_call(X, node.name, params, qs, :(Base.unsafe_convert(Ptr{Modifiers}, m)))
                push!(body, quote
                    ctrls = Ptr{Cvoid}[$(cq...)]
                    vals = Bool[$(node.ctrl_values...)]
                    GC.@preserve ctrls vals begin
                        m = $mods
                        GC.@preserve m $call
                    end
                end)
            else
                push!(body, _native_gate_call(X, node.name, params, qs, :(Ptr{Modifiers}(C_NULL))))
            end
        elseif record_observable!(obs, node)
        elseif node isa ExpvalNode
            nres += 1
            r = Symbol("res", nres)
            results[node.result] = r
            ts = terms(obs[node.obs])
            terms_expr = Any[]
            for t in ts
                push!(terms_expr, :($(real(t.coeff)) * $(X(:__catalyst__qis__Expval, :Float64, (:Int64,), (_native_word(X, t, qsym),)))))
            end
            push!(body, :($r = +($(terms_expr...))))
        end
    end
    push!(body, X(:__catalyst__rt__qubit_release_array, :Cvoid, (:(Ptr{Cvoid}),), (:qreg,)))
    push!(body, X(:__catalyst__rt__device_release, :Cvoid, (), ()))
    rets = [results[r] for r in prog.results]
    push!(body, prog.scalar_return ? :(return $(rets[1])) : :(return ($(rets...),)))
    Expr(:function, Expr(:call, name, sig...), Expr(:block, body...))
end

function _native_gate_call(X, name::Symbol, params, qs, mods)
    fsym = Symbol("__catalyst__qis__", name)
    if name === :MultiRZ
        return X(fsym, :Cvoid, (:Float64, :(Ptr{Modifiers}), :Int64, fill(:(Ptr{Cvoid}), length(qs))...), (params[1], mods, length(qs), qs...); nfixed=3)
    end
    def = GATES[name]
    X(fsym, :Cvoid, (fill(:Float64, def.nparams)..., fill(:(Ptr{Cvoid}), def.nqubits)..., :(Ptr{Modifiers})), (params..., qs..., mods))
end

# ObsIdType for one Pauli word: NamedObs per letter, TensorObs for products
function _native_word(X, t::PauliString, qsym)
    isempty(t.word) && return X(:__catalyst__qis__NamedObs, :Int64, (:Int64, :(Ptr{Cvoid})), (0, qsym[1]))
    ids = [X(:__catalyst__qis__NamedObs, :Int64, (:Int64, :(Ptr{Cvoid})), (OBS_CODES[l], qsym[w])) for (w, l) in t.word]
    length(ids) == 1 && return ids[1]
    X(:__catalyst__qis__TensorObs, :Int64, (:Int64, fill(:Int64, length(ids))...), (length(ids), ids...); nfixed=1)
end

"""
    native_function(prog::Program) -> Function

Evaluate `native_expr(prog)` into a generated function (call it with `Base.invokelatest` from
code compiled before the definition).
"""
function native_function(prog::Program; extern::Symbol=:ccall)
    ex = native_expr(prog; extern=extern)
    Core.eval(GENERATED, :(using PennyLane: Modifiers))
    Core.eval(GENERATED, ex)
end

"""
    JITDevice([nwires]; shots=0, kwargs=LIGHTNING_KWARGS)

Runs static expval programs on Lightning through Julia code generated with `native_expr`:
one straight-line function of runtime calls per program, compiled by Julia's JIT (no
interpreter, no MLIR). See also the GPUCompiler extension for standalone object files.
"""
struct JITDevice <: AbstractDevice
    nwires::Union{Nothing,Int}
    shots::Int
    kwargs::String
    cache::IdDict{Program,Any}
end
JITDevice(n::Union{Nothing,Integer}=nothing; shots::Integer=0, kwargs::AbstractString=LIGHTNING_KWARGS) =
    JITDevice(n === nothing ? nothing : Int(n), Int(shots), String(kwargs), IdDict{Program,Any}())
Base.show(io::IO, d::JITDevice) = print(io, "JITDevice(", d.nwires === nothing ? "" : d.nwires, ")")

function execute(dev::JITDevice, prog::Program, args::Vector{Any})
    f = get!(dev.cache, prog) do
        native_function(prog)
    end
    env = catalyst_env()
    rt_lib()
    rt_initialize!()
    lib, name, kw = env.lightning_plugin, "LightningSimulator", dev.kwargs
    ptrs = Any[a isa Float64 ? a : pointer(a::Vector{Float64}) for a in args]
    GC.@preserve lib name kw args begin
        Base.invokelatest(f, pointer(lib), pointer(name), pointer(kw), Int64(dev.shots), ptrs...)
    end
end
