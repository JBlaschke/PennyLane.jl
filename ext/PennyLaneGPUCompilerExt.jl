# T6: compile a traced program to a standalone native library with Julia's own compiler.
#
# `native_expr` turns a Program into straight-line Julia calling the Catalyst runtime C API;
# GPUCompiler's native target compiles that function without the Julia runtime, so the object
# file's only external symbols are the runtime's `__catalyst__*` functions and libm — a
# QIR-style program produced by Julia instead of MLIR. Activated by `using GPUCompiler`.
module PennyLaneGPUCompilerExt

using PennyLane
using GPUCompiler
using Libdl
using PennyLane: Program, QNode, GateNode, program, normalize_args, native_function, catalyst_env, rt_lib,
                 rt_initialize!, LIGHTNING_KWARGS, has_control_flow, has_mcm
import PennyLane: compile_native, native_llvm_ir

struct NativeParams <: GPUCompiler.AbstractCompilerParams end
# the Catalyst runtime's C API is the program's legitimate external interface
GPUCompiler.isintrinsic(@nospecialize(job::GPUCompiler.CompilerJob{<:Any,NativeParams}), fn::String) = startswith(fn, "__catalyst__")

_argtypes(prog::Program) = Tuple{Ptr{UInt8},Ptr{UInt8},Ptr{UInt8},Int64,(a.kind === :scalar ? Float64 : Ptr{Float64} for a in prog.args)...}

function _job(f, prog::Program, name::String)
    mi = GPUCompiler.methodinstance(typeof(f), _argtypes(prog))
    config = GPUCompiler.CompilerConfig(GPUCompiler.NativeCompilerTarget(), NativeParams();
                                        kernel=false, name=name, libraries=false)
    GPUCompiler.CompilerJob(mi, config)
end

function _check_static(prog::Program)
    prog.scalar_return && prog.result_specs[1].kind === :expval ||
        throw(ArgumentError("compile_native needs a QNode returning a single expval"))
end

_llvm_ir(f, prog, name) = GPUCompiler.JuliaContext() do ctx
    ir, _ = GPUCompiler.compile(:llvm, _job(f, prog, name))
    string(ir)
end
_object(f, prog, name) = GPUCompiler.JuliaContext() do ctx
    obj, _ = GPUCompiler.compile(:obj, _job(f, prog, name))
    obj
end

function PennyLane.native_llvm_ir(qn::QNode, args...; name::AbstractString="pl_" * string(qn.name))
    prog = program(qn, args...)
    _check_static(prog)
    f = native_function(prog; extern=:llvmcall)
    Base.invokelatest(_llvm_ir, f, prog, String(name))
end

"""A shared library compiled by `compile_native`; call it like the QNode."""
struct NativeLibrary
    prog::Program
    path::String
    handle::Ptr{Cvoid}
    fptr::Ptr{Cvoid}
    caller::Function
    shots::Int
    kwargs::String
end
Base.show(io::IO, l::NativeLibrary) = print(io, "NativeLibrary(", l.path, ")")

function PennyLane.compile_native(qn::QNode, args...; dir::AbstractString=mktempdir(), name::AbstractString="pl_" * string(qn.name),
                                  shots::Integer=0, kwargs::AbstractString=LIGHTNING_KWARGS)
    prog = program(qn, args...)
    _check_static(prog)
    f = native_function(prog; extern=:llvmcall)
    obj = Base.invokelatest(_object, f, prog, String(name))
    objpath = joinpath(dir, name * ".o")
    write(objpath, obj)
    env = catalyst_env()
    lib = joinpath(dir, "lib" * name * "." * Libdl.dlext)
    flags = ["-shared", "-o", lib, objpath, "-L$(env.libdir)", "-Wl,-rpath,$(env.libdir)", "-lrt_capi"]
    Sys.islinux() && push!(flags, "-lm")
    run(`$(PennyLane._linker()) $flags`)
    handle = Libdl.dlopen(lib, Libdl.RTLD_NOW | Libdl.RTLD_GLOBAL)
    fptr = Libdl.dlsym(handle, Symbol(name))
    # a ccall with this program's exact signature
    argsyms = [Symbol("a", i) for i in eachindex(prog.args)]
    types = [a.kind === :scalar ? :Float64 : :(Ptr{Float64}) for a in prog.args]
    caller = Core.eval(@__MODULE__, :((fptr, lib, dev, kw, shots, $(argsyms...)) ->
        ccall(fptr, Float64, (Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Int64, $(types...)), lib, dev, kw, shots, $(argsyms...))))
    NativeLibrary(prog, lib, handle, fptr, caller, Int(shots), String(kwargs))
end

function (l::NativeLibrary)(args...)
    a = normalize_args(args)
    map(PennyLane.argsig, a) == [(s.kind, s.kind === :scalar ? 0 : s.length) for s in l.prog.args] ||
        throw(ArgumentError("arguments do not match the compiled program's signature"))
    env = catalyst_env()
    rt_lib()
    rt_initialize!()
    lib, name, kw = env.lightning_plugin, "LightningSimulator", l.kwargs
    ptrs = Any[x isa Float64 ? x : pointer(x::Vector{Float64}) for x in a]
    GC.@preserve lib name kw a begin
        Base.invokelatest(l.caller, l.fptr, pointer(lib), pointer(name), pointer(kw), Int64(l.shots), ptrs...)
    end
end

end # module
