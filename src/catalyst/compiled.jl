# `CatalystDevice`: Julia -> Catalyst MLIR -> `catalyst` CLI -> clang -> shared library -> ccall.

"""
    CatalystDevice([nwires]; shots=0, kwargs=LIGHTNING_KWARGS, keep_intermediate=false)

Compiles programs with the Catalyst compiler and runs them on Lightning through the Catalyst
runtime. Gradients use Catalyst's adjoint differentiation (`gradient.grad`).
"""
struct CatalystDevice <: AbstractDevice
    nwires::Union{Nothing,Int}
    shots::Int
    kwargs::String
    keep_intermediate::Bool
    cache::IdDict{Program,Any}
end
CatalystDevice(n::Union{Nothing,Integer}=nothing; shots::Integer=0, kwargs::AbstractString=LIGHTNING_KWARGS, keep_intermediate::Bool=false) =
    CatalystDevice(n === nothing ? nothing : Int(n), Int(shots), String(kwargs), keep_intermediate, IdDict{Program,Any}())
Base.show(io::IO, d::CatalystDevice) = print(io, "CatalystDevice(", d.nwires === nothing ? "" : d.nwires, ")")

mutable struct CompiledModule
    prog::Program
    dir::String
    mlir::String
    dylib::String
    handle::Ptr{Cvoid}
    fn::Ptr{Cvoid}
    grad_fn::Ptr{Cvoid}
end
Base.show(io::IO, m::CompiledModule) = print(io, "CompiledModule(", m.prog.name, ", ", m.dylib, ")")

function _linker()
    cc = get(ENV, "CATALYST_CC", nothing)
    cc !== nothing && return cc
    for c in ("clang", "gcc", "cc")
        p = Sys.which(c)
        p === nothing || return p
    end
    error("no C compiler found for linking (set CATALYST_CC)")
end

# mirrors catalyst/compiler.py LinkerDriver.get_default_flags
function link_flags(env::CatalystEnv)
    flags = ["-shared", "-rdynamic",
             "-Wl,-rpath,$(env.libdir)", "-L$(env.libdir)",
             "-Wl,-rpath,$(env.utilsdir)", "-L$(env.utilsdir)",
             "-lrt_capi", "-lpthread", "-lmlir_c_runner_utils"]
    if Sys.isapple()
        push!(flags, "-llapacke.3", "-Wl,-arch_errors_fatal")
    else
        push!(flags, "-Wl,-no-as-needed", "-Wl,--disable-new-dtags")
        blasdir = joinpath(env.sitepackages, "scipy_openblas32", "lib")
        libs = isdir(blasdir) ? filter(f -> occursin("openblas", f) && endswith(f, Libdl.dlext), readdir(blasdir)) : String[]
        isempty(libs) && error("OpenBLAS from scipy_openblas32 not found in $blasdir (needed to link Catalyst programs on Linux)")
        push!(flags, "-Wl,-rpath,$blasdir", "-L$blasdir", "-l" * libs[1][4:end-length(Libdl.dlext)-1])
    end
    custom = joinpath(env.utilsdir, "libcustom_calls.so")          # named .so on every platform
    push!(flags, isfile(custom) ? custom : "-lcustom_calls")
    push!(flags, "-lmlir_async_runtime")
    for extra in ("librt_rsdecomp", "librt_OQD_capi")
        isfile(joinpath(env.libdir, extra * "." * Libdl.dlext)) && push!(flags, "-l" * extra[4:end])
    end
    flags
end

"""Compile a program with the Catalyst CLI, link it, and load it."""
function compile(dev::CatalystDevice, prog::Program)
    env = catalyst_env()
    # Catalyst's gradient lowering handles scf.if/scf.for with qubit iteration arguments but not
    # scf.while (it needs register threading there); no compiled gradient in that case.
    grad = prog.scalar_return && prog.result_specs[1].kind === :expval && !isempty(prog.args) && !has_mcm(prog) &&
           !any(n -> n isa WhileNode, all_nodes(prog))
    src = to_mlir(prog; device_lib=env.lightning_plugin, device_kwargs=dev.kwargs, shots=dev.shots, grad=grad)
    dir = mktempdir(; cleanup=!dev.keep_intermediate)
    name = string(prog.name)
    mlirfile = joinpath(dir, "$name.mlir")
    write(mlirfile, src)
    log = IOBuffer()
    cmd = `$(env.cli) --module-name=$name --workspace=$dir -o $(joinpath(dir, "$name.ll")) $mlirfile`
    ok = success(pipeline(ignorestatus(cmd); stdout=log, stderr=log))
    obj = joinpath(dir, "$name.o")
    (ok && isfile(obj)) || error("Catalyst compilation failed:\n$(String(take!(log)))\n--- MLIR ---\n$src")
    dylib = joinpath(dir, "lib$name." * Libdl.dlext)
    run(pipeline(`$(_linker()) $(link_flags(env)) -o $dylib $obj`; stdout=log, stderr=log))
    handle = Libdl.dlopen(dylib, Libdl.RTLD_NOW | Libdl.RTLD_GLOBAL)
    fn = Libdl.dlsym(handle, Symbol("_catalyst_pyface_jit_$name"))
    gfn = grad ? Libdl.dlsym(handle, Symbol("_catalyst_pyface_jit_$(name)_grad")) : Ptr{Cvoid}(C_NULL)
    CompiledModule(prog, dir, src, dylib, handle, fn, gfn)
end
compiled(dev::CatalystDevice, prog::Program) = get!(dev.cache, prog) do
    compile(dev, prog)
end

struct MemRef0{T}
    allocated::Ptr{T}
    aligned::Ptr{T}
    offset::Int64
end

# Call a `_catalyst_pyface_*` entry point: (void* results, void** args), memref descriptors
# for arguments, and a packed sequence of result descriptors owned by the runtime's memory
# manager until `_mlir_memory_transfer`.
# `shapes`: per result (element type :f64 | :c64, dims) with dims (), (n,) or (m, n) [row-major].
const FFIShape = Tuple{Symbol,Tuple{Vararg{Int}}}

function call_ffi(fn::Ptr{Cvoid}, args::Vector{Any}, shapes::Vector{FFIShape})
    keep = Any[]
    ptrs = Vector{Ptr{Cvoid}}(undef, length(args))
    for (i, a) in enumerate(args)
        if a isa Float64
            r = Ref(a)
            push!(keep, r)
            p = Base.unsafe_convert(Ptr{Float64}, r)
            d = Ref(MemRef0{Float64}(p, p, 0))
            push!(keep, d)
            ptrs[i] = Ptr{Cvoid}(Base.unsafe_convert(Ptr{MemRef0{Float64}}, d))
        else
            v = a::Vector{Float64}
            push!(keep, v)
            d = Ref(memref1d(v))
            push!(keep, d)
            ptrs[i] = Ptr{Cvoid}(Base.unsafe_convert(Ptr{MemRef1D{Float64}}, d))
        end
    end
    nwords = sum(3 + 2 * length(dims) for (_, dims) in shapes; init=0)
    resbuf = zeros(UInt64, nwords)
    GC.@preserve keep ptrs resbuf begin
        ccall(fn, Cvoid, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), pointer(resbuf), pointer(ptrs))
    end
    out = Any[]
    off = 1
    for (elt, dims) in shapes
        alloc = Ptr{Cvoid}(resbuf[off])
        T = elt === :f64 ? Float64 : ComplexF64
        p = Ptr{T}(resbuf[off+1])
        offset = Int(resbuf[off+2])
        rank = length(dims)
        if rank == 0
            push!(out, unsafe_load(p, offset + 1))
        elseif rank == 1
            sz, stride = Int(resbuf[off+3]), Int(resbuf[off+4])
            sz == dims[1] || error("compiled result has length $sz, expected $(dims[1])")
            push!(out, T[unsafe_load(p, offset + 1 + (j - 1) * stride) for j in 1:sz])
        else
            s0, s1, t0, t1 = Int(resbuf[off+3]), Int(resbuf[off+4]), Int(resbuf[off+5]), Int(resbuf[off+6])
            (s0, s1) == dims || error("compiled result has shape ($s0, $s1), expected $dims")
            push!(out, T[unsafe_load(p, offset + 1 + (i - 1) * t0 + (j - 1) * t1) for i in 1:s0, j in 1:s1])
        end
        off += 3 + 2rank
        if alloc != C_NULL
            ccall(rt_sym(:_mlir_memory_transfer), Cvoid, (Ptr{Cvoid},), alloc)
            Libc.free(alloc)
        end
    end
    out
end

function _result_shape(r::ResultSpec, shots::Int)::FFIShape
    r.kind in (:expval, :var) && return (:f64, ())
    r.kind === :probs && return (:f64, (r.size,))
    r.kind === :state && return (:c64, (r.size,))
    (:f64, (shots, r.size))
end
_arg_shape(a::ArgSpec)::FFIShape = a.kind === :scalar ? (:f64, ()) : (:f64, (a.length,))

function execute(dev::CatalystDevice, prog::Program, args::Vector{Any})
    mod = compiled(dev, prog)
    rt_initialize!()
    out = call_ffi(mod.fn, args, FFIShape[_result_shape(r, dev.shots) for r in prog.result_specs])
    for (k, r) in enumerate(prog.result_specs)
        r.kind === :sample && (out[k] = Int.(round.(out[k])))
    end
    prog.scalar_return ? out[1] : Tuple(out)
end

"""Gradient through Catalyst's compiled adjoint differentiation."""
function compiled_gradient(qn::QNode, args...)
    a = normalize_args(args)
    prog = program(qn, a...)
    qn.dev isa CatalystDevice || throw(ArgumentError("the :adjoint gradient is compiled by Catalyst and needs a CatalystDevice; use :parameter_shift or :finitediff on $(typeof(qn.dev))"))
    mod = compiled(qn.dev, prog)
    mod.grad_fn == C_NULL && throw(ArgumentError("compiled gradients need a QNode with arguments returning a single expval, without mid-circuit measurements or @trace while loops"))
    rt_initialize!()
    out = call_ffi(mod.grad_fn, a, FFIShape[_arg_shape(s) for s in prog.args])
    length(out) == 1 ? out[1] : Tuple(out)
end
