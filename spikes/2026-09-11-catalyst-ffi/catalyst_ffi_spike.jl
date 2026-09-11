# Spike (2026-09-11): drive PennyLane-Catalyst from plain Julia, no Python at runtime.
#
#   Path B  : hand-written Catalyst MLIR  --catalyst CLI-->  .o  --clang-->  .dylib  --ccall-->  Julia
#   Path A' : Julia --ccall--> Catalyst runtime C API (librt_capi) --> Lightning device plugin
#
# Requirements: a Python venv with `pennylane` and `pennylane-catalyst` installed (see README.md),
# pointed to by PENNYLANE_JL_VENV (defaults to ../../python/.venv). Only Libdl + clang are needed.

using Libdl, Printf

const VENV = get(ENV, "PENNYLANE_JL_VENV", normpath(joinpath(@__DIR__, "..", "..", "python", ".venv")))
isdir(VENV) || error("venv not found at $VENV -- set PENNYLANE_JL_VENV (see README.md)")
const PY   = joinpath(VENV, "bin", "python")
const SP   = readchomp(`$PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"`)
const CLI  = joinpath(VENV, "bin", "catalyst")
const CATALYST_LIB   = joinpath(SP, "catalyst", "lib")
const CATALYST_UTILS = joinpath(SP, "catalyst", "utils")
# Same query the Catalyst frontend makes: (device name, path of the runtime device plugin)
const LIGHT = strip(readchomp(`$PY -W ignore -c "import pennylane as qml; print(qml.device('lightning.qubit', wires=1).get_c_interface()[1])"`))
const KW    = "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"
const WS    = mktempdir()

# ---------------------------------------------------------------------------------------------
# 1. Emit MLIR in Catalyst's quantum/gradient dialects (format taken from catalyst 0.15.0 output)
# ---------------------------------------------------------------------------------------------
mlir = """
module @jlprobe {
  func.func public @jit_circuit(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circuit::@circuit(%arg0) : (tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func public @jit_dcircuit(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = gradient.grad "auto" @module_circuit::@circuit(%arg0) {diffArgIndices = dense<0> : tensor<1xi64>} : (tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  module @module_circuit {
    func.func public @circuit(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, quantum.node} {
      %c0_i64 = arith.constant 0 : i64
      quantum.device shots(%c0_i64) ["$LIGHT", "LightningSimulator", "$KW"]
      %0 = quantum.alloc( 2) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %extracted = tensor.extract %arg0[] : tensor<f64>
      %out_qubits = quantum.custom "RX"(%extracted) %1 : !quantum.bit
      %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
      %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits, %2 : !quantum.bit, !quantum.bit
      %3 = quantum.namedobs %out_qubits_0#1[ PauliZ] : !quantum.obs
      %4 = quantum.expval %3 : f64
      %from_elements = tensor.from_elements %4 : tensor<f64>
      %5 = quantum.insert %0[ 0], %out_qubits_0#0 : !quantum.reg, !quantum.bit
      %6 = quantum.insert %5[ 1], %out_qubits_0#1 : !quantum.reg, !quantum.bit
      quantum.dealloc %6 : !quantum.reg
      quantum.device_release
      return %from_elements : tensor<f64>
    }
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}
"""
mlir_path = joinpath(WS, "jlprobe.mlir")
write(mlir_path, mlir)

# ---------------------------------------------------------------------------------------------
# 2. Compile with the Catalyst CLI (MLIR -> LLVM -> object) and link like Catalyst's LinkerDriver
# ---------------------------------------------------------------------------------------------
t_compile = @elapsed run(pipeline(`$CLI --module-name=jlprobe --workspace=$WS -o $(joinpath(WS, "jlprobe.ll")) $mlir_path`; stderr=devnull))
obj = joinpath(WS, "jlprobe.o")
dylib = joinpath(WS, "libjlprobe.dylib")
link_flags = ["-shared", "-rdynamic",
    "-Wl,-rpath,$CATALYST_LIB", "-L$CATALYST_LIB",
    "-Wl,-rpath,$CATALYST_UTILS", "-L$CATALYST_UTILS",
    "-lrt_capi", "-lpthread", "-lmlir_c_runner_utils", "-llapacke.3",
    joinpath(CATALYST_UTILS, "libcustom_calls.so"),   # yes, `.so` on macOS too
    "-lmlir_async_runtime", "-lrt_rsdecomp"]
Sys.isapple() && push!(link_flags, "-Wl,-arch_errors_fatal")
Sys.islinux() && append!(link_flags, ["-Wl,-no-as-needed", "-Wl,--disable-new-dtags"])
t_link = @elapsed run(`clang $link_flags -o $dylib $obj`)
@printf("catalyst CLI compile: %.2f s, link: %.2f s\n", t_compile, t_link)

# ---------------------------------------------------------------------------------------------
# 3. Call the compiled entry points through the C interface (memref descriptors)
# ---------------------------------------------------------------------------------------------
struct MemRef0{T}          # rank-0 memref descriptor produced by MLIR bufferization
    allocated::Ptr{T}
    aligned::Ptr{T}
    offset::Int64
end

lib = dlopen(dylib, RTLD_NOW | RTLD_GLOBAL)
argv = ["jitted-function"]
GC.@preserve argv begin
    p = Base.unsafe_convert(Ptr{Cstring}, Base.cconvert(Ptr{Cstring}, argv))
    ccall(dlsym(lib, :setup), Cint, (Cint, Ptr{Cstring}), 1, p)       # -> __catalyst__rt__initialize
end
const transfer = dlsym(lib, :_mlir_memory_transfer)

function call_scalar(lib, name::Symbol, theta::Float64)
    fn  = dlsym(lib, name)
    x   = Ref(theta)
    res = Ref(MemRef0{Float64}(C_NULL, C_NULL, 0))
    GC.@preserve x res begin
        px  = Base.unsafe_convert(Ptr{Float64}, x)
        arg = Ref(MemRef0{Float64}(px, px, 0))
        ccall(fn, Cvoid, (Ptr{MemRef0{Float64}}, Ptr{MemRef0{Float64}}), res, arg)
    end
    r   = res[]
    val = unsafe_load(r.aligned, r.offset + 1)
    ccall(transfer, Cvoid, (Ptr{Cvoid},), r.allocated)  # take ownership from the runtime's memory manager
    Libc.free(r.allocated)
    return val
end

println("== Path B: compiled Catalyst MLIR -> dylib -> ccall ==")
θ  = 0.3
ev = call_scalar(lib, :_catalyst_ciface_jit_circuit,  θ)
gr = call_scalar(lib, :_catalyst_ciface_jit_dcircuit, θ)
@printf("expval = %.12f  (cos θ  = %.12f)\n", ev, cos(θ))
@printf("grad   = %.12f  (-sin θ = %.12f)\n", gr, -sin(θ))
@assert isapprox(ev, cos(θ); atol=1e-12) && isapprox(gr, -sin(θ); atol=1e-12)
n = 2000
t = @elapsed for i in 1:n; call_scalar(lib, :_catalyst_ciface_jit_circuit, θ + 1e-6i); end
@printf("latency: %.1f µs / call (includes device init+release per call)\n", 1e6 * t / n)
ccall(dlsym(lib, :teardown), Cvoid, ())                                 # -> __catalyst__rt__finalize

# ---------------------------------------------------------------------------------------------
# 4. Path A': skip the compiler entirely and talk to the runtime C API directly
# ---------------------------------------------------------------------------------------------
println("\n== Path A': Julia -> Catalyst runtime C API -> Lightning ==")
rt = dlopen(joinpath(CATALYST_LIB, "librt_capi." * Libdl.dlext), RTLD_NOW | RTLD_GLOBAL)
S(f) = dlsym(rt, f)
ccall(S(:__catalyst__rt__initialize), Cvoid, (Ptr{UInt32},), C_NULL)
ccall(S(:__catalyst__rt__device_init), Cvoid, (Cstring, Cstring, Cstring, Int64, Bool),
      LIGHT, "LightningSimulator", KW, 0, false)
function direct_expval(θ)
    qreg = ccall(S(:__catalyst__rt__qubit_allocate_array), Ptr{Cvoid}, (Int64,), 2)
    q(i) = unsafe_load(ccall(S(:__catalyst__rt__array_get_element_ptr_1d), Ptr{Ptr{Cvoid}}, (Ptr{Cvoid}, Int64), qreg, i))
    q0, q1 = q(0), q(1)
    ccall(S(:__catalyst__qis__RX),   Cvoid, (Float64, Ptr{Cvoid}, Ptr{Cvoid}), θ, q0, C_NULL)   # NULL Modifiers
    ccall(S(:__catalyst__qis__CNOT), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), q0, q1, C_NULL)
    obs = ccall(S(:__catalyst__qis__NamedObs), Int64, (Int64, Ptr{Cvoid}), 3, q1)  # 3 = PauliZ
    ev  = ccall(S(:__catalyst__qis__Expval), Float64, (Int64,), obs)
    ccall(S(:__catalyst__rt__qubit_release_array), Cvoid, (Ptr{Cvoid},), qreg)
    return ev
end
ev2 = direct_expval(θ)
@printf("expval = %.12f  (cos θ  = %.12f)\n", ev2, cos(θ))
@assert isapprox(ev2, cos(θ); atol=1e-12)
t = @elapsed for i in 1:n; direct_expval(θ + 1e-6i); end
@printf("latency: %.1f µs / circuit\n", 1e6 * t / n)
ccall(S(:__catalyst__rt__device_release), Cvoid, ())
ccall(S(:__catalyst__rt__finalize), Cvoid, ())
println("\nOK -- both paths agree with the analytic result.")
