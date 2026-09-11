# Bindings to the Catalyst runtime C API (librt_capi) and the `LightningDevice` simulator.
#
# The runtime is a QIR-style C library; devices are plugins it loads at `device_init`. Driving it
# from Julia needs no compiler and no Python process, only the wheel's shared libraries.

struct Modifiers                         # runtime/include/Types.h
    adjoint::Bool
    num_controlled::Csize_t
    controlled_wires::Ptr{Ptr{Cvoid}}
    controlled_values::Ptr{Bool}
end
struct MemRef1D{T}                       # MemRefT_<T>_1d
    allocated::Ptr{T}
    aligned::Ptr{T}
    offset::Csize_t
    size::Csize_t
    stride::Csize_t
end
memref1d(v::Vector{T}) where {T} = MemRef1D{T}(pointer(v), pointer(v), 0, length(v), 1)
struct MemRef2D{T}                       # MemRefT_<T>_2d (row-major sizes/strides)
    allocated::Ptr{T}
    aligned::Ptr{T}
    offset::Csize_t
    size0::Csize_t
    size1::Csize_t
    stride0::Csize_t
    stride1::Csize_t
end

const _RT_LIB = Ref{Ptr{Cvoid}}(C_NULL)
const _RT_INIT = Ref(false)

function rt_lib()
    if _RT_LIB[] == C_NULL
        env = catalyst_env()
        _RT_LIB[] = Libdl.dlopen(joinpath(env.libdir, "librt_capi." * Libdl.dlext), Libdl.RTLD_NOW | Libdl.RTLD_GLOBAL)
    end
    _RT_LIB[]
end
rt_sym(name::Symbol) = Libdl.dlsym(rt_lib(), name)

"""Initialise the runtime once per process (both `LightningDevice` and `CatalystDevice` share it)."""
function rt_initialize!()
    _RT_INIT[] && return
    ccall(rt_sym(:__catalyst__rt__initialize), Cvoid, (Ptr{UInt32},), C_NULL)
    _RT_INIT[] = true
    nothing
end
function rt_finalize!()
    _RT_INIT[] || return
    ccall(rt_sym(:__catalyst__rt__finalize), Cvoid, ())
    _RT_INIT[] = false
    nothing
end

rt_device_init(lib, name, kwargs, shots) =
    ccall(rt_sym(:__catalyst__rt__device_init), Cvoid, (Cstring, Cstring, Cstring, Int64, Bool), lib, name, kwargs, shots, false)
rt_device_release() = ccall(rt_sym(:__catalyst__rt__device_release), Cvoid, ())
rt_qubit_allocate_array(n) = ccall(rt_sym(:__catalyst__rt__qubit_allocate_array), Ptr{Cvoid}, (Int64,), n)
rt_qubit_release_array(a) = ccall(rt_sym(:__catalyst__rt__qubit_release_array), Cvoid, (Ptr{Cvoid},), a)
rt_array_get_element(a, i) =
    unsafe_load(ccall(rt_sym(:__catalyst__rt__array_get_element_ptr_1d), Ptr{Ptr{Cvoid}}, (Ptr{Cvoid}, Int64), a, i))

# gates: __catalyst__qis__<Name>(params..., qubits..., const Modifiers*)
for def in collect(values(GATES))
    def.nqubits == -1 && continue
    fsym = QuoteNode(Symbol("__catalyst__qis__", def.name))
    ptypes = Any[fill(:Float64, def.nparams); fill(:(Ptr{Cvoid}), def.nqubits); :(Ptr{Modifiers})]
    pargs = [:(p[$i]) for i in 1:def.nparams]
    qargs = [:(q[$i]) for i in 1:def.nqubits]
    @eval rt_gate!(::Val{$(QuoteNode(def.name))}, p::Vector{Float64}, q::Vector{Ptr{Cvoid}}, m::Ptr{Modifiers}) =
        ccall(rt_sym($fsym), Cvoid, ($(ptypes...),), $(pargs...), $(qargs...), m)
end

# variadic C functions need one method per arity (ccall cannot splat)
const MAX_VARARGS = 32
for n in 1:MAX_VARARGS
    qargs = [:(q[$i]) for i in 1:n]
    iargs = [:(ids[$i]) for i in 1:n]
    @eval begin
        rt_gate!(::Val{:MultiRZ}, p::Vector{Float64}, q::Vector{Ptr{Cvoid}}, m::Ptr{Modifiers}, ::Val{$n}) =
            ccall(rt_sym(:__catalyst__qis__MultiRZ), Cvoid, (Float64, Ptr{Modifiers}, Int64, Ptr{Cvoid}...), p[1], m, $n, $(qargs...))
        _rt_tensor_obs(ids::Vector{Int64}, ::Val{$n}) =
            ccall(rt_sym(:__catalyst__qis__TensorObs), Int64, (Int64, Int64...), $n, $(iargs...))
        _rt_hamiltonian_obs(m::Ref{MemRef1D{Float64}}, ids::Vector{Int64}, ::Val{$n}) =
            ccall(rt_sym(:__catalyst__qis__HamiltonianObs), Int64, (Ptr{MemRef1D{Float64}}, Int64, Int64...), m, $n, $(iargs...))
        _rt_probs(m::Ref{MemRef1D{Float64}}, q::Vector{Ptr{Cvoid}}, ::Val{$n}) =
            ccall(rt_sym(:__catalyst__qis__Probs), Cvoid, (Ptr{MemRef1D{Float64}}, Int64, Ptr{Cvoid}...), m, $n, $(qargs...))
        _rt_sample(m::Ref{MemRef2D{Float64}}, q::Vector{Ptr{Cvoid}}, ::Val{$n}) =
            ccall(rt_sym(:__catalyst__qis__Sample), Cvoid, (Ptr{MemRef2D{Float64}}, Int64, Ptr{Cvoid}...), m, $n, $(qargs...))
    end
end
_checkarity(n) = n <= MAX_VARARGS || throw(ArgumentError("at most $MAX_VARARGS operands are supported here, got $n"))
rt_gate!(v::Val{:MultiRZ}, p::Vector{Float64}, q::Vector{Ptr{Cvoid}}, m::Ptr{Modifiers}) =
    (_checkarity(length(q)); rt_gate!(v, p, q, m, Val(length(q))))

function with_modifiers(f, adjoint::Bool, ctrls::Vector{Ptr{Cvoid}}, vals::Vector{Bool})
    (!adjoint && isempty(ctrls)) && return f(Ptr{Modifiers}(C_NULL))
    GC.@preserve ctrls vals begin
        m = Ref(Modifiers(adjoint, length(ctrls), isempty(ctrls) ? C_NULL : pointer(ctrls), isempty(vals) ? C_NULL : pointer(vals)))
        GC.@preserve m f(Base.unsafe_convert(Ptr{Modifiers}, m))
    end
end

const OBS_CODES = Dict(:Identity => 0, :X => 1, :Y => 2, :Z => 3, :Hadamard => 4)
rt_named_obs(code::Integer, q::Ptr{Cvoid}) = ccall(rt_sym(:__catalyst__qis__NamedObs), Int64, (Int64, Ptr{Cvoid}), code, q)
rt_tensor_obs(ids::Vector{Int64}) = (_checkarity(length(ids)); _rt_tensor_obs(ids, Val(length(ids))))
function rt_hamiltonian_obs(coeffs::Vector{Float64}, ids::Vector{Int64})
    _checkarity(length(ids))
    GC.@preserve coeffs begin
        _rt_hamiltonian_obs(Ref(memref1d(coeffs)), ids, Val(length(ids)))
    end
end
rt_expval(id::Int64) = ccall(rt_sym(:__catalyst__qis__Expval), Float64, (Int64,), id)
rt_var(id::Int64) = ccall(rt_sym(:__catalyst__qis__Variance), Float64, (Int64,), id)
function rt_probs(q::Vector{Ptr{Cvoid}})
    _checkarity(length(q))
    out = Vector{Float64}(undef, 1 << length(q))
    GC.@preserve out begin
        _rt_probs(Ref(memref1d(out)), q, Val(length(q)))
    end
    out
end
function rt_sample(shots::Int, q::Vector{Ptr{Cvoid}})
    _checkarity(length(q))
    n = length(q)
    out = Matrix{Float64}(undef, n, shots)          # C row-major (shots × n) == Julia column-major (n × shots)
    GC.@preserve out begin
        _rt_sample(Ref(MemRef2D{Float64}(pointer(out), pointer(out), 0, shots, n, n, 1)), q, Val(n))
    end
    Int.(round.(permutedims(out)))
end
function rt_state(n::Int)
    out = Vector{ComplexF64}(undef, 1 << n)
    GC.@preserve out begin
        ccall(rt_sym(:__catalyst__qis__State), Cvoid, (Ptr{MemRef1D{ComplexF64}}, Int64), Ref(memref1d(out)), 0)
    end
    out
end

# ---- LightningDevice --------------------------------------------------------------------------
"""
    LightningDevice([nwires]; shots=0, kwargs=LIGHTNING_KWARGS)

PennyLane-Lightning's state-vector simulator, driven through the Catalyst runtime C API.
"""
struct LightningDevice <: AbstractSimulator
    nwires::Union{Nothing,Int}
    shots::Int
    kwargs::String
end
LightningDevice(n::Union{Nothing,Integer}=nothing; shots::Integer=0, kwargs::AbstractString=LIGHTNING_KWARGS) =
    LightningDevice(n === nothing ? nothing : Int(n), Int(shots), String(kwargs))
Base.show(io::IO, d::LightningDevice) = print(io, "LightningDevice(", d.nwires === nothing ? "" : d.nwires, ")")

mutable struct RTState
    qreg::Ptr{Cvoid}
    qubits::Vector{Ptr{Cvoid}}
    n::Int
    active::Bool
end

function sim_allocate(dev::LightningDevice, n::Int)
    env = catalyst_env()
    rt_initialize!()
    rt_device_init(env.lightning_plugin, "LightningSimulator", dev.kwargs, dev.shots)
    qreg = rt_qubit_allocate_array(n)
    RTState(qreg, Ptr{Cvoid}[rt_array_get_element(qreg, i - 1) for i in 1:n], n, true)
end
function sim_release!(::LightningDevice, st::RTState)
    st.active || return
    rt_qubit_release_array(st.qreg)
    rt_device_release()
    st.active = false
    nothing
end
function sim_apply!(::LightningDevice, st::RTState, name::Symbol, params::Vector{Float64}, wires::Vector{Int};
                    adjoint::Bool=false, ctrl_wires::Vector{Int}=Int[], ctrl_values::Vector{Bool}=Bool[])
    q = Ptr{Cvoid}[st.qubits[w] for w in wires]
    c = Ptr{Cvoid}[st.qubits[w] for w in ctrl_wires]
    with_modifiers(adjoint, c, collect(Bool, ctrl_values)) do m
        rt_gate!(Val(name), params, q, m)
    end
    st
end
function rt_word_id(st::RTState, p::PauliString)
    isempty(p.word) && return rt_named_obs(0, st.qubits[1])
    ids = Int64[rt_named_obs(OBS_CODES[l], st.qubits[w]) for (w, l) in p.word]
    length(ids) == 1 ? ids[1] : rt_tensor_obs(ids)
end
function rt_obs_id(st::RTState, o::Observable)
    ts = terms(o)
    length(ts) == 1 && ts[1].coeff == 1 && return rt_word_id(st, ts[1])
    rt_hamiltonian_obs(Float64[real(t.coeff) for t in ts], Int64[rt_word_id(st, t) for t in ts])
end
# ⟨H⟩ is linear: evaluate term by term (no arity limit, same work Lightning does internally)
sim_expval(::LightningDevice, st::RTState, o::Observable) =
    sum(real(t.coeff) * rt_expval(rt_word_id(st, t)) for t in terms(o); init=0.0)
sim_var(dev::LightningDevice, st::RTState, o::Observable) =
    length(terms(o)) <= MAX_VARARGS ? rt_var(rt_obs_id(st, o)) : invoke(sim_var, Tuple{AbstractSimulator,Any,Observable}, dev, st, o)
sim_probs(::LightningDevice, st::RTState, wires::Vector{Int}) = rt_probs(Ptr{Cvoid}[st.qubits[w] for w in wires])
function sim_sample(dev::LightningDevice, st::RTState, wires::Vector{Int})
    dev.shots > 0 || throw(ArgumentError("sample() needs shots: use LightningDevice(shots=n)"))
    rt_sample(dev.shots, Ptr{Cvoid}[st.qubits[w] for w in wires])
end
sim_state(::LightningDevice, st::RTState) = rt_state(st.n)
