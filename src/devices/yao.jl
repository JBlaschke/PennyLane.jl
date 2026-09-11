# Yao.jl backend — the implementation lives in ext/PennyLaneYaoExt.jl and is activated by `using Yao`.

"""
    YaoDevice([nwires]; shots=0, rng=Random.default_rng())

State-vector simulation with Yao.jl's `ArrayReg` and `instruct!` kernels (requires `using Yao`).
Demonstrates the `AbstractSimulator` interface; results match `StateVector` element-wise.
"""
struct YaoDevice <: AbstractSimulator
    nwires::Union{Nothing,Int}
    shots::Int
    rng::Random.AbstractRNG
end
YaoDevice(n::Union{Nothing,Integer}=nothing; shots::Integer=0, rng::Random.AbstractRNG=Random.default_rng()) =
    YaoDevice(n === nothing ? nothing : Int(n), Int(shots), rng)
Base.show(io::IO, d::YaoDevice) = print(io, "YaoDevice(", d.nwires === nothing ? "" : d.nwires, d.shots == 0 ? "" : "; shots=$(d.shots)", ")")
sim_allocate(::YaoDevice, n) = error("YaoDevice needs the Yao package: add Yao to your project and `import Yao`")
