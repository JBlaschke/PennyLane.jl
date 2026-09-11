using PennyLane
using Test
using LinearAlgebra
using Random

const σ = PennyLane.PAULI_MATRICES

include("circuits.jl")

@testset "PennyLane.jl" begin
    include("test_operators.jl")
    include("test_tracer.jl")
    include("test_statevector.jl")
    include("test_gradients.jl")
    include("test_shots.jl")
    include("test_vqe.jl")
    include("test_controlflow.jl")
    include("test_qaoa.jl")
    include("test_catalyst.jl")
    include("test_pythoncall.jl")
end
