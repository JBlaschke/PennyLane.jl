@testset "T1: VQE for H₂" begin
    H, n = h2_hamiltonian()
    @test n == 4
    @test length(terms(H)) == 15
    @test H' == H
    @test eigmin(Hermitian(matrix(H, 4))) ≈ H2_GROUND_ENERGY atol = 1e-10

    dev = StateVector()
    @qnode dev function energy(θ)
        h2_energy(θ)
    end
    @test energy(0.0) ≈ -1.117348921135931 atol = 1e-9       # Hartree–Fock energy
    @test energy(0.3) ≈ -1.132688823446468 atol = 1e-9       # value from PennyLane

    # plain gradient descent with parameter-shift gradients
    θ = 0.0
    for it in 1:60
        θ -= 0.4 * gradient(energy, θ)
    end
    @test θ ≈ 0.20973287 atol = 1e-4
    @test energy(θ) ≈ H2_GROUND_ENERGY atol = 1e-6

    # hardware-efficient ansatz: gradients agree and descent lowers the energy
    @qnode dev function hea(θ)
        h2_hea(θ)
    end
    w = 0.1 .* collect(1:8)
    @test gradient(hea, w) ≈ gradient(hea, w; method=:finitediff) atol = 1e-6
    e0 = hea(w)
    for it in 1:20
        w -= 0.2 * gradient(hea, w)
    end
    @test hea(w) < e0
end
