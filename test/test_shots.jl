@testset "shots and sampling" begin
    dev = StateVector(shots=20_000, rng=MersenneTwister(7))
    @qnode dev function bell(θ)
        a, b = qubits(2)
        a = RX(θ, a)
        a, b = CNOT(a, b)
        return expval(Z(b)), probs(), sample(), var(Z(b))
    end
    e, p, s, v = bell(0.3)
    @test e ≈ cos(0.3) atol = 0.03
    @test p ≈ [cos(0.15)^2, 0, 0, sin(0.15)^2] atol = 0.02
    @test sum(p) ≈ 1
    @test size(s) == (20_000, 2) && all(x -> x in (0, 1), s)
    @test all(s[:, 1] .== s[:, 2])                    # RX–CNOT: both bits always agree
    @test v ≈ 1 - cos(0.3)^2 atol = 0.03

    # deterministic with a seeded generator
    d1 = StateVector(shots=100, rng=MersenneTwister(1)); d2 = StateVector(shots=100, rng=MersenneTwister(1))
    @qnode d1 function s1(); a = qubits(1)[1]; a = Hadamard(a); return sample(a); end
    @qnode d2 function s2(); a = qubits(1)[1]; a = Hadamard(a); return sample(a); end
    @test s1() == s2()

    # sampled expectation of a Hamiltonian, and parameter-shift on noisy estimates
    dev2 = StateVector(shots=50_000, rng=MersenneTwister(3))
    @qnode dev2 function e(θ)
        h2_energy(θ)
    end
    @qnode StateVector() function e_exact(θ)
        h2_energy(θ)
    end
    @test e(0.2) ≈ e_exact(0.2) atol = 0.01
    @test gradient(e, 0.2) ≈ gradient(e_exact, 0.2) atol = 0.02

    # analytic devices refuse to sample
    @qnode StateVector() function nos(); a = qubits(1)[1]; return sample(a); end
    @test_throws ArgumentError nos()

    @test occursin("shots=20000", string(dev))
end
