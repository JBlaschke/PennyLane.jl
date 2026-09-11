@testset "adjoint gradients (StateVector)" begin
    dev = StateVector()
    @qnode dev function allg(θ); all_gates_circuit(θ); end
    θ = collect(range(0.15, 2.9; length=ALL_GATES_NPARAMS))
    ga = gradient(allg, θ; method=:adjoint)
    gp = gradient(allg, θ; method=:parameter_shift)
    @test ga ≈ gp atol = 1e-9
    @test gradient(allg, θ) ≈ gp atol = 1e-9                     # :auto picks adjoint on an analytic simulator

    @qnode dev function e(θ); h2_energy(θ); end
    @test gradient(e, 0.3; method=:adjoint) ≈ gradient(e, 0.3; method=:parameter_shift) atol = 1e-10

    @qnode dev function qu(γ, β); qaoa_unrolled(γ, β); end
    γ, β = [0.4, 0.9], [0.3, 0.6]
    ga = gradient(qu, γ, β; method=:adjoint)
    gp = gradient(qu, γ, β; method=:parameter_shift)
    @test ga[1] ≈ gp[1] atol = 1e-9
    @test ga[2] ≈ gp[2] atol = 1e-9

    @qnode dev function rnd(θ); random_circuit(θ)[1]; end
    θr = 2π .* rand(MersenneTwister(9), RANDOM_NPARAMS)
    @test gradient(rnd, θr; method=:adjoint) ≈ gradient(rnd, θr; method=:parameter_shift) atol = 1e-9

    @qnode dev function bell(θ); a, b = qubits(2); a = RX(θ, a); a, b = CNOT(a, b); return expval(Z(b)); end
    @test gradient(bell, 0.3; method=:adjoint) ≈ -sin(0.3) atol = 1e-12
    @qnode dev function adj(θ); a = qubits(1)[1]; a = RX(θ)'(a); return expval(Y(a)); end
    @test gradient(adj, 0.3; method=:adjoint) ≈ cos(0.3) atol = 1e-12

    @qnode dev function v(θ); a = qubits(1)[1]; a = RX(θ, a); return var(Z(a)); end
    @test_throws ArgumentError gradient(v, 0.3; method=:adjoint)
    @qnode StateVector(shots=100) function sh(θ); a = qubits(1)[1]; a = RX(θ, a); return expval(Z(a)); end
    @test_throws ArgumentError gradient(sh, 0.3; method=:adjoint)
    @qnode dev function tpz(θ); teleport(θ)[1]; end
    @test_throws ArgumentError gradient(tpz, 0.3; method=:adjoint)
end
