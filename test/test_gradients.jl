@testset "parameter-shift gradients" begin
    dev = StateVector()

    @qnode dev function allg(θ)
        all_gates_circuit(θ)
    end
    θ = collect(range(0.15, 2.9; length=ALL_GATES_NPARAMS))
    gps = gradient(allg, θ)                          # parameter shift (default on simulators)
    gfd = gradient(allg, θ; method=:finitediff)
    @test length(gps) == ALL_GATES_NPARAMS
    @test gps ≈ gfd atol = 1e-6
    @test any(abs.(gps) .> 0.05)                     # not a trivial point

    # two-term rule is exact for a plain rotation: compare with the analytic derivative
    @qnode dev function bell(θ)
        a, b = qubits(2)
        a = RX(θ, a)
        a, b = CNOT(a, b)
        return expval(Z(b))
    end
    @test gradient(bell, 0.3) ≈ -sin(0.3) atol = 1e-12
    @test gradient(bell, 0.3; method=:parameter_shift) ≈ -sin(0.3) atol = 1e-12

    # chain rule through traced arithmetic and scalar + vector arguments
    @qnode dev function mixed(θ, ϕ)
        a, b = qubits(2)
        a = RY(2θ[1] * ϕ + cos(θ[2]), a)
        a, b = CRX(ϕ / 3, a, b)
        return expval(Z(a) * Z(b) + 0.5 * X(b))
    end
    gθ, gϕ = gradient(mixed, [0.4, 1.1], 0.7)
    fθ, fϕ = gradient(mixed, [0.4, 1.1], 0.7; method=:finitediff)
    @test gθ ≈ fθ atol = 1e-6
    @test gϕ ≈ fϕ atol = 1e-6

    # constant parameters cost nothing and unsupported outputs are rejected clearly
    @qnode dev function consts(θ)
        a = qubits(1)[1]
        a = RX(0.3, a)
        a = RY(θ, a)
        return expval(Z(a))
    end
    @test gradient(consts, 0.2) ≈ -cos(0.3) * sin(0.2) atol = 1e-12
    @qnode dev function varq(θ)
        a = qubits(1)[1]
        a = RX(θ, a)
        return var(Z(a))
    end
    @test_throws ArgumentError gradient(varq, 0.2)
    @test gradient(varq, 0.2; method=:finitediff) ≈ 2 * sin(0.2) * cos(0.2) atol = 1e-6
    @test gradient(bell, 0.3; method=:adjoint) ≈ -sin(0.3) atol = 1e-12
    @test_throws ArgumentError gradient(bell, 0.3; method=:nonsense)

    # the same rules on the H₂ energy
    @qnode dev function e(θ)
        h2_energy(θ)
    end
    @test gradient(e, 0.1) ≈ gradient(e, 0.1; method=:finitediff) atol = 1e-6
end
