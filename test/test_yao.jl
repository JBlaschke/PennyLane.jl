if Base.find_package("Yao") === nothing
    @warn "skipping YaoDevice tests (Yao not available in this environment)"
else
    import Yao
    @testset "YaoDevice (Yao.jl backend)" begin
        dev = YaoDevice()
        @qnode dev function refq(θ, ϕ); reference(θ, ϕ); end
        e, v, p = refq(REF_ARGS...)
        @test e ≈ REF_EXPVAL atol = 1e-7
        @test v ≈ REF_VAR atol = 1e-7
        @test p ≈ REF_PROBS atol = 1e-7

        @qnode dev function rnd_y(θ); random_circuit(θ); end
        @qnode StateVector() function rnd_s(θ); random_circuit(θ); end
        for seed in 1:3
            θ = 2π .* rand(MersenneTwister(seed), RANDOM_NPARAMS)
            a, b = rnd_y(θ), rnd_s(θ)
            @test a[1] ≈ b[1] atol = 1e-10
            @test a[2] ≈ b[2] atol = 1e-10
            @test a[3] ≈ b[3] atol = 1e-10
        end
        @qnode dev function allg(θ); all_gates_circuit(θ); end
        @qnode StateVector() function allg_s(θ); all_gates_circuit(θ); end
        θ = collect(range(0.15, 2.9; length=ALL_GATES_NPARAMS))
        @test allg(θ) ≈ allg_s(θ) atol = 1e-10
        @test gradient(allg, θ) ≈ gradient(allg_s, θ) atol = 1e-8        # parameter shift on the Yao backend

        @qnode dev function st(); q = qubits(3); q[1] = Hadamard(q[1]); q[1], q[2] = CNOT(q[1], q[2]); q[2], q[3] = CRX(0.4, q[2], q[3]); return state(); end
        @qnode StateVector() function st_s(); q = qubits(3); q[1] = Hadamard(q[1]); q[1], q[2] = CNOT(q[1], q[2]); q[2], q[3] = CRX(0.4, q[2], q[3]); return state(); end
        @test st() ≈ st_s() atol = 1e-12

        @qnode YaoDevice(rng=MersenneTwister(3)) function tp(θ); teleport(θ); end
        for _ in 1:3
            ez, ey = tp(0.9)
            @test ez ≈ cos(0.9) atol = 1e-10
            @test ey ≈ -sin(0.9) atol = 1e-10
        end
        sdev = YaoDevice(shots=2000, rng=MersenneTwister(4))
        @qnode sdev function smp(θ); a, b = qubits(2); a = RX(θ, a); a, b = CNOT(a, b); return sample(), expval(Z(b)); end
        s, ez = smp(0.3)
        @test size(s) == (2000, 2) && all(s[:, 1] .== s[:, 2])
        @test ez ≈ cos(0.3) atol = 0.1
    end
end
