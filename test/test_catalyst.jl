if !has_catalyst()
    @warn "Catalyst binaries not found; skipping LightningDevice/CatalystDevice tests (run `uv sync --project python`)"
else
    @testset "LightningDevice (runtime FFI)" begin
        dev = LightningDevice()
        @qnode dev function bell(θ)
            a, b = qubits(2)
            a = RX(θ, a)
            a, b = CNOT(a, b)
            return expval(Z(b))
        end
        @test bell(0.3) ≈ cos(0.3)
        @test gradient(bell, 0.3) ≈ -sin(0.3) atol = 1e-6

        @qnode dev function adjtest(θ)
            a = qubits(1)[1]
            a = RX(θ)'(a)
            return expval(Y(a))
        end
        @test adjtest(0.3) ≈ sin(0.3)

        @qnode dev function refq(θ, ϕ)
            reference(θ, ϕ)
        end
        e, v, p = refq(REF_ARGS...)
        @test e ≈ REF_EXPVAL atol = 1e-7
        @test v ≈ REF_VAR atol = 1e-7
        @test p ≈ REF_PROBS atol = 1e-7

        @qnode dev function st()
            q = qubits(2)
            q[1] = Hadamard(q[1])
            q[1], q[2] = CNOT(q[1], q[2])
            return state()
        end
        @test st() ≈ [1 / √2, 0, 0, 1 / √2]

        # parameter shift through the runtime, T1 energy, sampling
        @qnode dev function allg_l(θ); all_gates_circuit(θ); end
        @qnode StateVector() function allg_s(θ); all_gates_circuit(θ); end
        θa = collect(range(0.15, 2.9; length=ALL_GATES_NPARAMS))
        @test allg_l(θa) ≈ allg_s(θa) atol = 1e-10
        @test gradient(allg_l, θa) ≈ gradient(allg_s, θa) atol = 1e-8
        @qnode dev function e_l(θ); h2_energy(θ); end
        @test e_l(0.3) ≈ -1.132688823446468 atol = 1e-9
        ldev = LightningDevice(shots=400)
        @qnode ldev function smp(); q = qubits(2); q[1] = Hadamard(q[1]); q[1], q[2] = CNOT(q[1], q[2]); return sample(); end
        s = smp()
        @test size(s) == (400, 2) && all(x -> x in (0, 1), s) && all(s[:, 1] .== s[:, 2])

        # adjoint differentiation through the runtime's tape recorder
        @test gradient(allg_l, θa; method=:adjoint) ≈ gradient(allg_s, θa; method=:parameter_shift) atol = 1e-8
        @test gradient(allg_l, θa) ≈ gradient(allg_s, θa) atol = 1e-8                         # :auto → adjoint
        @test gradient(e_l, 0.3; method=:adjoint) ≈ gradient(e_l, 0.3; method=:parameter_shift) atol = 1e-10
        @qnode dev function rnd_grad(θ); random_circuit(θ)[1]; end
        θg = 2π .* rand(MersenneTwister(7), RANDOM_NPARAMS)
        @test gradient(rnd_grad, θg; method=:adjoint) ≈ gradient(rnd_grad, θg; method=:parameter_shift) atol = 1e-8

        # mid-circuit measurements and control flow through the runtime (T3)
        @qnode dev function tp_l(θ); teleport(θ); end
        for θ in (0.3, 2.1), _ in 1:3
            ez, ey = tp_l(θ)
            @test ez ≈ cos(θ) atol = 1e-10
            @test ey ≈ -sin(θ) atol = 1e-10
        end
        @qnode dev function rus_l(); rus(); end
        for _ in 1:3
            @test rus_l() ≈ -1 atol = 1e-10
        end

        # random circuits: Lightning vs the Julia simulator
        @qnode dev function rnd_l(θ); random_circuit(θ); end
        @qnode StateVector() function rnd_s(θ); random_circuit(θ); end
        for seed in 1:3
            θ = 2π .* rand(RANDOM_NPARAMS)
            a, b = rnd_l(θ), rnd_s(θ)
            @test a[1] ≈ b[1] atol = 1e-10
            @test a[2] ≈ b[2] atol = 1e-10
            @test a[3] ≈ b[3] atol = 1e-10
        end
    end

    @testset "CatalystDevice (compiled)" begin
        dev = CatalystDevice()
        @qnode dev function bell(θ)
            a, b = qubits(2)
            a = RX(θ, a)
            a, b = CNOT(a, b)
            return expval(Z(b))
        end
        @test bell(0.3) ≈ cos(0.3)
        @test gradient(bell, 0.3) ≈ -sin(0.3)                       # compiled adjoint
        @test gradient(bell, 0.3; method=:finitediff) ≈ -sin(0.3) atol = 1e-6
        @test length(dev.cache) == 1                                  # one compilation per signature

        @qnode dev function refq(θ, ϕ)
            reference(θ, ϕ)
        end
        e, v, p = refq(REF_ARGS...)
        @test e ≈ REF_EXPVAL atol = 1e-7
        @test v ≈ REF_VAR atol = 1e-7
        @test p ≈ REF_PROBS atol = 1e-7

        # vector + scalar arguments, compiled adjoint gradient vs finite differences on StateVector
        @qnode dev function vq(θ, ϕ)
            q = qubits(3)
            for i in 1:3
                q[i] = RY(θ[i], q[i])
            end
            q[1], q[2] = CNOT(q[1], q[2])
            q[2], q[3] = CRX(ϕ, q[2], q[3])
            return expval(0.5 * Z(q[1]) * Z(q[3]) + X(q[2]))
        end
        @qnode StateVector() function vs(θ, ϕ)
            q = qubits(3)
            for i in 1:3
                q[i] = RY(θ[i], q[i])
            end
            q[1], q[2] = CNOT(q[1], q[2])
            q[2], q[3] = CRX(ϕ, q[2], q[3])
            return expval(0.5 * Z(q[1]) * Z(q[3]) + X(q[2]))
        end
        θ = [0.3, 0.8, 1.1]; ϕ = 0.6
        @test vq(θ, ϕ) ≈ vs(θ, ϕ)
        gθ, gϕ = gradient(vq, θ, ϕ)
        fθ, fϕ = gradient(vs, θ, ϕ)
        @test gθ ≈ fθ atol = 1e-6
        @test gϕ ≈ fϕ atol = 1e-6

        @qnode dev function st()
            q = qubits(2)
            q[1] = Hadamard(q[1])
            q[1], q[2] = CNOT(q[1], q[2])
            return state()
        end
        @test st() ≈ [1 / √2, 0, 0, 1 / √2]

        # T1 on the compiled path: energy and compiled adjoint gradient vs parameter shift
        @qnode dev function e_c(θ); h2_energy(θ); end
        @qnode StateVector() function e_s(θ); h2_energy(θ); end
        @test e_c(0.3) ≈ -1.132688823446468 atol = 1e-9
        @test gradient(e_c, 0.3) ≈ gradient(e_s, 0.3) atol = 1e-8
        @qnode dev function allg_c(θ); all_gates_circuit(θ); end
        @qnode StateVector() function allg_s(θ); all_gates_circuit(θ); end
        θ = collect(range(0.15, 2.9; length=ALL_GATES_NPARAMS))
        @test allg_c(θ) ≈ allg_s(θ) atol = 1e-10
        @test gradient(allg_c, θ) ≈ gradient(allg_s, θ) atol = 1e-7

        # control flow and mid-circuit measurements compiled by Catalyst (T3, T2 loop version)
        @qnode dev function tp_c(θ); teleport(θ); end
        for θ in (0.3, 2.1), _ in 1:3
            ez, ey = tp_c(θ)
            @test ez ≈ cos(θ) atol = 1e-10
            @test ey ≈ -sin(θ) atol = 1e-10
        end
        @qnode dev function rus_c(); rus(); end
        for _ in 1:3
            @test rus_c() ≈ -1 atol = 1e-10
        end
        @qnode dev function ql_c(γ, β); qaoa_looped(γ, β); end
        @qnode StateVector() function qu_s(γ, β); qaoa_unrolled(γ, β); end
        γ, β = [0.4, 0.9], [0.3, 0.6]
        @test ql_c(γ, β) ≈ qu_s(γ, β) atol = 1e-10
        @test gradient(ql_c, γ, β)[1] ≈ gradient(qu_s, γ, β)[1] atol = 1e-7      # compiled adjoint through scf.for
        @qnode dev function acc_c(θ)
            q = qubits(1)[1]
            s = 0.0
            @trace for i in 1:4
                s = s + θ[i] * 0.5
            end
            q = RX(s, q)
            k = 0
            @trace while k < 3
                q = RY(θ[1], q)
                k = k + 1
            end
            return expval(Z(q))
        end
        θ4 = [0.2, 0.4, 0.6, 0.8]
        @test acc_c(θ4) ≈ cos(sum(θ4) * 0.5) * cos(3 * θ4[1]) atol = 1e-10
        @test_throws ArgumentError gradient(acc_c, θ4)                            # no compiled gradient through scf.while
        @test_throws ArgumentError gradient(tp_c, 0.3)                            # no compiled gradient with MCM

        # shots: sampling through the compiled program
        sdev = CatalystDevice(shots=300)
        @qnode sdev function smp(); q = qubits(2); q[1] = Hadamard(q[1]); q[1], q[2] = CNOT(q[1], q[2]); return sample(), expval(Z(q[1])); end
        s, ez = smp()
        @test size(s) == (300, 2) && all(x -> x in (0, 1), s) && all(s[:, 1] .== s[:, 2])
        @test abs(ez) < 0.25
    end
end
