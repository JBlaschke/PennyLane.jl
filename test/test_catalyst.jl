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

        # shots: sampling through the compiled program
        sdev = CatalystDevice(shots=300)
        @qnode sdev function smp(); q = qubits(2); q[1] = Hadamard(q[1]); q[1], q[2] = CNOT(q[1], q[2]); return sample(), expval(Z(q[1])); end
        s, ez = smp()
        @test size(s) == (300, 2) && all(x -> x in (0, 1), s) && all(s[:, 1] .== s[:, 2])
        @test abs(ez) < 0.25
    end
end
