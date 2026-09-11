@testset "T2: QAOA MaxCut and time evolution" begin
    # evolve against dense matrix exponentials
    H = 0.7 * Z(1) * Z(2) + 0.3 * X(1) - 0.2 * Y(2)
    @qnode StateVector() function ev(t)
        q = qubits(2)
        q[1] = Hadamard(q[1]); q[2] = RY(0.4, q[2])
        q = collect(evolve(H, t, q...; steps=2000))
        return state()
    end
    ψ0 = kron(matrix(Hadamard()), matrix(RY(0.4))) * [1, 0, 0, 0]
    @test ev(0.9) ≈ exp(-im * 0.9 * Matrix(matrix(H, 2))) * ψ0 atol = 1e-4       # first-order Trotter, 2000 steps
    Hc = 0.7 * Z(1) * Z(2) + 0.3 * Z(1)                                            # commuting terms: exact
    @qnode StateVector() function evc(t)
        q = qubits(2)
        q[1] = Hadamard(q[1]); q[2] = Hadamard(q[2])
        q = collect(evolve(Hc, t, q...))
        return state()
    end
    @test evc(1.3) ≈ exp(-im * 1.3 * Matrix(matrix(Hc, 2))) * fill(0.5, 4) atol = 1e-12
    U = exp(-im * 0.6 * Hc)
    @test matrix(U) ≈ exp(-im * 0.6 * Matrix(matrix(Hc, 2)))
    @qnode StateVector() function evb()
        q = qubits(2)
        q[1] = Hadamard(q[1]); q[2] = Hadamard(q[2])
        q = collect(U(q...))
        return state()
    end
    @test evb() ≈ matrix(U) * fill(0.5, 4) atol = 1e-12
    @test_throws ArgumentError exp(0.3 * Z(1) + Z(2))

    # QAOA: unrolled vs @trace for, against a dense-matrix reference
    C = maxcut_cost(RING)
    @test C ≈ 2.0 * Identity() - 0.5 * (Z(1) * Z(2) + Z(2) * Z(3) + Z(3) * Z(4) + Z(4) * Z(1))
    dev = StateVector()
    @qnode dev function qu(γ, β); qaoa_unrolled(γ, β); end
    @qnode dev function ql(γ, β); qaoa_looped(γ, β); end
    γ, β = [0.4, 0.9], [0.3, 0.6]
    Cm = Matrix(matrix(C, 4))
    Bm = Matrix(matrix(mixer(4), 4))
    ψ = fill(0.25 + 0im, 16)
    for l in 1:2
        ψ = exp(-im * β[l] * Bm) * (exp(-im * γ[l] * Cm) * ψ)
    end
    ref = real(dot(ψ, Cm * ψ))
    @test qu(γ, β) ≈ ref atol = 1e-10
    @test ql(γ, β) ≈ ref atol = 1e-10
    @test count(n -> n isa PennyLane.ForNode, program(ql, γ, β).nodes) == 1
    @test 0 < ref < 4

    # gradients of the unrolled circuit by parameter shift; optimisation improves the cut
    g = gradient(qu, γ, β)
    gf = gradient(qu, γ, β; method=:finitediff)
    @test g[1] ≈ gf[1] atol = 1e-6
    @test g[2] ≈ gf[2] atol = 1e-6
    x = vcat(γ, β)
    for _ in 1:25
        gγ, gβ = gradient(qu, x[1:2], x[3:4])
        x += 0.1 * vcat(gγ, gβ)                 # maximise ⟨C⟩
    end
    @test qu(x[1:2], x[3:4]) > ref

    # sampled cut values agree with ⟨C⟩ on average
    sdev = StateVector(shots=4000, rng=MersenneTwister(5))
    @qnode sdev function qs(γ, β)
        q = qubits(4)
        for i in 1:4
            q[i] = Hadamard(q[i])
        end
        q = collect(evolve(C, γ, q...))
        q = collect(evolve(mixer(4), β, q...))
        return sample()
    end
    bits = qs(0.4, 0.3)
    cuts = [sum(bits[k, i] != bits[k, j] for (i, j) in RING) for k in 1:size(bits, 1)]
    @qnode dev function q1(γ, β); qaoa_unrolled([γ], [β]); end
    @test sum(cuts) / length(cuts) ≈ q1(0.4, 0.3) atol = 0.1
end
