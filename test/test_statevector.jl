@testset "StateVector simulator" begin
    dev = StateVector()
    @qnode dev function bell(θ)
        a, b = qubits(2)
        a = RX(θ, a)
        a, b = CNOT(a, b)
        return expval(Z(b))
    end
    @test bell(0.3) ≈ cos(0.3)
    @test bell(1.0) ≈ cos(1.0)
    @test gradient(bell, 0.3) ≈ -sin(0.3) atol = 1e-6

    @qnode dev function single(θ)
        a = qubits(1)[1]
        a = RY(θ, a)
        return expval(X(a)), expval(Z(a)), expval(Y(a)), var(Z(a))
    end
    ex, ez, ey, vz = single(0.7)
    @test ex ≈ sin(0.7) && ez ≈ cos(0.7) && ey ≈ 0 && vz ≈ 1 - cos(0.7)^2

    @qnode dev function ghz()
        q = qubits(3)
        q[1] = Hadamard(q[1])
        q[1], q[2] = CNOT(q[1], q[2])
        q[2], q[3] = CNOT(q[2], q[3])
        return probs(), state()
    end
    p, ψ = ghz()
    @test p ≈ [0.5, 0, 0, 0, 0, 0, 0, 0.5]
    @test ψ ≈ [1 / √2, 0, 0, 0, 0, 0, 0, 1 / √2]

    # controlled gates and control values against dense matrices
    @qnode dev function ctrltest(θ)
        a, b = qubits(2)
        a = Hadamard(a)
        a, b = ctrl(RX(θ), a; values=[false])(b)
        return state()
    end
    ψ = ctrltest(0.9)
    U = kron([1 0; 0 0], matrix(RX(0.9))) + kron([0 0; 0 1], I(2))     # controlled on |0⟩
    @test ψ ≈ U * kron(matrix(Hadamard()), I(2)) * [1, 0, 0, 0]

    @qnode dev function adjtest(θ)
        a = qubits(1)[1]
        a = RX(θ)'(a)
        return expval(Y(a))
    end
    @test adjtest(0.3) ≈ sin(0.3)             # RX(θ)|0⟩ has ⟨Y⟩ = -sin θ; the adjoint flips it

    # MultiRZ and PauliRot vs matrices, on 3 qubits
    @qnode dev function mrz(θ)
        q = qubits(3)
        q[1] = Hadamard(q[1]); q[2] = Hadamard(q[2]); q[3] = Hadamard(q[3])
        q[1], q[3] = MultiRZ(θ, q[1], q[3])
        q[2], q[3] = PauliRot(θ / 2, "XZ", q[2], q[3])
        return state()
    end
    ψ = mrz(0.6)
    plus = fill(1 / √8, 8)
    U1 = PennyLane.apply_matrix!(copy(plus) .+ 0im, matrix(MultiRZ(0.6), 2), [1, 3], 3)
    U2 = PennyLane.apply_matrix!(U1, matrix(PauliRot(0.3, "XZ")), [2, 3], 3)
    @test ψ ≈ U2

    @qnode dev function refq(θ, ϕ)
        reference(θ, ϕ)
    end
    e, v, p = refq(REF_ARGS...)
    @test e ≈ REF_EXPVAL atol = 1e-7
    @test v ≈ REF_VAR atol = 1e-7
    @test p ≈ REF_PROBS atol = 1e-7

    # exp(-i t P) as a gate
    @qnode dev function evolve(t)
        a, b = qubits(2)
        a = Hadamard(a)
        a, b = exp(-im * 0.35 * Z(1) * Z(2))(a, b)
        return expval(X(a))
    end
    @test evolve(0.0) ≈ cos(0.7)

    # finite-difference gradient of a vector-parameter circuit
    @qnode dev function rnd(θ)
        random_circuit(θ)[1]
    end
    θ = collect(range(0.1, 1.9; length=RANDOM_NPARAMS))
    g = gradient(rnd, θ)
    @test length(g) == RANDOM_NPARAMS
end
