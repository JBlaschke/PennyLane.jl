@testset "operator algebra" begin
    @test X(1) * X(1) == Identity()
    @test X(1) * Y(1) == im * Z(1)
    @test Y(1) * X(1) == -im * Z(1)
    @test Y(1) * Z(1) == im * X(1)
    @test Z(1) * X(1) == im * Y(1)
    @test commutator(X(1), Y(1)) == 2im * Z(1)
    @test iszero(commutator(X(1), Z(2)))
    @test iszero(anticommutator(X(1), Y(1)))
    @test matrix(X(1) * Z(2)) == kron(σ[:X], σ[:Z])
    @test matrix(Z(2), 2) == kron(σ[:I], σ[:Z])
    @test matrix(X(1) ⊗ Z(2)) == matrix(X(1) * Z(2))
    @test_throws ArgumentError X(1) ⊗ Z(1)

    H = 0.5 * Z(1) * Z(2) + 0.3 * X(1) - 0.2 * Y(2)
    @test H isa PauliSum
    @test length(terms(H)) == 3
    @test wires(H) == [1, 2]
    @test matrix(H) ≈ 0.5 * kron(σ[:Z], σ[:Z]) + 0.3 * kron(σ[:X], σ[:I]) - 0.2 * kron(σ[:I], σ[:Y])
    @test H' == H
    @test iszero(H - H)
    @test (H + 1.0) - 1.0 ≈ H
    @test matrix(H * H) ≈ matrix(H)^2
    @test string(H) == "0.5 * Z(1) * Z(2) + 0.3 * X(1) - 0.2 * Y(2)"
    @test string(2 * X(1)) == "2 * X(1)"
    @test PauliSum([X(1), X(1)]) == 2 * X(1)

    U = exp(-im * 0.4 * X(1) * Y(2))
    @test matrix(U) ≈ exp(-im * 0.4 * kron(σ[:X], σ[:Y]))
    @test_throws ArgumentError exp(0.4 * X(1))

    @test matrix(RX(0.3)) ≈ exp(-im * 0.3 / 2 * σ[:X])
    @test matrix(RX(0.3)') ≈ matrix(RX(-0.3))
    @test matrix(CNOT()) == ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
    @test matrix(PauliRot(0.7, "ZZ")) ≈ matrix(IsingZZ(0.7))
    @test matrix(MultiRZ(0.7), 2) ≈ matrix(IsingZZ(0.7))
    @test matrix(Rot(0.1, 0.2, 0.3)) ≈ matrix(RZ(0.3)) * matrix(RY(0.2)) * matrix(RZ(0.1))
    for (name, def) in PennyLane.GATES
        def.nqubits == -1 && continue
        M = PennyLane.gate_matrix(name, fill(0.37, def.nparams), def.nqubits)
        @test M * M' ≈ I atol = 1e-12
    end
end
