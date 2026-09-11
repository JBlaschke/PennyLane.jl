@testset "tracing and value semantics" begin
    dev = StateVector()

    @qnode dev function bell(θ)
        a, b = qubits(2)
        a = RX(θ, a)
        a, b = CNOT(a, b)
        return expval(Z(b))
    end
    prog = program(bell, 0.3)
    @test prog.nqubits == 2
    @test count(n -> n isa PennyLane.GateNode, prog.nodes) == 2
    @test prog.scalar_return
    @test prog.diff_method == "adjoint"
    @test occursin("RX(arg1)", string(prog))

    # reusing a consumed qubit is an error
    @qnode dev function reuse(θ)
        a, b = qubits(2)
        RX(θ, a)
        a, b = CNOT(a, b)          # `a` was consumed by RX
        return expval(Z(b))
    end
    @test_throws QubitConsumedError reuse(0.1)

    # observables built from stale qubit values are rejected
    @qnode dev function stale(θ)
        a = qubits(1)[1]
        obs = Z(a)
        a = RX(θ, a)
        return expval(obs)
    end
    @test_throws QubitConsumedError stale(0.1)

    # wire-indexed (PennyLane) style and mixing
    @qnode StateVector(2) function wired(θ)
        RX(θ, 1)
        CNOT(1, 2)
        return expval(Z(2))
    end
    @test wired(0.3) ≈ cos(0.3)
    @test wired(0.3) ≈ bell(0.3)

    # gates after a terminal measurement are an error
    @qnode dev function late(θ)
        a = qubits(1)[1]
        e = expval(Z(a))
        PauliX(a)
        return e
    end
    @test_throws TraceError late(0.1)

    # traced arithmetic and vector arguments
    @qnode dev function arith(θ, ϕ)
        a = qubits(1)[1]
        a = RX(2θ[1] + sin(θ[2]) - ϕ / 2, a)
        return expval(Z(a))
    end
    θ = [0.1, 0.2]; ϕ = 0.5
    @test arith(θ, ϕ) ≈ cos(2θ[1] + sin(θ[2]) - ϕ / 2)
    @test_throws ArgumentError (@qnode dev function branch(θ)
        a = qubits(1)[1]
        θ > 0 && (a = PauliX(a))
        return expval(Z(a))
    end)(0.3)

    # the Pauli operators are not gates
    @qnode dev function wrongx(θ)
        a = qubits(1)[1]
        a = X(a)
        return expval(Z(a))
    end
    @test_throws ArgumentError wrongx(0.3)

    # outside a qnode
    @test_throws TraceError qubits(2)

    # MLIR text
    src = mlir(bell, 0.3)
    @test occursin("quantum.custom \"RX\"", src)
    @test occursin("quantum.custom \"CNOT\"", src)
    @test occursin("quantum.namedobs", src)
    @test occursin("llvm.emit_c_interface", src)
    @test occursin("gradient.grad", mlir(bell, 0.3; grad=true))
end
