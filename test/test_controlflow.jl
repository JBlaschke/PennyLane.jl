@testset "control flow and mid-circuit measurement" begin
    dev = StateVector(rng=MersenneTwister(11))

    # T3: teleportation gives the right state on every trajectory
    @qnode dev function tp(θ)
        teleport(θ)
    end
    for θ in (0.3, 1.1, 2.5), _ in 1:4
        ez, ey = tp(θ)
        @test ez ≈ cos(θ) atol = 1e-12
        @test ey ≈ -sin(θ) atol = 1e-12
    end
    prog = program(tp, 0.3)
    @test PennyLane.has_mcm(prog) && PennyLane.has_control_flow(prog)
    @test count(n -> n isa PennyLane.IfNode, prog.nodes) == 2
    @test occursin("if %v", string(prog))
    src = mlir(tp, 0.3)
    @test occursin("quantum.measure", src) && occursin("scf.if", src) && occursin("scf.yield", src)

    # T3: repeat until success
    @qnode dev function r()
        rus()
    end
    for _ in 1:5
        @test r() ≈ -1 atol = 1e-12
    end
    @test occursin("scf.while", mlir(r))

    # if/else with classical carried values and wire-indexed gates inside
    @qnode StateVector(2) function ifelse_(θ)
        m, _ = measure(1; postselect=1)         # |0⟩ postselected on 1 is impossible → error below; use H first
        return expval(Z(2))
    end
    @test_throws ArgumentError ifelse_(0.1)
    @qnode StateVector(2; rng=MersenneTwister(1)) function branches(θ)
        Hadamard(1)
        m, _ = measure(1)
        s = 0.0
        @trace if m
            RX(θ, 2)
            s = s + 1.0
        else
            RY(θ, 2)
            s = s - 1.0
        end
        RZ(s * 0.0, 2)                          # traced classical result used as a parameter
        return expval(Z(2))
    end
    for _ in 1:4
        @test branches(0.7) ≈ cos(0.7) atol = 1e-12
    end

    # postselect and reset
    @qnode StateVector() function ps()
        q = qubits(1)[1]
        q = Hadamard(q)
        m, q = measure(q; postselect=1)
        return expval(Z(q))
    end
    @test ps() ≈ -1
    @qnode StateVector(rng=MersenneTwister(2)) function rs()
        q = qubits(1)[1]
        q = Hadamard(q)
        m, q = measure(q; reset=true)
        return expval(Z(q))
    end
    @test rs() ≈ 1

    # traced for loop with dynamic parameter indexing equals the unrolled circuit
    @qnode StateVector() function loop(θ)
        q = qubits(2)
        @trace for i in 1:3
            q[1] = RY(θ[i], q[1])
            q[1], q[2] = CNOT(q[1], q[2])
        end
        return expval(Z(q[2])), probs()
    end
    @qnode StateVector() function unrolled(θ)
        q = qubits(2)
        for i in 1:3
            q[1] = RY(θ[i], q[1])
            q[1], q[2] = CNOT(q[1], q[2])
        end
        return expval(Z(q[2])), probs()
    end
    θ = [0.3, 0.8, 1.7]
    e1, p1 = loop(θ); e2, p2 = unrolled(θ)
    @test e1 ≈ e2 && p1 ≈ p2
    @test count(n -> n isa PennyLane.ForNode, program(loop, θ).nodes) == 1
    @test occursin("scf.for", mlir(loop, θ)) && occursin("tensor.extract %arg0[%", mlir(loop, θ))

    # loop-carried classical accumulator and a while loop with a counter
    @qnode StateVector() function acc(θ)
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
    @qnode StateVector() function acc_ref(θ)
        q = qubits(1)[1]
        q = RX(sum(θ) * 0.5, q)
        for _ in 1:3
            q = RY(θ[1], q)
        end
        return expval(Z(q))
    end
    θ4 = [0.2, 0.4, 0.6, 0.8]
    @test acc(θ4) ≈ acc_ref(θ4) atol = 1e-12
    @test occursin("scf.while", mlir(acc, θ4))

    # errors: terminal measurement inside a region, stale carried qubit, parameter shift on control flow
    @qnode StateVector() function bad1()
        q = qubits(1)[1]
        m, q = measure(q)
        @trace if m
            e = expval(Z(q))
        end
        return expval(Z(q))
    end
    @test_throws TraceError bad1()
    @test_throws ArgumentError gradient(tp, 0.3)
    @qnode dev function tpz(θ); teleport(θ)[1]; end
    @test_throws ArgumentError gradient(tpz, 0.3)
    @test gradient(tpz, 0.3; method=:finitediff) ≈ -sin(0.3) atol = 1e-5
    @qnode StateVector() function typechange()
        q = qubits(1)[1]
        m, q = measure(q)
        k = 0
        @trace if m
            k = 1.5
        end
        return expval(Z(q))
    end
    @test_throws TraceError typechange()
    @test_throws ArgumentError (@qnode StateVector() function dynwire(θ)
        q = qubits(2)
        @trace for i in 1:2
            q[i] = RX(θ, q[i])
        end
        return expval(Z(q[1]))
    end)(0.3)
end
