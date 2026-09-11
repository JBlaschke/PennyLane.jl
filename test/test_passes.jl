@testset "T5: cancel inverses and merge rotations" begin
    dev = StateVector()
    function redundant(θ)
        q = qubits(3)
        q[1] = Hadamard(q[1]); q[1] = Hadamard(q[1])                 # cancels
        q[2] = RX(θ[1], q[2]); q[2] = RX(θ[2], q[2])                 # merges
        q[1], q[2] = CNOT(q[1], q[2]); q[1], q[2] = CNOT(q[1], q[2]) # cancels
        q[3] = T(q[3]); q[3] = T()'(q[3])                            # cancels (adjoint pair)
        q[2] = RZ(0.3, q[2]); q[2] = RZ(0.3)'(q[2])                  # cancels
        q[1], q[3] = CRX(θ[3], q[1], q[3]); q[1], q[3] = CRX(0.2, q[1], q[3])   # merges
        q[2] = RY(θ[1], q[2])                                        # stays (different axis)
        q[1] = PauliX(q[1]); q[2] = PauliX(q[2]); q[1] = PauliX(q[1])            # X on q1 cancels, X on q2 stays
        return expval(0.5 * Z(q[1]) * Z(q[2]) + X(q[3]))
    end
    @qnode dev function plain(θ); redundant(θ); end
    @qnode dev passes = [:cancel_inverses, :merge_rotations] function opt(θ); redundant(θ); end
    θ = [0.4, 0.9, 1.3]
    @test plain(θ) ≈ opt(θ) atol = 1e-12
    p0, p1 = program(plain, θ), program(opt, θ)
    @test gate_count(p0) == 16
    @test gate_count(p1) == 4                                        # RX, CRX, RY, X
    @test occursin("RX(add(arg1[1], arg1[2]))", string(p1))
    @test gradient(plain, θ) ≈ gradient(opt, θ) atol = 1e-9
    @test gate_count(optimize(p0; passes=[:cancel_inverses])) == 6
    @test gate_count(optimize(p0; passes=[:merge_rotations])) == 14

    # nothing to do on an already minimal circuit; passes inside regions
    @qnode dev function bell(θ); a, b = qubits(2); a = RX(θ, a); a, b = CNOT(a, b); return expval(Z(b)); end
    @test gate_count(optimize(program(bell, 0.3))) == 2
    @qnode dev function inloop(θ)
        q = qubits(2)
        @trace for i in 1:3
            q[1] = RX(θ[i], q[1]); q[1] = RX(θ[i], q[1])
            q[2] = Hadamard(q[2]); q[2] = Hadamard(q[2])
        end
        return expval(Z(q[1]) * Z(q[2]))
    end
    pl = program(inloop, θ)
    po = optimize(pl)
    @test gate_count(pl) == 4 && gate_count(po) == 1
    @test execute(dev, po, Any[θ]) ≈ inloop(θ) atol = 1e-12

    # Catalyst's passes agree on the gate count of the same program
    if has_catalyst()
        env = catalyst_env()
        dir = mktempdir()
        src = mlir(plain, θ)
        write(joinpath(dir, "in.mlir"), src)
        out = read(pipeline(`$(env.cli) --tool=opt --catalyst-pipeline=pipe\(cancel-inverses\;merge-rotations\) $(joinpath(dir, "in.mlir"))`; stderr=devnull), String)
        ngates = count(l -> occursin("quantum.custom", l) || occursin("quantum.multirz", l), split(out, '\n'))
        @test ngates == gate_count(p1)
    end
end
