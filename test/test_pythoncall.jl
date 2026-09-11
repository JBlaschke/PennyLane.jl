const _py = PennyLane.python_exe()
if !isfile(_py) || Base.find_package("PythonCall") === nothing
    @warn "skipping PythonCall bridge tests (need python/.venv and the PythonCall package)"
else
    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
    ENV["JULIA_PYTHONCALL_EXE"] = _py
    using PythonCall
    @testset "PythonCall bridge (PennyLane as oracle)" begin
        dev = PyDevice("default.qubit")
        @qnode dev function refq(θ, ϕ)
            reference(θ, ϕ)
        end
        e, v, p = refq(REF_ARGS...)
        @test e ≈ REF_EXPVAL atol = 1e-7
        @test v ≈ REF_VAR atol = 1e-7
        @test p ≈ REF_PROBS atol = 1e-7

        @qnode dev function rnd_py(θ); random_circuit(θ); end
        @qnode StateVector() function rnd_s(θ); random_circuit(θ); end
        for seed in 1:3
            θ = 2π .* rand(MersenneTwister(seed), RANDOM_NPARAMS)
            a, b = rnd_py(θ), rnd_s(θ)
            @test a[1] ≈ b[1] atol = 1e-10
            @test a[2] ≈ b[2] atol = 1e-10
            @test a[3] ≈ b[3] atol = 1e-10
        end

        @qnode dev function allg(θ); all_gates_circuit(θ); end
        @qnode StateVector() function allg_s(θ); all_gates_circuit(θ); end
        θ = collect(range(0.15, 2.9; length=ALL_GATES_NPARAMS))
        @test allg(θ) ≈ allg_s(θ) atol = 1e-10
        @test gradient(allg, θ) ≈ gradient(allg_s, θ) atol = 1e-8       # parameter shift through PennyLane

        # qchem import matches the built-in fixture
        Hpy, n = molecular_hamiltonian(["H", "H"], [0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
        @test n == 4
        @test Hpy ≈ h2_hamiltonian()[1] atol = 1e-8

        @qnode dev function st(); q = qubits(2); q[1] = Hadamard(q[1]); q[1], q[2] = CNOT(q[1], q[2]); return state(); end
        @test st() ≈ [1 / √2, 0, 0, 1 / √2]

        shot_dev = PyDevice("default.qubit"; shots=500)
        @qnode shot_dev function smp(); q = qubits(2); q[1] = Hadamard(q[1]); q[1], q[2] = CNOT(q[1], q[2]); return sample(); end
        s = smp()
        @test size(s) == (500, 2) && all(s[:, 1] .== s[:, 2])

        @qnode dev function bell(θ); a, b = qubits(2); a = RX(θ, a); a, b = CNOT(a, b); return expval(Z(b)); end
        d = draw(bell, 0.3)
        @test occursin("RX", d) && occursin("<Z>", d)
        @test occursin("QuantumScript", string(pytype(to_pennylane(bell, 0.3))))
    end
end
