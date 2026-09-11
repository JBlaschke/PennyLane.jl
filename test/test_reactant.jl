if Base.find_package("Reactant") === nothing || !has_catalyst()
    @warn "skipping Reactant MLIR-pass tests: need Reactant and the Catalyst binaries"
else
    using Reactant
    @testset "Julia MLIR pass on Catalyst IR via Reactant" begin
        dev = StateVector()
        function redundant(θ)
            q = qubits(2)
            q[1] = Hadamard(q[1]); q[1] = Hadamard(q[1])
            q[1] = RX(θ, q[1]); q[1] = RX(θ)'(q[1])
            q[1], q[2] = CNOT(q[1], q[2]); q[1], q[2] = CNOT(q[1], q[2])
            q[2] = RY(θ, q[2])
            q[1] = T(q[1]); q[1] = T()'(q[1])
            return expval(Z(q[1]) * Z(q[2]))
        end
        @qnode dev function plain(θ); redundant(θ); end
        src = mlir(plain, 0.4)
        ngates(text) = count(l -> occursin("quantum.custom", l), split(text, '\n'))
        @test ngates(src) == 9
        out = mlir_pass(src, :cancel_inverses)
        @test ngates(out) == 1
        @test occursin("\"RY\"", out)
        @test_throws ArgumentError mlir_pass(src, :nonsense)
        # the rewritten (generic-form) module still compiles and runs through Catalyst
        cdev = CatalystDevice(mlir_transform=s -> mlir_pass(s, :cancel_inverses))
        @qnode cdev function opt(θ); redundant(θ); end
        @test opt(0.4) ≈ plain(0.4) atol = 1e-12
        @test opt(1.3) ≈ plain(1.3) atol = 1e-12
        @test gradient(opt, 0.4) ≈ gradient(plain, 0.4) atol = 1e-9
    end
end
