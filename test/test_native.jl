@testset "native code generation (JITDevice)" begin
    dev = StateVector()
    @qnode dev function bell_s(θ); a, b = qubits(2); a = RX(θ, a); a, b = CNOT(a, b); return expval(Z(b)); end
    ex = native_expr(program(bell_s, 0.3))
    src = string(ex)
    @test occursin("__catalyst__qis__RX", src) && occursin("__catalyst__qis__CNOT", src) && occursin("__catalyst__qis__Expval", src)
    @qnode dev function withvar(θ); a = qubits(1)[1]; a = RX(θ, a); return var(Z(a)); end
    @test_throws ArgumentError native_expr(program(withvar, 0.3))
    @qnode dev function tp(θ); teleport(θ)[1]; end
    @test_throws ArgumentError native_expr(program(tp, 0.3))

    if has_catalyst()
        jit = JITDevice()
        @qnode jit function bell(θ); a, b = qubits(2); a = RX(θ, a); a, b = CNOT(a, b); return expval(Z(b)); end
        @test bell(0.3) ≈ cos(0.3) atol = 1e-12
        @qnode jit function allg(θ); all_gates_circuit(θ); end
        @qnode dev function allg_s(θ); all_gates_circuit(θ); end
        θ = collect(range(0.15, 2.9; length=ALL_GATES_NPARAMS))
        @test allg(θ) ≈ allg_s(θ) atol = 1e-10                     # adjoint and controlled gates via Modifiers
        @qnode jit function e(θ); h2_energy(θ); end
        @test e(0.3) ≈ -1.132688823446468 atol = 1e-9
        @qnode jit function rnd(θ); random_circuit(θ)[1]; end
        @qnode dev function rnd_s(θ); random_circuit(θ)[1]; end
        θr = 2π .* rand(MersenneTwister(11), RANDOM_NPARAMS)
        @test rnd(θr) ≈ rnd_s(θr) atol = 1e-10
        @qnode jit function two(θ, ϕ); a, b = qubits(2); a = RY(θ, a); a, b = CRX(ϕ, a, b); return expval(Z(a)), expval(X(b) * Z(a)); end
        @qnode dev function two_s(θ, ϕ); a, b = qubits(2); a = RY(θ, a); a, b = CRX(ϕ, a, b); return expval(Z(a)), expval(X(b) * Z(a)); end
        r, rs = two(0.4, 1.1), two_s(0.4, 1.1)
        @test r[1] ≈ rs[1] atol = 1e-12
        @test r[2] ≈ rs[2] atol = 1e-12
        @test gradient(bell, 0.3; method=:parameter_shift) ≈ -sin(0.3) atol = 1e-12
    end
end

if Base.find_package("GPUCompiler") === nothing || !has_catalyst()
    @warn "skipping T6 (compile_native) tests: need GPUCompiler and the Catalyst binaries"
else
    using GPUCompiler
    @testset "T6: Julia -> native library with GPUCompiler" begin
        dev = StateVector()
        @qnode dev function bell(θ); a, b = qubits(2); a = RX(θ, a); a, b = CNOT(a, b); return expval(Z(b)); end
        ir = native_llvm_ir(bell, 0.3)
        @test occursin("call void @__catalyst__qis__RX", ir) || occursin("@__catalyst__qis__RX(", ir)
        @test occursin("__catalyst__qis__CNOT", ir) && occursin("__catalyst__qis__Expval", ir)
        @test !occursin("jl_gc", ir) && !occursin("ijl_", ir)             # no Julia runtime in the program
        lib = compile_native(bell, 0.3)
        @test isfile(lib.path)
        @test lib(0.3) ≈ cos(0.3) atol = 1e-12
        @test lib(1.1) ≈ cos(1.1) atol = 1e-12
        @qnode dev function e(θ); h2_energy(θ); end
        elib = compile_native(e, 0.3)
        @test elib(0.3) ≈ -1.132688823446468 atol = 1e-9
        @qnode dev function rnd(θ); random_circuit(θ)[1]; end
        θr = 2π .* rand(MersenneTwister(12), RANDOM_NPARAMS)
        rlib = compile_native(rnd, θr)
        @test rlib(θr) ≈ rnd(θr) atol = 1e-10
        @test_throws ArgumentError rlib(0.3)
        @qnode dev function adj(θ); a = qubits(1)[1]; a = RX(θ)'(a); return expval(Y(a)); end
        @test_throws ArgumentError compile_native(adj, 0.3)
        # only the runtime and libm are needed by the object
        syms = read(`nm -u $(lib.path)`, String)
        @test occursin("__catalyst__qis__RX", syms) && !occursin("jl_", syms)
    end
end
