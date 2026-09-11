# The three ways a traced program reaches the Catalyst runtime, and a Julia-written MLIR pass.
# Run with:  julia --project examples/compilers.jl        (GPUCompiler / Reactant parts need those packages)
using PennyLane, Printf

@qnode StateVector() function bell(θ)
    a, b = qubits(2)
    a = RX(θ, a)
    a = Hadamard(a); a = Hadamard(a)          # redundant pair, for the pass below
    a, b = CNOT(a, b)
    return expval(Z(b))
end

println("1. Julia code generated from the program (JITDevice runs this through Julia's JIT):\n")
println(native_expr(program(bell, 0.3)))

if has_catalyst()
    jit = QNode(bell.f, JITDevice(); name=:bell)
    @printf("\n   JITDevice: %.12f   (cos θ = %.12f)\n", jit(0.3), cos(0.3))

    if Base.find_package("GPUCompiler") !== nothing
        using GPUCompiler
        println("\n2. Same program compiled by Julia's compiler to a standalone shared library (no MLIR):\n")
        ir = native_llvm_ir(bell, 0.3)
        for l in split(ir, '\n')
            occursin("__catalyst__", l) && println("   ", strip(l))
        end
        lib = compile_native(bell, 0.3)
        @printf("\n   %s -> %.12f\n", lib, lib(0.3))
    end

    if Base.find_package("Reactant") !== nothing
        using Reactant
        println("\n3. Catalyst MLIR before/after a Julia-implemented cancel-inverses pass (via Reactant's MLIR):\n")
        src = mlir(bell, 0.3)
        out = mlir_pass(src, :cancel_inverses)
        ngates(t) = count(l -> occursin("quantum.custom", l), split(t, '\n'))
        @printf("   gates: %d -> %d\n", ngates(src), ngates(out))
        cdev = CatalystDevice(mlir_transform=s -> mlir_pass(s, :cancel_inverses))
        opt = QNode(bell.f, cdev; name=:bell)
        @printf("   compiled through Catalyst after the pass: %.12f\n", opt(0.3))
    end
end
