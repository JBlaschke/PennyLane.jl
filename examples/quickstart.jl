# Run with:  julia --project examples/quickstart.jl
using PennyLane, Printf

H = 0.5 * Z(1) * Z(2) + 0.3 * X(1)                 # operators are Julia values
println("H  = ", H)
println("H² = ", H * H)
println("[Z(1)Z(2), X(1)] = ", commutator(Z(1) * Z(2), X(1)))

function bell_body(θ)
    a, b = qubits(2)                                # qubits are values
    a    = RX(θ, a)                                 # gates consume and return them
    a, b = CNOT(a, b)
    return expval(Z(b))
end

for dev in (StateVector(), LightningDevice(), CatalystDevice())
    has_catalyst() || dev isa StateVector || continue
    qn = QNode(bell_body, dev; name=:bell)
    t = @elapsed v = qn(0.3)                        # first call traces (and compiles on CatalystDevice)
    g = gradient(qn, 0.3)
    @printf("%-18s expval = %.12f  grad = %.12f   (first call %.2f s)\n", string(dev), v, g, t)
end

@qnode StateVector() function bell(θ)
    bell_body(θ)
end
println("\nProgram IR:\n", program(bell, 0.3))
println("\nCatalyst MLIR:\n", mlir(bell, 0.3; grad=true))
