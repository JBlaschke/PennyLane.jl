# Mid-circuit measurements and traced control flow.  Run with:  julia --project examples/control_flow.jl
using PennyLane, Printf

# Teleport RX(θ)|0⟩ from wire 1 to wire 3. The corrections branch on measurement results, so
# `@trace if` keeps them in the program instead of unrolling; every trajectory yields the same state.
function teleport(θ)
    a, b, c = qubits(3)
    a = RX(θ, a)
    b = Hadamard(b); b, c = CNOT(b, c)          # Bell pair
    a, b = CNOT(a, b); a = Hadamard(a)
    m1, a = measure(a)
    m2, b = measure(b)
    @trace if m2
        c = PauliX(c)
    end
    @trace if m1
        c = PauliZ(c)
    end
    return expval(Z(c)), expval(Y(c))
end

# Repeat until success: measure |+⟩ until the outcome is 1.
function rus()
    q = qubits(1)[1]
    done = false
    @trace while !done
        q = Hadamard(q)
        done, q = measure(q)
    end
    return expval(Z(q))
end

# QAOA layers as a traced loop: γ[l] is read with a dynamic index inside scf.for.
edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
C = sum(0.5 * (1 - Z(i) * Z(j)) for (i, j) in edges)
B = sum(X(i) for i in 1:4)
function qaoa(γ, β)
    q = qubits(4)
    for i in 1:4
        q[i] = Hadamard(q[i])
    end
    @trace for l in 1:length(γ)
        q = collect(evolve(C, γ[l], q...))      # exp(-iγ C): exact, the ZZ terms commute
        q = collect(evolve(B, β[l], q...))
    end
    return expval(C)
end

devices = Any[StateVector()]
has_catalyst() && push!(devices, LightningDevice(), CatalystDevice())
for dev in devices
    tp = QNode(teleport, dev; name=:teleport)
    r = QNode(rus, dev; name=:rus)
    qa = QNode(qaoa, dev; name=:qaoa)
    ez, ey = tp(0.7)
    @printf("%-20s teleport: <Z> = %+.6f (cos θ = %+.6f)  <Y> = %+.6f   RUS: <Z> = %+.1f   QAOA(p=2): <C> = %.6f\n",
            string(dev), ez, cos(0.7), ey, r(), qa([0.4, 0.9], [0.3, 0.6]))
end

println("\nProgram IR of the teleportation circuit:")
println(program(QNode(teleport, StateVector(); name=:teleport), 0.7))
