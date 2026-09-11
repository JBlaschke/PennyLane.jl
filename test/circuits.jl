# Circuits shared by several test files.

# Reference circuit; values from PennyLane 0.45.1 + Catalyst 0.15.0 (lightning.qubit), wires shifted by one.
function reference(θ, ϕ)
    a, b, c = qubits(3)
    a = RX(2θ[1] + sin(θ[2]), a)
    b = RY(θ[3])'(b)
    a, c = ctrl(RZ(ϕ), a; values=[false])(c)
    a, b = PauliRot(ϕ, "XY", a, b)
    b, c = MultiRZ(θ[1], b, c)
    a, c = CRX(ϕ, a, c)
    c = Hadamard(c)
    H = 0.5 * Z(a) * Z(b) + 0.3 * X(a) - 0.2 * Y(c)
    return expval(H), var(Z(b)), probs(a, b)
end
const REF_ARGS = ([0.1, 0.2, 0.3], 0.4)
const REF_EXPVAL = 0.45699726
const REF_VAR = 0.21003733
const REF_PROBS = [0.90451144, 0.0199035, 0.03988778, 0.03569728]

function random_circuit(θ)
    q = qubits(4)
    k = 1
    for layer in 1:3
        for i in 1:4
            q[i] = RX(θ[k], q[i]); k += 1
            q[i] = RY(θ[k], q[i]); k += 1
        end
        for i in 1:3
            q[i], q[i+1] = CNOT(q[i], q[i+1])
        end
        q[1], q[4] = CRZ(θ[k], q[1], q[4]); k += 1
        q[2], q[3] = IsingXX(θ[k], q[2], q[3]); k += 1
    end
    return expval(0.7 * Z(q[1]) * X(q[2]) - 0.4 * Y(q[3]) + 1.3 * Z(q[4])), probs(q[2], q[4]), var(X(q[1]))
end
const RANDOM_NPARAMS = 3 * (8 + 2)

# every parametrised gate family, with a scalar output (for gradient cross-checks)
function all_gates_circuit(θ)
    q = qubits(4)
    q[1] = RX(θ[1], q[1]); q[2] = RY(θ[2], q[2]); q[3] = RZ(θ[3], q[3]); q[4] = Hadamard(q[4])
    q[1] = PhaseShift(θ[4], q[1])
    q[2] = Rot(θ[5], θ[6], θ[7], q[2])
    q[1], q[2] = CRX(θ[8], q[1], q[2])
    q[2], q[3] = CRY(θ[9], q[2], q[3])
    q[3], q[4] = CRZ(θ[10], q[3], q[4])
    q[1], q[3] = ControlledPhaseShift(θ[11], q[1], q[3])
    q[2], q[4] = IsingXX(θ[12], q[2], q[4])
    q[1], q[2] = IsingYY(θ[13], q[1], q[2])
    q[3], q[4] = IsingZZ(θ[14], q[3], q[4])
    q[1], q[2], q[3] = MultiRZ(θ[15], q[1], q[2], q[3])
    q[1], q[2] = CRot(θ[16], θ[17], θ[18], q[1], q[2])
    q[3], q[4] = SingleExcitation(θ[19], q[3], q[4])
    q[1], q[2], q[3], q[4] = DoubleExcitation(θ[20], q[1], q[2], q[3], q[4])
    q[2], q[3] = ctrl(RX(θ[21]), q[2]; values=[false])(q[3])
    q[4] = RY(θ[22])'(q[4])
    q[1], q[4] = PauliRot(θ[23], "XY", q[1], q[4])
    q[2] = RX(2θ[1] + sin(θ[24]) - θ[3] / 3, q[2])           # shared parameters and traced arithmetic
    return expval(0.5 * Z(q[1]) * Z(q[2]) + 0.3 * X(q[3]) - 0.2 * Y(q[4]) + Z(q[1]) * X(q[4]))
end
const ALL_GATES_NPARAMS = 24

# T1: H₂ with the Hartree–Fock state and a double excitation (exact for this molecule)
const H2 = h2_hamiltonian()[1]
function h2_energy(θ)
    q = qubits(4)
    q = collect(BasisState([1, 1, 0, 0], q...))
    q = collect(DoubleExcitation(θ, q...))
    return expval(H2)
end

# hardware-efficient ansatz for H₂: RY layer + CNOT ladder, repeated
function h2_hea(θ)
    q = qubits(4)
    k = 1
    for layer in 1:2
        for i in 1:4
            q[i] = RY(θ[k], q[i]); k += 1
        end
        for i in 1:3
            q[i], q[i+1] = CNOT(q[i], q[i+1])
        end
    end
    return expval(H2)
end
