# VQE for H₂ on every available backend.  Run with:  julia --project examples/vqe_h2.jl
using PennyLane, Printf

H, n = h2_hamiltonian()
println("H₂ Hamiltonian ($n qubits, $(length(terms(H))) terms):\n  ", H)
@printf("exact ground-state energy: %.10f Ha\n\n", H2_GROUND_ENERGY)

function ansatz(θ)
    q = qubits(4)
    q = collect(BasisState([1, 1, 0, 0], q...))        # Hartree–Fock reference |1100⟩
    q = collect(DoubleExcitation(θ, q...))             # UCCSD for H₂ is a single double excitation
    return expval(H)
end

function vqe(dev; iters=40, lr=0.4)
    energy = QNode(ansatz, dev; name=:h2)
    θ = 0.0
    for _ in 1:iters
        θ -= lr * gradient(energy, θ)                  # parameter shift; compiled adjoint on CatalystDevice
    end
    θ, energy(θ)
end

devices = Any[StateVector()]
has_catalyst() && push!(devices, LightningDevice(), CatalystDevice())
for dev in devices
    t = @elapsed (θ, E) = vqe(dev)
    @printf("%-20s θ* = %.6f   E = %.10f Ha   error = %.1e   (%.2f s)\n", string(dev), θ, E, E - H2_GROUND_ENERGY, t)
end

# shot-based estimation, as on hardware
dev = StateVector(shots=4000)
θ, E = vqe(dev; iters=30, lr=0.3)
@printf("%-20s θ* = %.6f   E = %.6f Ha (sampled)\n", string(dev), θ, E)
