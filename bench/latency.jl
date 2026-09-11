# Per-call latency and simulator throughput.  Run with:  julia --project bench/latency.jl
using PennyLane, Printf

function layered(θ, n, layers)
    q = qubits(n)
    k = 1
    for _ in 1:layers
        for i in 1:n
            q[i] = RX(θ[k], q[i]); k += 1
            q[i] = RY(θ[k], q[i]); k += 1
        end
        for i in 1:n-1
            q[i], q[i+1] = CNOT(q[i], q[i+1])
        end
    end
    return expval(Z(q[1]))
end

println("per-call latency (2 qubits, 1 layer; includes device init/release per call)")
devs = Any[StateVector()]
has_catalyst() && push!(devs, LightningDevice(), CatalystDevice())
for dev in devs
    qn = QNode(θ -> layered(θ, 2, 1), dev; name=:lat)
    θ = rand(4)
    qn(θ)
    n = 500
    t = @elapsed for _ in 1:n; qn(θ); end
    @printf("  %-20s %8.1f µs / call\n", string(dev), 1e6 * t / n)
end

println("\nstate-vector throughput (RX,RY per qubit + CNOT ladder, 4 layers); Julia threads: $(Threads.nthreads())")
for n in (8, 12, 16, 20, 22, 24)
    for dev in devs
        dev isa CatalystDevice && continue
        qn = QNode(θ -> layered(θ, n, 4), dev; name=Symbol(:tp, n))
        θ = rand(8n)
        qn(θ)
        reps = n <= 12 ? 20 : n <= 20 ? 3 : 1
        t = @elapsed for _ in 1:reps; qn(θ); end
        gates = 4 * (2n + n - 1)
        @printf("  n=%2d %-20s %9.2f ms / circuit  (%6.2f µs / gate)\n", n, string(dev), 1e3 * t / reps, 1e6 * t / reps / gates)
    end
end
