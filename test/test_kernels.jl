# slow reference for the kernels: explicit loop over all basis indices
function ref_apply(ψ, U, wires, n, cw=Int[], cv=Bool[])
    ψ = copy(ψ)
    k = length(wires); K = 1 << k
    tpos = [n - w for w in wires]
    tmask = sum(1 << p for p in tpos; init=0)
    cmask = sum(1 << (n - w) for w in cw; init=0)
    cval = sum(v ? 1 << (n - w) : 0 for (w, v) in zip(cw, cv); init=0)
    offs = [sum(((j >> (k - b)) & 1) << p for (b, p) in enumerate(tpos); init=0) for j in 0:K-1]
    for base in 0:(1<<n)-1
        (base & tmask) == 0 && (base & cmask) == cval || continue
        buf = [ψ[base+o+1] for o in offs]
        for j in 1:K
            ψ[base+offs[j]+1] = sum(U[j, l] * buf[l] for l in 1:K)
        end
    end
    ψ
end
randstate(rng, n) = (v = randn(rng, ComplexF64, 1 << n); v ./ norm(v))
randunitary(rng, d) = (A = randn(rng, ComplexF64, d, d); exp(im * Matrix(Hermitian(A + A'))))

@testset "state-vector kernels" begin
    rng = MersenneTwister(42)
    n = 6
    for k in 1:3, ctrl in (0, 1, 2), rep in 1:3
        wires = randperm(rng, n)[1:k+ctrl]
        targets, cw = wires[1:k], wires[k+1:end]
        cv = rand(rng, Bool, ctrl)
        U = randunitary(rng, 1 << k)
        ψ = randstate(rng, n)
        @test PennyLane.apply_matrix!(copy(ψ), U, targets, n, cw, cv) ≈ ref_apply(ψ, U, targets, n, cw, cv) atol = 1e-12
        @test PennyLane.apply_matrix!(copy(ψ), U, targets, n, cw, cv; threaded=true) ≈ ref_apply(ψ, U, targets, n, cw, cv) atol = 1e-12
        D = Diagonal(exp.(im .* randn(rng, 1 << k)))
        @test PennyLane.apply_matrix!(copy(ψ), D, targets, n, cw, cv) ≈ ref_apply(ψ, D, targets, n, cw, cv) atol = 1e-12
    end
    # single-pass Pauli expectation and application against dense matrices
    for rep in 1:5
        ψ = randstate(rng, n)
        word = [w => (:X, :Y, :Z)[rand(rng, 1:3)] for w in sort(randperm(rng, n)[1:rand(rng, 1:n)])]
        P = PauliString(1, word)
        M = matrix(P, n)
        @test PennyLane.pauli_expval(ψ, word, n) ≈ real(dot(ψ, M * ψ)) atol = 1e-12
        @test PennyLane.pauli_expval(ψ, word, n, true) ≈ real(dot(ψ, M * ψ)) atol = 1e-12
        @test PennyLane.apply_pauli(ψ, word, n) ≈ M * ψ atol = 1e-12
        H = 0.3 * P + 0.7 * Z(1) - 0.2 * X(2) * Y(3)
        @test PennyLane.apply_observable(ψ, H, n) ≈ matrix(H, n) * ψ atol = 1e-12
    end
    # threaded and serial simulators agree on a larger circuit
    @qnode StateVector(threads=true) function big_t(θ); random_circuit(θ); end
    @qnode StateVector(threads=false) function big_s(θ); random_circuit(θ); end
    θ = 2π .* rand(rng, RANDOM_NPARAMS)
    a, b = big_t(θ), big_s(θ)
    @test a[1] ≈ b[1] atol = 1e-12
    @test a[2] ≈ b[2] atol = 1e-12
    @test a[3] ≈ b[3] atol = 1e-12
    @qnode StateVector(threads=true) function wide(θ)
        q = qubits(15)
        for i in 1:15
            q[i] = RY(θ[i], q[i])
        end
        for i in 1:14
            q[i], q[i+1] = CNOT(q[i], q[i+1])
        end
        q[1], q[15] = CRZ(θ[1], q[1], q[15])
        return expval(Z(q[15]) * X(q[1])), probs(q[3], q[9])
    end
    @qnode StateVector(threads=false) function wide_s(θ)
        q = qubits(15)
        for i in 1:15
            q[i] = RY(θ[i], q[i])
        end
        for i in 1:14
            q[i], q[i+1] = CNOT(q[i], q[i+1])
        end
        q[1], q[15] = CRZ(θ[1], q[1], q[15])
        return expval(Z(q[15]) * X(q[1])), probs(q[3], q[9])
    end
    θ15 = rand(rng, 15)
    e1, p1 = wide(θ15); e2, p2 = wide_s(θ15)
    @test e1 ≈ e2 atol = 1e-12
    @test p1 ≈ p2 atol = 1e-12
end
