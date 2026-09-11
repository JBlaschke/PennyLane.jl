# Small built-in datasets so test problems run without Python.

"""
    h2_hamiltonian() -> (H::PauliSum, nqubits)

Electronic Hamiltonian of H₂ in the STO-3G basis at bond length 1.3228 Bohr (0.7 Å), in the
Jordan–Wigner qubit representation on 4 qubits (15 Pauli terms, wires 1–4). Generated with
PennyLane 0.45.1: `qml.qchem.molecular_hamiltonian(Molecule(["H","H"], [0,0,-0.6614, 0,0,0.6614]); method="dhf")`.
The Hartree–Fock state is |1100⟩; the exact ground-state energy is `H2_GROUND_ENERGY`.
"""
function h2_hamiltonian()
    c1, c2, c3, c4, c5, c6, c7 = 0.1777135822909175, 0.17059759276836803, -0.2427450126094143, 0.12293330449299361,
                                 0.16768338855601356, 0.044750084063019925, 0.1762766139418181
    H = -0.04207255194743914 * Identity() +
        c1 * Z(1) + c1 * Z(2) + c2 * Z(1) * Z(2) +
        c3 * Z(3) + c3 * Z(4) +
        c4 * Z(1) * Z(3) + c5 * Z(2) * Z(3) + c5 * Z(1) * Z(4) + c4 * Z(2) * Z(4) + c7 * Z(3) * Z(4) +
        c6 * Y(1) * X(2) * X(3) * Y(4) - c6 * Y(1) * Y(2) * X(3) * X(4) -
        c6 * X(1) * X(2) * Y(3) * Y(4) + c6 * X(1) * Y(2) * Y(3) * X(4)
    (H, 4)
end

"""Exact (full CI, STO-3G) ground-state energy of `h2_hamiltonian()` in Hartree."""
const H2_GROUND_ENERGY = -1.1361891625218796
