# PennyLane (Python) bridge — the implementation lives in ext/PennyLanePythonCallExt.jl and is
# activated by `using PythonCall`. This file holds the types, the user-facing stubs and the
# Python environment configuration so the core package never depends on Python.

const PYTHONCALL_UUID = Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d")
const CONDAPKG_UUID = Base.UUID("992eb4ea-22a4-4c89-a5bb-47a3300528ab")

"""Path of the Python interpreter in the uv-managed environment (see python/README.md)."""
python_exe() = Sys.iswindows() ? joinpath(default_venv(), "Scripts", "python.exe") : joinpath(default_venv(), "bin", "python")

"""
    setup_python!(; exe=python_exe())

Configure PythonCall (via Preferences in the active project) to use the uv-managed Python
environment and disable CondaPkg. Run once per project, then `using PythonCall`.
"""
function setup_python!(; exe::AbstractString=python_exe())
    isfile(exe) || error("Python not found at $exe. Create the environment first:\n    uv sync --project $(joinpath(pkgdir(@__MODULE__), "python"))")
    Preferences.set_preferences!(PYTHONCALL_UUID, "exe" => String(exe); force=true)
    Preferences.set_preferences!(CONDAPKG_UUID, "backend" => "Null"; force=true)
    @info "PythonCall will use $exe (CondaPkg disabled). Add PythonCall to your project and run `using PythonCall`."
    String(exe)
end

"""
    PyDevice(name; wires=nothing, shots=0, kwargs...)

Any PennyLane device, including hardware plugins, driven through PythonCall. Programs are
converted to a PennyLane tape and run with `qml.execute`, so this is the bridge to real quantum
computers, e.g. `PyDevice("qiskit.remote"; wires=5, shots=1000, backend=...)`. Requires
`using PythonCall` (see `setup_python!`). The Python device is created on first use with the
extra keyword arguments, one per qubit count unless `wires` is given (hardware devices usually
need `wires` and `shots`).
"""
struct PyDevice <: AbstractDevice
    name::String
    nwires::Union{Nothing,Int}
    shots::Int
    kwargs::Dict{Symbol,Any}
    handle::Base.RefValue{Any}
end
PyDevice(name::AbstractString; wires::Union{Nothing,Integer}=nothing, shots::Integer=0, kwargs...) =
    PyDevice(String(name), wires === nothing ? nothing : Int(wires), Int(shots), Dict{Symbol,Any}(kwargs), Ref{Any}(nothing))
Base.show(io::IO, d::PyDevice) = print(io, "PyDevice(\"", d.name, "\"", d.nwires === nothing ? "" : ", wires=$(d.nwires)",
                                       d.shots == 0 ? "" : ", shots=$(d.shots)", ")")

const _NEEDS_PYTHONCALL = "this needs PennyLane's Python bridge: run PennyLane.setup_python!() once, add PythonCall to your project and `using PythonCall`"
execute(dev::PyDevice, prog::Program, args::Vector{Any}) = _py_execute(dev, prog, args)
_py_execute(dev, prog, args) = error(_NEEDS_PYTHONCALL)

"""
    molecular_hamiltonian(symbols, coordinates; charge=0, mult=1, basis="sto-3g", unit="bohr", method="dhf") -> (PauliSum, nqubits)

Molecular Hamiltonian from PennyLane's `qchem` (needs `using PythonCall`), wires 1-based.
"""
molecular_hamiltonian(args...; kwargs...) = error(_NEEDS_PYTHONCALL)
"""`from_pennylane(op)`: convert a PennyLane observable to a `PauliSum` (needs `using PythonCall`)."""
from_pennylane(args...; kwargs...) = error(_NEEDS_PYTHONCALL)
"""`to_pennylane(qn, args...)`: the PennyLane tape (`QuantumScript`) of a QNode's program (needs `using PythonCall`)."""
to_pennylane(args...; kwargs...) = error(_NEEDS_PYTHONCALL)
"""`draw(qn, args...)`: text drawing of the circuit via PennyLane's drawer (needs `using PythonCall`)."""
draw(args...; kwargs...) = error(_NEEDS_PYTHONCALL)
