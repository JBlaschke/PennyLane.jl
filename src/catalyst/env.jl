# Locating the Catalyst binaries (CLI, runtime libraries, Lightning device plugin).
#
# They ship inside the `pennylane-catalyst` and `pennylane-lightning` wheels. We look for a uv
# environment (see python/README.md) and never start a Python interpreter.

struct CatalystEnv
    venv::String
    sitepackages::String
    cli::String
    libdir::String
    utilsdir::String
    lightning_plugin::String
end
Base.show(io::IO, e::CatalystEnv) = print(io, "CatalystEnv(", e.venv, ")")

const _CATALYST_ENV = Ref{Union{Nothing,CatalystEnv}}(nothing)

function default_venv()
    haskey(ENV, "PENNYLANE_JL_PYTHON") && return dirname(dirname(ENV["PENNYLANE_JL_PYTHON"]))
    haskey(ENV, "PENNYLANE_JL_VENV") && return ENV["PENNYLANE_JL_VENV"]
    normpath(joinpath(pkgdir(@__MODULE__), "python", ".venv"))
end

function find_sitepackages(venv::AbstractString)
    libdir = joinpath(venv, "lib")
    if isdir(libdir)
        for d in sort(readdir(libdir); rev=true)
            startswith(d, "python") || continue
            sp = joinpath(libdir, d, "site-packages")
            isdir(sp) && return sp
        end
    end
    sp = joinpath(venv, "Lib", "site-packages")     # Windows layout
    isdir(sp) ? sp : nothing
end

"""
    catalyst_env(; refresh=false) -> CatalystEnv

The Catalyst installation used by `LightningDevice` and `CatalystDevice`. Resolution order:
`PENNYLANE_JL_PYTHON`, `PENNYLANE_JL_VENV`, then `python/.venv` inside this package.
"""
function catalyst_env(; refresh::Bool=false)
    refresh || _CATALYST_ENV[] === nothing || return _CATALYST_ENV[]
    venv = default_venv()
    help = """
        PennyLane.jl could not find the Catalyst binaries (looked in $venv).
        Create the pinned Python environment with uv (no Conda needed):
            uv sync --project $(joinpath(pkgdir(@__MODULE__), "python"))
        or set PENNYLANE_JL_VENV (or PENNYLANE_JL_PYTHON) to an environment with pennylane-catalyst installed."""
    sp = find_sitepackages(venv)
    sp === nothing && error(help)
    cli = joinpath(venv, "bin", "catalyst")
    libdir = joinpath(sp, "catalyst", "lib")
    utils = joinpath(sp, "catalyst", "utils")
    plugin = joinpath(sp, "pennylane_lightning", "liblightning_qubit_catalyst." * Libdl.dlext)
    for (what, path) in (("catalyst CLI", cli), ("runtime library directory", libdir), ("Lightning device plugin", plugin))
        ispath(path) || error(help * "\n(missing $what: $path)")
    end
    _CATALYST_ENV[] = CatalystEnv(String(venv), sp, cli, libdir, utils, plugin)
end

"""`has_catalyst()`: whether the Catalyst binaries are available."""
has_catalyst() = (try catalyst_env(); true catch; false end)
