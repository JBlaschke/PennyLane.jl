# Python environment for PennyLane.jl

PennyLane.jl does not need Python to load or to run its own simulator. Python provides two optional things:

1. The **Catalyst compiler CLI, runtime libraries and Lightning device plugin**, which are shipped as binaries inside
   the `pennylane-catalyst` and `pennylane-lightning` wheels. `LightningDevice` and `CatalystDevice` need them.
2. **PennyLane itself**, used as a reference implementation in the test suite.

The environment is managed with [uv](https://docs.astral.sh/uv/) and pinned by `uv.lock`. No Conda.

```bash
uv sync --project python
```

creates `python/.venv`. PennyLane.jl finds it automatically; to use another interpreter set
`PENNYLANE_JL_PYTHON=/path/to/python`.
