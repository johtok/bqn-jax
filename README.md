# bqn-jax

BQN-flavored array programming on top of JAX.

`bqn-jax` provides:
- An interpreter path for language/runtime semantics.
- A JAX-lowered compile path for numerical workloads.

Spec reference used by this project:
https://mlochbaum.github.io/BQN/spec/index.html

## Highlights

- BQN syntax support across parser/runtime with an extensive test suite.
- One-shot and stateful evaluation APIs.
- Compiled expressions with `.jit()`, `.grad()`, and `.vmap()` wrappers.
- Pixi-first developer workflow with reproducible environments and tasks.

## Getting Started

### 1. Install Pixi (recommended)

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Alternative install methods:
https://pixi.sh/latest/#installation

### 2. Install and test the project

```bash
git clone <your-repo-url>
cd bqn-jax
pixi install
pixi run test
```

### 3. Evaluate your first BQN expressions

```python
from bqn_jax import evaluate

evaluate("1 + 2 × 3")
evaluate("≢ (2‿3 ⥊ ↕6)")
evaluate("+´ 1‿2‿3‿4")
```

### 4. Explore the BQN tutorial notebooks

Example notebooks that recreate the official tutorial REPL inputs are in:

`examples/tutorials/`

There are two sets:

- `examples/tutorials/one_shot/` uses direct one-shot `evaluate("...")` calls.
- `examples/tutorials/stateful/` uses persistent state with `stateful = evaluate(env)`.

Open these notebooks in any Jupyter environment with `bqn_jax` available.
If you want to run them from the project Pixi environment, use the dedicated `jupyter` feature:

```bash
pixi install -e jupyter
pixi run -e jupyter jupyter-lab
```

Optional: register a reusable kernel in your local Jupyter installation:

```bash
pixi run -e jupyter jupyter-kernel-install
```

## Installation Options

### pip from a local checkout

```bash
pip install .
```

### pip from git

```bash
pip install "git+https://github.com/johtok/bqn-jax.git"
```

### editable install for development

```bash
pip install -e .[dev]
```

Note: Pixi packaging is configured with `pixi-build-python` in `pyproject.toml`, while pip packaging remains standard PEP 517 (`setuptools`).
Pixi workspace platforms are configured as `win-64`, `linux-64`, `osx-64`, and `osx-arm64`.
The Pixi Python range is pinned to `<3.13` for broad JAX wheel availability across these targets.

## Build a Conda Package (Pixi Build)

This repository is configured to build a conda package via `pixi-build-python`.

Build:

```bash
pixi build
```

Find the built artifact:

```bash
find .pixi -type f -name "bqn-jax-*.conda"
```

Install that artifact in Pixi:

```bash
pixi add /absolute/path/to/bqn-jax-<version>-<build>.conda
```

Install that artifact in Conda:

```bash
conda install /absolute/path/to/bqn-jax-<version>-<build>.conda
```

## Core Usage

### One-shot and stateful evaluation

`evaluate` supports three forms:

- `evaluate(expr)`
- `evaluate(env)` -> returns a stateful callable
- `evaluate(expr, env)` -> evaluates in an existing environment

```python
from bqn_jax import EvaluationEnvironment, evaluate

# one-shot
evaluate("1 + 2 × 3")  # -> 7

# explicit environment
env = EvaluationEnvironment({"x": 4})
out, env = evaluate("x + 2", env)  # -> (6, env)

# stateful callable
stateful = evaluate(env)
stateful("a ← 10")
stateful("F ← {𝕩 + a}")
stateful("F 3")  # -> 13
```

### JAX compile path

```python
import jax.numpy as jnp
from bqn_jax import ShapePolicy, compile_expression

compiled = compile_expression(
    "(x × x) + y",
    arg_names=("x", "y"),
    shape_policy=ShapePolicy(kind="dynamic"),
)

x = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
y = jnp.asarray([10.0, 20.0], dtype=jnp.float32)
compiled(x, y)

jit_fn = compiled.jit()
grad_fn = compile_expression(
    "x × x + 1",
    arg_names=("x",),
    shape_policy=ShapePolicy(kind="static"),
).grad()
```

### String and character output (JAX limitation)

JAX arrays are numeric-only. That means BQN string results are represented as
JAX `int32` codepoint arrays, not native string dtypes.

Example:

```python
out = evaluate('<⟜\'a\'⊸/ "Big Questions Notation"')  # -> [66 32 81 32 78]
text = "".join(chr(int(c)) for c in out.tolist())     # -> "B Q N"
```

Character scalars are represented as `BQNChar`.

## Common Commands

- `pixi run lint`
- `pixi run test`
- `pixi run conformance`
- `pixi run spec-audit`
- `pixi run bench`
- `pixi run bench-md -- --ns 10,1000,10000`
- `pixi run bench-md-shape`
- `pixi run bench-md-trends`
- `pixi run mlp`

Benchmark outputs are written under `benchmarks/output/`.

## Compatibility Scope

- The interpreter covers a broad practical BQN subset with strict runtime tests.
- The JAX backend covers the shared numerical subset and rejects runtime-only features with explicit errors.
- System/process/file/network mutation APIs are intentionally excluded by host policy.

## AI-Assisted Implementation

This repository was developed iteratively with AI assistance, grounded in the official BQN specification:
https://mlochbaum.github.io/BQN/spec/index.html

Validation is test-first, including parser/runtime conformance checks, differential checks, and benchmark suites.
