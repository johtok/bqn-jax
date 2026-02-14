# BQN Tutorial Notebooks

The official tutorial examples have been rebuilt into two notebook sets:

- `one_shot/`: each example is a direct one-shot `evaluate("...")` call.
- `stateful/`: each example uses persistent state via `env = EvaluationEnvironment()` and `stateful = evaluate(env)`.

If you're new:

- Start with `one_shot/` to learn syntax without worrying about saved state.
- Then use `stateful/` to learn how definitions carry across cells.

Source tutorial index:
https://mlochbaum.github.io/BQN/tutorial/index.html

## Notebooks

Each set contains:

- `01_expressions.ipynb`
- `02_list_manipulation.ipynb`
- `03_combinators.ipynb`
- `04_variables.ipynb`

## Run

Open in any Jupyter environment with `bqn_jax` available.
For Pixi users:

```bash
pixi install -e jupyter
pixi run -e jupyter jupyter-lab
```

Optional: install a named local kernel for reuse across Jupyter sessions:

```bash
pixi run -e jupyter jupyter-kernel-install
```

When running a notebook, execute cells from top to bottom.
In `stateful/`, order matters because later cells may depend on earlier definitions.
