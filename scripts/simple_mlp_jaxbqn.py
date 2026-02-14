"""Terse jax-bqn MLP demo with Adam updates expressed in BQN."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from bqn_jax import ShapePolicy, compile_expression


E0 = "+´(v×÷⟜(1˙⊸+∘|)(w(+´∘×)x+b))+bo"
E1 = "((0.2 + (0.9 × (+´x))) - (0.3 × ((+´x) ⋆ 2))) + (0.4 × ((+´x) ⋆ 3))"
E2 = "+´×˜(p-y)÷≠y"
E3 = "√(+´×˜(p-y)÷≠y)"
E4 = "(beta1×m)+((1-beta1)×g)"
E5 = "(beta2×v)+((1-beta2)××˜g)"
E6 = "m÷(1-(beta1⋆t))"
E7 = "v÷(1-(beta2⋆t))"
E8 = "p-(lr×mhat÷((√vhat)+eps))"

_STATIC = ShapePolicy(kind="static")
_DYNAMIC = ShapePolicy(kind="dynamic")


def _ce_static(expr: str, arg_names: tuple[str, ...]):
    return compile_expression(expr, arg_names=arg_names, shape_policy=_STATIC)


def _ce_dynamic(expr: str, arg_names: tuple[str, ...]):
    return compile_expression(expr, arg_names=arg_names, shape_policy=_DYNAMIC)


F = _ce_dynamic(E0, ("x", "w", "b", "v", "bo"))
YF = _ce_dynamic(E1, ("x",))
LF = _ce_static(E2, ("p", "y"))
AF = _ce_dynamic(E3, ("p", "y"))
MUF = _ce_static(E4, ("m", "g", "beta1"))
VUF = _ce_static(E5, ("v", "g", "beta2"))
MHF = _ce_static(E6, ("m", "beta1", "t"))
VHF = _ce_static(E7, ("v", "beta2", "t"))
PUF = _ce_static(E8, ("p", "mhat", "vhat", "lr", "eps"))

Y_MAP = YF.vmap(in_axes=0, out_axes=0)
PRED_MAP = F.vmap(in_axes=(0, None, None, None, None), out_axes=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Train a terse array-first jax-bqn MLP on a polynomial mapping.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--num-samples", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--show-expression", action="store_true")
    a = p.parse_args()

    k = jax.random.PRNGKey(a.seed)
    x = jax.random.uniform(k, (a.num_samples, 1), minval=-1.5, maxval=1.5, dtype=jnp.float32)
    y = Y_MAP(x).astype(jnp.float32)
    n = int(0.8 * a.num_samples)
    xt, yt, xv, yv = x[:n], y[:n], x[n:], y[n:]

    k1, k2 = jax.random.split(jax.random.PRNGKey(a.seed + 1))
    w = (
        0.2 * jax.random.normal(k1, (1, 16), dtype=jnp.float32),
        jnp.zeros((16,), dtype=jnp.float32),
        0.2 * jax.random.normal(k2, (16,), dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    m_state = jax.tree_util.tree_map(jnp.zeros_like, w)
    v_state = jax.tree_util.tree_map(jnp.zeros_like, w)
    t = jnp.asarray(1.0, dtype=jnp.float32)
    lr = jnp.asarray(a.learning_rate, dtype=jnp.float32)
    beta1 = jnp.asarray(a.beta1, dtype=jnp.float32)
    beta2 = jnp.asarray(a.beta2, dtype=jnp.float32)
    eps = jnp.asarray(a.epsilon, dtype=jnp.float32)

    def pred(q, xb):
        return PRED_MAP(xb, q[0], q[1], q[2], q[3])

    def loss(q, xb, yb):
        return LF(pred(q, xb), yb)

    @jax.jit
    def step(q, m_acc, v_acc, step_t, xb, yb):
        l, grads = jax.value_and_grad(loss)(q, xb, yb)
        m_acc = jax.tree_util.tree_map(lambda m, g: MUF(m, g, beta1), m_acc, grads)
        v_acc = jax.tree_util.tree_map(lambda v, g: VUF(v, g, beta2), v_acc, grads)
        m_hat = jax.tree_util.tree_map(lambda m: MHF(m, beta1, step_t), m_acc)
        v_hat = jax.tree_util.tree_map(lambda v: VHF(v, beta2, step_t), v_acc)
        q = jax.tree_util.tree_map(lambda p, mh, vh: PUF(p, mh, vh, lr, eps), q, m_hat, v_hat)
        return q, m_acc, v_acc, step_t + 1.0, l

    def rmse(q, xb, yb):
        return float(AF(pred(q, xb), yb))

    print(f"epochs={a.epochs} lr={a.learning_rate} train={xt.shape[0]} test={xv.shape[0]}")
    if a.show_expression:
        print(
            f"mlp_expr_len={len(E0)}\n{E0}\ntarget_expr={E1}\nloss_expr={E2}\nrmse_expr={E3}\n"
            f"adam_m={E4}\nadam_v={E5}\nadam_mhat={E6}\nadam_vhat={E7}\nadam_update={E8}"
        )

    for e in range(a.epochs):
        w, m_state, v_state, t, l = step(w, m_state, v_state, t, xt, yt)
        if e % a.log_every == 0 or e == a.epochs - 1:
            print(f"epoch={e:4d} loss={float(l):.6f} train_rmse={rmse(w, xt, yt):.4f} test_rmse={rmse(w, xv, yv):.4f}")


if __name__ == "__main__":
    main()
