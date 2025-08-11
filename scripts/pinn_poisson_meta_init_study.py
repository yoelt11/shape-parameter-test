#!/usr/bin/env python3
"""
Meta-initialization study for PINN Poisson RBF models.
Runs multiple seeds per model to assess which model yields tighter parameter
distributions and more consistent convergence — a proxy for easier targets
for meta-initialization.

Outputs under results/pinn_poisson_meta/<timestamp>/ with:
- Per-seed metrics (JSON) and parameters (NPY)
- Aggregate parameter distribution plots (box/violin) and summary markdown
"""

import os
import json
from datetime import datetime
from typing import Dict, Tuple, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)


# ------------------------ Config ------------------------ #
# Choose a configuration similar to the "old-like good performance" the user noted
N_KERNELS = 64
EPOCHS = 300
LR = 1e-2
SEEDS = list(range(10))
WAVE_NUMBERS = [(1, 1), (2, 1), (2, 2)]

# Old-like behavior
EVAL_SMOOTH_STANDARD = False
EVAL_SMOOTH_ADVANCED = False
PROJECTION_EVERY_N = 1
WIDEN_BOUNDS = False


# ------------------------ Domain / GT ------------------------ #
def make_domain(n: int = 32, low: float = -1.0, high: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.linspace(low, high, n)
    y = jnp.linspace(low, high, n)
    X, Y = jnp.meshgrid(x, y)
    pts = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    return X, Y, pts


def poisson_gt_u_and_f(X: jnp.ndarray, Y: jnp.ndarray, kx: int = 1, ky: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    u = jnp.sin(kx * jnp.pi * X) * jnp.sin(ky * jnp.pi * Y)
    f = - (kx**2 + ky**2) * (jnp.pi ** 2) * u
    return u, f


# ------------------------ Projection helpers ------------------------ #
def apply_projection_standard(params: jnp.ndarray, X_grid: jnp.ndarray, Y_grid: jnp.ndarray) -> jnp.ndarray:
    x_min, x_max = X_grid.min(), X_grid.max()
    y_min, y_max = Y_grid.min(), Y_grid.max()
    n_points = X_grid.size
    out = params
    out = out.at[:, 0:2].set(jnp.clip(out[:, 0:2], jnp.array([x_min, y_min]), jnp.array([x_max, y_max])))
    domain_width = jnp.maximum(x_max - x_min, y_max - y_min)
    avg_point_spacing = domain_width / jnp.sqrt(n_points)
    if WIDEN_BOUNDS:
        min_sigma = avg_point_spacing / 4.0
        max_sigma = domain_width
    else:
        min_sigma = avg_point_spacing / 2.0
        max_sigma = domain_width / 2.0
    out = out.at[:, 2:4].set(jnp.clip(out[:, 2:4], jnp.log(min_sigma), jnp.log(max_sigma)))
    return out


def apply_projection_advanced(params: jnp.ndarray, X_grid: jnp.ndarray, Y_grid: jnp.ndarray) -> jnp.ndarray:
    x_min, x_max = X_grid.min(), X_grid.max()
    y_min, y_max = Y_grid.min(), Y_grid.max()
    n_points = X_grid.size
    out = params
    out = out.at[:, 0:2].set(jnp.clip(out[:, 0:2], jnp.array([x_min, y_min]), jnp.array([x_max, y_max])))
    domain_width = jnp.maximum(x_max - x_min, y_max - y_min)
    avg_point_spacing = domain_width / jnp.sqrt(n_points)
    if WIDEN_BOUNDS:
        min_scale = (avg_point_spacing / 4.0)
        max_scale = (domain_width)
    else:
        min_scale = (avg_point_spacing / 2.0)
        max_scale = (domain_width / 2.0)
    out = out.at[:, 3].set(jnp.clip(out[:, 3], min_scale, max_scale))
    return out


# ------------------------ Models ------------------------ #
def standard_init(n_kernels: int, key: jax.Array) -> jnp.ndarray:
    params = jnp.zeros((n_kernels, 6))
    key, k1 = jax.random.split(key)
    params = params.at[:, 0:2].set(jax.random.uniform(k1, (n_kernels, 2), minval=-0.8, maxval=0.8))
    params = params.at[:, 2].set(jnp.log(0.3) * jnp.ones(n_kernels))  # log_sigma_x
    params = params.at[:, 3].set(jnp.log(0.3) * jnp.ones(n_kernels))  # log_sigma_y
    params = params.at[:, 4].set(jnp.zeros(n_kernels))  # angle
    key, k2 = jax.random.split(key)
    params = params.at[:, 5].set(jax.random.normal(k2, (n_kernels,)) * 0.1)  # weights
    return params


def standard_eval_and_laplacian(X_pts: jnp.ndarray, params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x_min, x_max = jnp.min(X_pts[:, 0]), jnp.max(X_pts[:, 0])
    y_min, y_max = jnp.min(X_pts[:, 1]), jnp.max(X_pts[:, 1])
    mu_raw = params[:, 0:2]
    sigx_raw = params[:, 2]
    sigy_raw = params[:, 3]
    theta = params[:, 4]
    weights = params[:, 5]

    if EVAL_SMOOTH_STANDARD:
        n_points = X_pts.shape[0]
        domain_width = jnp.maximum(x_max - x_min, y_max - y_min)
        avg_point_spacing = domain_width / jnp.sqrt(n_points)
        sigma_min = avg_point_spacing / 2.0
        sigma_max = domain_width / 2.0
        mus = jnp.array([x_min, y_min]) + jax.nn.sigmoid(mu_raw) * jnp.array([x_max - x_min, y_max - y_min])
        sigma_x = sigma_min + jax.nn.sigmoid(sigx_raw) * (sigma_max - sigma_min)
        sigma_y = sigma_min + jax.nn.sigmoid(sigy_raw) * (sigma_max - sigma_min)
    else:
        mus = mu_raw
        sigma_x = jnp.exp(sigx_raw)
        sigma_y = jnp.exp(sigy_raw)

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    R = jnp.stack([jnp.stack([cos_t, -sin_t], axis=1), jnp.stack([sin_t, cos_t], axis=1)], axis=1)
    S = jnp.stack([jnp.stack([sigma_x ** 2, jnp.zeros_like(sigma_x)], axis=1),
                   jnp.stack([jnp.zeros_like(sigma_y), sigma_y ** 2], axis=1)], axis=1)
    C = jnp.matmul(jnp.matmul(R, S), jnp.swapaxes(R, 1, 2))
    A = jnp.linalg.inv(C)

    d = X_pts[:, None, :] - mus[None, :, :]
    Ad = jnp.einsum('kij,nkj->nki', A, d)
    quad = jnp.einsum('nki,nki->nk', d, Ad)
    phi = jnp.exp(-0.5 * quad)

    u_pred = jnp.dot(phi, weights)
    norm_Ad_sq = jnp.sum(Ad ** 2, axis=2)
    trace_A = A[:, 0, 0] + A[:, 1, 1]
    lap_phi = phi * (norm_Ad_sq - trace_A[None, :])
    lap_u = jnp.dot(lap_phi, weights)
    return u_pred, lap_u


def advanced_init(n_kernels: int, key: jax.Array) -> jnp.ndarray:
    params = jnp.zeros((n_kernels, 5))
    key, k1 = jax.random.split(key)
    params = params.at[:, 0:2].set(jax.random.uniform(k1, (n_kernels, 2), minval=-0.8, maxval=0.8))
    eps = jnp.linspace(0, 2 * jnp.pi, n_kernels, endpoint=False)
    params = params.at[:, 2].set(eps)
    params = params.at[:, 3].set(0.1 * jnp.ones(n_kernels))
    key, k2 = jax.random.split(key)
    params = params.at[:, 4].set(jax.random.normal(k2, (n_kernels,)) * 0.1)
    return params


def advanced_eval_and_laplacian(X_pts: jnp.ndarray, params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mu_raw = params[:, 0:2]
    eps = params[:, 2]
    scale_raw = params[:, 3]
    weights = params[:, 4]

    if EVAL_SMOOTH_ADVANCED:
        x_min, x_max = jnp.min(X_pts[:, 0]), jnp.max(X_pts[:, 0])
        y_min, y_max = jnp.min(X_pts[:, 1]), jnp.max(X_pts[:, 1])
        n_points = X_pts.shape[0]
        domain_width = jnp.maximum(x_max - x_min, y_max - y_min)
        avg_point_spacing = domain_width / jnp.sqrt(n_points)
        sigma_min = avg_point_spacing / 2.0
        sigma_max = domain_width / 2.0
        lam_min = 1.0 / (sigma_max ** 2)
        lam_max = 1.0 / (sigma_min ** 2)
        lam_mid = jnp.sqrt(lam_min * lam_max)

        mus = jnp.array([x_min, y_min]) + jax.nn.sigmoid(mu_raw) * jnp.array([x_max - x_min, y_max - y_min])
        r = (1.0 / (domain_width ** 2 + 1e-12)) * (1.0 + jax.nn.softplus(scale_raw))
        inv11 = jnp.abs(r * (1.0 + jnp.sin(eps))) + 1e-6
        inv22 = jnp.abs(r * (1.0 + jnp.cos(eps))) + 1e-6
        inv12 = (0.05 * (1.0 + jax.nn.softplus(scale_raw))) * jnp.sin(2 * eps)
        A = jnp.zeros((params.shape[0], 2, 2))
        A = A.at[:, 0, 0].set(inv11)
        A = A.at[:, 1, 1].set(inv22)
        A = A.at[:, 0, 1].set(inv12)
        A = A.at[:, 1, 0].set(inv12)
        w, V = jnp.linalg.eigh(A)
        beta = 0.5
        w_smooth = lam_min + (lam_max - lam_min) * jax.nn.sigmoid(beta * (w - lam_mid))
        A = jnp.einsum('kij,kj,kjl->kil', V, w_smooth, jnp.swapaxes(V, 1, 2))
    else:
        mus = mu_raw
        r = 100.0 * scale_raw
        inv11 = jnp.clip(jnp.abs(r * (1.0 + jnp.sin(eps))) + 1e-6, 1e-6, 1e6)
        inv22 = jnp.clip(jnp.abs(r * (1.0 + jnp.cos(eps))) + 1e-6, 1e-6, 1e6)
        inv12_raw = 10.0 * scale_raw * jnp.sin(2 * eps)
        max_b = jnp.sqrt(jnp.maximum(inv11 * inv22 - 1e-12, 0.0))
        inv12 = jnp.clip(inv12_raw, -max_b, max_b)
        A = jnp.zeros((params.shape[0], 2, 2))
        A = A.at[:, 0, 0].set(inv11)
        A = A.at[:, 1, 1].set(inv22)
        A = A.at[:, 0, 1].set(inv12)
        A = A.at[:, 1, 0].set(inv12)

    d = X_pts[:, None, :] - mus[None, :, :]
    Ad = jnp.einsum('kij,nkj->nki', A, d)
    quad = jnp.einsum('nki,nki->nk', d, Ad)
    phi = jnp.exp(-0.5 * quad)

    u_pred = jnp.dot(phi, weights)
    norm_Ad_sq = jnp.sum(Ad ** 2, axis=2)
    trace_A = A[:, 0, 0] + A[:, 1, 1]
    lap_phi = phi * (norm_Ad_sq - trace_A[None, :])
    lap_u = jnp.dot(lap_phi, weights)
    return u_pred, lap_u


# ------------------------ Training ------------------------ #
def train_pinn(model_name: str,
               init_fn,
               eval_fn,
               X_pts: jnp.ndarray,
               f_rhs: jnp.ndarray,
               n_kernels: int = 64,
               epochs: int = 300,
               seed: int = 0,
               lr: float = 1e-2,
               X_grid: jnp.ndarray | None = None,
               Y_grid: jnp.ndarray | None = None) -> Dict:
    key = jax.random.PRNGKey(seed)
    params = init_fn(n_kernels, key)
    params0 = params

    def loss_fn(p):
        _, lap_u = eval_fn(X_pts, p)
        res = lap_u - f_rhs
        return jnp.mean(res ** 2)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, s):
        l, g = jax.value_and_grad(loss_fn)(p)
        up, s = opt.update(g, s)
        p = optax.apply_updates(p, up)
        return p, s, l

    losses: List[float] = []
    best = 1e9
    pat = 25
    patience = 0
    for e in range(epochs):
        params, opt_state, l = step(params, opt_state)
        # Projection each step (old-like) if desired
        if PROJECTION_EVERY_N and (e % PROJECTION_EVERY_N == 0):
            if model_name == 'Standard (Full)':
                params = apply_projection_standard(params, X_grid, Y_grid)
            else:
                params = apply_projection_advanced(params, X_grid, Y_grid)
        lv = float(l)
        losses.append(lv)
        if lv < best - 1e-8:
            best = lv
            patience = 0
        else:
            patience += 1
        if patience >= pat:
            break

    return {
        'model': model_name,
        'final_loss': float(losses[-1]),
        'loss_history': losses,
        'epochs_run': len(losses),
        'params0': np.array(params0),
        'params': np.array(params),
    }


# ------------------------ Aggregation / Plots ------------------------ #
def aggregate_and_plot(model_name: str, all_runs: List[Dict], outdir: str, wave_tag: str):
    # Collect final losses
    final_losses = np.array([r['final_loss'] for r in all_runs])
    # Collect parameter arrays
    params_arr = np.array([r['params'] for r in all_runs])  # shape: (S, K, D)

    # Build per-parameter-type arrays
    if model_name == 'Standard (Full)':
        mus = params_arr[:, :, 0:2].reshape(len(all_runs), -1)
        log_sigx = params_arr[:, :, 2].reshape(len(all_runs), -1)
        log_sigy = params_arr[:, :, 3].reshape(len(all_runs), -1)
        angles = params_arr[:, :, 4].reshape(len(all_runs), -1)
        weights = params_arr[:, :, 5].reshape(len(all_runs), -1)

        # Derived: anisotropy ratio in sigma-space
        sigx = np.exp(log_sigx)
        sigy = np.exp(log_sigy)
        anis_ratio = sigx / (sigy + 1e-12)

        to_plot = {
            'mu_x': mus[:, 0::2],
            'mu_y': mus[:, 1::2],
            'sigma_x': sigx,
            'sigma_y': sigy,
            'anisotropy_ratio': anis_ratio,
            'angle': angles,
            'weight': weights,
        }
    else:
        mus = params_arr[:, :, 0:2].reshape(len(all_runs), -1)
        eps = params_arr[:, :, 2].reshape(len(all_runs), -1)
        scale = params_arr[:, :, 3].reshape(len(all_runs), -1)
        weights = params_arr[:, :, 4].reshape(len(all_runs), -1)
        to_plot = {
            'mu_x': mus[:, 0::2],
            'mu_y': mus[:, 1::2],
            'epsilon': eps,
            'scale': scale,
            'weight': weights,
        }

    # Boxplots
    plt.style.use('default')
    fig, axes = plt.subplots(1, len(to_plot), figsize=(4 * len(to_plot), 4), constrained_layout=True)
    if len(to_plot) == 1:
        axes = [axes]
    for ax, (name, arr) in zip(axes, to_plot.items()):
        # Flatten across kernels and seeds
        data = arr.flatten()
        # Downsample for speed if huge
        if data.size > 20000:
            data = np.random.choice(data, size=20000, replace=False)
        ax.boxplot(data, vert=True)
        ax.set_title(name)
    fig.suptitle(f'{model_name} parameter distributions ({wave_tag})')
    fig.savefig(os.path.join(outdir, f'{wave_tag}_{model_name.replace(" ", "_")}_param_boxplots.png'), dpi=200)
    plt.close(fig)

    # Summary json
    summary = {
        'model': model_name,
        'wave': wave_tag,
        'final_loss_mean': float(np.mean(final_losses)),
        'final_loss_std': float(np.std(final_losses)),
        'num_seeds': len(all_runs),
    }
    with open(os.path.join(outdir, f'{wave_tag}_{model_name.replace(" ", "_")}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'pinn_poisson_meta', timestamp)
    os.makedirs(outdir, exist_ok=True)

    X, Y, P = make_domain(n=32)

    reports = []

    for (kx, ky) in WAVE_NUMBERS:
        wave_tag = f'kx{kx}_ky{ky}'
        case_dir = os.path.join(outdir, wave_tag)
        os.makedirs(case_dir, exist_ok=True)
        U_true, f = poisson_gt_u_and_f(X, Y, kx=kx, ky=ky)
        f_rhs = f.flatten()

        model_specs = [
            ('Standard (Full)', standard_init, standard_eval_and_laplacian),
            ('Advanced Shape Transform', advanced_init, advanced_eval_and_laplacian),
        ]

        for model_name, init_fn, eval_fn in model_specs:
            runs = []
            for seed in SEEDS:
                res = train_pinn(model_name, init_fn, eval_fn, P, f_rhs,
                                  n_kernels=N_KERNELS, epochs=EPOCHS, seed=seed, lr=LR,
                                  X_grid=X, Y_grid=Y)
                runs.append(res)
                # Save per-seed params
                np.save(os.path.join(case_dir, f'{model_name.replace(" ", "_")}_seed{seed}_params.npy'), res['params'])
                np.save(os.path.join(case_dir, f'{model_name.replace(" ", "_")}_seed{seed}_params0.npy'), res['params0'])
                with open(os.path.join(case_dir, f'{model_name.replace(" ", "_")}_seed{seed}_metrics.json'), 'w') as f:
                    json.dump({k: v for k, v in res.items() if k not in ('params', 'params0')}, f, indent=2)

            aggregate_and_plot(model_name, runs, case_dir, wave_tag)

            # Report
            final_losses = [r['final_loss'] for r in runs]
            reports.append({
                'wave': wave_tag,
                'model': model_name,
                'loss_mean': float(np.mean(final_losses)),
                'loss_std': float(np.std(final_losses)),
            })

    # Markdown conclusions
    md = []
    md.append('# Meta-initialization Study (Poisson PINN)\n')
    md.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    md.append('Config: old-like (raw eval, projection every step, narrow bounds)\n')
    md.append(f'Seeds: {len(SEEDS)}, Kernels: {N_KERNELS}, Epochs: {EPOCHS}\n')
    md.append('## Summary (final loss mean ± std)\n')
    for r in reports:
        md.append(f"- {r['wave']} — {r['model']}: {r['loss_mean']:.3e} ± {r['loss_std']:.2e}\n")
    md.append('\n')
    md.append('Interpretation: Tighter parameter distributions and lower loss variance across seeds suggest a model that is an easier meta-target. Use the saved per-seed parameter arrays to fit a meta-initializer (e.g., fit priors or medians per-parameter-type). Note kernel permutation invariance — aggregate distributions ignoring kernel index or align kernels via clustering before averaging.\n')
    with open(os.path.join(outdir, 'CONCLUSIONS.md'), 'w') as f:
        f.write('\n'.join(md))

    print('Saved meta study to', outdir)


if __name__ == '__main__':
    main()


