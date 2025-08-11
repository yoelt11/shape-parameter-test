#!/usr/bin/env python3
"""
Allen–Cahn (steady, forced) PINN comparison between Standard (full covariance) and Advanced Shape Transform RBF models.

We solve the manufactured steady Allen–Cahn:
  eps^2 * Δu + u - u^3 = g(x,y)
where u_true(x,y) = sin(kx π x) sin(ky π y), and g is computed from u_true.

Loss: mean squared residual of model u_pred:
  r = eps^2 * Δu_pred + u_pred - u_pred^3 - g

Outputs per wave number (kx,ky): loss curves, solution/error plots with kernel ellipses, and RESULTS.md.
"""

import os
import time
from datetime import datetime
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

jax.config.update('jax_enable_x64', True)

# Old-like config similar to good-performing Poisson run
EVAL_SMOOTH_STANDARD: bool = False
EVAL_SMOOTH_ADVANCED: bool = False
PROJECTION_EVERY_N: int = 1
WIDEN_BOUNDS: bool = False


def make_domain(n: int = 32, low: float = -1.0, high: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.linspace(low, high, n)
    y = jnp.linspace(low, high, n)
    X, Y = jnp.meshgrid(x, y)
    pts = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    return X, Y, pts


def allen_cahn_forcing(X: jnp.ndarray, Y: jnp.ndarray, eps: float, kx: int = 1, ky: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Manufactured solution u_true and forcing g for steady Allen–Cahn:
       eps^2 Δu + u - u^3 = g
       with u_true = sin(kx π x) sin(ky π y)
    """
    u_true = jnp.sin(kx * jnp.pi * X) * jnp.sin(ky * jnp.pi * Y)
    # Δu_true = - (kx^2 + ky^2) π^2 u_true
    lap_u_true = - (kx**2 + ky**2) * (jnp.pi ** 2) * u_true
    g = (eps**2) * lap_u_true + u_true - (u_true ** 3)
    return u_true, g


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
    mu_raw = params[:, 0:2]
    sigx_raw = params[:, 2]
    sigy_raw = params[:, 3]
    theta = params[:, 4]
    weights = params[:, 5]

    if EVAL_SMOOTH_STANDARD:
        x_min, x_max = jnp.min(X_pts[:, 0]), jnp.max(X_pts[:, 0])
        y_min, y_max = jnp.min(X_pts[:, 1]), jnp.max(X_pts[:, 1])
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
    epsv = params[:, 2]
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
        inv11 = jnp.abs(r * (1.0 + jnp.sin(epsv))) + 1e-6
        inv22 = jnp.abs(r * (1.0 + jnp.cos(epsv))) + 1e-6
        inv12 = (0.05 * (1.0 + jax.nn.softplus(scale_raw))) * jnp.sin(2 * epsv)
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
        inv11 = jnp.clip(jnp.abs(r * (1.0 + jnp.sin(epsv))) + 1e-6, 1e-6, 1e6)
        inv22 = jnp.clip(jnp.abs(r * (1.0 + jnp.cos(epsv))) + 1e-6, 1e-6, 1e6)
        inv12_raw = 10.0 * scale_raw * jnp.sin(2 * epsv)
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


def train_allen_cahn(model_name: str,
                     init_fn,
                     eval_fn,
                     X_pts: jnp.ndarray,
                     g_rhs: jnp.ndarray,
                     eps: float,
                     n_kernels: int = 64,
                     epochs: int = 300,
                     seed: int = 42,
                     lr: float = 1e-2,
                     X_grid: jnp.ndarray | None = None,
                     Y_grid: jnp.ndarray | None = None) -> Dict:
    key = jax.random.PRNGKey(seed)
    params = init_fn(n_kernels, key)

    def loss_fn(p):
        u_pred, lap_u = eval_fn(X_pts, p)
        r = (eps**2) * lap_u + u_pred - (u_pred ** 3) - g_rhs
        return jnp.mean(r ** 2)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, s):
        l, g = jax.value_and_grad(loss_fn)(p)
        up, s = opt.update(g, s)
        p = optax.apply_updates(p, up)
        return p, s, l

    losses = []
    t0 = time.time()
    best = 1e9
    pat = 25
    patience = 0
    for e in range(epochs):
        params, opt_state, l = step(params, opt_state)
        # projection cadence
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

    train_time = time.time() - t0
    iters = len(losses)
    it_per_s = iters / train_time if train_time > 0 else float('inf')
    return {
        'model': model_name,
        'final_loss': float(losses[-1]),
        'loss_history': losses,
        'epochs_run': iters,
        'train_time_s': train_time,
        'iters_per_s': it_per_s,
        'params': params,
    }


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'allen_cahn', timestamp)
    os.makedirs(outdir, exist_ok=True)

    X, Y, P = make_domain(n=32)
    n_k = 64
    epochs = 300
    lr = 1e-2
    eps_values = [0.05]
    wave_numbers = [(1,1), (2,1), (2,2)]

    def draw_ellipses_standard(ax, params, color='k', alpha=0.25):
        mus = np.array(params[:, 0:2])
        sigma_x = np.array(np.exp(np.array(params[:, 2])))
        sigma_y = np.array(np.exp(np.array(params[:, 3])))
        theta = np.array(params[:, 4])
        for (mx, my), sx, sy, th in zip(mus, sigma_x, sigma_y, theta):
            e = Ellipse(xy=(mx, my), width=2*sx, height=2*sy, angle=np.degrees(th),
                        edgecolor=color, facecolor='none', lw=0.8, alpha=alpha)
            ax.add_patch(e)

    def draw_ellipses_advanced(ax, params, color='k', alpha=0.25):
        mus = np.array(params[:, 0:2])
        epsv = np.array(params[:, 2])
        scales = np.array(params[:, 3])
        r = 100.0 * scales
        inv11 = np.clip(np.abs(r * (1.0 + np.sin(epsv))) + 1e-6, 1e-6, 1e6)
        inv22 = np.clip(np.abs(r * (1.0 + np.cos(epsv))) + 1e-6, 1e-6, 1e6)
        inv12_raw = 10.0 * scales * np.sin(2 * epsv)
        max_b = np.sqrt(np.maximum(inv11 * inv22 - 1e-12, 0.0))
        inv12 = np.clip(inv12_raw, -max_b, max_b)
        for i, (mx, my) in enumerate(mus):
            A = np.array([[inv11[i], inv12[i]], [inv12[i], inv22[i]]], dtype=float)
            w, V = np.linalg.eigh(A)
            w = np.clip(w, 1e-12, None)
            cov_eigs = 1.0 / w
            axes = 2 * np.sqrt(cov_eigs)
            angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            e = Ellipse(xy=(mx, my), width=axes[1], height=axes[0], angle=angle,
                        edgecolor=color, facecolor='none', lw=0.8, alpha=alpha)
            ax.add_patch(e)

    # Iterate cases
    for eps in eps_values:
        for (kx, ky) in wave_numbers:
            case_dir = os.path.join(outdir, f'eps{eps}_kx{kx}_ky{ky}')
            os.makedirs(case_dir, exist_ok=True)
            U_true, g = allen_cahn_forcing(X, Y, eps=eps, kx=kx, ky=ky)
            g_rhs = g.flatten()

            std_res = train_allen_cahn('Standard (Full)', standard_init, standard_eval_and_laplacian, P, g_rhs, eps,
                                       n_kernels=n_k, epochs=epochs, seed=42, lr=lr, X_grid=X, Y_grid=Y)
            adv_res = train_allen_cahn('Advanced Shape Transform', advanced_init, advanced_eval_and_laplacian, P, g_rhs, eps,
                                       n_kernels=n_k, epochs=epochs, seed=42, lr=lr, X_grid=X, Y_grid=Y)

            # Loss curves
            plt.figure(figsize=(7,4))
            plt.plot(std_res['loss_history'], label='Standard (Full)')
            plt.plot(adv_res['loss_history'], label='Advanced Shape Transform')
            plt.yscale('log')
            plt.xlabel('Epoch'); plt.ylabel('AC residual MSE')
            plt.title(f'Allen–Cahn Loss (eps={eps}, kx={kx}, ky={ky})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(case_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Predictions
            def eval_u(eval_fn, params):
                u_pred, _ = eval_fn(P, params)
                return u_pred.reshape(X.shape)

            U_std = eval_u(standard_eval_and_laplacian, std_res['params'])
            U_adv = eval_u(advanced_eval_and_laplacian, adv_res['params'])
            E_std = jnp.abs(U_std - U_true)
            E_adv = jnp.abs(U_adv - U_true)

            fig, axes = plt.subplots(3, 2, figsize=(10, 12))
            c0 = axes[0, 0].contourf(X, Y, U_std, levels=40, cmap='viridis')
            axes[0, 0].set_title('Standard: Prediction')
            fig.colorbar(c0, ax=axes[0, 0], fraction=0.046, pad=0.04)
            c1 = axes[0, 1].contourf(X, Y, U_adv, levels=40, cmap='viridis')
            axes[0, 1].set_title('Advanced: Prediction')
            fig.colorbar(c1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            c2 = axes[1, 0].contourf(X, Y, E_std, levels=40, cmap='magma')
            axes[1, 0].set_title('|Error| Standard')
            fig.colorbar(c2, ax=axes[1, 0], fraction=0.046, pad=0.04)
            c3 = axes[1, 1].contourf(X, Y, E_adv, levels=40, cmap='magma')
            axes[1, 1].set_title('|Error| Advanced')
            fig.colorbar(c3, ax=axes[1, 1], fraction=0.046, pad=0.04)
            axes[2, 0].contourf(X, Y, U_true, levels=20, cmap='Greys', alpha=0.3)
            draw_ellipses_standard(axes[2, 0], std_res['params'], color='cyan', alpha=0.6)
            axes[2, 0].set_title('Standard: Kernel ellipses (1σ)')
            axes[2, 0].set_aspect('equal', adjustable='box')
            axes[2, 1].contourf(X, Y, U_true, levels=20, cmap='Greys', alpha=0.3)
            draw_ellipses_advanced(axes[2, 1], adv_res['params'], color='lime', alpha=0.6)
            axes[2, 1].set_title('Advanced: Kernel ellipses (1σ)')
            axes[2, 1].set_aspect('equal', adjustable='box')
            for ax in axes.ravel():
                ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(False)
            fig.suptitle(f'Allen–Cahn comparison (eps={eps}, kx={kx}, ky={ky})', y=0.95)
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, 'solution_comparison_contourf.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # Simple overall report
    md = []
    md.append('# Allen–Cahn PINN Comparison\n')
    md.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    md.append('- Domain: [-1,1]^2, grid 32x32\n')
    md.append('- Steady forced Allen–Cahn: eps^2 Δu + u - u^3 = g (manufactured)\n')
    md.append('- Models: Standard (Full), Advanced Shape Transform\n')
    with open(os.path.join(outdir, 'RESULTS.md'), 'w', encoding='utf-8') as fmd:
        fmd.write('\n'.join(md))

    print('Saved results to', outdir)


if __name__ == '__main__':
    main()


