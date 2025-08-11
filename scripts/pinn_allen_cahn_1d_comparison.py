#!/usr/bin/env python3
"""
1D Allen–Cahn PINN comparison using 2D RBFs over (t, x).

PDE: u_t = eps * u_xx + u - u^3,  x in [-1,1], t in [0,1]
IC: u(0,x) = x^2 cos(pi x)
BC: u(t,-1) = 0, u(t,1) = 0

We compare Standard (full covariance) vs Advanced Shape Transform kernels,
following the structure of the 1D wave experiment. Since no closed-form
solution is provided, we visualize predicted u(t,x) and residual heatmaps.
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

# Old-like configuration to match prior stable runs
EVAL_SMOOTH_STANDARD: bool = False
EVAL_SMOOTH_ADVANCED: bool = False
PROJECTION_EVERY_N: int = 1
WIDEN_BOUNDS: bool = False


def make_domain(n_t: int = 64, n_x: int = 128) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    t = jnp.linspace(0.0, 1.0, n_t)
    x = jnp.linspace(-1.0, 1.0, n_x)
    T, X = jnp.meshgrid(t, x, indexing='ij')
    pts = jnp.stack([T.flatten(), X.flatten()], axis=1)
    return T, X, pts


# ------------------------ Projection helpers ------------------------ #
def apply_projection_standard(params: jnp.ndarray, T_grid: jnp.ndarray, X_grid: jnp.ndarray) -> jnp.ndarray:
    # params: [mu_t, mu_x, log_sigma_t, log_sigma_x, angle, weight]
    t_min, t_max = 0.0, 1.0
    x_min, x_max = -1.0, 1.0
    out = params
    out = out.at[:, 0].set(jnp.clip(out[:, 0], t_min, t_max))
    out = out.at[:, 1].set(jnp.clip(out[:, 1], x_min, x_max))
    dt = (t_max - t_min) / max(T_grid.shape[0]-1, 1)
    dx = (x_max - x_min) / max(X_grid.shape[1]-1, 1)
    if WIDEN_BOUNDS:
        min_sig_t, max_sig_t = dt / 4.0, (t_max - t_min)
        min_sig_x, max_sig_x = dx / 4.0, (x_max - x_min)
    else:
        min_sig_t, max_sig_t = dt / 2.0, (t_max - t_min) / 2.0
        min_sig_x, max_sig_x = dx / 2.0, (x_max - x_min) / 2.0
    out = out.at[:, 2].set(jnp.clip(out[:, 2], jnp.log(min_sig_t), jnp.log(max_sig_t)))
    out = out.at[:, 3].set(jnp.clip(out[:, 3], jnp.log(min_sig_x), jnp.log(max_sig_x)))
    return out


def apply_projection_advanced(params: jnp.ndarray, T_grid: jnp.ndarray, X_grid: jnp.ndarray) -> jnp.ndarray:
    # params: [mu_t, mu_x, epsilon, scale, weight]
    t_min, t_max = 0.0, 1.0
    x_min, x_max = -1.0, 1.0
    out = params
    out = out.at[:, 0].set(jnp.clip(out[:, 0], t_min, t_max))
    out = out.at[:, 1].set(jnp.clip(out[:, 1], x_min, x_max))
    dt = (t_max - t_min) / max(T_grid.shape[0]-1, 1)
    dx = (x_max - x_min) / max(X_grid.shape[1]-1, 1)
    dom = jnp.maximum(t_max - t_min, x_max - x_min)
    if WIDEN_BOUNDS:
        min_scale, max_scale = (min(dt, dx) / 4.0), dom
    else:
        min_scale, max_scale = (min(dt, dx) / 2.0), dom / 2.0
    out = out.at[:, 3].set(jnp.clip(out[:, 3], min_scale, max_scale))
    return out


# ------------------------ Models and eval (like wave) ------------------------ #
def standard_init(n_kernels: int, key: jax.Array) -> jnp.ndarray:
    params = jnp.zeros((n_kernels, 6))
    key, k1 = jax.random.split(key)
    params = params.at[:, 0:2].set(jax.random.uniform(k1, (n_kernels, 2), minval=0.0, maxval=1.0) * jnp.array([1.0, 2.0]) + jnp.array([0.0, -1.0]))
    # map to t in [0,1], x in [-1,1]
    # log sigmas
    params = params.at[:, 2].set(jnp.log(0.2) * jnp.ones(n_kernels))
    params = params.at[:, 3].set(jnp.log(0.2) * jnp.ones(n_kernels))
    params = params.at[:, 4].set(jnp.zeros(n_kernels))
    key, k2 = jax.random.split(key)
    params = params.at[:, 5].set(jax.random.normal(k2, (n_kernels,)) * 0.1)
    return params


def standard_eval_with_derivs(P: jnp.ndarray, params: jnp.ndarray):
    mu = params[:, 0:2]
    log_sig_t = params[:, 2]
    log_sig_x = params[:, 3]
    theta = params[:, 4]
    weights = params[:, 5]

    if EVAL_SMOOTH_STANDARD:
        t_min, t_max = 0.0, 1.0
        x_min, x_max = -1.0, 1.0
        mu = jnp.array([t_min, x_min]) + jax.nn.sigmoid(mu) * jnp.array([t_max - t_min, x_max - x_min])
        sigma_min_t, sigma_max_t = 1.0/64.0, 0.5
        sigma_min_x, sigma_max_x = 2.0/128.0, 1.0
        sig_t = sigma_min_t + jax.nn.sigmoid(log_sig_t) * (sigma_max_t - sigma_min_t)
        sig_x = sigma_min_x + jax.nn.sigmoid(log_sig_x) * (sigma_max_x - sigma_min_x)
    else:
        sig_t = jnp.exp(log_sig_t)
        sig_x = jnp.exp(log_sig_x)

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    R = jnp.stack([jnp.stack([cos_t, -sin_t], axis=1), jnp.stack([sin_t, cos_t], axis=1)], axis=1)
    S = jnp.stack([jnp.stack([sig_t ** 2, jnp.zeros_like(sig_t)], axis=1),
                   jnp.stack([jnp.zeros_like(sig_x), sig_x ** 2], axis=1)], axis=1)
    C = jnp.matmul(jnp.matmul(R, S), jnp.swapaxes(R, 1, 2))
    A = jnp.linalg.inv(C)

    d = P[:, None, :] - mu[None, :, :]
    Ad = jnp.einsum('kij,nkj->nki', A, d)
    quad = jnp.einsum('nki,nki->nk', d, Ad)
    phi = jnp.exp(-0.5 * quad)

    u = jnp.dot(phi, weights)
    grad_phi = -phi[:, :, None] * Ad
    du_dt = jnp.dot(grad_phi[:, :, 0], weights)
    du_dx = jnp.dot(grad_phi[:, :, 1], weights)
    A_tt = A[:, 0, 0]
    A_xx = A[:, 1, 1]
    d2u_dt2 = jnp.dot(phi * (Ad[:, :, 0] ** 2 - A_tt[None, :]), weights)
    d2u_dx2 = jnp.dot(phi * (Ad[:, :, 1] ** 2 - A_xx[None, :]), weights)
    return u, du_dt, du_dx, d2u_dt2, d2u_dx2


def advanced_init(n_kernels: int, key: jax.Array) -> jnp.ndarray:
    params = jnp.zeros((n_kernels, 5))
    key, k1 = jax.random.split(key)
    params = params.at[:, 0:2].set(jax.random.uniform(k1, (n_kernels, 2), minval=0.0, maxval=1.0) * jnp.array([1.0, 2.0]) + jnp.array([0.0, -1.0]))
    eps = jnp.linspace(0, 2 * jnp.pi, n_kernels, endpoint=False)
    params = params.at[:, 2].set(eps)
    params = params.at[:, 3].set(0.1 * jnp.ones(n_kernels))
    key, k2 = jax.random.split(key)
    params = params.at[:, 4].set(jax.random.normal(k2, (n_kernels,)) * 0.1)
    return params


def advanced_eval_with_derivs(P: jnp.ndarray, params: jnp.ndarray):
    mu = params[:, 0:2]
    epsv = params[:, 2]
    scale_raw = params[:, 3]
    weights = params[:, 4]

    if EVAL_SMOOTH_ADVANCED:
        mu = jax.nn.sigmoid(mu) * jnp.array([1.0, 2.0]) + jnp.array([0.0, -1.0])
        r = (1.0) * (1.0 + jax.nn.softplus(scale_raw))
        inv11 = jnp.abs(r * (1.0 + jnp.sin(epsv))) + 1e-6
        inv22 = jnp.abs(r * (1.0 + jnp.cos(epsv))) + 1e-6
        inv12 = (0.05 * (1.0 + jax.nn.softplus(scale_raw))) * jnp.sin(2 * epsv)
        A = jnp.zeros((params.shape[0], 2, 2))
        A = A.at[:, 0, 0].set(inv11)
        A = A.at[:, 1, 1].set(inv22)
        A = A.at[:, 0, 1].set(inv12)
        A = A.at[:, 1, 0].set(inv12)
    else:
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

    d = P[:, None, :] - mu[None, :, :]
    Ad = jnp.einsum('kij,nkj->nki', A, d)
    quad = jnp.einsum('nki,nki->nk', d, Ad)
    phi = jnp.exp(-0.5 * quad)

    u = jnp.dot(phi, weights)
    grad_phi = -phi[:, :, None] * Ad
    du_dt = jnp.dot(grad_phi[:, :, 0], weights)
    du_dx = jnp.dot(grad_phi[:, :, 1], weights)
    A_tt = A[:, 0, 0]
    A_xx = A[:, 1, 1]
    d2u_dt2 = jnp.dot(phi * (Ad[:, :, 0] ** 2 - A_tt[None, :]), weights)
    d2u_dx2 = jnp.dot(phi * (Ad[:, :, 1] ** 2 - A_xx[None, :]), weights)
    return u, du_dt, du_dx, d2u_dt2, d2u_dx2


# ------------------------ Training ------------------------ #
def train_allen_cahn(model_name: str,
                     init_fn,
                     eval_fn,
                     P: jnp.ndarray,
                     T: jnp.ndarray,
                     X: jnp.ndarray,
                     eps: float,
                     n_kernels: int = 64,
                     epochs: int = 400,
                     seed: int = 42,
                     lr: float = 1e-2) -> Dict:
    key = jax.random.PRNGKey(seed)
    params = init_fn(n_kernels, key)

    # Collocation sets
    P_res = P
    # IC at t=0
    x_ic = jnp.linspace(-1.0, 1.0, X.shape[1])
    P_ic = jnp.stack([jnp.zeros_like(x_ic), x_ic], axis=1)
    u0 = (x_ic ** 2) * jnp.cos(jnp.pi * x_ic)
    # BC x=-1 and x=1 for all t
    t_bc = jnp.linspace(0.0, 1.0, T.shape[0])
    P_bc0 = jnp.stack([t_bc, -jnp.ones_like(t_bc)], axis=1)
    P_bc1 = jnp.stack([t_bc,  jnp.ones_like(t_bc)], axis=1)

    def residual_and_terms(p):
        u, du_dt, _, _, d2u_dx2 = eval_fn(P_res, p)
        res = du_dt - eps * d2u_dx2 - (u - u ** 3)
        return res

    def loss_fn(p):
        # PDE residual
        res = residual_and_terms(p)
        loss_res = jnp.mean(res ** 2)
        # IC
        u_ic, *_ = eval_fn(P_ic, p)
        loss_ic = jnp.mean((u_ic - u0) ** 2)
        # BCs
        u_bc0, *_ = eval_fn(P_bc0, p)
        u_bc1, *_ = eval_fn(P_bc1, p)
        loss_bc = jnp.mean(u_bc0 ** 2) + jnp.mean(u_bc1 ** 2)
        return loss_res + loss_ic + loss_bc

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, s):
        l, g = jax.value_and_grad(loss_fn)(p)
        up, s = opt.update(g, s)
        p = optax.apply_updates(p, up)
        return p, s, l

    losses = []
    best = 1e9
    pat = 30
    patience = 0
    for e in range(epochs):
        params, opt_state, l = step(params, opt_state)
        if PROJECTION_EVERY_N and (e % PROJECTION_EVERY_N == 0):
            if model_name == 'Standard (Full)':
                params = apply_projection_standard(params, T, X)
            else:
                params = apply_projection_advanced(params, T, X)
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
        'params': params,
    }


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'allen_cahn_1d', timestamp)
    os.makedirs(outdir, exist_ok=True)

    T, X, P = make_domain(n_t=64, n_x=128)
    n_k = 64
    epochs = 400
    lr = 1e-2
    # eps choices from literature
    eps_values = [0.0001 / (jnp.pi ** 2), 0.01]

    for eps in eps_values:
        case_dir = os.path.join(outdir, f'eps{float(eps):.6f}')
        os.makedirs(case_dir, exist_ok=True)

        std_res = train_allen_cahn('Standard (Full)', standard_init, standard_eval_with_derivs, P, T, X,
                                   eps=float(eps), n_kernels=n_k, epochs=epochs, seed=42, lr=lr)
        adv_res = train_allen_cahn('Advanced Shape Transform', advanced_init, advanced_eval_with_derivs, P, T, X,
                                   eps=float(eps), n_kernels=n_k, epochs=epochs, seed=42, lr=lr)

        # Loss curves
        plt.figure(figsize=(7,4))
        plt.plot(std_res['loss_history'], label='Standard (Full)')
        plt.plot(adv_res['loss_history'], label='Advanced Shape Transform')
        plt.yscale('log')
        plt.xlabel('Epoch'); plt.ylabel('AC residual + IC/BC MSE')
        plt.title(f'Allen–Cahn 1D PINN Loss (eps={float(eps):.6f})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(case_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Predictions and residual heatmaps
        def eval_grid(eval_fn, params):
            u, du_dt, _, _, d2u_dx2 = eval_fn(P, params)
            res = du_dt - float(eps) * d2u_dx2 - (u - u ** 3)
            return u.reshape(T.shape), res.reshape(T.shape)

        U_std, R_std = eval_grid(standard_eval_with_derivs, std_res['params'])
        U_adv, R_adv = eval_grid(advanced_eval_with_derivs, adv_res['params'])

        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        im0 = axes[0, 0].imshow(U_std, origin='lower', aspect='auto', extent=[-1,1,0,1], cmap='viridis')
        axes[0, 0].set_title('Standard: u(t,x)')
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        im1 = axes[0, 1].imshow(U_adv, origin='lower', aspect='auto', extent=[-1,1,0,1], cmap='viridis')
        axes[0, 1].set_title('Advanced: u(t,x)')
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        im2 = axes[1, 0].imshow(jnp.abs(R_std), origin='lower', aspect='auto', extent=[-1,1,0,1], cmap='magma')
        axes[1, 0].set_title('|Residual| Standard')
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        im3 = axes[1, 1].imshow(jnp.abs(R_adv), origin='lower', aspect='auto', extent=[-1,1,0,1], cmap='magma')
        axes[1, 1].set_title('|Residual| Advanced')
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # Kernel ellipses on (t,x)
        def draw_ellipses_standard(ax, params, color='cyan', alpha=0.5):
            mus = np.array(params[:, 0:2])
            sig_t = np.array(np.exp(np.array(params[:, 2])))
            sig_x = np.array(np.exp(np.array(params[:, 3])))
            theta = np.array(params[:, 4])
            for (mt, mx), st, sx, th in zip(mus, sig_t, sig_x, theta):
                e = Ellipse((mx, mt), width=2*sx, height=2*st, angle=np.degrees(th),
                            edgecolor=color, facecolor='none', lw=0.6, alpha=alpha)
                ax.add_patch(e)

        def draw_ellipses_advanced(ax, params, color='lime', alpha=0.5):
            mus = np.array(params[:, 0:2])
            epsv = np.array(params[:, 2])
            scales = np.array(params[:, 3])
            r = 100.0 * scales
            inv11 = np.clip(np.abs(r * (1.0 + np.sin(epsv))) + 1e-6, 1e-6, 1e6)
            inv22 = np.clip(np.abs(r * (1.0 + np.cos(epsv))) + 1e-6, 1e-6, 1e6)
            inv12_raw = 10.0 * scales * np.sin(2 * epsv)
            max_b = np.sqrt(np.maximum(inv11 * inv22 - 1e-12, 0.0))
            inv12 = np.clip(inv12_raw, -max_b, max_b)
            for i, (mt, mx) in enumerate(mus):
                A = np.array([[inv11[i], inv12[i]], [inv12[i], inv22[i]]], dtype=float)
                w, V = np.linalg.eigh(A)
                w = np.clip(w, 1e-12, None)
                cov_eigs = 1.0 / w
                axes_len = 2 * np.sqrt(cov_eigs)
                angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
                e = Ellipse((mx, mt), width=axes_len[1], height=axes_len[0], angle=angle,
                            edgecolor=color, facecolor='none', lw=0.6, alpha=alpha)
                ax.add_patch(e)

        axes[2, 0].imshow(jnp.zeros_like(U_std), origin='lower', aspect='auto', extent=[-1,1,0,1], cmap='Greys', alpha=0.1)
        draw_ellipses_standard(axes[2, 0], std_res['params'])
        axes[2, 0].set_title('Standard: kernel ellipses (t,x)')
        axes[2, 1].imshow(jnp.zeros_like(U_adv), origin='lower', aspect='auto', extent=[-1,1,0,1], cmap='Greys', alpha=0.1)
        draw_ellipses_advanced(axes[2, 1], adv_res['params'])
        axes[2, 1].set_title('Advanced: kernel ellipses (t,x)')
        for ax in axes.ravel():
            ax.set_xlabel('x'); ax.set_ylabel('t')
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, 'solution_and_residual_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Report
    md = []
    md.append('# 1D Allen–Cahn PINN Comparison\n')
    md.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    md.append('- Domain: t in [0,1], x in [-1,1], grid 64x128\n')
    md.append('- PDE: u_t = eps u_xx + u - u^3; IC u(0,x)=x^2 cos(pi x); BC u(t,±1)=0\n')
    md.append('- Models: Standard (Full), Advanced Shape Transform\n')
    with open(os.path.join(outdir, 'RESULTS.md'), 'w', encoding='utf-8') as fmd:
        fmd.write('\n'.join(md))

    print('Saved results to', outdir)


if __name__ == '__main__':
    main()


